// src/kernels/cuda/hc_q8.cuh - the hyper-connection read from the GGUF's Q8_0 projections (STRATA_HC_Q8=1).
//
// A FRAGMENT, like fused_gr_common.cuh (whose names it uses): fused_gr.cu includes it inside its anonymous namespace,
// and tests/cuda_emu/hc_q8_emu_test.cpp compiles the same device code for the CPU.
//
// ================================ S23 experiment (STRATA_HC_Q8=1): the read with the GGUF's Q8_0 projections ======
// The pack holds the hyper-connection projections as BF16 (iq_pack.py --compat-bf16 rounds the GGUF's Q8_0 values to
// BF16: 1.20 GiB per verify step); this read takes the Q8_0 bytes as stored (0.65 GiB - and the GGUF's own values,
// not a rounding of them).  The layout is the stream split of the v3 read above, at 4 chunks per stream: `down` runs
// on (10 groups of 32 rows + the inject rows) x 16 chunks of 640 columns = 176 blocks, each staging its chunk of
// R' * w_norm (unscaled) for the T tokens and writing the chunk's partial dots and sums of squares; `up` sums them
// per stream in a fixed order, applies rs, SwiGLU-free silu, and runs the default epilogue.  Another summation order
// and other weights than the default read: an output-changing step (measured, quality-checked).
constexpr int Q8B = 34;                             // Q8_0 block: fp16 d + 32 int8
constexpr int Q8_RPW = 4;                           // down rows per warp
constexpr int Q8_RG = LR / (WARPS * Q8_RPW);        // 10 row groups (+1: the inject rows)
constexpr int Q8_KC = 640, Q8_NKC = D / Q8_KC;      // 16 chunks, 4 per stream
constexpr int Q8_CPS = N / Q8_KC;                   // chunks per stream
constexpr int Q8_SPB = Q8_KC / 128;                 // 5 steps of 4 Q8_0 blocks (8 lanes x 4 values each)
__device__ __forceinline__ uint32_t q8_ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ float q8_half(uint32_t h) { return __half2float(__ushort_as_half((unsigned short) h)); }
__device__ __forceinline__ float4 q8_four(uint32_t q, float d) {
    return make_float4(d * (float) (int8_t) (q & 255u), d * (float) (int8_t) ((q >> 8) & 255u),
                       d * (float) (int8_t) ((q >> 16) & 255u), d * (float) (int8_t) (q >> 24));
}

template <int T>
__global__ void __launch_bounds__(THREADS) gr_down_q8_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    STRATA_SHARED(float, xs, [T][Q8_KC]);
    STRATA_SHARED(float, red, [WARPS][T]);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, kc = blockIdx.y, c = kc / Q8_CPS;
    const bool inj = rg == Q8_RG;
    // S25: the inject rows are F32 in the GGUF with BF16-exact values, so by default they are read as the pack's BF16
    // (exact); a Q8_0 copy (STRATA_HC_Q8_INJECT=1) is the other option
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : Q8_RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * Q8_RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;       // lanes 8b..8b+7: one Q8_0 block, 4 values each
    uint32_t q[Q8_RPW][Q8_SPB];
    float dq[Q8_RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        ssp[k] = 0.0f;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[t * Q8_NKC + kc] = s;
    }
    if (nrows == 0) return;
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows) break;
        float acc[T];
#pragma unroll
        for (int k = 0; k < T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const int v = 32 * (4 * s + bl) + 4 * sub;
            float4 w;
            if (inj_bf16) {
                const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
            } else {
                w = q8_four(q[r][s], dq[r][s]);
            }
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
            }
        }
        const int prow = inj ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[((size_t) k * Q8_NKC + kc) * PR + prow] = v;
        }
    }
}

template <int T>
__global__ void __launch_bounds__(THREADS) gr_up_q8_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    STRATA_SHARED(float, lo, [T][LR]);
    STRATA_SHARED(float, rsS, [T][HC]);
    STRATA_SHARED(float, g, [T][HC][UPM_COLS]);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = blockIdx.x * UPM_COLS;
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += ssg[k * Q8_NKC + c * Q8_CPS + j];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += THREADS) {
        const int k = i / PR, r = i - k * PR;
        if (r >= LR && (blockIdx.x != 0 || m.a[k].w_inject == nullptr)) continue;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += part[((size_t) k * Q8_NKC + c * Q8_CPS + j) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        if (r < LR) {
            const float x = sum / (float) HC;
            lo[k][r] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[r - LR] = sum;
        }
    }
    __syncthreads();
    constexpr int RPW8 = HC * UPM_COLS / WARPS;     // 8 rows per warp
#pragma unroll 1
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                w[s] = q8_four(q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16, q8_half(q8_ld16(blk)));
            } else {
                w[s] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc = fmaf(w[s].x, x.x, acc); acc = fmaf(w[s].y, x.y, acc);
                    acc = fmaf(w[s].z, x.z, acc); acc = fmaf(w[s].w, x.w, acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

#if !defined(__HIPCC__)
// ================================ STRATA_HC_Q8_FUSED=1: the same read in ONE launch =====================================
// gr_down_q8_kernel -> gr_up_q8_kernel is two launches with the up kernel redoing, in each of its 160 blocks, the reduction of
// every row's 16 partial dots (the 324 x T lo / inject values: ~3 MB x T of L2 reads) before it can start on its own weights,
// and with its weights only requested after that.  This kernel runs both phases as tasks of one grid and computes every
// output with EXACTLY the same operations in the same order (so its outputs are bit for bit gr_down_q8_kernel +
// gr_up_q8_kernel's; tests/cuda_emu/hc_q8_emu_test.cpp checks that on the CPU, src/kernels/hc_q8_parity.cpp on a GPU):
//
//   task 0 .. 175   (down)  one (row group, 640-column chunk) each, the old down block.  After its partial dots are written
//                           the block counts itself into its row group; the LAST of the group's 16 chunk blocks reduces the
//                           group's rows (p over the chunk's four partials, fmaf with rs, silu / inject: the old up kernel's
//                           prologue, once per row instead of once per up block) into the caller's `lo`, `inject_out` and
//                           (group 0) `rs`, then counts the group into `ready`.
//   task 176 .. 335 (up)    the old up block (16 columns x 4 streams).  It loads its weights and its own inputs first, waits
//                           until `ready` reaches 11 (all row groups reduced), then reads the finished `lo` (T x 320 floats).
//
// A block does not know its task from blockIdx but from a ticket (atomicAdd on sync_[0]) taken as it starts, so the tasks
// start in ascending order whatever order the hardware dispatches the blocks in: an up task exists only after all 176
// down tasks have started, so the down tasks it waits for are running (or done) and never wait for anything themselves.
// There is no need for the grid to fit on the GPU at once.  The last block to finish zeroes the counters for the next launch
// (one launch at a time per `sync_`; a graph replays it in stream order).  The wait gives up (__trap) after ~2 s.
//
// Read-after-write across blocks: data is written, __threadfence()d, then published with an atomic; a reader that has seen the
// count fences, and reads with __ldcg (through L2, never a stale L1 line).
constexpr int Q8_NRG = Q8_RG + 1;                  // 11 row groups: 10 of 32 down rows, and the inject rows
constexpr int Q8F_DOWN = Q8_NRG * Q8_NKC;          // 176 down tasks
constexpr int Q8F_UP = UPM_BLOCKS;                 // 160 up tasks
constexpr int Q8F_GRID = Q8F_DOWN + Q8F_UP;        // 336 blocks
constexpr int SYNC_READY = 2, SYNC_RG = 3;   // sync_[SYNC_RG + row group]; (SYNC_TICKET, SYNC_DONE: fused_gr_common.cuh)
static_assert(SYNC_RG + Q8_NRG <= kFusedGrSyncWords, "the counters fit the words the caller allocates");

template <int T>
__device__ __forceinline__ void gr_q8f_down(const GrMulti& m, float* __restrict__ part, float* __restrict__ ssg,
                                            unsigned* sync_, int rg, int kc) {
    STRATA_SHARED(float, xs, [T][Q8_KC]);
    STRATA_SHARED(float, red, [WARPS][T]);
    STRATA_SHARED(int, last, );
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c = kc / Q8_CPS;
    const bool inj = rg == Q8_RG;
    const bool inj_bf16 = inj && m.a[0].q8_inject == nullptr;
    const int nrows = inj ? (((m.a[0].q8_inject != nullptr || m.a[0].w_inject != nullptr) && warp < HC) ? 1 : 0) : Q8_RPW;
    const int row0 = inj ? warp : (rg * WARPS + warp) * Q8_RPW;
    const uint8_t* wb = inj ? m.a[0].q8_inject : m.a[0].q8_down;
    const int sub = lane & 7, bl = lane >> 3;
    uint32_t q[Q8_RPW][Q8_SPB];
    float dq[Q8_RPW][Q8_SPB];
#pragma unroll
    for (int r = 0; r < Q8_RPW; ++r) {
        if (r >= nrows || inj_bf16) break;
        const uint8_t* row = wb + (size_t) (row0 + r) * (D / 32) * Q8B + (size_t) kc * (Q8_KC / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < Q8_SPB; ++s) {
            const uint8_t* blk = row + (4 * s + bl) * Q8B;
            dq[r][s] = q8_half(q8_ld16(blk));
            q[r][s] = q8_ld16(blk + 2 + 4 * sub) | q8_ld16(blk + 4 + 4 * sub) << 16;
        }
    }
    float ssp[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        ssp[k] = 0.0f;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
        for (int i = t; i < Q8_KC / 4; i += THREADS) {
            const int col = kc * Q8_KC + 4 * i;
            float4 r = *reinterpret_cast<const float4*>(a.R + col);
            if (a.apply) {
                const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + (col - c * N));
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = *reinterpret_cast<const float4*>(a.w_norm + col);
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            *reinterpret_cast<float4*>(&xs[k][4 * i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (t < T) {   // the chunk's sum of squares: every row group keeps its own copy (the same bits in all of them)
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += red[w][t];
        ssg[((size_t) rg * T + t) * Q8_NKC + kc] = s;
    }
    if (nrows != 0) {
#pragma unroll
        for (int r = 0; r < Q8_RPW; ++r) {
            if (r >= nrows) break;
            float acc[T];
#pragma unroll
            for (int k = 0; k < T; ++k) acc[k] = 0.0f;
#pragma unroll
            for (int s = 0; s < Q8_SPB; ++s) {
                const int v = 32 * (4 * s + bl) + 4 * sub;
                float4 w;
                if (inj_bf16) {
                    const uint2 b = *reinterpret_cast<const uint2*>(m.a[0].w_inject + (size_t) row0 * D + (size_t) kc * Q8_KC + v);
                    w = make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u),
                                    __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
                } else {
                    w = q8_four(q[r][s], dq[r][s]);
                }
#pragma unroll
                for (int k = 0; k < T; ++k) {
                    const float4 x = *reinterpret_cast<const float4*>(&xs[k][v]);
                    acc[k] = fmaf(w.x, x.x, acc[k]); acc[k] = fmaf(w.y, x.y, acc[k]);
                    acc[k] = fmaf(w.z, x.z, acc[k]); acc[k] = fmaf(w.w, x.w, acc[k]);
                }
            }
            const int prow = inj ? LR + warp : row0 + r;
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float v = warp_sum(acc[k]);
                if (lane == 0) part[((size_t) k * Q8_NKC + kc) * PR + prow] = v;
            }
        }
    }
    // publish this chunk; the last chunk of the row group reduces the group's rows
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(&sync_[SYNC_RG + rg], 1u) == (unsigned) (Q8_NKC - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int nr = inj ? HC : WARPS * Q8_RPW;
    const int prow0 = inj ? LR : rg * (WARPS * Q8_RPW);
    for (int i = t; i < T * nr; i += THREADS) {
        const int k = i / nr, prow = prow0 + (i - k * nr);
        if (prow >= LR && m.a[k].w_inject == nullptr) continue;
        float rsv[HC];
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) {
            float ss = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) ss += __ldcg(ssg + ((size_t) rg * T + k) * Q8_NKC + cc * Q8_CPS + j);
            rsv[cc] = rsqrtf(ss / (float) N + m.a[k].eps);
        }
        float sum = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) {
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < Q8_CPS; ++j) p += __ldcg(part + ((size_t) k * Q8_NKC + cc * Q8_CPS + j) * PR + prow);
            sum = fmaf(rsv[cc], p, sum);
        }
        if (prow < LR) {
            const float x = sum / (float) HC;
            m.a[k].lo[prow] = x / (1.0f + __expf(-x));
        } else {
            m.a[k].inject_out[prow - LR] = sum;
        }
    }
    if (rg == 0 && t < T * HC) {   // rs for the up tasks (and the caller): the same expression as above
        const int k = t / HC, cc = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < Q8_CPS; ++j) ss += __ldcg(ssg + ((size_t) rg * T + k) * Q8_NKC + cc * Q8_CPS + j);
        m.a[k].rs[cc] = rsqrtf(ss / (float) N + m.a[k].eps);
    }
    __threadfence();
    __syncthreads();
    if (t == 0) atomicAdd(&sync_[SYNC_READY], 1u);
}

template <int T>
__device__ __forceinline__ void gr_q8f_up(const GrMulti& m, const unsigned* sync_, int u) {
    STRATA_SHARED(float, lo, [T][LR]);
    STRATA_SHARED(float, rsS, [T][HC]);
    STRATA_SHARED(float, g, [T][HC][UPM_COLS]);
    STRATA_SHARED(float, rvS, [T][HC * UPM_COLS]);   // R at this block's 64 (stream, column) pairs, before the wait
    STRATA_SHARED(float, boS, [T][UPM_COLS]);
    STRATA_SHARED(float, ipS, [T][HC]);
    STRATA_SHARED(float, wnS, [HC * UPM_COLS]);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = u * UPM_COLS;
    constexpr int RPW8 = HC * UPM_COLS / WARPS;     // 8 rows per warp
    // 1. this block's weights, requested before anything is waited for
    uint32_t pq[RPW8][3];
    float pd[RPW8][3];
#pragma unroll
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint8_t* row = m.a[0].q8_up + (size_t) i * (LR / 32) * Q8B;
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int v = 4 * lane + 128 * s;
            pq[qq][s] = 0u; pd[qq][s] = 0.0f;
            if (v < LR) {
                const uint8_t* blk = row + (v >> 5) * Q8B;
                pd[qq][s] = q8_half(q8_ld16(blk));
                pq[qq][s] = q8_ld16(blk + 2 + (v & 31)) | q8_ld16(blk + 4 + (v & 31)) << 16;
            }
        }
    }
    // 2. its inputs (R is rewritten only by this block's own epilogue, after the wait; the down tasks only read it)
    for (int i = t; i < T * HC * UPM_COLS; i += THREADS) {
        const int k = i / (HC * UPM_COLS), r = i - k * (HC * UPM_COLS), c = r / UPM_COLS, dd = r - c * UPM_COLS;
        rvS[k][r] = m.a[k].R[c * N + d0 + dd];
    }
    for (int r = t; r < HC * UPM_COLS; r += THREADS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS;
        wnS[r] = m.a[0].w_norm[c * N + d0 + dd];
    }
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, dd = i - k * UPM_COLS;
        boS[k][dd] = m.a[k].apply ? m.a[k].bo_prev[d0 + dd] : 0.0f;
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        ipS[k][c] = m.a[k].apply ? m.a[k].inj_prev[c] : 0.0f;
    }
    // 3. the down phase done: every row group reduced
    if (t == 0) gr_wait_ge(sync_ + SYNC_READY, (unsigned) Q8_NRG);
    __syncthreads();
    __threadfence();
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, j = i - k * LR;
        lo[k][j] = __ldcg(m.a[k].lo + j);
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        rsS[k][c] = __ldcg(m.a[k].rs + c);
    }
    __syncthreads();
#pragma unroll
    for (int qq = 0; qq < RPW8; ++qq) {
        const int r = warp + qq * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        float4 w[3];
#pragma unroll
        for (int s = 0; s < 3; ++s) {   // lane: values 4 lane + 128 s (the third step: lanes 0-15)
            const int v = 4 * lane + 128 * s;
            w[s] = v < LR ? q8_four(pq[qq][s], pd[qq][s]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            rv = rvS[lane][r];
            wn = wnS[r];
            apply = m.a[lane].apply;
            if (apply) { bo = boS[lane][dd]; ip = ipS[lane][c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 3; ++s) {
                const int v = 4 * lane + 128 * s;
                if (v < LR) {
                    const float4 x = *reinterpret_cast<const float4*>(&lo[k][v]);
                    acc = fmaf(w[s].x, x.x, acc); acc = fmaf(w[s].y, x.y, acc);
                    acc = fmaf(w[s].z, x.z, acc); acc = fmaf(w[s].w, x.w, acc);
                }
            }
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

// (THREADS, 2): the down task keeps its 40 weight registers in flight (128 registers in all, no spills at any T; ptxas on
// sm_86); asking for 3 blocks per SM (85 registers) spills 170-250 bytes per thread, so the grid (336 blocks) runs as
// 136 resident blocks and the rest as those finish.  The ticket order below makes that safe.
template <int T>
__global__ void __launch_bounds__(THREADS, 2) gr_q8_fused_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg,
                                                                 unsigned* sync_) {
    const unsigned task = gr_take_ticket(sync_);
    if (task < (unsigned) Q8F_DOWN) gr_q8f_down<T>(m, part, ssg, sync_, (int) (task % Q8_NRG), (int) (task / Q8_NRG));
    else gr_q8f_up<T>(m, sync_, (int) (task - Q8F_DOWN));
    gr_retire(sync_, (unsigned) Q8F_GRID);
}
#endif   // !__HIPCC__
