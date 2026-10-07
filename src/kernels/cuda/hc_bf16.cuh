// src/kernels/cuda/hc_bf16.cuh - the hyper-connection read from the BF16 projections: the multi-launch variants (split norm,
// staged down, multi up) and, behind STRATA_HC_FUSED=1, the same read as one launch.
//
// A FRAGMENT, like fused_gr_common.cuh (whose names it uses): fused_gr.cu includes it inside its anonymous namespace, and
// tests/cuda_emu/hc_q8_emu_test.cpp compiles the same device code for the CPU.
//
constexpr int kHcPlain = 1, kHcSplit = 2, kHcStaged = 3;
constexpr int H_TILE = 1280;                          // staged tile: half a stream = 160 chunks of 8, 5 per lane
constexpr int HQ = H_TILE / 8 / 32;
constexpr int N_HTILES = D / H_TILE;                  // 8
static_assert(N % H_TILE == 0, "a staged tile never straddles two streams");
static_assert(H_TILE % 256 == 0, "a staged tile holds whole rounds of 32 chunks: the plain read's lane order");

// one (token k, stream c) block's work of gr_norm_split_kernel (also a task of the one-launch read, below)
__device__ __forceinline__ void gr_norm_split_task(const GrMulti& m, int k, int c) {
    STRATA_SHARED(float, part, [WARPS]);
    STRATA_SHARED(float, s_rs, );
    const FusedGrArgs& a = m.a[k];
    float* xn = m.xn + (size_t) k * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss += sq;
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        const float px = r.x * g.x, py = r.y * g.y, pz = r.z * g.z, pw = r.w * g.w;
        *reinterpret_cast<float4*>(xn + i) = make_float4(px * rs, py * rs, pz * rs, pw * rs);
    }
}
__global__ void __launch_bounds__(THREADS) gr_norm_split_kernel(GrMulti m) { gr_norm_split_task(m, blockIdx.x, blockIdx.y); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__)
#define STRATA_GR_CP_ASYNC 1
#endif
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
#if defined(STRATA_GR_CP_ASYNC)
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait0() {
#if defined(STRATA_GR_CP_ASYNC)
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// `dot8` with its 8 activations as two float4: the same eight fmaf in the same order
__device__ __forceinline__ float dot8v(const uint4 w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc);
    acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc);
    acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc);
    acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc);
    acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

// Stage tile `h` of every token into `buf`: [T][2 planes][160 chunks] float4, plane 0 = floats 0-3 of a chunk.
__device__ __forceinline__ void stage_htile(const GrMulti& m, int T, int h, float4* buf, int t) {
    for (int i = t; i < T * (H_TILE / 4); i += THREADS) {
        const int k = i / (H_TILE / 4), s4 = i - k * (H_TILE / 4);   // s4: float4 of the tile, chunk s4/2, half s4&1
        const float* src = m.xn + (size_t) k * D + (size_t) h * H_TILE + (size_t) s4 * 4;
        cp_async16(buf + (size_t) k * (H_TILE / 4) + (s4 & 1) * (H_TILE / 8) + (s4 >> 1), src);
    }
}

// One block's work of gr_down_staged_kernel (bx: its block index).  WAIT (the one-launch read): the weights are requested
// first, then the block waits for the norm tasks (every token's xn written: `norm_done` reaches `n_norm`) before staging
// the activations, and counts itself into `down_done` when its rows are written.
template <int MAX_T, bool EXACT_T, bool WAIT>
__device__ __forceinline__ void gr_down_staged_task(const GrMulti& m, int bx, unsigned* norm_done, unsigned n_norm,
                                                    unsigned* down_done) {
    STRATA_DYN_SHARED(float4, hbuf);                    // 2 buffers x [T][2][160] float4
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const bool inject_block = bx == DOWN_BLOCKS;
    const int row = inject_block ? warp : bx * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    const size_t buf_f4 = (size_t) T * (H_TILE / 4);    // float4 per buffer (the second one follows the first)
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    uint4 wv[HQ], wnext[HQ];
    if (active) {
#pragma unroll
        for (int q = 0; q < HQ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    }
    if (WAIT) {
#if !defined(__HIPCC__)
        if (t == 0) gr_wait_ge(norm_done, n_norm);
        __syncthreads();
        __threadfence();
#endif
    }
    stage_htile(m, T, 0, hbuf, t);
    cp_async_commit();
    stage_htile(m, T, 1, hbuf + buf_f4, t);
    cp_async_commit();
#pragma unroll 1
    for (int h = 0; h < N_HTILES; ++h) {
        if (active && h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wnext[q] = __ldg(w4 + (h + 1) * (H_TILE / 8) + lane + 32 * q);
        }
        if (h + 1 < N_HTILES) cp_async_wait1();         // tile h has landed (h + 1 may still be on its way)
        else cp_async_wait0();
        __syncthreads();
        const float4* cur = hbuf + (h & 1) * buf_f4;
        if (active) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) {
                const int j = lane + 32 * q;
                const Bf16x8 wvq = unpack8(wv[q]);
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (EXACT_T || k < T) {
                        const float4* pk = cur + (size_t) k * (H_TILE / 4);
                        acc[k] += dot8u(wvq, pk[j], pk[H_TILE / 8 + j]);
                    }
                }
            }
        }
        __syncthreads();                                // every warp is done with buffer h & 1
        if (h + 2 < N_HTILES) stage_htile(m, T, h + 2, hbuf + (h & 1) * buf_f4, t);
        cp_async_commit();                              // an empty group at the end keeps the wait counts simple
        if (h + 1 < N_HTILES) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv[q] = wnext[q];
        }
    }
    if (active) {
        float s[MAX_T];
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) s[k] = (EXACT_T || k < T) ? warp_sum(acc[k]) : 0.0f;
        // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if ((!EXACT_T && k >= T) || lane != k) continue;
            if (inject_block) {
                m.a[k].inject_out[row] = s[k];
            } else {
                const float x = s[k] / (float) HC;
                m.a[k].lo[row] = x / (1.0f + __expf(-x));
            }
        }
    }
    if (WAIT) {
#if !defined(__HIPCC__)
        __threadfence();
        __syncthreads();
        if (t == 0) atomicAdd(down_done, 1u);
#endif
    }
}
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_down_staged_kernel(GrMulti m) {
    gr_down_staged_task<MAX_T, EXACT_T, false>(m, blockIdx.x, nullptr, 0u, nullptr);
}

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by xor so every lane holds every
// sum, and lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
template <int MAX_T = kFusedGrMaxT, bool EXACT_T = false>
__global__ void __launch_bounds__(THREADS) gr_up_multi_kernel(GrMulti m) {
    STRATA_SHARED(float, lo, [MAX_T][LR]);
    STRATA_SHARED(float, g, [MAX_T][HC][UPM_COLS]);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        const Bf16x8 wa = unpack8(__ldg(w4 + lane));
        const Bf16x8 wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        // the epilogue inputs of this lane's token, fetched while the dots run
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            rv = a.R[i];
            wn = a.w_norm[i];
            rsc = a.rs[c];
            apply = a.apply;
            if (apply) { bo = a.bo_prev[d0 + dd]; ip = a.inj_prev[c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

#if !defined(__HIPCC__)
// ================================ STRATA_HC_FUSED=1: the BF16 read as ONE launch =========================================
// The multi-launch read above - gr_norm_split_kernel, gr_down_staged_kernel, gr_up_multi_kernel - as the tasks of one grid, with
// each task's operations (and so every output bit) the kernel's own: the norm and down tasks are the very functions the three
// kernels now call (gr_norm_split_task, gr_down_staged_task), the up task is gr_up_multi_kernel's with its weights and its
// epilogue inputs requested before it waits.  What changes is only who starts when:
//
//   tasks 0 .. 4T-1            norm   one (token, stream) each; counts itself into `norm_done`
//   next 41                    down   8 rows each (the last: the inject rows); asks for its weights, waits for all 4T norm
//                                     tasks, stages the activations; counts itself into `down_done`
//   last 160                   up     16 columns x 4 streams each; asks for its weights, waits for all 41 down tasks, then
//                                     reads the finished `lo` (T x 320 floats)
//
// The three phases' launch gaps and ramp-up/drain (about 3 x 3 us) go, and the down and up weights stream in while the previous
// phase is still running.  Tickets and the retire step: fused_gr_common.cuh.  Dynamic shared memory is the staged down read's
// (2 * T * 5 KiB per block, 40 KiB at 4 tokens: below the 48 KiB that needs no opt-in); more tokens keep the three launches.
constexpr int F1_MAX_T = 4;
constexpr int SYNC_NORM = 2, SYNC_DOWN = 3;

template <int T>
__device__ __forceinline__ void gr_f1_up_task(const GrMulti& m, const unsigned* down_done, int u) {
    STRATA_SHARED(float, lo, [T][LR]);
    STRATA_SHARED(float, g, [T][HC][UPM_COLS]);
    STRATA_SHARED(float, rvS, [T][HC * UPM_COLS]);   // R at this block's 64 (stream, column) pairs, read before the wait
    STRATA_SHARED(float, wnS, [HC * UPM_COLS]);
    STRATA_SHARED(float, boS, [T][UPM_COLS]);
    STRATA_SHARED(float, ipS, [T][HC]);
    STRATA_SHARED(float, rsS, [T][HC]);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int d0 = u * UPM_COLS;
    constexpr int RPW8 = HC * UPM_COLS / WARPS;      // 8 rows per warp
    // 1. this block's weights (a row's 320 bf16: lane l holds chunk l, and lanes 0-7 chunk 32 + l), before anything is waited for
    uint4 pa[RPW8], pb[RPW8];
#pragma unroll
    for (int j = 0; j < RPW8; ++j) {
        const int r = warp + j * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        pa[j] = __ldg(w4 + lane);
        pb[j] = lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0);
    }
    // 2. the epilogue inputs that the norm and down tasks do not produce (R is rewritten only by this block's own epilogue)
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
    // 3. the down phase done: every row of `lo` (and `rs`, from the norm phase) is written
    if (t == 0) gr_wait_ge(down_done, (unsigned) (DOWN_BLOCKS + 1));
    __syncthreads();
    __threadfence();
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = __ldcg(m.a[i / LR].lo + i % LR);
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        rsS[k][c] = __ldcg(m.a[k].rs + c);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < RPW8; ++j) {
        const int r = warp + j * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const Bf16x8 wa = unpack8(pa[j]);
        const Bf16x8 wb = unpack8(pb[j]);
        float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            rv = rvS[lane][r];
            wn = wnS[r];
            rsc = rsS[lane][c];
            apply = m.a[lane].apply;
            if (apply) { bo = boS[lane][dd]; ip = ipS[lane][c]; }
        }
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            float acc = dot8u_ptr(wa, lo[k] + lane * 8);
            if (lane < LR / 8 - 32) acc += dot8u_ptr(wb, lo[k] + (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        if (lane < T) {
            if (apply) {
                rv = fmaf(bo, 2.0f * sigmoidf_(ip / (float) HC), rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsc;
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) gr_q8_tail(m, d0);   // S26 STRATA_QFUSE
}

template <int T>
constexpr unsigned kF1Grid = (unsigned) (T * HC + (DOWN_BLOCKS + 1) + UPM_BLOCKS);   // blocks of gr_hc_fused_kernel<T>

template <int T>
__global__ void __launch_bounds__(THREADS) gr_hc_fused_kernel(GrMulti m, unsigned* sync_) {
    const unsigned task = gr_take_ticket(sync_);
    constexpr unsigned NNORM = T * HC, NDOWN = DOWN_BLOCKS + 1;
    if (task < NNORM) {
        gr_norm_split_task(m, (int) (task / HC), (int) (task % HC));
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(&sync_[SYNC_NORM], 1u);
    } else if (task < NNORM + NDOWN) {
        gr_down_staged_task<T, true, true>(m, (int) (task - NNORM), &sync_[SYNC_NORM], NNORM, &sync_[SYNC_DOWN]);
    } else {
        gr_f1_up_task<T>(m, &sync_[SYNC_DOWN], (int) (task - NNORM - NDOWN));
    }
    gr_retire(sync_, kF1Grid<T>);
}
#endif   // !__HIPCC__
