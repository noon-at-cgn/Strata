// src/kernels/cuda/fused_gr_common.cuh - the hyper-connection read's constants and small device helpers.
//
// A FRAGMENT: fused_gr.cu includes it inside its anonymous namespace (after <cuda_fp16.h> and fused_gr.hpp), and so
// does tests/cuda_emu/hc_q8_emu_test.cpp, which compiles the same device code for the CPU (tests/cuda_emu/cuda_emu.hpp
// supplies the CUDA names).  It declares nothing outside that namespace.
//
// STRATA_SHARED(type, name, dims): one statically sized __shared__ variable.  On the GPU it is exactly
// `__shared__ __align__(16) type name dims;`; the CPU emulation gives every block its own copy.
#ifndef STRATA_SHARED
#define STRATA_SHARED(type, name, dims) __shared__ __align__(16) type name dims
#endif
// the launch's dynamic shared memory (one array per kernel), as `type* name`
#ifndef STRATA_DYN_SHARED
#define STRATA_DYN_SHARED(type, name) extern __shared__ __align__(16) type name[]
#endif

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160
constexpr int PR = LR + HC;                        // partial rows per (token, stream): 320 down + 4 inject
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};


// S26 STRATA_QFUSE=1: the q8_1 image of `mixed`, written by the up kernels below. A q8_1 block is 32 columns and an
// up block owns UPM_COLS = 16, so the second of the two blocks that own a 32-column group (a per-group counter,
// incremented after the block's writes are fenced, reset by that block for the next launch) reads the 32 values back
// and quantizes them with native_quantize_q8_1_kernel's quantizer: one warp per token, the same XOR-tree max and sum,
// d = amax / 127, roundf(x / d), ds = (d, sum) - the bytes the separate launch writes.
struct GrQ81 { half2 ds; int8_t qs[32]; };
static_assert(2 * 16 == 32, "two up blocks per q8_1 group");
__device__ __forceinline__ void gr_q8_tail(const GrMulti& m, int d0) {
    STRATA_SHARED(int, last, );
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* c = m.a[0].q8_cnt + d0 / 32;
        last = atomicAdd(c, 1u) == 1u;
        if (last) atomicExch(c, 0u);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp >= m.T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[warp].mixed + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    GrQ81* y = reinterpret_cast<GrQ81*>(m.a[warp].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = make_half2(d, sum);
}

// 8 bf16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {
        make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                    __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u)),
        make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                    __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u))
    };
}
__device__ __forceinline__ float dot8u(const Bf16x8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc);
    acc = fmaf(w.a.y, x0.y, acc);
    acc = fmaf(w.a.z, x0.z, acc);
    acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc);
    acc = fmaf(w.b.y, x1.y, acc);
    acc = fmaf(w.b.z, x1.z, acc);
    acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float dot8u_ptr(const Bf16x8& w, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    return dot8u(w, x4[0], x4[1]);
}

// 8 bf16 packed in a uint4 against 8 16-byte-aligned floats (same 8-FMA order).
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    return dot8u_ptr(unpack8(w), x);
}

#if !defined(__HIPCC__)
// ================================ STRATA_HC_FUSED=1: the blocks of one launch wait on each other ==========================
// The counters (FusedGrArgs::hc_sync, kFusedGrSyncWords words, zero between launches) and the two helpers every fused read
// uses; hc_q8.cuh and hc_bf16.cuh say what each phase counts.  A block's task comes from an atomic ticket taken as it
// starts (so tasks start in ascending order whatever order the hardware dispatches blocks in, and a task only ever waits for
// tasks with lower numbers, which are running or done), and the last block out zeroes the counters.
constexpr int SYNC_TICKET = 0, SYNC_DONE = 1;
__device__ __forceinline__ unsigned gr_ld_acquire(const unsigned* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(STRATA_CUDA_EMU)
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
#else
    return *reinterpret_cast<const volatile unsigned*>(p);
#endif
}
/// thread 0 only (the caller's __syncthreads() then releases the block); gives up with __trap() after ~2 s: a block that
/// never started should fail loudly, not hang the card
__device__ __forceinline__ void gr_wait_ge(const unsigned* p, unsigned v) {
#if defined(STRATA_CUDA_EMU)
    emu::wait_until([&] { return gr_ld_acquire(p) >= v; });
#else
    const long long t0 = clock64();
    while (gr_ld_acquire(p) < v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(40);
#endif
        if (clock64() - t0 > 4000000000ll) __trap();
    }
#endif
}
/// the block's ticket (call from every thread; one atomic per block)
__device__ __forceinline__ unsigned gr_take_ticket(unsigned* sync_) {
    STRATA_SHARED(unsigned, ticket, );
    if (threadIdx.x == 0) ticket = atomicAdd(&sync_[SYNC_TICKET], 1u);
    __syncthreads();
    return ticket;
}
/// after the block's last work, from every thread: the last of `grid` blocks zeroes the counters for the next launch
__device__ __forceinline__ void gr_retire(unsigned* sync_, unsigned grid) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&sync_[SYNC_DONE], 1u) == grid - 1)
            for (int i = 0; i < kFusedGrSyncWords; ++i) atomicExch(&sync_[i], 0u);
    }
}
#endif   // !__HIPCC__
