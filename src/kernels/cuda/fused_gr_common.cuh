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

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int UPM_COLS = 16;                      // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;          // 160
constexpr int PR = LR + HC;                        // partial rows per (token, stream): 320 down + 4 inject

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
