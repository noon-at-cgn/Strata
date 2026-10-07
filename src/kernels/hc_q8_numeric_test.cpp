// src/kernels/hc_q8_numeric_test.cpp - what STRATA_HC_Q8=1 changes in the numbers (CPU only, no model, no GPU).
//
// The pack (tools/iq_pack.py --compat-bf16) stores the GGUF's Q8_0 hyper-connection projections as BF16: every
// weight is dequantized (fp16 scale d x int8 q, exact in fp32) and rounded to BF16 (nearest even).  STRATA_HC_Q8=1
// reads the Q8_0 bytes instead and dequantizes them in the kernel the same way (d * (float) q).  So the two reads
// differ by exactly that BF16 rounding of the weights, whenever the product d * q needs more than the 8 significant
// bits of a BF16 - which, for an fp16 scale (11 significant bits) times an int8 (up to 7), is almost always.
//
// Checks (all exhaustive or fixed-seed, deterministic):
//   1. d * q is exact in fp32 for EVERY finite fp16 d and every int8 q (so the kernel's dequantization is the
//      GGUF's value, not an approximation of it);
//   2. the BF16 copy is NOT bit-identical: over all (d, q) the fraction of values BF16 holds exactly is printed and
//      required to be well below 1, and the worst relative error of the rounding is required to be <= 2^-8;
//   3. on real-shaped weights (blocks quantized the way ggml's quantize_row_q8_0 does, normally distributed, a few
//      outliers) the BF16 copy's worst per-weight relative error and the resulting relative difference of a K = 10240
//      dot product (the hc down projection's shape) against the exact Q8_0 dot are printed and bounded.
//
// This test documents the size of the difference; it does not make it zero.  The kernels' summation order is a
// separate difference (STRATA_HC_Q8's kernels sum per 640-column chunk), covered by gr_q8_parity on a GPU.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using strata::kernels::bf16_from_f32;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::kernels::f32_from_f16;

namespace {
int failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

// ggml quantize_row_q8_0 (reference implementation): d = amax / 127 stored as fp16; q = round(x / d) with the fp32 reciprocal.
struct Q8Block { uint16_t d; int8_t q[32]; };
Q8Block quantize_block(const float* x) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    Q8Block b{};
    b.d = f16_from_f32(d);
    for (int i = 0; i < 32; ++i) b.q[i] = (int8_t) std::lround(x[i] * id);
    return b;
}
}  // namespace

int main() {
    // ---- 1 + 2: every finite fp16 scale x every int8 -------------------------------------------------------------
    uint64_t total = 0, exact_bf16 = 0, prod_not_exact_f32 = 0;
    double worst_rel = 0.0;
    for (uint32_t h = 0; h < 65536; ++h) {
        if (((h >> 10) & 0x1f) == 31) continue;                       // inf / NaN
        const float d = f32_from_f16((uint16_t) h);
        if (d == 0.0f) continue;
        for (int q = -127; q <= 127; ++q) {
            if (q == 0) continue;
            const float p = d * (float) q;                            // the kernel's q8_four()
            const double exact = (double) d * (double) q;
            if ((double) p != exact) ++prod_not_exact_f32;            // fp32 overflow/underflow is the only way
            ++total;
            const float b = f32_from_bf16(bf16_from_f32(p));
            if (b == p) ++exact_bf16;
            worst_rel = std::max(worst_rel, std::fabs((double) b - exact) / std::fabs(exact));
        }
    }
    std::printf("all (fp16 d, int8 q) pairs: %llu, fp32 product inexact: %llu, held exactly by BF16: %.2f%%, worst BF16 relative error: %.3e (2^-8 = %.3e)\n",
                (unsigned long long) total, (unsigned long long) prod_not_exact_f32, 100.0 * (double) exact_bf16 / (double) total,
                worst_rel, std::ldexp(1.0, -8));
    // subnormal fp16 scales times small q can fall below fp32's range only for d < 2^-126: impossible for fp16 (min 2^-24)
    CHECK(prod_not_exact_f32 == 0);
    CHECK(exact_bf16 < total / 2);                                    // the BF16 copy is NOT the Q8_0 value, in general
    CHECK(exact_bf16 > 0);
    CHECK(worst_rel <= std::ldexp(1.0, -8) * (1.0 + 1e-12));
    CHECK(worst_rel > std::ldexp(1.0, -9));                           // and the bound is approached

    // ---- 3: real-shaped blocks, a K = 10240 dot ------------------------------------------------------------------
    std::mt19937 rng(20261007u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    constexpr int K = 10240, ROWS = 4000;
    double worst_w = 0.0, worst_dot = 0.0, sum_dot = 0.0;
    std::vector<double> dots;
    dots.reserve(ROWS);
    for (int r = 0; r < ROWS; ++r) {
        const float sigma = 0.004f * std::exp(0.8f * nd(rng));        // per-row scale spread
        double dot_q8 = 0.0, dot_bf = 0.0, mag = 0.0;
        for (int blk = 0; blk < K / 32; ++blk) {
            float x[32];
            for (int i = 0; i < 32; ++i) x[i] = sigma * nd(rng);
            if ((blk % 37) == 0) x[blk % 32] *= 8.0f;                 // an outlier now and then
            const Q8Block b = quantize_block(x);
            const float d = f32_from_f16(b.d);
            for (int i = 0; i < 32; ++i) {
                const float w = d * (float) b.q[i];
                const float wb = f32_from_bf16(bf16_from_f32(w));
                const double a = nd(rng);                             // the activation (normalized hc stream x w_norm)
                dot_q8 += (double) w * a;
                dot_bf += (double) wb * a;
                mag += std::fabs((double) w * a);
                if (w != 0.0f) worst_w = std::max(worst_w, std::fabs((double) wb - (double) w) / std::fabs((double) w));
            }
        }
        // relative to the sum of |terms| (what a dot's rounding error scales with), and to the dot itself
        const double rel_mag = std::fabs(dot_bf - dot_q8) / mag;
        worst_dot = std::max(worst_dot, rel_mag);
        sum_dot += rel_mag;
        dots.push_back(std::fabs(dot_bf - dot_q8) / std::max(std::fabs(dot_q8), 1e-30));
    }
    std::sort(dots.begin(), dots.end());
    std::printf("real-shaped rows (%d x K %d): worst per-weight BF16 error %.3e; BF16-vs-Q8_0 dot difference / sum|terms|: mean %.3e worst %.3e; / |dot|: median %.3e p99 %.3e\n",
                ROWS, K, worst_w, sum_dot / ROWS, worst_dot, dots[dots.size() / 2], dots[dots.size() * 99 / 100]);
    CHECK(worst_w <= std::ldexp(1.0, -8) * (1.0 + 1e-12));
    CHECK(worst_dot < 2e-3);                                          // the weights' 2^-8 bound, averaged by K = 10240 random signs

    if (failures) { std::printf("hc_q8_numeric_test: %d failure(s)\n", failures); return 1; }
    std::printf("hc_q8_numeric_test: ok\n");
    return 0;
}
