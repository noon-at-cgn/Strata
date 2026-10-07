// src/kernels/cpu/kq_avx2.cpp - Unsloth UD-Q4_K_XL's expert rows (Q4_K gate/up, Q5_1 and Q8_0 down) for several
// tokens at once, BIT-EXACT against ggml-cpu's own dot products.
//
// ggml-cpu is built for AVX2 here (STRATA_PORTABLE: GGML_AVX2=ON, GGML_AVX512=OFF), so its vec_dot for these formats
// is the __AVX2__ branch of ggml-cpu/arch/x86/quants.c: ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q5_1_q8_1,
// ggml_vec_dot_q8_0_q8_0.  The functions below are those branches with the loop over tokens moved inside the loop
// over blocks: everything that depends only on the weights (the 4-bit unpacking, the 6-bit scales and mins, the
// fifth bits, the absolute values of the Q8_0 weights, the fp16 scales) is done once per block, and every token then
// runs exactly ggml's integer chain and ggml's float operations in ggml's order - so each token's result is ggml's
// to the bit (native_expert_parity checks it), whatever the number of tokens.  Unlike the i-quant kernels (#152)
// the group size therefore never changes an answer.
//
// The gain is the weight-side work and the weight bytes, read once per verify window instead of once per token.
#include "strata/kernels/cpu/kq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

constexpr int MAXT = 8;   // the verify window's tokens per group (kVerifyMaxT)

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} / {d, m} / {d, s} a block starts with (a union in ggml-common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 / non-VNNI forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                                0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}
inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}
inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}
alignas(32) const uint8_t k_shuffle[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle + i); }

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
void q4k_dot(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[MAXT];
    __m128 acc_m[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, __AVX2__)
void q5_1_dot(const block_q5_1* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    float summs[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(half_at(&x[ib], 0));
        const float xm = half_at(&x[ib], 1);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        for (int t = 0; t < nt; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * half_at(y, 1);
            const __m256 dy = _mm256_set1_ps(half_at(y, 0));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = mul_sum_us8_pairs_float(qx, qy);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, __AVX2__; mul_sum_i8_pairs_float without AVX-VNNI-INT8)
void q8_0_dot(const block_q8_0* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * h2f(y->d));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy);
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

// ---- the "fast" row kernels (STRATA_KQ_KERNEL=fast): the same integer chains and the same float operations in the
// same order per row and token as q4k_dot / q5_1_dot above (so, bit for bit, ggml's), but arranged for throughput.
// ggml's dot is one row at a time: ~100 instructions per Q4_K block in one dependent chain (loads -> maddubs -> madd
// -> adds -> convert -> fma), the weights' scale unpacking in scalar code.  kq_fast_parity --bench prints what each
// arrangement gets on a given CPU.  Here R rows (and NT tokens) go through each block together:
//   - the activation's loads (q8, bsums, the fp16 scale pair) happen once for R rows instead of once per row;
//   - R independent accumulator chains (R * NT float accumulators) overlap each other's latencies;
//   - the 6-bit scales and mins of a Q4_K block are unpacked with four vector operations on all four words
//     (ggml: ~16 scalar operations and four lane inserts) - integer arithmetic, the same eight scales and mins;
//   - the next row set's bytes are prefetched while this one is computed (the rows of an expert are contiguous).
// Every integer result is exact (no int16 saturation either: the same maddubs/madd as ggml) and the float part keeps
// ggml's order, so the result of one row does not depend on R, NT or the row range.
#define KQF_UNROLL _Pragma("GCC unroll 8")
constexpr int kFastMaxNt = 4;   // tokens per pass; larger groups run in several passes over the (L1-resident) rows

inline void prefetch_t0(const void* p) { _mm_prefetch((const char*) p, _MM_HINT_T0); }

// ggml's utmp arithmetic (get_scale_min_k4 packed as four words) on a whole vector: `sc` is the block's 12 scale bytes,
// 4 more bytes (the start of qs) are readable.  scales: 8 x int16 in both 128-bit lanes (ggml's `scales`),
// mins: 8 x int16.
inline void q4k_scales_fast(const uint8_t* sc, __m256i& scales, __m128i& mins) {
    const __m128i v = _mm_loadu_si128((const __m128i*) sc);                       // u0 u1 u2 (u3: junk)
    const __m128i mlo = _mm_setr_epi32(0x3f3f3f3f, 0x0f0f0f0f, 0x3f3f3f3f, 0x0f0f0f0f);
    const __m128i mhi = _mm_setr_epi32(0, 0x30303030, 0, 0x30303030);
    // lo = u0 & k1 | u2 & k2 | u1 & k1 | (u2 >> 4) & k2
    __m128i lo = _mm_srlv_epi32(_mm_shuffle_epi32(v, _MM_SHUFFLE(2, 1, 2, 0)), _mm_setr_epi32(0, 0, 0, 4));
    lo = _mm_and_si128(lo, mlo);
    // hi = 0 | ((u0 >> 6) & k3) << 4 | 0 | ((u1 >> 6) & k3) << 4  ==  (u >> 2) & 0x30303030
    __m128i hi = _mm_srli_epi32(_mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 0, 0)), 2);
    hi = _mm_and_si128(hi, mhi);
    const __m256i ms = _mm256_cvtepu8_epi16(_mm_or_si128(lo, hi));
    mins = _mm256_extracti128_si256(ms, 1);
    scales = _mm256_permute2x128_si256(ms, ms, 0x00);
}

// R weight rows (each `nb` Q4_K blocks) against NT activations (Q8_K rows, `nb` blocks each).
template <int R, int NT>
inline void q4k_rows_fast(const uint8_t* const* wr, const uint8_t* const* wn, int nb, const void* const* act, float (*res)[NT]) {
    __m256 acc[R][NT];
    __m128 accm[R][NT];
    KQF_UNROLL
    for (int r = 0; r < R; ++r)
        KQF_UNROLL
        for (int t = 0; t < NT; ++t) { acc[r][t] = _mm256_setzero_ps(); accm[r][t] = _mm_setzero_ps(); }
    const __m256i m4 = _mm256_set1_epi8(0xF);
    for (int i = 0; i < nb; ++i) {
        KQF_UNROLL
        for (int r = 0; r < R; ++r) {   // the next row set (rows are contiguous): its blocks arrive while this one computes
            const char* p = (const char*) wn[r] + (size_t) i * sizeof(block_q4_K);
            prefetch_t0(p);
            prefetch_t0(p + 64);
            prefetch_t0(p + 128);
        }
        const block_q8_K* y[NT];
        __m128i q8s[NT];
        KQF_UNROLL
        for (int t = 0; t < NT; ++t) {
            y[t] = (const block_q8_K*) act[t] + i;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[t]->bsums);
            q8s[t] = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        }
        KQF_UNROLL
        for (int r = 0; r < R; ++r) {
            const block_q4_K* x = (const block_q4_K*) wr[r] + i;
            uint32_t dd;
            std::memcpy(&dd, x, 4);   // d, dmin
            const __m128 dm = _mm_cvtph_ps(_mm_cvtsi32_si128((int) dd));
            const float xd = _mm_cvtss_f32(dm), xdmin = _mm_cvtss_f32(_mm_movehdup_ps(dm));
            __m256i scales;
            __m128i mins;
            q4k_scales_fast(x->scales, scales, mins);
            __m256i sumi[NT];
            KQF_UNROLL
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            const uint8_t* q4 = x->qs;
            KQF_UNROLL
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (q4 + 32 * j));
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                KQF_UNROLL
                for (int t = 0; t < NT; ++t) {
                    const int8_t* q8 = y[t]->qs + 64 * j;
                    const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                    const __m256i q8h = _mm256_loadu_si256((const __m256i*) (q8 + 32));
                    const __m256i p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
                    const __m256i p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p16l, p16h));
                }
            }
            KQF_UNROLL
            for (int t = 0; t < NT; ++t) {
                const float d = y[t]->d * xd;
                const float dmin = -y[t]->d * xdmin;
                const __m128i prod = _mm_madd_epi16(mins, q8s[t]);
                accm[r][t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), accm[r][t]);
                acc[r][t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[t]), acc[r][t]);
            }
        }
    }
    KQF_UNROLL
    for (int r = 0; r < R; ++r)
        KQF_UNROLL
        for (int t = 0; t < NT; ++t) {
            __m128 m = accm[r][t];
            m = _mm_add_ps(m, _mm_movehl_ps(m, m));
            m = _mm_add_ss(m, _mm_movehdup_ps(m));
            res[r][t] = hsum_float_8(acc[r][t]) + _mm_cvtss_f32(m);
        }
}

// R (1, 2 or 4) weight rows (each `nb` Q5_1 blocks) against NT activations (Q8_1 rows).  The fp16 scale pairs {d, m} of
// the R rows' block are converted by one vcvtph2ps ([d0 m0 d1 m1 d2 m2 d3 m3]); d * dy is one vector multiply and
// summs += m * ys one vector fma over all rows (the odd lanes) - the same two roundings per row as ggml's scalar code
// (a rounded product d * dy for the block's fma, and the fused m * ys + summs), only fewer instructions.
template <int R, int NT>
inline void q5_1_rows_fast(const uint8_t* const* wr, const uint8_t* const* wn, int nb, const void* const* act, float (*res)[NT]) {
    static_assert(R == 1 || R == 2 || R == 4, "q5_1_rows_fast: R must be 1, 2 or 4");
    __m256 acc[R][NT];
    __m256 summs[NT];
    KQF_UNROLL
    for (int t = 0; t < NT; ++t) summs[t] = _mm256_setzero_ps();
    KQF_UNROLL
    for (int r = 0; r < R; ++r)
        KQF_UNROLL
        for (int t = 0; t < NT; ++t) acc[r][t] = _mm256_setzero_ps();
    const __m256i lowMask = _mm256_set1_epi8(0xF), hiBit = _mm256_set1_epi8(0x10);
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    const __m256i all1 = _mm256_set1_epi64x(-1);
    const __m256i nib_shift = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);
    for (int ib = 0; ib < nb; ++ib) {
        if ((ib & 7) == 0) {   // the next row set's matching 192 bytes (eight blocks), three cache lines
            KQF_UNROLL
            for (int r = 0; r < R; ++r) {
                const char* p = (const char*) wn[r] + (size_t) ib * sizeof(block_q5_1);
                prefetch_t0(p);
                prefetch_t0(p + 64);
                prefetch_t0(p + 128);
            }
        }
        __m256i qx[R];
        const block_q5_1* x[R];
        KQF_UNROLL
        for (int r = 0; r < R; ++r) {
            x[r] = (const block_q5_1*) wr[r] + ib;
            uint32_t qh;
            std::memcpy(&qh, x[r]->qh, 4);
            // the nibbles: low half of the 32 values from the low nibbles, high half from the high nibbles
            const __m256i q16 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) x[r]->qs));
            const __m256i nib = _mm256_and_si256(_mm256_srlv_epi32(q16, nib_shift), lowMask);
            // the fifth bits: ggml's bytes_from_bits_32, then 0x10 where the bit is set
            __m256i bits = _mm256_shuffle_epi8(_mm256_set1_epi32((int) qh), shuf_mask);
            bits = _mm256_cmpeq_epi8(_mm256_or_si256(bits, bit_mask), all1);
            qx[r] = _mm256_or_si256(nib, _mm256_and_si256(bits, hiBit));
        }
        uint32_t dd[R];
        KQF_UNROLL
        for (int r = 0; r < R; ++r) std::memcpy(&dd[r], x[r], 4);   // d, m
        __m128i dmh = _mm_cvtsi32_si128((int) dd[0]);
        if constexpr (R >= 2) dmh = _mm_insert_epi32(dmh, (int) dd[1], 1);
        if constexpr (R == 4) {
            dmh = _mm_insert_epi32(dmh, (int) dd[2], 2);
            dmh = _mm_insert_epi32(dmh, (int) dd[3], 3);
        }
        const __m256 dm = _mm256_cvtph_ps(dmh);   // [d0 m0 d1 m1 d2 m2 d3 m3]
        KQF_UNROLL
        for (int t = 0; t < NT; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            uint32_t ds;
            std::memcpy(&ds, y, 4);   // d, s
            const __m128 dsf = _mm_cvtph_ps(_mm_cvtsi32_si128((int) ds));
            const __m256 dy = _mm256_broadcastss_ps(dsf);
            const __m256 ys = _mm256_broadcastss_ps(_mm_movehdup_ps(dsf));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            summs[t] = _mm256_fmadd_ps(dm, ys, summs[t]);
            const __m256 dxdy = _mm256_mul_ps(dm, dy);
            KQF_UNROLL
            for (int r = 0; r < R; ++r) {
                const __m256 q = mul_sum_us8_pairs_float(qx[r], qy);
                const __m256 s = R == 1 ? _mm256_broadcastss_ps(_mm256_castps256_ps128(dxdy))
                                        : _mm256_permutevar8x32_ps(dxdy, _mm256_set1_epi32(2 * r));
                acc[r][t] = _mm256_fmadd_ps(q, s, acc[r][t]);
            }
        }
    }
    KQF_UNROLL
    for (int t = 0; t < NT; ++t) {
        alignas(32) float sm[8];
        _mm256_store_ps(sm, summs[t]);
        KQF_UNROLL
        for (int r = 0; r < R; ++r) res[r][t] = hsum_float_8(acc[r][t]) + sm[2 * r + 1];
    }
}

// R rows x NT tokens of the fast kernel for one weight type; res[r][t].
template <int R, int NT>
inline void fast_set(int type, const uint8_t* const* wr, const uint8_t* const* wn, int nb, const void* const* act,
                     float (*res)[NT]) {
    if (type == 12) q4k_rows_fast<R, NT>(wr, wn, nb, act, res);
    else q5_1_rows_fast<R, NT>(wr, wn, nb, act, res);
}

inline float swiglu(float g, float u) { return (g / (1.f + std::exp(-g))) * u; }   // native_gu_rows' expression

// gate/up rows [r0, r1), the tokens act[0..nt) (nt <= kFastMaxNt), rows interleaved: one token takes four rows (gate and
// up of two row indices) per pass, two tokens two rows (gate and up of one index), three or four tokens one row.
template <int NT>
void fast_gu_pass(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int nb, const void* const* act,
                  float* const* ff, int r0, int r1) {
    const uint8_t* up = blob + up_off;
    int r = r0;
    if constexpr (NT == 1) {
        for (; r + 2 <= r1; r += 2) {
            const uint8_t* wr[4] = {blob + (size_t) r * gu_row, up + (size_t) r * gu_row, blob + (size_t) (r + 1) * gu_row,
                                    up + (size_t) (r + 1) * gu_row};
            const uint8_t* wn[4] = {blob + (size_t) (r + 2) * gu_row, up + (size_t) (r + 2) * gu_row,
                                    blob + (size_t) (r + 3) * gu_row, up + (size_t) (r + 3) * gu_row};
            float res[4][1];
            fast_set<4, 1>(type, wr, wn, nb, act, res);
            ff[0][r] = swiglu(res[0][0], res[1][0]);
            ff[0][r + 1] = swiglu(res[2][0], res[3][0]);
        }
    }
    if constexpr (NT <= 2) {
        for (; r < r1; ++r) {
            const uint8_t* wr[2] = {blob + (size_t) r * gu_row, up + (size_t) r * gu_row};
            const uint8_t* wn[2] = {blob + (size_t) (r + 1) * gu_row, up + (size_t) (r + 1) * gu_row};
            float res[2][NT];
            fast_set<2, NT>(type, wr, wn, nb, act, res);
            for (int t = 0; t < NT; ++t) ff[t][r] = swiglu(res[0][t], res[1][t]);
        }
    } else {
        for (; r < r1; ++r) {
            const uint8_t* g = blob + (size_t) r * gu_row;
            const uint8_t* u = up + (size_t) r * gu_row;
            const uint8_t* gn = g + gu_row;
            const uint8_t* un = u + gu_row;
            float rg[1][NT], ru[1][NT];
            fast_set<1, NT>(type, &g, &gn, nb, act, rg);
            fast_set<1, NT>(type, &u, &un, nb, act, ru);
            for (int t = 0; t < NT; ++t) ff[t][r] = swiglu(rg[0][t], ru[0][t]);
        }
    }
}

// down rows [r0, r1) of w, same shapes (one token: four rows per pass; two: two; three or four: one).
template <int NT>
void fast_rows_pass(int type, const uint8_t* w, size_t row_bytes, int nb, const void* const* act, float* const* out,
                    int r0, int r1) {
    int r = r0;
    if constexpr (NT == 1) {
        for (; r + 4 <= r1; r += 4) {
            const uint8_t* wr[4] = {w + (size_t) r * row_bytes, w + (size_t) (r + 1) * row_bytes,
                                    w + (size_t) (r + 2) * row_bytes, w + (size_t) (r + 3) * row_bytes};
            const uint8_t* wn[4] = {w + (size_t) (r + 4) * row_bytes, w + (size_t) (r + 5) * row_bytes,
                                    w + (size_t) (r + 6) * row_bytes, w + (size_t) (r + 7) * row_bytes};
            float res[4][1];
            fast_set<4, 1>(type, wr, wn, nb, act, res);
            for (int k = 0; k < 4; ++k) out[0][r + k] = res[k][0];
        }
    }
    if constexpr (NT <= 2) {
        for (; r + 2 <= r1; r += 2) {
            const uint8_t* wr[2] = {w + (size_t) r * row_bytes, w + (size_t) (r + 1) * row_bytes};
            const uint8_t* wn[2] = {w + (size_t) (r + 2) * row_bytes, w + (size_t) (r + 3) * row_bytes};
            float res[2][NT];
            fast_set<2, NT>(type, wr, wn, nb, act, res);
            for (int k = 0; k < 2; ++k)
                for (int t = 0; t < NT; ++t) out[t][r + k] = res[k][t];
        }
    }
    for (; r < r1; ++r) {
        const uint8_t* wr = w + (size_t) r * row_bytes;
        const uint8_t* wn = wr + row_bytes;
        float res[1][NT];
        fast_set<1, NT>(type, &wr, &wn, nb, act, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[0][t];
    }
}

void dot_rows(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (type) {
        case 12: q4k_dot((const block_q4_K*) row, n / QK_K, act, nt, res); break;
        case 7: q5_1_dot((const block_q5_1*) row, n / QK5_1, act, nt, res); break;
        case 8: q8_0_dot((const block_q8_0*) row, n / QK8_0, act, nt, res); break;
        default: break;
    }
}

}  // namespace

bool kq256_supported(int type) noexcept { return type == 12 || type == 7 || type == 8; }

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    // each row is read once for all `nt` (<= 8) tokens: the router (2.6 MB per layer) streams once per prediction
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc[8];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        for (int c = 0; c + 8 <= cols; c += 8) {
            const __m256 wf = _mm256_castsi256_ps(
                _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c))), 16));
            for (int t = 0; t < nt; ++t) acc[t] = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + (size_t) t * cols + c), acc[t]);
        }
        for (int t = 0; t < nt; ++t) out[(size_t) t * rows + r] = hsum_float_8(acc[t]);   // cols % 8 == 0 (2560)
    }
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c)));
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(h, 16)), _mm256_loadu_ps(x + c), acc);
        }
        float s = hsum_float_8(acc);
        for (; c < cols; ++c) {
            const uint32_t b = (uint32_t) wr[c] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            s += f * x[c];
        }
        out[r] = s;
    }
}

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float g[MAXT], u[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, blob + (size_t) r * gu_row, n, act + t0, m, g);
            dot_rows(type, blob + up_off + (size_t) r * gu_row, n, act + t0, m, u);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
            for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float s[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, w + (size_t) r * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

bool kqfast_supported(int type) noexcept { return type == 12 || type == 7; }

void kqfast_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                    float* const* ff, int r0, int r1) {
    const int nb = n / (type == 12 ? QK_K : QK5_1);
    for (int t0 = 0; t0 < nt; t0 += kFastMaxNt) {
        const int m = nt - t0 < kFastMaxNt ? nt - t0 : kFastMaxNt;
        switch (m) {
            case 1: fast_gu_pass<1>(type, blob, gu_row, up_off, nb, act + t0, ff + t0, r0, r1); break;
            case 2: fast_gu_pass<2>(type, blob, gu_row, up_off, nb, act + t0, ff + t0, r0, r1); break;
            case 3: fast_gu_pass<3>(type, blob, gu_row, up_off, nb, act + t0, ff + t0, r0, r1); break;
            default: fast_gu_pass<4>(type, blob, gu_row, up_off, nb, act + t0, ff + t0, r0, r1); break;
        }
    }
}

void kqfast_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    const int nb = n / (type == 12 ? QK_K : QK5_1);
    for (int t0 = 0; t0 < nt; t0 += kFastMaxNt) {
        const int m = nt - t0 < kFastMaxNt ? nt - t0 : kFastMaxNt;
        switch (m) {
            case 1: fast_rows_pass<1>(type, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
            case 2: fast_rows_pass<2>(type, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
            case 3: fast_rows_pass<3>(type, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
            default: fast_rows_pass<4>(type, w, row_bytes, nb, act + t0, out + t0, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
