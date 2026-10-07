// src/kernels/hc_q8_ref.hpp - host-side fixture and double-precision reference of the hyper-connection read (header-only).
//
// Shared by tests/cuda_emu/hc_q8_emu_test.cpp (the device code on the CPU) and src/kernels/hc_q8_parity.cpp (on a GPU):
// random Q8_0 weights (real-shaped blocks, quantized the way ggml's quantize_row_q8_0 does), the BF16 inject rows,
// activations, the previous half's outputs, and the read computed in double from the dequantized values.
#pragma once

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace hcref {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;

struct Fixture {
    int T = 3;
    std::vector<uint8_t> q8_down, q8_up, q8_inj;   // Q8_0 [320][10240], [10240][320], [4][10240]
    std::vector<uint16_t> w_inj;                   // bf16 [4][10240]
    std::vector<float> w_norm, R, bo, inj_prev;
    float eps = 1e-6f;
    std::vector<float> dq_down, dq_up, dq_inj;     // the dequantized values (exact products), for the reference

    static void make_q8(std::mt19937& rng, size_t rows, size_t cols, float sigma, std::vector<uint8_t>& bytes,
                        std::vector<float>& deq) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        bytes.assign(rows * cols / 32 * 34, 0);
        deq.assign(rows * cols, 0.0f);
        for (size_t r = 0; r < rows; ++r)
            for (size_t b = 0; b < cols / 32; ++b) {
                float x[32], amax = 0.0f;
                for (int i = 0; i < 32; ++i) { x[i] = sigma * nd(rng); amax = std::max(amax, std::fabs(x[i])); }
                const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
                const uint16_t dh = strata::kernels::f16_from_f32(d);
                uint8_t* blk = bytes.data() + (r * (cols / 32) + b) * 34;
                std::memcpy(blk, &dh, 2);
                for (int i = 0; i < 32; ++i) {
                    const int8_t q = (int8_t) std::lround(x[i] * id);
                    blk[2 + i] = (uint8_t) q;
                    deq[r * cols + b * 32 + i] = strata::kernels::f32_from_f16(dh) * (float) q;
                }
            }
    }
    void build(int tokens, unsigned seed) {
        T = tokens;
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        make_q8(rng, LR, D, 0.02f, q8_down, dq_down);
        make_q8(rng, D, LR, 0.05f, q8_up, dq_up);
        make_q8(rng, HC, D, 0.02f, q8_inj, dq_inj);
        w_inj.resize((size_t) HC * D);
        for (size_t i = 0; i < w_inj.size(); ++i) w_inj[i] = strata::kernels::bf16_from_f32(0.02f * nd(rng));
        w_norm.resize(D);
        for (auto& v : w_norm) v = 1.0f + 0.3f * nd(rng);
        R.resize((size_t) T * D); bo.resize((size_t) T * N); inj_prev.resize((size_t) T * HC);
        for (auto& v : R) v = 2.0f * nd(rng);
        for (auto& v : bo) v = nd(rng);
        for (auto& v : inj_prev) v = 2.0f * nd(rng);
    }
    /// the weights as the pack holds them (--compat-bf16): every Q8_0 value rounded to BF16 (nearest even)
    static std::vector<uint16_t> bf16_copy(const std::vector<float>& deq) {
        std::vector<uint16_t> o(deq.size());
        for (size_t i = 0; i < deq.size(); ++i) o[i] = strata::kernels::bf16_from_f32(deq[i]);
        return o;
    }
};

struct Ref { std::vector<double> mixed, Rout, rs, inject, lo; };

/// The read of include/strata/kernels/fused_gr.hpp in double: R' = R (+ bo * 2 sigmoid(inj_prev / hc) when `apply`), rs per stream,
/// xn = R' * w_norm * rs, lo = silu((W_down . xn) / hc), inject = W_inject . xn, mixed = mean_c xn * sigmoid(W_up . lo).
/// `down` / `up` hold the weights as floats [LR][D] / [D][LR] (the dequantized Q8_0 values, or the BF16 copy).
inline Ref reference(const Fixture& f, bool apply, bool inject, bool inject_q8, const std::vector<float>& down,
                     const std::vector<float>& up) {
    const int T = f.T;
    Ref r;
    r.mixed.assign((size_t) T * N, 0); r.Rout.assign((size_t) T * D, 0); r.rs.assign((size_t) T * HC, 0);
    r.inject.assign((size_t) T * HC, 0); r.lo.assign((size_t) T * LR, 0);
    for (int t = 0; t < T; ++t) {
        std::vector<double> Rp(D), xn(D);
        for (int ci = 0; ci < HC; ++ci) {
            const double gw = apply ? 2.0 / (1.0 + std::exp(-(double) f.inj_prev[(size_t) t * HC + ci] / HC)) : 0.0;
            double ss = 0;
            for (int d = 0; d < N; ++d) {
                const size_t i = (size_t) ci * N + d;
                Rp[i] = (double) f.R[(size_t) t * D + i] + (apply ? (double) f.bo[(size_t) t * N + d] * gw : 0.0);
                ss += Rp[i] * Rp[i];
            }
            const double rs = 1.0 / std::sqrt(ss / N + (double) f.eps);
            r.rs[(size_t) t * HC + ci] = rs;
            for (int d = 0; d < N; ++d) {
                const size_t i = (size_t) ci * N + d;
                xn[i] = Rp[i] * (double) f.w_norm[i] * rs;
                r.Rout[(size_t) t * D + i] = Rp[i];
            }
        }
        for (int row = 0; row < LR; ++row) {
            double a = 0;
            for (int i = 0; i < D; ++i) a += (double) down[(size_t) row * D + i] * xn[i];
            const double x = a / HC;
            r.lo[(size_t) t * LR + row] = x / (1.0 + std::exp(-x));
        }
        if (inject)
            for (int ci = 0; ci < HC; ++ci) {
                double a = 0;
                for (int i = 0; i < D; ++i) {
                    const double w = inject_q8 ? (double) f.dq_inj[(size_t) ci * D + i]
                                               : (double) strata::kernels::f32_from_bf16(f.w_inj[(size_t) ci * D + i]);
                    a += w * xn[i];
                }
                r.inject[(size_t) t * HC + ci] = a;
            }
        for (int d = 0; d < N; ++d) {
            double s = 0;
            for (int ci = 0; ci < HC; ++ci) {
                double a = 0;
                for (int k = 0; k < LR; ++k) a += (double) up[((size_t) ci * N + d) * LR + k] * r.lo[(size_t) t * LR + k];
                s += xn[(size_t) ci * N + d] / (1.0 + std::exp(-a));
            }
            r.mixed[(size_t) t * N + d] = s / HC;
        }
    }
    return r;
}

/// worst |got - ref| / max |ref|
inline double worst(const std::vector<float>& got, const std::vector<double>& ref) {
    double w = 0, mag = 1e-30;
    for (double v : ref) mag = std::max(mag, std::fabs(v));
    for (size_t i = 0; i < ref.size(); ++i) w = std::max(w, std::fabs((double) got[i] - ref[i]));
    return w / mag;
}
inline bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace hcref
