// dense_mmq_numerics_test - what STRATA_PREFILL_DENSE_MMQ=1 changes in the arithmetic of a dense Q8_0 projection, with no
// GPU: the activations are rounded to q8_1 (a scale per 32 values, int8 codes: ggml-cuda's quantize_mmq_q8_1, D4 layout)
// and multiplied against the int8 weights in integers, where the default path rounds the weights (scale x code) to FP16
// and multiplies FP16 x FP16 on the tensor cores with FP32 accumulation.
//
// The data are shaped like the engine's: K = 2560 (every projection's input: the hyper-connection read's normalized row)
// and K = 6144 (ssm_out / attn_output: the attention half's output), tokens whose rows differ in scale by an order of
// magnitude, per-channel gains that are log-normal with a few percent of channels far above the rest (the "massive
// activation" channels of trained transformers), Q8_0 weights quantized as ggml does (a scale per 32 values, the scale
// rounded to FP16).  `--dump FILE` replaces the synthetic activations by real ones: the file STRATA_PREFILL_DUMP_R writes
// (per position: int64 position, then the 10240 FP32 values of the final multi-stream residual), each of its four 2560-value
// streams RMS-normalized as the projections' inputs are.
//
// Three products per output are compared in double precision: the exact one (FP16-rounded activations, exact Q8_0 weights:
// what both paths aim at), the default path's (weights rounded to FP16) and the MMQ path's (activations rounded to q8_1).
// Exits 0 when the MMQ path's relative error is inside the bound documented in docs/OPTIMIZATION_KNOBS.md.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// round-to-nearest-even FP32 -> FP16 -> FP32 (the images the GEMM reads), normal range and subnormals, saturating at 65504
float round_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = u & 0x80000000u;
    const int32_t e = (int32_t) ((u >> 23) & 0xff) - 127;
    float a = std::fabs(f);
    if (!(a == a)) return f;
    if (a >= 65520.0f) return sign ? -65504.0f : 65504.0f;
    if (a < 5.9604645e-8f * 0.5f) return sign ? -0.0f : 0.0f;
    // the spacing of FP16 values around a: 2^(e-10) for normals (e >= -14), 2^-24 below
    const int ee = std::max(e, -14);
    const float ulp = std::ldexp(1.0f, ee - 10);
    const float r = std::nearbyint(a / ulp) * ulp;   // nearbyint: ties to even in the default rounding mode
    return sign ? -r : r;
}

struct Q8Row {   // one weight row: K/32 blocks
    std::vector<float> d;    // the block scale as stored (FP16)
    std::vector<int8_t> q;   // K codes
};

Q8Row quantize_q8_0(const float* w, int K) {
    Q8Row r;
    r.d.resize((size_t) K / 32);
    r.q.resize((size_t) K);
    for (int b = 0; b < K / 32; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(w[b * 32 + i]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        r.d[(size_t) b] = round_f16(d);
        for (int i = 0; i < 32; ++i) r.q[(size_t) (b * 32 + i)] = (int8_t) std::nearbyint(w[b * 32 + i] * id);
    }
    return r;
}

struct Stats {
    double err2 = 0, ref2 = 0, max_abs = 0;
    std::vector<double> tok_err2, tok_ref2;   // per token
    void add(size_t t, double got, double ref) {
        const double d = got - ref;
        err2 += d * d;
        ref2 += ref * ref;
        max_abs = std::max(max_abs, std::fabs(d));
        if (tok_err2.size() <= t) { tok_err2.resize(t + 1, 0.0); tok_ref2.resize(t + 1, 0.0); }
        tok_err2[t] += d * d;
        tok_ref2[t] += ref * ref;
    }
    double rel() const { return std::sqrt(err2 / ref2); }
    double tok_rel(double q) const {   // quantile of the per-token relative errors
        std::vector<double> v;
        for (size_t i = 0; i < tok_err2.size(); ++i) v.push_back(std::sqrt(tok_err2[i] / std::max(tok_ref2[i], 1e-300)));
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t) (q * (double) v.size()))];
    }
};

// activations: T rows of K values
std::vector<float> synthetic_x(int T, int K, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::vector<float> gain((size_t) K);
    for (int k = 0; k < K; ++k) gain[(size_t) k] = std::exp(0.8f * n01(g));                       // log-normal channel gains
    for (int k = 0; k < K; ++k) if (g() % 1000 < 4) gain[(size_t) k] *= 30.0f;                   // 0.4% massive channels
    std::vector<float> x((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        const float rowscale = std::exp(1.2f * n01(g));                                          // rows differ ~10x in scale
        for (int k = 0; k < K; ++k) x[(size_t) t * K + k] = rowscale * gain[(size_t) k] * n01(g);
    }
    return x;
}

bool dump_x(const char* path, int T, int K, std::vector<float>& x) {   // STRATA_PREFILL_DUMP_R records -> K = 2560 rows
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    constexpr int D = 10240, N = 2560;
    std::vector<float> rec((size_t) D);
    x.clear();
    int64_t pos;
    while ((int) (x.size() / (size_t) K) < T && std::fread(&pos, 8, 1, f) == 1 && std::fread(rec.data(), 4, (size_t) D, f) == (size_t) D) {
        for (int s = 0; s < D / N && (int) (x.size() / (size_t) K) < T; ++s) {
            double ss = 0;
            for (int k = 0; k < N; ++k) ss += (double) rec[(size_t) (s * N + k)] * rec[(size_t) (s * N + k)];
            const float inv = 1.0f / std::sqrt((float) (ss / N) + 1e-6f);
            for (int k = 0; k < K; ++k) x.push_back(rec[(size_t) (s * N + k % N)] * inv);
        }
    }
    std::fclose(f);
    return !x.empty();
}

struct Result {
    double mmq_rel, cublas_rel, mmq_vs_cublas, mmq_tok_p99, mmq_max_over_rms;
};

Result run_shape(const char* name, int T, int N_out, int K, const std::vector<float>& x32, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    const int rows = (int) (x32.size() / (size_t) K);
    // the activations as the GEMM reads them (FP16), and as MMQ rounds them (q8_1 of those FP16 values)
    std::vector<float> x16((size_t) rows * K), xq((size_t) rows * K), xs((size_t) rows * (K / 32));
    std::vector<int8_t> xc((size_t) rows * K);
    for (size_t i = 0; i < x16.size(); ++i) x16[i] = round_f16(x32[i]);
    for (int t = 0; t < rows; ++t)
        for (int b = 0; b < K / 32; ++b) {
            const float* v = &x16[(size_t) t * K + b * 32];
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
            const float d_inv = 127.0f / amax, d = 1.0f / d_inv;   // ggml-cuda: float scale, roundf (ties away from zero)
            xs[(size_t) t * (K / 32) + b] = amax == 0.0f ? 0.0f : d;
            for (int i = 0; i < 32; ++i) xc[(size_t) t * K + b * 32 + i] = amax == 0.0f ? 0 : (int8_t) std::round(v[i] * d_inv);
        }
    Stats mmq, cub, diff;
    std::vector<float> w((size_t) K);
    const int use_rows = std::min(rows, T);
    for (int o = 0; o < N_out; ++o) {
        // Gaussian weights with a scale that differs per block (trained rows are not stationary)
        for (int b = 0; b < K / 32; ++b) {
            const float s = 0.02f * std::exp(0.5f * n01(g));
            for (int i = 0; i < 32; ++i) w[(size_t) (b * 32 + i)] = s * n01(g);
        }
        const Q8Row q = quantize_q8_0(w.data(), K);
        std::vector<double> wexact((size_t) K);
        std::vector<float> w16((size_t) K);
        for (int k = 0; k < K; ++k) {
            wexact[(size_t) k] = (double) q.d[(size_t) (k / 32)] * q.q[(size_t) k];
            w16[(size_t) k] = round_f16((float) wexact[(size_t) k]);   // the dequantized FP16 weight the default path multiplies
        }
        for (int t = 0; t < use_rows; ++t) {
            double ref = 0, via_cublas = 0, via_mmq = 0;
            for (int b = 0; b < K / 32; ++b) {
                int32_t isum = 0;
                for (int i = 0; i < 32; ++i) {
                    const size_t k = (size_t) (b * 32 + i);
                    ref += wexact[k] * (double) x16[(size_t) t * K + k];
                    via_cublas += (double) w16[k] * (double) x16[(size_t) t * K + k];
                    isum += (int32_t) q.q[k] * (int32_t) xc[(size_t) t * K + k];
                }
                via_mmq += (double) q.d[(size_t) b] * (double) xs[(size_t) t * (K / 32) + b] * (double) isum;
            }
            mmq.add((size_t) t, via_mmq, ref);
            cub.add((size_t) t, via_cublas, ref);
            diff.add((size_t) t, via_mmq, via_cublas);
        }
    }
    const double rms = std::sqrt(mmq.ref2 / ((double) N_out * use_rows));
    Result r{mmq.rel(), cub.rel(), diff.rel(), mmq.tok_rel(0.99), mmq.max_abs / rms};
    std::printf("%-14s K %5d  rows %4d x %4d outputs: relative L2 error vs exact: MMQ %.3f%%, default (FP16 weights) %.3f%%; "
                "MMQ vs default %.3f%%; MMQ worst token p99 %.3f%%, worst single output %.2f%% of the output rms\n",
                name, K, use_rows, N_out, 100 * r.mmq_rel, 100 * r.cublas_rel, 100 * r.mmq_vs_cublas, 100 * r.mmq_tok_p99,
                100 * r.mmq_max_over_rms);
    (void) T;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    const char* dump = nullptr;
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--dump")) dump = argv[i + 1];
    int failures = 0;
    // 1. the rounding helper against known FP16 values
    {
        struct { float in, out; } cases[] = {{1.0f, 1.0f}, {1.0f + 1.0f / 2048, 1.0f}, {1.0f + 3.0f / 2048, 1.0f + 2.0f / 1024},
                                             {65519.0f, 65504.0f}, {70000.0f, 65504.0f}, {6.1e-5f, 1023.0f / 16777216.0f},
                                             {6.1035156e-5f, 6.1035156e-5f}, {0.0f, 0.0f},
                                             {-2.5f, -2.5f}, {1e-8f, 0.0f}};
        for (const auto& c : cases)
            if (std::fabs(round_f16(c.in) - c.out) > 1e-12f * std::max(1.0f, std::fabs(c.out))) {
                std::printf("FAIL round_f16(%.9g) = %.9g, want %.9g\n", c.in, round_f16(c.in), c.out);
                ++failures;
            }
    }
    // 2. the integer dot is exact: a quantized activation row against a quantized weight row in integers equals the sum of
    // the dequantized products in double (so all of MMQ's error is the activation rounding)
    {
        std::vector<float> x = synthetic_x(1, 256, 1), w(256);
        std::mt19937 g(7);
        std::normal_distribution<float> n01(0, 1);
        for (auto& v : w) v = 0.02f * n01(g);
        const Q8Row q = quantize_q8_0(w.data(), 256);
        double a = 0, b = 0;
        for (int blk = 0; blk < 8; ++blk) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[(size_t) (blk * 32 + i)]));
            const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
            int32_t isum = 0;
            for (int i = 0; i < 32; ++i) {
                const int xq = (int) std::round(x[(size_t) (blk * 32 + i)] * d_inv);
                isum += xq * q.q[(size_t) (blk * 32 + i)];
                a += (double) q.d[(size_t) blk] * q.q[(size_t) (blk * 32 + i)] * (double) d * xq;
            }
            b += (double) q.d[(size_t) blk] * (double) d * isum;
        }
        if (std::fabs(a - b) > 1e-9 * std::max(1.0, std::fabs(a))) {
            std::printf("FAIL integer dot %.12g vs dequantized %.12g\n", b, a);
            ++failures;
        }
    }
    // 3. the error at the engine's shapes
    constexpr int T = 96, NOUT = 48;
    std::vector<float> x2560 = synthetic_x(T, 2560, 11), x6144 = synthetic_x(T, 6144, 12);
    std::vector<float> real;
    if (dump && !dump_x(dump, T, 2560, real)) {
        std::printf("cannot read the dump %s\n", dump);
        return 2;
    }
    std::vector<Result> rs;
    rs.push_back(run_shape("synthetic", T, NOUT, 2560, x2560, 21));
    rs.push_back(run_shape("synthetic", T, NOUT, 6144, x6144, 22));
    if (!real.empty()) rs.push_back(run_shape("real (dump)", T, NOUT, 2560, real, 23));
    // a bound on the activation rounding: q8_1 steps are amax/127 per 32 values, so the output error relative to the
    // output is ~ (amax / rms of the block) / (127 sqrt(12)) / sqrt(32)-ish averaging; measured here, and bounded with margin
    for (const Result& r : rs) {
        if (r.cublas_rel > 1.5e-3) { std::printf("FAIL default path's error %.4f%% above 0.15%%\n", 100 * r.cublas_rel); ++failures; }
        if (r.mmq_rel > 2.0e-2) { std::printf("FAIL MMQ path's error %.3f%% above 2%%\n", 100 * r.mmq_rel); ++failures; }
        if (r.mmq_rel <= r.cublas_rel) { std::printf("FAIL MMQ error not above the default's: the test is not measuring what it says\n"); ++failures; }
    }
    std::printf(failures ? "dense_mmq_numerics_test: %d failure(s)\n" : "dense_mmq_numerics_test passed\n", failures);
    return failures ? 1 : 0;
}
