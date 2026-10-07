// prefill_dense_mmq_test - the prompt path's dense GGUF projections through int8 MMQ (STRATA_PREFILL_DENSE_MMQ=1) against the
// default path (dequantize to FP16 + cuBLAS, FP32 accumulate) and against a double-precision reference, on synthetic Q8_0
// weights at the engine's real projection shapes (Qwen3.8-Flash-Next: 2560 hidden, GDN qkv 10240 / gate 6144 / out 2560 x
// 6144, QSA q 12288 / k, v 512 / out, shared expert 640 and its K = 640 down, which MMQ must refuse and leave to cuBLAS).
//
//   prefill_dense_mmq_test [--tokens T] [--bench] [--reps N]
//
// Default: T = 2048 tokens, correctness only (about 0.35 GiB of VRAM: the largest shape is attn_q, 12288 x 2560: Y 2048 x
// 12288 FP32 = 100 MiB, X, the 33 MiB weight, Gemm's 64 MiB dequantization scratch and 32 MiB cuBLAS workspace, and MMQ's ~10 MiB
// quantized-activation buffer).  --bench runs T = 8192 (the production chunk) unless --tokens is given and times both paths per shape with CUDA
// events (median of --reps, default 7) - about 0.75 GiB at T = 8192, 0.45 GiB at T = 4096 - then adds up one stage's dense work
// per chunk (18 GDN + 6 QSA layers) for both paths.  Exit 77 when there is no CUDA device or the device is below sm_80.
//
// Pass: the default path within 0.15% (relative L2) of the double-precision reference, the MMQ path within 2%, the refused
// shape (K = 640) bitwise equal between the two settings.
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::prefill::Gemm;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { ck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

struct Shape {
    const char* name;
    int64_t N, K;
    int per_gdn, per_qsa;   // calls per layer of each kind
};

// the engine's dense GGUF projections (prefill.cpp: native_proj call sites); the shared expert gate and up are one shape
const Shape kShapes[] = {
    {"attn_qkv", 10240, 2560, 1, 0},     {"attn_gate(z)", 6144, 2560, 1, 0}, {"ssm_out", 2560, 6144, 1, 0},
    {"attn_q", 12288, 2560, 0, 1},       {"attn_k", 512, 2560, 0, 1},        {"attn_v", 512, 2560, 0, 1},
    {"attn_output", 2560, 6144, 0, 1},   {"shexp gate/up", 640, 2560, 2, 2}, {"shexp down (K=640)", 2560, 640, 1, 1},
};

// a matrix of any ggml type from a smooth deterministic pattern (for the timing; layout, not values, matters there)
std::vector<uint8_t> make_q(ggml_type t, int64_t N, int64_t K, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    const auto* tr = ggml_get_type_traits(t);
    const size_t rb = ggml_row_size(t, K);
    std::vector<uint8_t> out((size_t) N * rb);
    std::vector<float> row((size_t) K);
    for (int64_t r = 0; r < N; ++r) {
        for (auto& v : row) v = 0.02f * n01(g);
        tr->from_float_ref(row.data(), out.data() + (size_t) r * rb, K);
    }
    return out;
}

// activations like the engine's: rows differ in scale, channel gains are log-normal, a few channels are massive
std::vector<float> make_x(int64_t T, int64_t K, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::vector<float> gain((size_t) K);
    for (auto& v : gain) v = std::exp(0.8f * n01(g));
    for (auto& v : gain) if (g() % 1000 < 4) v *= 30.0f;
    std::vector<float> x((size_t) (T * K));
    for (int64_t t = 0; t < T; ++t) {
        const float rs = std::exp(1.2f * n01(g));
        for (int64_t k = 0; k < K; ++k) x[(size_t) (t * K + k)] = rs * gain[(size_t) k] * n01(g);
    }
    return x;
}

std::vector<uint8_t> make_q8(int64_t N, int64_t K, uint32_t seed, std::vector<float>& deq) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    const auto* tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, K);
    std::vector<uint8_t> out((size_t) N * rb);
    std::vector<float> row((size_t) K);
    deq.assign((size_t) (N * K), 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t b = 0; b < K / 32; ++b) {
            const float s = 0.02f * std::exp(0.5f * n01(g));
            for (int i = 0; i < 32; ++i) row[(size_t) (b * 32 + i)] = s * n01(g);
        }
        tr->from_float_ref(row.data(), out.data() + (size_t) r * rb, K);
        tr->to_float(out.data() + (size_t) r * rb, deq.data() + (size_t) (r * K), K);
    }
    return out;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int64_t T = 0;
    bool bench = false;
    int reps = 7;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) T = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    }
    if (T <= 0) T = bench ? 8192 : 2048;
    try {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("no CUDA device\n"); return 77; }
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, 0), "properties");
        if (prop.major * 10 + prop.minor < 80) { std::printf("%s is below sm_80: skipped\n", prop.name); return 77; }
        if (!strata::prefill::mmq::built()) { std::printf("this build has no MMQ kernels\n"); return 77; }
        std::printf("%s, T = %lld tokens%s\n", prop.name, (long long) T, bench ? ", timing" : "");

        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        Gemm gm;
        std::string err;
        constexpr int64_t kScratch = 32ll << 20;   // the engine's GEMM_SCRATCH: FP16 elements
        if (!gm.init(s, kScratch, err)) throw std::runtime_error(err);
        gm.set_dense_mmq_min_n(0);   // every shape through MMQ here, to check and time them; the engine's default keeps N < 2048 on cuBLAS
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event");
        ck(cudaEventCreate(&e1), "event");

        int failures = 0;
        double stage_ms[2] = {0, 0};   // one stage's dense GGUF work per chunk: default, MMQ
        for (const Shape& sh : kShapes) {
            const int64_t N = sh.N, K = sh.K;
            std::vector<float> deq;
            const std::vector<uint8_t> w = make_q8(N, K, 100 + (uint32_t) N, deq);
            const std::vector<float> x = make_x(T, K, 200 + (uint32_t) K);
            std::vector<__half> x16((size_t) (T * K));
            for (size_t i = 0; i < x16.size(); ++i) x16[i] = __float2half(x[i]);
            Dev dw(w.size() + 4096), dx(x16.size() * 2), dy0((size_t) (T * N) * 4), dy1((size_t) (T * N) * 4);
            ck(cudaMemset(dw.p, 0, dw.n), "zero w");
            ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
            ck(cudaMemcpy(dx.p, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice), "x");
            ck(cudaDeviceSynchronize(), "uploads");   // the stream below is non-blocking: it does not wait for the legacy stream

            auto run = [&](bool mmq, Dev& y) {
                gm.set_dense_mmq(mmq);
                ck(cudaMemsetAsync(y.p, 0xff, y.n, s), "sentinel");
                gm.native((const uint16_t*) dx.p, GGML_TYPE_Q8_0, dw.p, (float*) y.p, T, N, K);
            };
            const int64_t mmq_before = gm.dense_mmq_calls();
            run(false, dy0);
            run(true, dy1);
            ck(cudaStreamSynchronize(s), "sync");
            const bool took_mmq = gm.dense_mmq_calls() > mmq_before;
            std::vector<float> y0((size_t) (T * N)), y1((size_t) (T * N));
            ck(cudaMemcpy(y0.data(), dy0.p, y0.size() * 4, cudaMemcpyDeviceToHost), "y0");
            ck(cudaMemcpy(y1.data(), dy1.p, y1.size() * 4, cudaMemcpyDeviceToHost), "y1");

            // double-precision reference on a sample of tokens (every output column)
            std::vector<int64_t> toks;
            for (int64_t t = 0; t < T; t += std::max<int64_t>(1, T / 24)) toks.push_back(t);
            double e_def = 0, e_mmq = 0, e_dd = 0, r2 = 0;
            bool finite = true, same = true;
            for (int64_t t : toks)
                for (int64_t o = 0; o < N; ++o) {
                    double ref = 0;
                    for (int64_t k = 0; k < K; ++k)
                        ref += (double) deq[(size_t) (o * K + k)] * (double) __half2float(x16[(size_t) (t * K + k)]);
                    const double a = y0[(size_t) (t * N + o)], b = y1[(size_t) (t * N + o)];
                    finite = finite && std::isfinite(a) && std::isfinite(b);
                    e_def += (a - ref) * (a - ref);
                    e_mmq += (b - ref) * (b - ref);
                    e_dd += (a - b) * (a - b);
                    r2 += ref * ref;
                }
            if (std::memcmp(y0.data(), y1.data(), y0.size() * 4) != 0) same = false;
            const double rel_def = std::sqrt(e_def / r2), rel_mmq = std::sqrt(e_mmq / r2), rel_dd = std::sqrt(e_dd / r2);
            std::printf("%-20s N %5lld K %4lld  via %-5s  rel L2 vs double: default %.4f%%  MMQ %.4f%%  (default vs MMQ %.4f%%)%s\n", sh.name,
                        (long long) N, (long long) K, took_mmq ? "MMQ" : "cuBLAS", 100 * rel_def, 100 * rel_mmq, 100 * rel_dd,
                        same ? "  bitwise equal" : "");
            if (!finite) { std::printf("  FAIL: non-finite or unwritten output\n"); ++failures; }
            if (rel_def > 1.5e-3) { std::printf("  FAIL: the default path is %.4f%% off the reference (bound 0.15%%)\n", 100 * rel_def); ++failures; }
            if (rel_mmq > 2e-2) { std::printf("  FAIL: the MMQ path is %.3f%% off the reference (bound 2%%)\n", 100 * rel_mmq); ++failures; }
            const bool should_take = K % 256 == 0;
            if (should_take != took_mmq) { std::printf("  FAIL: MMQ %s this shape (K %% 256 == %lld)\n", took_mmq ? "took" : "refused", (long long) (K % 256)); ++failures; }
            if (!should_take && !same) { std::printf("  FAIL: a refused shape differs between the two settings\n"); ++failures; }

            if (bench) {
                double ms[2] = {0, 0};
                for (int m = 0; m < 2; ++m) {
                    std::vector<double> v;
                    gm.set_dense_mmq(m == 1);
                    for (int r = -1; r < reps; ++r) {   // one warm-up
                        ck(cudaEventRecord(e0, s), "rec");
                        gm.native((const uint16_t*) dx.p, GGML_TYPE_Q8_0, dw.p, (float*) dy0.p, T, N, K);
                        ck(cudaEventRecord(e1, s), "rec");
                        ck(cudaEventSynchronize(e1), "sync");
                        float t = 0;
                        ck(cudaEventElapsedTime(&t, e0, e1), "elapsed");
                        if (r >= 0) v.push_back(t);
                    }
                    ms[m] = median(v);
                }
                const double tflops0 = 2.0 * (double) T * (double) N * (double) K / (ms[0] * 1e-3) / 1e12;
                const double tflops1 = 2.0 * (double) T * (double) N * (double) K / (ms[1] * 1e-3) / 1e12;
                std::printf("    time per call at T=%lld: default %.3f ms (%.1f TFLOP/s), MMQ %.3f ms (%.1f TFLOP/s)%s\n", (long long) T, ms[0],
                            tflops0, ms[1], tflops1, should_take ? "" : "  [K=640: same path]");
                // a stage holds 18 GDN + 6 QSA layers (layer split 24 / 24, a QSA layer every 4th)
                stage_ms[0] += ms[0] * (18 * sh.per_gdn + 6 * sh.per_qsa);
                stage_ms[1] += ms[1] * (18 * sh.per_gdn + 6 * sh.per_qsa);
            }
        }
        if (bench)
            std::printf("one stage's dense GGUF projections per %lld-token chunk (18 GDN + 6 QSA layers): default %.0f ms, MMQ %.0f ms "
                        "(%.2fx); excludes the BF16 hyper-connection products and the FP16->FP32 widening MMQ's call includes\n",
                        (long long) T, stage_ms[0], stage_ms[1], stage_ms[1] > 0 ? stage_ms[0] / stage_ms[1] : 0.0);
        if (bench) {
            // The other GEMM-shaped work of a stage's chunk, for the phase split: one MMQ group of 16 experts (gate/up Q4_K 1280 x 2560,
            // down Q5_1 2560 x 640, T x 10 / 512 = 160 rows each as at T = 8192 with 10 experts per token) as prefill.cpp's compute()
            // runs it (gate/up product, SwiGLU, q8_1 quantize, down product), and the BF16 hyper-connection products.
            namespace mmq = strata::prefill::mmq;
            if (mmq::supported(GGML_TYPE_Q4_K) && mmq::supported(GGML_TYPE_Q5_1)) {
                constexpr int G = 16;
                const int rows_per = (int) std::max<int64_t>(1, T * 10 / 512), rows = G * rows_per;
                std::vector<uint8_t> wgu, wd;
                std::vector<float> junk;
                for (int e = 0; e < G; ++e) {
                    std::vector<uint8_t> a = make_q(GGML_TYPE_Q4_K, 1280, 2560, 300 + e), b = make_q(GGML_TYPE_Q5_1, 2560, 640, 400 + e);
                    wgu.insert(wgu.end(), a.begin(), a.end());
                    wd.insert(wd.end(), b.begin(), b.end());
                }
                wgu.resize(wgu.size() + 4096, 0);
                wd.resize(wd.size() + 4096, 0);
                const std::vector<float> x = make_x(rows, 2560, 77);
                std::vector<int32_t> bounds((size_t) G + 1), ident((size_t) rows);
                for (int e = 0; e <= G; ++e) bounds[(size_t) e] = e * rows_per;
                for (int i = 0; i < rows; ++i) ident[(size_t) i] = i;
                Dev dgu(wgu.size()), dd(wd.size()), dx(x.size() * 4), db(bounds.size() * 4), di(ident.size() * 4),
                    dxq(mmq::q8_bytes(rows, 2560)), dhq(mmq::q8_bytes(rows, 640)), dGU((size_t) rows * 1280 * 4),
                    dH((size_t) rows * 640 * 4), dY((size_t) rows * 2560 * 4);
                ck(cudaMemcpy(dgu.p, wgu.data(), wgu.size(), cudaMemcpyHostToDevice), "gu");
                ck(cudaMemcpy(dd.p, wd.data(), wd.size(), cudaMemcpyHostToDevice), "d");
                ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
                ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "b");
                ck(cudaMemcpy(di.p, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice), "i");
                ck(cudaDeviceSynchronize(), "uploads");
                mmq::Context ctx;
                auto group = [&]() {
                    mmq::quantize((const float*) dx.p, nullptr, dxq.p, GGML_TYPE_Q4_K, 2560, 2560, rows, s);
                    mmq::Product p;
                    p.w = dgu.p; p.type = GGML_TYPE_Q4_K; p.w_rows = 1280; p.w_cols = 2560;
                    p.expert_bytes = mmq::matrix_bytes(GGML_TYPE_Q4_K, 1280, 2560); p.n = G; p.xq = dxq.p;
                    p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) di.p; p.total_rows = rows; p.max_rows = rows_per;
                    p.dst = (float*) dGU.p; p.ld_dst = 1280;
                    ctx.run(p, s);
                    mmq::swiglu((const float*) dGU.p, (float*) dH.p, rows, 640, false, s);
                    mmq::quantize((const float*) dH.p, nullptr, dhq.p, GGML_TYPE_Q5_1, 640, 640, rows, s);
                    mmq::Product q;
                    q.w = dd.p; q.type = GGML_TYPE_Q5_1; q.w_rows = 2560; q.w_cols = 640;
                    q.expert_bytes = mmq::matrix_bytes(GGML_TYPE_Q5_1, 2560, 640); q.n = G; q.xq = dhq.p;
                    q.bounds = (const int32_t*) db.p; q.ids = (const int32_t*) di.p; q.total_rows = rows; q.max_rows = rows_per;
                    q.dst = (float*) dY.p; q.ld_dst = 2560;
                    ctx.run(q, s);
                };
                std::vector<double> v;
                for (int r = -1; r < reps; ++r) {
                    ck(cudaEventRecord(e0, s), "rec");
                    group();
                    ck(cudaEventRecord(e1, s), "rec");
                    ck(cudaEventSynchronize(e1), "sync");
                    float t = 0;
                    ck(cudaEventElapsedTime(&t, e0, e1), "elapsed");
                    if (r >= 0) v.push_back(t);
                }
                const double g = median(v);
                std::printf("MoE group of %d experts x %d rows: %.3f ms (%.1f int8-equivalent TFLOP/s); a layer is 32 such groups, a stage's "
                            "24 layers %.0f ms per chunk\n", G, rows_per, g,
                            2.0 * rows * (1280.0 * 2560 + 2560.0 * 640) / (g * 1e-3) / 1e12, g * 32 * 24);
            }
            {   // BF16 products of a hyper-connection read (down 320 x 10240, up 10240 x 320, inject 4 x 10240), two per layer
                struct B { const char* name; int64_t N, K; } bs[] = {{"hc down", 320, 10240}, {"hc up", 10240, 320}, {"hc inject", 4, 10240}};
                double per_layer = 0;
                for (const B& b : bs) {
                    Dev w((size_t) (b.N * b.K) * 2), xx((size_t) (T * b.K) * 2), y((size_t) (T * b.N) * 4);
                    ck(cudaMemset(w.p, 0x3c, w.n), "w");
                    ck(cudaMemset(xx.p, 0x3c, xx.n), "x");
                    ck(cudaDeviceSynchronize(), "sync");
                    std::vector<double> v;
                    for (int r = -1; r < reps; ++r) {
                        ck(cudaEventRecord(e0, s), "rec");
                        gm.bf16((const uint16_t*) xx.p, (const uint16_t*) w.p, (float*) y.p, T, b.N, b.K);
                        ck(cudaEventRecord(e1, s), "rec");
                        ck(cudaEventSynchronize(e1), "sync");
                        float t = 0;
                        ck(cudaEventElapsedTime(&t, e0, e1), "elapsed");
                        if (r >= 0) v.push_back(t);
                    }
                    const double m = median(v);
                    std::printf("%-10s BF16 N %5lld K %5lld: %.3f ms (%.1f TFLOP/s)\n", b.name, (long long) b.N, (long long) b.K, m,
                                2.0 * (double) T * (double) b.N * (double) b.K / (m * 1e-3) / 1e12);
                    per_layer += 2 * m;   // attention half and MoE half
                }
                std::printf("hyper-connection BF16 products: %.2f ms per layer, %.0f ms per stage chunk (24 layers)\n", per_layer, per_layer * 24);
            }
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        std::printf(failures ? "prefill_dense_mmq_test: %d failure(s)\n" : "prefill_dense_mmq_test passed\n", failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_dense_mmq_test failed: %s\n", e.what());
        return 1;
    }
}
