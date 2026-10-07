// src/kernels/cpu/pool_layer_test.cpp - the expert pool's two layer drains (ExpertPool::LayerDrain: Barriered, Counters) and the
// three Q4_K/Q5_1 row kernels (native_set_kq_kernel) on a native layer, bit for bit, and their time.  No GPU, no model.
//
//     pool_layer_test
//         random layers (1..40 experts, 1..8 tokens each, several worker counts, the host working or not, chunk sizes from 4
//         rows up, a pool whose workers sleep between layers) through run_split_multi_native in both drains and all kernel
//         modes, switched at random between layers; every output memcmp-equal to a single-thread reference built from
//         native_gu_rows / native_quant_h / native_down_rows.  Exit code 1 on any difference or a stalled layer.
//     pool_layer_test --bench [--epl 3.3] [--nt 1] [--layers 2000] [--mb 768] [--drain barriered,counters]
//                             [--kernel ggml,fast] [--gu-rows 40] [--down-rows 160] [--workers N] [--no-host]
//         the real pool (workers pinned to physical cores as in the engine) on a pool of --mb MiB of random expert blobs:
//         ms per layer (mean, p50, p90, p99), ms per distinct expert and GB/s of expert bytes for each drain x kernel, as
//         the engine's "CPU pool call" and "ms per distinct CPU expert" measure them.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
using Drain = cpu::ExpertPool::LayerDrain;

namespace {

constexpr int kH = cpu::H, kFF = cpu::FF;
std::mt19937_64 g_rng(0xC0FFEE);
uint32_t rnd() { return (uint32_t) g_rng(); }
float rnd_f(float lo, float hi) { return lo + (hi - lo) * (float) (rnd() >> 8) / (float) (1u << 24); }

void put_half(uint8_t* p, float v) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(v);
    std::memcpy(p, &h, 2);
}

// a UD-Q4_K_XL expert (Q4_K gate/up; Q5_1 or Q8_0 down): random bytes, valid fp16 scales
void fill_blob(const cpu::NativeFmt& f, uint8_t* blob) {
    for (size_t i = 0; i < f.bytes; ++i) blob[i] = (uint8_t) rnd();
    const int nbg = (int) (f.n_embd / 256);
    for (int part = 0; part < 2; ++part)
        for (int r = 0; r < kFF; ++r)
            for (int b = 0; b < nbg; ++b) {
                uint8_t* x = blob + (size_t) part * f.up_off + (size_t) r * f.gu_row + (size_t) b * 144;
                put_half(x, rnd_f(0.0005f, 0.06f));
                put_half(x + 2, rnd_f(-0.05f, 0.08f));
            }
    const size_t bs = f.d_type == 7 ? 24 : 34;
    for (int r = 0; r < kH; ++r)
        for (int b = 0; b < (int) (f.n_ff / 32); ++b) {
            uint8_t* x = blob + f.down_off + (size_t) r * f.d_row + (size_t) b * bs;
            put_half(x, rnd_f(-0.06f, 0.06f));
            if (f.d_type == 7) put_half(x + 2, rnd_f(-0.5f, 0.5f));
        }
}

struct Layer {
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<std::vector<float>> out;   // [expert * MAXT + t][H]
    std::vector<float> ff;
};

// the reference for one expert and token: the whole layer on one thread
void reference(const cpu::NativeFmt& f, const uint8_t* blob, const void* act, float* out) {
    std::vector<float> ff((size_t) kFF);
    float* ffp[1] = {ff.data()};
    const void* ap[1] = {act};
    cpu::native_gu_rows(f, blob, ap, 1, ffp, 0, kFF);
    alignas(64) uint8_t hq[cpu::kNativeHBytes];
    cpu::native_quant_h(f, ff.data(), hq);
    const void* hp[1] = {hq};
    float* op[1] = {out};
    cpu::native_down_rows(f, blob, hp, 1, op, 0, kH);
}

struct Case {
    int workers;
    bool host;
    int gu_rows, down_rows;
    int spin_us;   // STRATA_POOL_SPIN_US: 0 = workers sleep between layers
};

long g_layers = 0, g_bad = 0;

int test_case(const Case& c, const cpu::NativeFmt& f, int layers) {
    char us[16];
    std::snprintf(us, sizeof us, "%d", c.spin_us);
    setenv("STRATA_POOL_SPIN_US", us, 1);
    cpu::ExpertPool pool(c.workers, false, c.host);
    pool.set_layer_chunks(c.gu_rows, c.down_rows);
    const int nblobs = 48;
    std::vector<uint8_t> blobs((size_t) nblobs * f.bytes);
    for (int i = 0; i < nblobs; ++i) fill_blob(f, blobs.data() + (size_t) i * f.bytes);
    std::vector<float> x((size_t) cpu::MAXT * f.n_embd);
    for (float& v : x) v = rnd_f(-2.f, 2.f);
    std::vector<uint8_t> act((size_t) cpu::MAXT * cpu::kNativeActBytes);
    for (int t = 0; t < cpu::MAXT; ++t)
        cpu::native_quant_act(f, x.data() + (size_t) t * f.n_embd, act.data() + (size_t) t * cpu::kNativeActBytes);
    // reference rows for every (blob, token)
    std::vector<float> ref((size_t) nblobs * cpu::MAXT * kH);
    cpu::native_set_kq_kernel(0);
    for (int b = 0; b < nblobs; ++b)
        for (int t = 0; t < cpu::MAXT; ++t)
            reference(f, blobs.data() + (size_t) b * f.bytes, act.data() + (size_t) t * cpu::kNativeActBytes, ref.data() + ((size_t) b * cpu::MAXT + t) * kH);
    int bad = 0;
    for (int l = 0; l < layers; ++l) {
        const int n = 1 + (int) (rnd() % 40);
        std::vector<cpu::ExpertJobMulti> jobs((size_t) n);
        std::vector<int> which((size_t) n);
        std::vector<float> out((size_t) n * cpu::MAXT * kH, 777.f);
        // distinct blobs, as the engine's jobs are
        std::vector<int> pick((size_t) nblobs);
        for (int i = 0; i < nblobs; ++i) pick[(size_t) i] = i;
        std::shuffle(pick.begin(), pick.end(), g_rng);
        for (int e = 0; e < n; ++e) {
            which[(size_t) e] = pick[(size_t) (e % nblobs)];
            cpu::ExpertJobMulti& j = jobs[(size_t) e];
            j.blob = blobs.data() + (size_t) which[(size_t) e] * f.bytes;
            j.nt = 1 + (int) (rnd() % (rnd() % 4 == 0 ? 8 : 2));
            for (int t = 0; t < j.nt; ++t) {
                const int tok = (int) (rnd() % cpu::MAXT);
                j.nact[t] = act.data() + (size_t) tok * cpu::kNativeActBytes;
                j.out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * kH;
                (void) tok;
            }
        }
        // remember which token each slot used, for the reference
        std::vector<int> tokof((size_t) n * cpu::MAXT, 0);
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[(size_t) e].nt; ++t)
                tokof[(size_t) e * cpu::MAXT + t] = (int) (((const uint8_t*) jobs[(size_t) e].nact[t] - act.data()) / cpu::kNativeActBytes);
        cpu::native_set_kq_kernel((int) (rnd() % 3));
        pool.set_layer_drain((rnd() & 1) ? Drain::Counters : Drain::Barriered);
        pool.run_split_multi_native(f, jobs.data(), n);
        ++g_layers;
        for (int e = 0; e < n && bad < 8; ++e)
            for (int t = 0; t < jobs[(size_t) e].nt; ++t) {
                const float* want = ref.data() + ((size_t) which[(size_t) e] * cpu::MAXT + tokof[(size_t) e * cpu::MAXT + t]) * kH;
                const float* got = jobs[(size_t) e].out[t];
                if (std::memcmp(want, got, (size_t) kH * 4) != 0) {
                    int r = 0;
                    while (r < kH && std::memcmp(want + r, got + r, 4) == 0) ++r;
                    std::printf("FAIL workers %d host %d chunks %d/%d spin %d: layer %d expert %d/%d token %d row %d: got %.9g want %.9g (drain %d kernel %d)\n",
                                c.workers, (int) c.host, c.gu_rows, c.down_rows, c.spin_us, l, e, n, t, r, got[r], want[r],
                                (int) pool.layer_drain(), cpu::native_kq_kernel());
                    ++bad;
                    break;
                }
            }
        if (bad >= 8) break;
    }
    cpu::native_set_kq_kernel(0);
    g_bad += bad;
    return bad;
}

// ---- bench
std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    std::string cur;
    for (char ch : s + ",") {
        if (ch != ',') { cur += ch; continue; }
        v.push_back(cur);
        cur.clear();
    }
    return v;
}

int bench_main(int argc, char** argv) {
    double epl = 3.3;
    int nt = 1, layers = 2000, mb = 768, gu_rows = 40, down_rows = 160, workers = 0;
    bool host = true;
    std::vector<std::string> drains = {"barriered", "counters"}, kernels = {"ggml", "fast"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--epl") epl = std::atof(next());
        else if (a == "--nt") nt = std::atoi(next());
        else if (a == "--layers") layers = std::atoi(next());
        else if (a == "--mb") mb = std::atoi(next());
        else if (a == "--gu-rows") gu_rows = std::atoi(next());
        else if (a == "--down-rows") down_rows = std::atoi(next());
        else if (a == "--workers") workers = std::atoi(next());
        else if (a == "--no-host") host = false;
        else if (a == "--drain") drains = split(next());
        else if (a == "--kernel") kernels = split(next());
    }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(12, 7, 2560, 640, f, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 2; }
    const size_t nblobs = ((size_t) mb << 20) / f.bytes;
    uint8_t* pool_mem = (uint8_t*) std::aligned_alloc(4096, nblobs * f.bytes);
    for (size_t i = 0; i < nblobs; ++i) fill_blob(f, pool_mem + i * f.bytes);
    std::vector<float> x((size_t) cpu::MAXT * f.n_embd);
    for (float& v : x) v = rnd_f(-1.f, 1.f);
    std::vector<uint8_t> act((size_t) cpu::MAXT * cpu::kNativeActBytes);
    for (int t = 0; t < cpu::MAXT; ++t)
        cpu::native_quant_act(f, x.data() + (size_t) t * f.n_embd, act.data() + (size_t) t * cpu::kNativeActBytes);
    std::vector<float> out((size_t) 16 * cpu::MAXT * kH);
    cpu::ExpertPool pool(workers, true, host);
    pool.set_layer_chunks(gu_rows, down_rows);
    std::printf("pool: %d workers, host works %d; %zu expert blobs of %zu B (%d MiB); %.1f experts per layer x %d token(s); counters chunks %d/%d rows\n",
                pool.workers(), (int) pool.host_works(), nblobs, f.bytes, mb, epl, nt, gu_rows, down_rows);
    for (const std::string& kn : kernels) {
        for (const std::string& dn : drains) {
            cpu::native_set_kq_kernel(kn == "ggml" ? 0 : kn == "kq256" ? 1 : 2);
            pool.set_layer_drain(dn == "counters" ? Drain::Counters : Drain::Barriered);
            std::mt19937 rg(1);   // the same layers for every combination
            std::vector<cpu::ExpertJobMulti> jobs(16);
            std::vector<double> lay;
            lay.reserve((size_t) layers);
            double bytes = 0, experts = 0;
            for (int l = -50; l < layers; ++l) {   // 50 warm-up layers
                int n = (int) epl;
                if ((rg() % 1000) < (epl - (int) epl) * 1000) ++n;
                if (n < 1) n = 1;
                for (int e = 0; e < n; ++e) {
                    cpu::ExpertJobMulti& j = jobs[(size_t) e];
                    j.blob = pool_mem + (size_t) (rg() % nblobs) * f.bytes;
                    j.nt = nt;
                    for (int t = 0; t < nt; ++t) {
                        j.nact[t] = act.data() + (size_t) t * cpu::kNativeActBytes;
                        j.out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * kH;
                    }
                }
                const auto t0 = Clock::now();
                pool.run_split_multi_native(f, jobs.data(), n);
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                if (l < 0) continue;
                lay.push_back(ms);
                bytes += (double) n * (double) f.bytes;
                experts += n;
            }
            double tot = 0;
            for (double v : lay) tot += v;
            std::sort(lay.begin(), lay.end());
            const size_t N = lay.size();
            std::printf("kernel %-5s drain %-9s: %.3f ms/layer (p50 %.3f p90 %.3f p99 %.3f) = %.4f ms per distinct expert, %.1f GB/s of expert bytes\n",
                        kn.c_str(), dn.c_str(), tot / (double) N, lay[N / 2], lay[N * 9 / 10], lay[N * 99 / 100], tot / experts,
                        bytes / 1e6 / tot);
            std::fflush(stdout);
        }
    }
    cpu::native_set_kq_kernel(0);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--bench") return bench_main(argc, argv);
    cpu::NativeFmt f, f8;
    std::string err;
    if (!cpu::native_fmt(12, 7, 2560, 640, f, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 2; }
    if (!cpu::native_fmt(12, 8, 2560, 640, f8, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 2; }
    const Case cases[] = {
        {1, true, 40, 160, 20000},  {2, true, 4, 4, 20000},     {3, false, 8, 12, 20000}, {5, true, 40, 160, 20000},
        {5, true, 64, 256, 0},      {8, true, 40, 160, 20000},  {8, false, 100, 600, 0},  {16, true, 4, 8, 20000},
        {4, true, 640, 2560, 20000},
    };
    int bad = 0;
    for (const Case& c : cases) {
        bad += test_case(c, f, 150);
        bad += test_case(c, f8, 40);   // Q8_0 down rows
    }
    std::printf("pool_layer_test: %ld layers, %ld differences\n", g_layers, g_bad);
    return bad == 0 ? 0 : 1;
}
