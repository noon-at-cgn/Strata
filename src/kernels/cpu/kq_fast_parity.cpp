// src/kernels/cpu/kq_fast_parity.cpp - the AVX-2 row kernels of Unsloth UD-Q4_K_XL experts (Q4_K gate/up, Q5_1 down) bit for
// bit against ggml-cpu's vec_dot, and their speed.  No GPU, no model: random blocks (every scale / min / nibble / fifth-bit
// pattern, extreme-value rows that would show an int16 saturation difference) and activations quantized the way the engine
// quantizes them (native_quant_act / native_quant_h).
//
//     kq_fast_parity
//         for every geometry, every group size 1..8, whole and partial row ranges (odd starts and ends): the ggml-cpu
//         reference (one vec_dot per row and token, native_gu_rows' SwiGLU) against kq256_*, kqfast_* and the engine's
//         dispatch in each mode (native_set_kq_kernel 0/1/2); memcmp, i.e. zero tolerance (the repository's parity rule
//         for these formats: native_expert_parity compares kq256 the same way); and the fast kernel's rows over [0, N)
//         against the union of random sub-ranges (partitioning the rows over workers changes no bit).  Exit code 1 on
//         any difference.
//     kq_fast_parity --bench [--mode ggml,kq256,fast] [--nt 1,2,3,4] [--threads T] [--cpu0 C] [--cpus a,b,c] [--mb 512]
//                            [--experts 300] [--reps 3]
//         weights streamed from DRAM (a pool of --mb MiB of random expert blobs, far above the L3), one expert per step in
//         random order, activations of --nt tokens; prints, per mode and group size, GB/s of weight bytes per core for the
//         gate/up rows, the down rows and the whole expert (gate/up, quantize h, down) - and the aggregate over --threads
//         threads (each pinned to its own CPU: --cpus, or consecutive from --cpu0) - as the pool's workers run it.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kMaxT = 8;

std::mt19937_64 g_rng(0x5eed);
uint32_t rnd() { return (uint32_t) g_rng(); }
float rnd_f(float lo, float hi) { return lo + (hi - lo) * (float) (rnd() >> 8) / (float) (1u << 24); }

void put_half(uint8_t* p, float v) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(v);
    std::memcpy(p, &h, 2);
}

// ---- weights: valid fp16 scales, everything else random (or extreme)
enum class Fill { Random, Max, Min };

void fill_q4k_row(uint8_t* row, int nb, Fill fill) {
    for (int b = 0; b < nb; ++b) {
        uint8_t* x = row + (size_t) b * 144;
        for (int i = 4; i < 144; ++i) x[i] = fill == Fill::Max ? 0xff : fill == Fill::Min ? 0x00 : (uint8_t) rnd();
        put_half(x, rnd_f(0.0005f, 0.06f) * (rnd() % 8 == 0 ? -1.f : 1.f));
        put_half(x + 2, rnd_f(-0.05f, 0.08f));
    }
}
void fill_q5_1_row(uint8_t* row, int nb, Fill fill) {
    for (int b = 0; b < nb; ++b) {
        uint8_t* x = row + (size_t) b * 24;
        for (int i = 4; i < 24; ++i) x[i] = fill == Fill::Max ? 0xff : fill == Fill::Min ? 0x00 : (uint8_t) rnd();
        put_half(x, rnd_f(-0.06f, 0.06f));
        put_half(x + 2, rnd_f(-0.5f, 0.5f));
    }
}
void fill_blob(const cpu::NativeFmt& f, uint8_t* blob, Fill fill) {
    const int nbg = (int) (f.n_embd / 256), nbd = (int) (f.n_ff / 32);
    for (int r = 0; r < (int) f.n_ff; ++r) {
        fill_q4k_row(blob + (size_t) r * f.gu_row, nbg, fill);
        fill_q4k_row(blob + f.up_off + (size_t) r * f.gu_row, nbg, fill);
    }
    for (int r = 0; r < (int) f.n_embd; ++r) fill_q5_1_row(blob + f.down_off + (size_t) r * f.d_row, nbd, fill);
}

// ---- activations
struct Acts {
    std::vector<uint8_t> a, h;   // nt quantized gate/up activations and nt quantized down activations
    std::vector<float> x, hf;
};
enum class ActKind { Gauss, Big, Sign, Ones, Zero };
void make_acts(const cpu::NativeFmt& f, int nt, ActKind kind, Acts& A) {
    A.a.assign((size_t) nt * cpu::kNativeActBytes, 0);
    A.h.assign((size_t) nt * cpu::kNativeHBytes, 0);
    A.x.assign((size_t) nt * f.n_embd, 0.f);
    A.hf.assign((size_t) nt * f.n_ff, 0.f);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto gen = [&](float* v, int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            switch (kind) {
                case ActKind::Gauss: v[i] = nd(g_rng) * (rnd() % 64 == 0 ? 20.f : 1.f); break;
                case ActKind::Big: v[i] = nd(g_rng) * 1000.f; break;
                case ActKind::Sign: v[i] = (i & 1) ? 1.f : -1.f; break;
                case ActKind::Ones: v[i] = 1.f; break;
                case ActKind::Zero: v[i] = 0.f; break;
            }
        }
    };
    for (int t = 0; t < nt; ++t) {
        gen(A.x.data() + (size_t) t * f.n_embd, f.n_embd);
        gen(A.hf.data() + (size_t) t * f.n_ff, f.n_ff);
        cpu::native_quant_act(f, A.x.data() + (size_t) t * f.n_embd, A.a.data() + (size_t) t * cpu::kNativeActBytes);
        cpu::native_quant_h(f, A.hf.data() + (size_t) t * f.n_ff, A.h.data() + (size_t) t * cpu::kNativeHBytes);
    }
}

// ---- the reference: ggml-cpu's vec_dot, one row and one token at a time
struct Ref {
    std::vector<std::vector<float>> ff, out;   // [t][row]
};
void reference(const cpu::NativeFmt& f, const uint8_t* blob, const Acts& A, int nt, Ref& R) {
    const ggml_type_traits_cpu* tg = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    const ggml_type_traits_cpu* td = ggml_get_type_traits_cpu((ggml_type) f.d_type);
    R.ff.assign((size_t) nt, std::vector<float>((size_t) f.n_ff));
    R.out.assign((size_t) nt, std::vector<float>((size_t) f.n_embd));
    for (int t = 0; t < nt; ++t) {
        const void* a = A.a.data() + (size_t) t * cpu::kNativeActBytes;
        const void* h = A.h.data() + (size_t) t * cpu::kNativeHBytes;
        for (int r = 0; r < (int) f.n_ff; ++r) {
            float g = 0.f, u = 0.f;
            tg->vec_dot((int) f.n_embd, &g, 0, blob + (size_t) r * f.gu_row, 0, a, 0, 1);
            tg->vec_dot((int) f.n_embd, &u, 0, blob + f.up_off + (size_t) r * f.gu_row, 0, a, 0, 1);
            R.ff[(size_t) t][(size_t) r] = (g / (1.f + std::exp(-g))) * u;
        }
        for (int r = 0; r < (int) f.n_embd; ++r) {
            float s = 0.f;
            td->vec_dot((int) f.n_ff, &s, 0, blob + f.down_off + (size_t) r * f.d_row, 0, h, 0, 1);
            R.out[(size_t) t][(size_t) r] = s;
        }
    }
}

struct Out {
    std::vector<std::vector<float>> v;
    std::vector<float*> p;
    void init(int nt, size_t n) {
        v.assign((size_t) nt, std::vector<float>(n, 12345.f));   // sentinel: untouched rows must stay as they were
        p.resize((size_t) nt);
        for (int t = 0; t < nt; ++t) p[(size_t) t] = v[(size_t) t].data();
    }
};

long g_checks = 0, g_fail = 0;
void check(const char* what, const char* geom, int nt, int r0, int r1, const Out& o, const std::vector<std::vector<float>>& ref) {
    ++g_checks;
    bool bad = false;
    for (size_t t = 0; t < (size_t) nt && !bad; ++t) {
        for (int r = r0; r < r1; ++r)
            if (std::memcmp(&o.v[t][(size_t) r], &ref[t][(size_t) r], 4) != 0) {
                std::printf("FAIL %s [%s] nt=%d rows [%d,%d): token %zu row %d: got %.9g want %.9g\n", what, geom, nt, r0, r1, t, r,
                            o.v[t][(size_t) r], ref[t][(size_t) r]);
                bad = true;
                break;
            }
        // rows outside [r0, r1) must not have been written
        for (size_t r = 0; r < o.v[t].size() && !bad; ++r)
            if ((int) r < r0 || (int) r >= r1)
                if (o.v[t][r] != 12345.f) {
                    std::printf("FAIL %s [%s] nt=%d rows [%d,%d): token %zu wrote row %zu outside the range\n", what, geom, nt, r0, r1, t, r);
                    bad = true;
                }
    }
    if (bad) ++g_fail;
}

void test_geometry(int64_t n_embd, int64_t n_ff) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(12, 7, n_embd, n_ff, f, err)) { std::printf("native_fmt(%lld,%lld): %s\n", (long long) n_embd, (long long) n_ff, err.c_str()); std::exit(2); }
    char geom[64];
    std::snprintf(geom, sizeof geom, "%lld/%lld", (long long) n_embd, (long long) n_ff);
    std::vector<uint8_t> blob(f.bytes + 64);   // slack: the kernels may prefetch (not read) past the end
    const int rows_gu = (int) n_ff, rows_d = (int) n_embd;
    struct Case { Fill fill; ActKind act; };
    const Case cases[] = {{Fill::Random, ActKind::Gauss}, {Fill::Random, ActKind::Gauss}, {Fill::Random, ActKind::Big},
                          {Fill::Max, ActKind::Sign},     {Fill::Max, ActKind::Ones},     {Fill::Min, ActKind::Sign},
                          {Fill::Random, ActKind::Zero}};
    for (const Case& c : cases) {
        fill_blob(f, blob.data(), c.fill);
        for (int nt = 1; nt <= kMaxT; ++nt) {
            Acts A;
            make_acts(f, nt, c.act, A);
            Ref R;
            reference(f, blob.data(), A, nt, R);
            const void* ap[kMaxT];
            const void* hp[kMaxT];
            for (int t = 0; t < nt; ++t) {
                ap[t] = A.a.data() + (size_t) t * cpu::kNativeActBytes;
                hp[t] = A.h.data() + (size_t) t * cpu::kNativeHBytes;
            }
            // row ranges: the whole thing, and random partial ones (odd starts / lengths, single rows)
            struct Rg { int a, b; };
            std::vector<Rg> gu_rg = {{0, rows_gu}}, d_rg = {{0, rows_d}};
            for (int k = 0; k < 6; ++k) {
                int a = (int) (rnd() % (uint32_t) rows_gu), b = a + 1 + (int) (rnd() % (uint32_t) (rows_gu - a));
                gu_rg.push_back({a, b});
                a = (int) (rnd() % (uint32_t) rows_d); b = a + 1 + (int) (rnd() % (uint32_t) (rows_d - a));
                d_rg.push_back({a, b});
            }
            gu_rg.push_back({rows_gu - 1, rows_gu});
            gu_rg.push_back({0, 1});
            d_rg.push_back({rows_d - 3, rows_d});
            for (const Rg& g : gu_rg) {
                Out o;
                o.init(nt, (size_t) rows_gu);
                cpu::kq256_gu_rows(12, blob.data(), f.gu_row, f.up_off, (int) n_embd, ap, nt, o.p.data(), g.a, g.b);
                check("kq256 gate/up", geom, nt, g.a, g.b, o, R.ff);
                o.init(nt, (size_t) rows_gu);
                cpu::kqfast_gu_rows(12, blob.data(), f.gu_row, f.up_off, (int) n_embd, ap, nt, o.p.data(), g.a, g.b);
                check("fast gate/up", geom, nt, g.a, g.b, o, R.ff);
                for (int mode = 0; mode <= 2; ++mode) {
                    cpu::native_set_kq_kernel(mode);
                    o.init(nt, (size_t) rows_gu);
                    cpu::native_gu_rows(f, blob.data(), ap, nt, o.p.data(), g.a, g.b);
                    check(mode == 0 ? "dispatch ggml gate/up" : mode == 1 ? "dispatch kq256 gate/up" : "dispatch fast gate/up", geom, nt, g.a, g.b, o, R.ff);
                }
            }
            for (const Rg& g : d_rg) {
                Out o;
                o.init(nt, (size_t) rows_d);
                cpu::kq256_rows(7, blob.data() + f.down_off, f.d_row, (int) n_ff, hp, nt, o.p.data(), g.a, g.b);
                check("kq256 down", geom, nt, g.a, g.b, o, R.out);
                o.init(nt, (size_t) rows_d);
                cpu::kqfast_rows(7, blob.data() + f.down_off, f.d_row, (int) n_ff, hp, nt, o.p.data(), g.a, g.b);
                check("fast down", geom, nt, g.a, g.b, o, R.out);
                for (int mode = 0; mode <= 2; ++mode) {
                    cpu::native_set_kq_kernel(mode);
                    o.init(nt, (size_t) rows_d);
                    cpu::native_down_rows(f, blob.data(), hp, nt, o.p.data(), g.a, g.b);
                    check(mode == 0 ? "dispatch ggml down" : mode == 1 ? "dispatch kq256 down" : "dispatch fast down", geom, nt, g.a, g.b, o, R.out);
                }
            }
            // partitioning: the union of random consecutive chunks is the whole
            {
                Out og, od;
                og.init(nt, (size_t) rows_gu);
                od.init(nt, (size_t) rows_d);
                for (int r = 0; r < rows_gu;) {
                    const int e = std::min(rows_gu, r + 1 + (int) (rnd() % 37));
                    cpu::kqfast_gu_rows(12, blob.data(), f.gu_row, f.up_off, (int) n_embd, ap, nt, og.p.data(), r, e);
                    r = e;
                }
                for (int r = 0; r < rows_d;) {
                    const int e = std::min(rows_d, r + 1 + (int) (rnd() % 37));
                    cpu::kqfast_rows(7, blob.data() + f.down_off, f.d_row, (int) n_ff, hp, nt, od.p.data(), r, e);
                    r = e;
                }
                check("fast gate/up in chunks", geom, nt, 0, rows_gu, og, R.ff);
                check("fast down in chunks", geom, nt, 0, rows_d, od, R.out);
            }
        }
    }
    cpu::native_set_kq_kernel(0);
}

// ---- bench ---------------------------------------------------------------------------------------------------------
std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}
void pin(int cpu_id) {
#if defined(__linux__)
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu_id, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
#else
    (void) cpu_id;
#endif
}

struct Bench {
    cpu::NativeFmt f;
    uint8_t* pool = nullptr;
    size_t nblobs = 0;
};

struct Times { double gu = 0, down = 0, whole = 0; };   // seconds per expert

Times run_thread(const Bench& B, int mode, int nt, int experts, int reps, uint32_t seed, int rows_gu, int rows_down) {
    const cpu::NativeFmt& f = B.f;
    std::mt19937 rg(seed);
    std::vector<float> x((size_t) nt * f.n_embd), ffb((size_t) nt * f.n_ff);
    std::vector<uint8_t> act((size_t) nt * cpu::kNativeActBytes), hq((size_t) nt * cpu::kNativeHBytes);
    std::vector<float> outb((size_t) nt * f.n_embd);
    for (float& v : x) v = (float) (rg() % 2000) / 1000.f - 1.f;
    const void* ap[kMaxT];
    const void* hp[kMaxT];
    float* ffp[kMaxT];
    float* op[kMaxT];
    for (int t = 0; t < nt; ++t) {
        cpu::native_quant_act(f, x.data() + (size_t) t * f.n_embd, act.data() + (size_t) t * cpu::kNativeActBytes);
        ap[t] = act.data() + (size_t) t * cpu::kNativeActBytes;
        hp[t] = hq.data() + (size_t) t * cpu::kNativeHBytes;
        ffp[t] = ffb.data() + (size_t) t * f.n_ff;
        op[t] = outb.data() + (size_t) t * f.n_embd;
    }
    for (int t = 0; t < nt; ++t) cpu::native_quant_h(f, ffp[t], hq.data() + (size_t) t * cpu::kNativeHBytes);   // zeros: fine for timing
    std::vector<size_t> order((size_t) experts);
    Times best{1e9, 1e9, 1e9};
    cpu::native_set_kq_kernel(mode);
    for (int rep = 0; rep < reps + 1; ++rep) {   // the first rep warms the code; the best of the others is reported
        for (size_t& o : order) o = rg() % B.nblobs;
        Times tm;
        for (int part = 0; part < 3; ++part) {
            const auto t0 = Clock::now();
            for (size_t e : order) {
                const uint8_t* blob = B.pool + e * f.bytes;
                if (part == 0) cpu::native_gu_rows(f, blob, ap, nt, ffp, 0, rows_gu);
                else if (part == 1) cpu::native_down_rows(f, blob, hp, nt, op, 0, rows_down);
                else {
                    cpu::native_gu_rows(f, blob, ap, nt, ffp, 0, rows_gu);
                    for (int t = 0; t < nt; ++t) cpu::native_quant_h(f, ffp[t], hq.data() + (size_t) t * cpu::kNativeHBytes);
                    cpu::native_down_rows(f, blob, hp, nt, op, 0, rows_down);
                }
            }
            const double s = std::chrono::duration<double>(Clock::now() - t0).count() / experts;
            (part == 0 ? tm.gu : part == 1 ? tm.down : tm.whole) = s;
        }
        if (rep > 0) {
            best.gu = std::min(best.gu, tm.gu);
            best.down = std::min(best.down, tm.down);
            best.whole = std::min(best.whole, tm.whole);
        }
    }
    return best;
}

int bench_main(int argc, char** argv) {
    std::vector<int> modes = {0, 1, 2}, nts = {1, 2, 3, 4}, cpus;
    int threads = 1, cpu0 = 2, mb = 512, experts = 300, reps = 3, rows = 0;   // rows > 0: only the first `rows` gate/up rows (and 4x as many down rows) of each expert: cache-resident
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--mode") {
            modes.clear();
            std::string cur;
            for (char ch : std::string(next()) + ",") {
                if (ch != ',') { cur += ch; continue; }
                modes.push_back(cur == "ggml" || cur == "0" ? 0 : cur == "kq256" || cur == "1" ? 1 : 2);
                cur.clear();
            }
        } else if (a == "--nt") nts = parse_list(next());
        else if (a == "--threads") threads = std::atoi(next());
        else if (a == "--cpu0") cpu0 = std::atoi(next());
        else if (a == "--cpus") cpus = parse_list(next());
        else if (a == "--mb") mb = std::atoi(next());
        else if (a == "--experts") experts = std::atoi(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a == "--rows") rows = std::atoi(next());
    }
    if (cpus.empty()) for (int t = 0; t < threads; ++t) cpus.push_back(cpu0 + t);
    threads = (int) cpus.size();
    Bench B;
    std::string err;
    if (!cpu::native_fmt(12, 7, 2560, 640, B.f, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 2; }
    B.nblobs = ((size_t) mb << 20) / B.f.bytes;
    B.pool = (uint8_t*) std::aligned_alloc(4096, B.nblobs * B.f.bytes);
    for (size_t i = 0; i < B.nblobs; ++i) fill_blob(B.f, B.pool + i * B.f.bytes, Fill::Random);
    std::printf("expert blob %zu B (gate/up %zu B/row x 2 x %lld, down %zu B/row x %lld); pool %zu experts = %.0f MiB; %d thread(s) on cpus",
                B.f.bytes, B.f.gu_row, (long long) B.f.n_ff, B.f.d_row, (long long) B.f.n_embd, B.nblobs, (double) B.nblobs * B.f.bytes / 1048576.0, threads);
    for (int c : cpus) std::printf(" %d", c);
    std::printf("\n");
    const int rows_gu = rows > 0 ? std::min(rows, (int) B.f.n_ff) : (int) B.f.n_ff, rows_dn = rows > 0 ? std::min(4 * rows, (int) B.f.n_embd) : (int) B.f.n_embd;
    const double gu_b = 2.0 * (double) rows_gu * (double) B.f.gu_row, d_b = (double) rows_dn * (double) B.f.d_row, all_b = gu_b + d_b;
    static const char* names[] = {"ggml  ", "kq256 ", "fast  "};
    for (int nt : nts) {
        for (int mode : modes) {
            std::vector<Times> res((size_t) threads);
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> th;
            for (int k = 0; k < threads; ++k)
                th.emplace_back([&, k] {
                    pin(cpus[(size_t) k]);
                    ready.fetch_add(1);
                    while (!go.load()) {}
                    res[(size_t) k] = run_thread(B, mode, nt, experts, reps, 1000u + (uint32_t) k, rows_gu, rows_dn);
                });
            while (ready.load() < threads) {}
            go.store(true);
            for (auto& t : th) t.join();
            double gu = 0, dn = 0, wh = 0;
            for (const Times& t : res) { gu += t.gu; dn += t.down; wh += t.whole; }
            gu /= threads; dn /= threads; wh /= threads;
            std::printf("nt=%d %s gate/up %6.2f GB/s/core (%.3f ms)  down %6.2f GB/s/core (%.3f ms)  whole expert %6.2f GB/s/core (%.3f ms)  | %d thread(s): %.1f GB/s aggregate\n",
                        nt, names[mode], gu_b / gu / 1e9, gu * 1e3, d_b / dn / 1e9, dn * 1e3, all_b / wh / 1e9, wh * 1e3, threads,
                        all_b / wh / 1e9 * threads);
            std::fflush(stdout);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--bench") return bench_main(argc, argv);
    if (!cpu::cpu_avx2_ok()) { std::printf("kq_fast_parity: this CPU has no AVX2, nothing to test (skipped)\n"); return 0; }
    // the group sizes 1..8 and several widths (n_embd must be a multiple of 256, n_ff of 32)
    test_geometry(2560, 640);
    test_geometry(768, 96);
    test_geometry(256, 32);
    test_geometry(1536, 224);
    std::printf("kq_fast_parity: %ld checks, %ld differences\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
