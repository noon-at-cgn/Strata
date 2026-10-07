#include "strata/program/stage_sim.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::program;

#define CHECK(c, ...)                                                           \
    do {                                                                        \
        if (!(c)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s  (", __FILE__, __LINE__, #c); \
            std::fprintf(stderr, __VA_ARGS__);                                  \
            std::fprintf(stderr, ")\n");                                        \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static const StagePolicy kPolicies[] = {StagePolicy::Serial, StagePolicy::Pipelined, StagePolicy::Priority,
                                        StagePolicy::Quantum, StagePolicy::Fifo};

// per-layer costs that differ layer to layer (a deterministic mix), stage totals as given
static StageSimParams ragged(int n0, double g0, double c0, int n1, double g1, double c1, unsigned seed) {
    StageSimParams p = stage_sim_uniform(n0, g0, c0, n1, g1, c1);
    unsigned s = seed;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return 0.2 + 1.6 * ((s >> 8) & 0xffff) / 65535.0; };   // 0.2 .. 1.8
    for (int st = 0; st < 2; ++st)
        for (double& v : p.gpu[st]) v *= rnd();
    for (int st = 0; st < 2; ++st)
        for (double& v : p.cpu[st]) v *= rnd();
    return p;
}

int main() {
    long checks = 0;
    const StageSimParams brief = stage_sim_brief(), rec = stage_sim_recording();

    // ---- 1. the serial policy reproduces the solo numbers: 40.6 ms (the brief) and 43.7 ms (the recording) ----
    {
        const StageSimResult a = stage_sim_run(brief, StagePolicy::Serial, 0.5, 0.2, 500);   // h and d are ignored by serial
        CHECK(std::fabs(a.ms_per_window - 40.6) < 1e-6, "brief serial %.6f", a.ms_per_window);
        CHECK(a.onpath == 0 && a.doomed == 0 && a.fresh == 500, "serial speculated");
        CHECK(stage_sim_check(brief, a).empty(), "%s", stage_sim_check(brief, a).c_str());
        const StageSimResult b = stage_sim_run(rec, StagePolicy::Serial, 0.0, 0.0, 500);
        CHECK(std::fabs(b.ms_per_window - 43.7) < 1e-6, "recording serial %.6f", b.ms_per_window);
        // the ceilings the report derived: S0 22.9, S1 17.7, P 20.1 -> floor 22.9 (1.77x), pool alone 2.02x
        CHECK(std::fabs(stage_sim_floor(brief) - 22.9) < 1e-9, "floor");
        CHECK(std::fabs(brief.pool_total() - 20.1) < 1e-9, "pool total");
        CHECK(std::fabs(a.ms_per_window / stage_sim_floor(brief) - 1.773) < 0.001, "ceiling 1.77x");
        checks += 6;
    }

    // ---- 2. the invariants hold: every preset, a ragged one, every policy, h x d, jitter, stretch, host times ----
    {
        std::vector<StageSimParams> ps = {brief, rec, ragged(24, 10.6, 12.3, 24, 9.9, 7.8, 7), ragged(30, 8.0, 15.0, 18, 11.0, 6.0, 99),
                                          ragged(5, 1.0, 3.0, 9, 2.0, 1.0, 5)};
        for (size_t i = 0; i < ps.size(); ++i)
            for (int variant = 0; variant < 4; ++variant) {
                StageSimParams p = ps[i];
                if (variant & 1) { p.jitter = 0.3; p.overlap_stretch = 1.15; }
                if (variant & 2) { p.fixed_ms = 0.7; p.verdict_ms = 0.2; p.undo_ms = 1.5; p.hop_ms = 0.3; p.skip_ms = 0.01; p.jitter = 0.6; p.seed = 42 + i; }
                for (StagePolicy pol : kPolicies)
                    for (double h : {0.0, 0.3, 0.5, 0.7, 1.0})
                        for (double d : {0.0, 0.15, 0.4}) {
                            if (h + d > 1.0) continue;
                            const StageSimResult r = stage_sim_run(p, pol, h, d, 150);
                            const std::string e = stage_sim_check(p, r);
                            CHECK(e.empty(), "preset %zu variant %d %s h %.2f d %.2f: %s", i, variant, stage_policy_name(pol), h, d, e.c_str());
                            CHECK(r.ms_per_window > 0, "no time");
                            if (pol != StagePolicy::Serial) {
                                CHECK(r.onpath + r.fresh == 150, "window classes");
                                CHECK(std::abs(r.onpath - (int) std::lround(h * 150)) <= 3, "on-path share %d of 150 for h %.2f", r.onpath, h);
                            }
                            ++checks;
                        }
            }
    }

    // ---- 3. the checker catches broken schedules (it is not vacuous) ----
    {
        StageSimResult r = stage_sim_run(brief, StagePolicy::Pipelined, 0.5, 0.15, 100);
        CHECK(stage_sim_check(brief, r).empty(), "clean run");
        {   // two pool layers at once
            StageSimResult x = r;
            for (SimLayer& L : x.layers) if (L.stage == 1 && L.pool_end > L.pool_start) { L.pool_start = x.layers[0].pool_start; L.pool_end = x.layers[0].pool_end; break; }
            CHECK(!stage_sim_check(brief, x).empty(), "pool overlap not caught");
        }
        {   // a layer starting before the previous layer's pool result
            StageSimResult x = r;
            SimLayer* prev = nullptr;
            for (SimLayer& L : x.layers) { if (prev && prev->win == L.win && prev->stage == L.stage && L.layer == 5) { L.gpu_start = prev->pool_end - 0.2; break; } if (L.layer == 4) prev = &L; }
            CHECK(!stage_sim_check(brief, x).empty(), "early layer not caught");
        }
        {   // stage 1 running before the verdict of the window before it
            StageSimResult x = r;
            for (SimWin& w : x.wins) if (!w.doomed && w.k == 10) w.s1_start = x.wins[0].s0_start;
            CHECK(!stage_sim_check(brief, x).empty(), "stage 1 before the verdict not caught");
        }
        {   // below the floor
            StageSimResult x = r;
            x.ms_per_window = 1.0;
            CHECK(!stage_sim_check(brief, x).empty(), "floor not caught");
        }
        {   // stage 1 running a doomed window
            StageSimResult x = r;
            for (SimWin& w : x.wins) if (w.doomed) { w.s1_start = 1.0; break; }
            CHECK(!stage_sim_check(brief, x).empty(), "stage 1 on a doomed window not caught");
        }
        checks += 6;
    }

    // ---- 4. what the model says about speculation (jitter 0: exact numbers; checks use margins) ----
    {
        // h = 1, d = 0: never below the floor, within 1.4x of it (stage 0 is the long stage)
        const StageSimResult top = stage_sim_run(brief, StagePolicy::Pipelined, 1.0, 0.0, 1000);
        CHECK(top.ms_per_window >= stage_sim_floor(brief) - 1e-9 && top.ms_per_window < 1.05 * stage_sim_floor(brief),
              "h=1: %.3f vs floor %.3f", top.ms_per_window, stage_sim_floor(brief));
        // the gain falls with h: monotone, for all policies and both presets, with and without noise
        for (const StageSimParams* base : {&brief, &rec})
            for (double jit : {0.0, 0.3}) {
                StageSimParams p = *base;
                p.jitter = jit;
                for (StagePolicy pol : {StagePolicy::Pipelined, StagePolicy::Priority, StagePolicy::Quantum, StagePolicy::Fifo}) {
                    double last = stage_sim_run(p, pol, 0.0, 0.0, 600).ms_per_window;
                    for (double h : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}) {
                        const double ms = stage_sim_run(p, pol, h, 0.0, 600).ms_per_window;
                        CHECK(ms <= last + 0.05, "not monotone: h %.1f %.3f after %.3f", h, ms, last);
                        last = ms;
                        ++checks;
                    }
                }
            }
        // a wrong guess costs about undo + a little (the doomed window is killed at the verdict), never more than the window it holds up
        for (double d : {0.1, 0.3, 0.6}) {
            const double ser = stage_sim_run(brief, StagePolicy::Serial, 0, 0, 800).ms_per_window;
            const double ms = stage_sim_run(brief, StagePolicy::Pipelined, 0.0, d, 800).ms_per_window;
            const double per = (ms - ser) / d;
            CHECK(per > brief.undo_ms - 1e-6 && per < 8.0, "a doomed window costs %.2f ms", per);
            ++checks;
        }
        // with the report's estimate h = 0.4, d = 0.15 the gain on the recording is +10..+25% (its central +16%)
        const double s = stage_sim_run(rec, StagePolicy::Serial, 0, 0, 1000).ms_per_window;
        const double m = stage_sim_run(rec, StagePolicy::Pipelined, 0.4, 0.15, 1000).ms_per_window;
        CHECK(s / m - 1.0 > 0.10 && s / m - 1.0 < 0.25, "gain %.3f at h 0.4 d 0.15", s / m - 1.0);
        ++checks;
        // pool-bound: when the pool is the long resource the speed cannot pass P, whatever the stages do
        StageSimParams pb = stage_sim_uniform(24, 4.0, 30.0, 24, 4.0, 30.0);
        const double pbm = stage_sim_run(pb, StagePolicy::Pipelined, 1.0, 0.0, 600).ms_per_window;
        CHECK(pbm >= 60.0 - 1e-6 && pbm < 62.0, "pool-bound window %.3f (P = 60)", pbm);
        ++checks;
    }

    // ---- 5. the pool's service order: with one request a stage at a time and a GPU phase before each, there is never a
    //         choice to make (the layer a stage just had served is on its GPU), so every policy gives the same schedule;
    //         and when the pool is the bottleneck a work-conserving order cannot change the throughput either ----
    {
        for (const StageSimParams* base : {&brief, &rec})
            for (double jit : {0.0, 0.5})
                for (double stretch : {1.0, 1.2}) {
                    StageSimParams p = *base;
                    p.jitter = jit;
                    p.overlap_stretch = stretch;
                    const double ref = stage_sim_run(p, StagePolicy::Pipelined, 0.5, 0.2, 400).ms_per_window;
                    for (StagePolicy pol : {StagePolicy::Priority, StagePolicy::Quantum, StagePolicy::Fifo}) {
                        const double ms = stage_sim_run(p, pol, 0.5, 0.2, 400).ms_per_window;
                        CHECK(std::fabs(ms - ref) < 1e-9, "%s differs: %.9f vs %.9f", stage_policy_name(pol), ms, ref);
                        ++checks;
                    }
                }
        StageSimParams nogpu = brief;   // no GPU time at all: both stages always ask for the pool, so the order is a real choice
        nogpu.gpu[0].assign(24, 0.0);
        nogpu.gpu[1].assign(24, 0.0);
        double lo = 1e30, hi = 0;
        for (StagePolicy pol : {StagePolicy::Pipelined, StagePolicy::Priority, StagePolicy::Quantum, StagePolicy::Fifo}) {
            const StageSimResult r = stage_sim_run(nogpu, pol, 1.0, 0.0, 400);
            CHECK(stage_sim_check(nogpu, r).empty(), "nogpu %s: %s", stage_policy_name(pol), stage_sim_check(nogpu, r).c_str());
            lo = std::min(lo, r.ms_per_window);
            hi = std::max(hi, r.ms_per_window);
            ++checks;
        }
        CHECK(hi - lo < 1e-6 && std::fabs(hi - brief.pool_total()) < 1e-6, "pool-bound policies %.4f..%.4f (P = %.2f)", lo, hi, brief.pool_total());
    }

    // ---- 6. determinism and the knob effects ----
    {
        StageSimParams p = rec;
        p.jitter = 0.4;
        const StageSimResult a = stage_sim_run(p, StagePolicy::Priority, 0.4, 0.15, 300), b = stage_sim_run(p, StagePolicy::Priority, 0.4, 0.15, 300);
        CHECK(a.ms_per_window == b.ms_per_window && a.layers.size() == b.layers.size(), "not deterministic");
        StageSimParams q = rec;   // a slower pool while both stages run can only slow it down
        q.overlap_stretch = 1.3;
        CHECK(stage_sim_run(q, StagePolicy::Pipelined, 0.5, 0.1, 300).ms_per_window >
              stage_sim_run(rec, StagePolicy::Pipelined, 0.5, 0.1, 300).ms_per_window, "stretch has no effect");
        // moving two layers from stage 0 to stage 1 (the report's D2: 22 or 23 layers on stage 0 is "layer_split") evens the stages
        StageSimParams mv = stage_sim_uniform(22, 10.42 * 22 / 24, 14.07 * 22 / 24, 26, 9.35 * 26 / 24, 9.38 * 26 / 24);
        mv.fixed_ms = rec.fixed_ms;
        const double a0 = stage_sim_run(rec, StagePolicy::Pipelined, 0.5, 0.15, 800).ms_per_window;
        const double a1 = stage_sim_run(mv, StagePolicy::Pipelined, 0.5, 0.15, 800).ms_per_window;
        CHECK(a1 < a0 + 0.5, "moving layers to the lighter stage: %.3f -> %.3f", a0, a1);
        checks += 3;
    }

    // ---- the table the coordinator compares with the live measurement ----
    std::printf("== recording 20261007-021355 (43.7 ms/window, 2.34 tokens/window), d = 0.15, jitter +-30%%\n");
    StageSimParams show = rec;
    show.jitter = 0.3;
    stage_sim_print_table(stdout, show, 0.15, 2.34, 2000);
    std::printf("== the brief's numbers (40.6 ms/window)\n");
    show = brief;
    show.jitter = 0.3;
    stage_sim_print_table(stdout, show, 0.15, 2.34, 2000);
    std::printf("stage_sim_test: %ld checks passed\n", checks);
    return 0;
}
