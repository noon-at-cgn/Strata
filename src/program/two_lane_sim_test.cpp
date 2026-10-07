#include "strata/program/two_lane_sim.hpp"

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

// a model with no noise: every number deterministic
static TwoLaneParams quiet(TwoLaneParams p) {
    p.gpu_jitter = 0.0;
    p.expert_sigma = 0.0;
    return p;
}

// explicit uniform per-layer costs through the trace path: stage s has n layers of g ms and c ms of pool each
static TwoLaneParams traced(int n0, double g0, double c0, int n1, double g1, double c1) {
    TwoLaneParams p;
    p.layers[0] = n0; p.layers[1] = n1;
    LaneWindowTrace w;
    w.gpu[0].assign((size_t) n0, g0); w.cpu[0].assign((size_t) n0, c0);
    w.gpu[1].assign((size_t) n1, g1); w.cpu[1].assign((size_t) n1, c1);
    p.trace.push_back(w);
    p.verdict_ms = 0; p.hop_ms = 0; p.fixed_host_ms = 0; p.draft_ms = 0; p.fixed_lat_ms = 0;
    p.rows = p.rows_ref = 2.0;
    p.tokens_override = 1.0;
    p.windows = 400;
    return p;
}

int main() {
    long checks = 0;

    // ---- 1. tokens per window: 1 + the chain of accepted proposals ----
    {
        TwoLaneParams p;
        p.accept = 0.8;
        p.rows = 1.0; CHECK(std::fabs(p.tokens_per_window() - 1.0) < 1e-12, "rows 1");
        p.rows = 2.0; CHECK(std::fabs(p.tokens_per_window() - 1.8) < 1e-12, "rows 2: %f", p.tokens_per_window());
        p.rows = 3.0; CHECK(std::fabs(p.tokens_per_window() - (1 + 0.8 + 0.64)) < 1e-12, "rows 3");
        p.rows = 1.5; CHECK(std::fabs(p.tokens_per_window() - 1.4) < 1e-12, "rows 1.5: %f", p.tokens_per_window());
        p.tokens_override = 3.38; CHECK(p.tokens_per_window() == 3.38, "override");
        checks += 5;
    }

    // ---- 2. one lane alone is the sum of its parts (no noise): stages, pool, the loop; and equals stage_sim's serial window ----
    {
        TwoLaneParams p = traced(24, 0.4, 0.2, 24, 0.35, 0.15);
        p.lanes = 1;
        const TwoLaneResult r = two_lane_run(p);
        CHECK(r.error.empty(), "%s", r.error.c_str());
        const double expect = 24 * (0.4 + 0.2) + 24 * (0.35 + 0.15);
        CHECK(std::fabs(r.lane_ms_per_window[0] - expect) < 1e-6, "traced one lane %.6f vs %.6f", r.lane_ms_per_window[0], expect);
        // the same layers through stage_sim's serial policy: S0 + S1 + verdict + fixed
        StageSimParams q = stage_sim_uniform(24, 24 * 0.4, 24 * 0.2, 24, 24 * 0.35, 24 * 0.15);
        const StageSimResult s = stage_sim_run(q, StagePolicy::Serial, 0, 0, 400);
        CHECK(std::fabs(s.ms_per_window - r.lane_ms_per_window[0]) < 1e-6, "stage_sim serial %.6f vs two lane solo %.6f", s.ms_per_window, r.lane_ms_per_window[0]);
        // the loop adds its parts
        p.verdict_ms = 0.1; p.hop_ms = 0.2; p.fixed_host_ms = 0.9; p.draft_ms = 1.1; p.fixed_lat_ms = 2.2;
        const TwoLaneResult r2 = two_lane_run(p);
        CHECK(std::fabs(r2.lane_ms_per_window[0] - (expect + 0.1 + 0.2 + 0.9 + 1.1 + 2.2)) < 1e-6, "loop %.6f", r2.lane_ms_per_window[0]);
        // the model's own mean for the generated (non-trace) windows
        TwoLaneParams g = quiet(two_lane_lane_preset(true));
        g.lanes = 1; g.windows = 300;
        const TwoLaneResult rg = two_lane_run(g);
        CHECK(std::fabs(rg.lane_ms_per_window[0] / two_lane_solo_mean(g) - 1.0) < 0.005, "generated %.6f vs mean %.6f (the Poisson draws stay)", rg.lane_ms_per_window[0], two_lane_solo_mean(g));
        checks += 5;
    }

    // ---- 3. the presets reproduce the measured windows (the numbers of the production log) within 2% ----
    {
        struct { TwoLaneParams p; double ms; const char* n; } c[] = {
            {two_lane_merged_preset(true), 43.8, "merged warm"},
            {two_lane_merged_preset(false), 57.4, "merged typical"},
            {[] { auto p = two_lane_lane_preset(true); p.lanes = 1; return p; }(), 33.5, "lane warm"},
            {[] { auto p = two_lane_lane_preset(false); p.lanes = 1; return p; }(), 38.5, "lane typical"},
            {two_lane_serve_solo_preset(), 28.0, "serve solo"},
        };
        for (auto& x : c) {
            x.p.windows = 2000;
            const TwoLaneResult r = two_lane_run(x.p);
            CHECK(r.error.empty(), "%s", r.error.c_str());
            CHECK(std::fabs(r.lane_ms_per_window[0] / x.ms - 1.0) < 0.02, "%s: model %.2f vs measured %.2f", x.n, r.lane_ms_per_window[0], x.ms);
            ++checks;
        }
    }

    // ---- 4. the invariants hold in every configuration ----
    {
        const StagePolicy pols[] = {StagePolicy::Fifo, StagePolicy::Pipelined, StagePolicy::Priority, StagePolicy::Quantum};
        int n = 0;
        for (int lanes = 1; lanes <= 3; ++lanes)
            for (StagePolicy pol : pols)
                for (int variant = 0; variant < 8; ++variant) {
                    TwoLaneParams p = two_lane_lane_preset(variant & 1);
                    p.lanes = lanes; p.policy = pol; p.windows = 120; p.record = true; p.seed = 11 + (uint64_t) variant;
                    p.draft_exclusive = !(variant & 2);
                    p.draft_blocks_host = !(variant & 4);
                    if (variant == 3) { p.host_fifo = true; p.overlap_stretch = 1.2; }
                    if (variant == 5) { p.rows = 3.0; p.hop_ms = 0.3; }
                    if (variant == 6) { p.draft_ms = 0; p.fixed_host_ms = 0; p.fixed_lat_ms = 0; p.verdict_ms = 0; }
                    if (variant == 7) { p.tokens_override = 2.0; p.gpu_scale = 0.7; p.pool_scale = 1.5; p.gpu_delta_ms = -3.5; }
                    const TwoLaneResult r = two_lane_run(p);
                    const std::string e = two_lane_check(p, r);
                    CHECK(e.empty(), "lanes %d policy %s variant %d: %s", lanes, stage_policy_name(pol), variant, e.c_str());
                    ++n;
                }
        // a trace replay too
        TwoLaneParams p = traced(24, 0.4, 0.2, 24, 0.35, 0.15);
        p.lanes = 2; p.record = true; p.windows = 100; p.draft_ms = 1.0; p.fixed_host_ms = 0.5; p.fixed_lat_ms = 1.0;
        const TwoLaneResult r = two_lane_run(p);
        CHECK(two_lane_check(p, r).empty(), "trace: %s", two_lane_check(p, r).c_str());
        checks += n + 1;
    }

    // ---- 5. the two-lane facts ----
    {
        // (a) with two lanes the pool order cannot matter: at most one lane is ever waiting while the other is served
        TwoLaneParams p = two_lane_lane_preset(true);
        p.windows = 800;
        double ref = -1, refw = -1;
        for (StagePolicy pol : {StagePolicy::Fifo, StagePolicy::Pipelined, StagePolicy::Priority, StagePolicy::Quantum}) {
            p.policy = pol;
            const TwoLaneResult r = two_lane_run(p);
            if (ref < 0) { ref = r.agg_tok_s; refw = r.pool_wait_per_window_ms; }
            CHECK(std::fabs(r.agg_tok_s - ref) < 1e-9 && std::fabs(r.pool_wait_per_window_ms - refw) < 1e-9, "policy %s differs with 2 lanes", stage_policy_name(pol));
            ++checks;
        }
        // (b) never above the ceiling 2 x one lane, and never below one lane alone
        TwoLaneParams one = two_lane_lane_preset(true); one.lanes = 1; one.windows = 800;
        const TwoLaneResult r1 = two_lane_run(one), r2 = two_lane_run(p);
        CHECK(r2.agg_tok_s <= 2 * r1.agg_tok_s * 1.005, "above the ceiling: %.2f vs %.2f", r2.agg_tok_s, 2 * r1.agg_tok_s);
        CHECK(r2.agg_tok_s >= r1.agg_tok_s, "two lanes slower than one: %.2f vs %.2f", r2.agg_tok_s, r1.agg_tok_s);
        checks += 2;
        // (c) a lane loop much longer than the stages: the lanes never meet, 2x exactly (within noise)
        TwoLaneParams lo = traced(24, 0.2, 0.0, 24, 0.2, 0.0);
        lo.fixed_lat_ms = 40.0; lo.lanes = 2;
        TwoLaneParams lo1 = lo; lo1.lanes = 1;
        const TwoLaneResult a2 = two_lane_run(lo), a1 = two_lane_run(lo1);
        CHECK(std::fabs(a2.agg_tok_s / (2 * a1.agg_tok_s) - 1.0) < 0.01, "independent lanes %.3f vs %.3f", a2.agg_tok_s, 2 * a1.agg_tok_s);
        ++checks;
        // (d) stage-bound: no pool, no loop, stage windows of 10 ms each: the cards alternate perfectly, 2 windows per 20 ms
        TwoLaneParams sb = traced(10, 1.0, 0.0, 10, 1.0, 0.0);
        sb.lanes = 2;
        const TwoLaneResult b2 = two_lane_run(sb);
        CHECK(std::fabs(b2.lane_ms_per_window[0] - 20.0) < 0.1 && std::fabs(b2.agg_tok_s - 2 * 1000.0 / 20.0) < 0.5, "stage-bound %.4f ms, %.3f tok/s", b2.lane_ms_per_window[0], b2.agg_tok_s);
        CHECK(b2.stage_util[0] > 0.999 && b2.stage_util[1] > 0.999, "cards not full: %.3f %.3f", b2.stage_util[0], b2.stage_util[1]);
        checks += 3;
        // (e) pool-bound: almost no GPU time, 4 ms of pool a layer: windows follow the pool, the pool is 100% busy
        TwoLaneParams pb = traced(10, 0.001, 0.4, 10, 0.001, 0.4);
        pb.lanes = 2;
        const TwoLaneResult c2 = two_lane_run(pb);
        const double per_window = 20 * 0.4;   // all of a window's pool time
        // (all of a window's pool time is `per_window`; the checks below are the pool's busy share and the rate it allows)
        CHECK(c2.pool_util > 0.99, "pool-bound: pool %.3f", c2.pool_util);
        // two windows' pool work per (2 x per_window) ms: aggregate = 1000 / per_window tokens/s at 1 token a window
        CHECK(std::fabs(c2.agg_tok_s - 1000.0 / per_window) < 0.01 * 1000.0 / per_window, "pool-bound rate %.3f vs %.3f", c2.agg_tok_s, 1000.0 / per_window);
        checks += 3;
        // (f) a single lane's pool is never contended: no waits at all
        CHECK(r1.pool_wait_per_window_ms == 0.0 && r1.stage_wait_per_window_ms == 0.0, "a lone lane waited");
        ++checks;
    }

    // ---- 6. what-ifs move the result the right way ----
    {
        TwoLaneParams p = two_lane_lane_preset(true);
        p.windows = 1500;
        const double base = two_lane_run(p).agg_tok_s;
        TwoLaneParams q = p; q.pool_scale = 0.7;
        CHECK(two_lane_run(q).agg_tok_s > base, "a faster pool kernel did not help");
        q = p; q.gpu_delta_ms = -3.5;
        CHECK(two_lane_run(q).agg_tok_s > base, "dense fusion did not help");
        q = p; q.overlap_stretch = 1.25;
        CHECK(two_lane_run(q).agg_tok_s < base, "a pool stretch did not hurt");
        q = p;
        q.experts[0] *= 2; q.experts[1] *= 2;
        CHECK(two_lane_run(q).agg_tok_s < base, "more CPU experts did not hurt");
        q = p; q.draft_exclusive = false; q.draft_blocks_host = false;
        CHECK(two_lane_run(q).agg_tok_s >= base * 0.999, "an async draft hurt");
        q = p; q.draft_exclusive = true; q.draft_blocks_host = true; q.fixed_host_ms = 4.2 - 1.1; q.fixed_lat_ms = 0.0;
        CHECK(two_lane_run(q).agg_tok_s < base, "a host-blocking loop did not hurt");
        // the same seed, the same run; another seed, a nearby one
        q = p;
        CHECK(two_lane_run(q).agg_tok_s == base, "not deterministic");
        q.seed = 99;
        const double other = two_lane_run(q).agg_tok_s;
        CHECK(other != base && std::fabs(other / base - 1.0) < 0.03, "seed changes: %.3f vs %.3f", other, base);
        checks += 8;
    }

    // ---- 7. the verdict the model gives from the production numbers (a regression guard on the presets): two lanes beat the
    //         merged batch by 1.1-1.45x, not 1.5x, in both tiers; one lane through the batch path is slower than the serve path ----
    {
        for (int warm = 0; warm < 2; ++warm) {
            TwoLaneParams lane = two_lane_lane_preset(warm == 1);
            lane.windows = 2000;
            TwoLaneParams one = lane; one.lanes = 1;
            TwoLaneParams m = two_lane_merged_preset(warm == 1); m.windows = 2000;
            const double two = two_lane_run(lane).agg_tok_s, merged = two_lane_run(m).agg_tok_s, solo = two_lane_run(one).agg_tok_s;
            TwoLaneParams sv = two_lane_serve_solo_preset(); sv.windows = 2000;
            const double serve = two_lane_run(sv).agg_tok_s;
            CHECK(two / merged > 1.1 && two / merged < 1.45, "warm=%d: two lanes %.1f vs merged %.1f = %.2fx", warm, two, merged, two / merged);
            CHECK(2 * solo / merged < 1.6, "ceiling %.2fx", 2 * solo / merged);
            CHECK(solo < serve * 0.85, "a lone lane %.1f vs the serve path %.1f", solo, serve);
            checks += 3;
        }
    }

    // ---- 8. bad input is refused with an error, not a crash ----
    {
        TwoLaneParams p; p.windows = 5;
        CHECK(!two_lane_run(p).error.empty(), "windows < 20 accepted");
        p.windows = 100; p.layers[0] = 0;
        CHECK(!two_lane_run(p).error.empty(), "no layers accepted");
        checks += 2;
    }

    std::printf("two_lane_sim_test: %ld checks passed\n", checks);
    return 0;
}
