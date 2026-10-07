// two_lane_sim: what two independent MTP lanes (--batch-groups 2 beside --batch-mtp) could give two concurrent users on a
// layer split, against today's merged-row 2-slot batch and time-sliced solo.  Trace-driven from the serve log's numbers
// (docs/TWO_LANE_MODEL.md has each one's source).
//   two_lane_sim --report [--windows N]        the whole study: the four questions and the go / no-go line
//   two_lane_sim [--tier warm|typical] [--lanes N] [--policy fifo|stage0|priority|quantum] [--rows R] [--accept p]
//                [--experts-scale f] [--gpu-scale f] [--gpu-delta ms] [--pool-scale f] [--stretch f]
//                [--draft exclusive|free|async] [--lat ms] [--host ms] [--windows N] [--seed S] [--trace file.csv]
// With --trace the per-layer costs come from a CSV (one line per layer: window,stage,layer,gpu_ms,cpu_ms), replayed per lane.
#include "strata/program/two_lane_sim.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

using namespace strata::program;

namespace {

constexpr double kGoRatio = 1.5;   // the go bar: two lanes' aggregate >= 1.5 x today's merged-row batch of the same tier

void scale_experts(TwoLaneParams& p, double f) { p.experts[0] *= f; p.experts[1] *= f; }

TwoLaneResult run(TwoLaneParams p, int windows) { p.windows = windows; return two_lane_run(p); }

// the merged-row batch under the same what-ifs as `lane` (gpu_scale, gpu_delta, pool_scale and the expert scale)
TwoLaneParams merged_like(const TwoLaneParams& lane, bool warm, double expert_scale) {
    TwoLaneParams m = two_lane_merged_preset(warm);
    m.gpu_scale = lane.gpu_scale; m.gpu_delta_ms = lane.gpu_delta_ms; m.pool_scale = lane.pool_scale;
    scale_experts(m, expert_scale);
    m.seed = lane.seed;
    return m;
}

bool load_trace(const char* path, std::vector<LaneWindowTrace>& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::map<int, LaneWindowTrace> wins;
    std::string ln;
    while (std::getline(f, ln)) {
        if (ln.empty() || ln[0] == '#') continue;
        std::stringstream ss(ln);
        std::string a, b, c, d, e;
        if (!std::getline(ss, a, ',') || !std::getline(ss, b, ',') || !std::getline(ss, c, ',') || !std::getline(ss, d, ',') || !std::getline(ss, e, ',')) continue;
        if (a == "window") continue;
        const int w = std::atoi(a.c_str()), s = std::atoi(b.c_str());
        if (s < 0 || s > 1) continue;
        wins[w].gpu[s].push_back(std::atof(d.c_str()));
        wins[w].cpu[s].push_back(std::atof(e.c_str()));
    }
    for (auto& kv : wins) out.push_back(kv.second);
    return !out.empty();
}

void row(const char* name, const TwoLaneResult& r, double ref_tok_s, double ref2_tok_s = 0) {
    std::printf("  %-52s %6.1f ms/window  %5.2f tok/window  %6.1f tok/s", name, r.lane_ms_per_window[0], r.tokens_per_window, r.agg_tok_s);
    if (ref_tok_s > 0) std::printf("  %5.2fx", r.agg_tok_s / ref_tok_s);
    if (ref2_tok_s > 0) std::printf("   %5.2fx of the lean merged batch (%.1f tok/s)", r.agg_tok_s / ref2_tok_s, ref2_tok_s);
    std::printf("\n");
}

// a lane window as lean as the serve path's (the batch path's host-visible GPU waits are 20% above it: 18.15 ms against 15.08 a
// window in the same tier) with a 2.4 ms loop (commit/emit 0.5 + draft 1.1 + 0.8) instead of the batch path's 4.2
TwoLaneParams lean(TwoLaneParams p) {
    p.gpu_total[0] *= 15.08 / 18.15; p.gpu_total[1] *= 15.08 / 18.15; p.tail_ms[1] *= 15.08 / 18.15;
    p.fixed_host_ms = 0.5; p.draft_ms = 1.1; p.fixed_lat_ms = 0.8;
    return p;
}

void report(int W) {
    std::printf("two_lane_sim: trace-driven model of two independent MTP lanes on one shared CPU pool (docs/TWO_LANE_MODEL.md)\n");
    std::printf("Rates are in-window (decode windows back to back).  Today's merged batch measured 57-67 tok/s at the clients; the model's in-window\n"
                "rate for it is 61 (typical tier) and 77 (warm tier).  Every ratio below is against the merged batch of the same tier.\n\n");
    std::printf("== 0. calibration: the model's single lane against the measured windows (mean of %d windows)\n", W);
    {
        struct { const char* n; TwoLaneParams p; double measured; } c[] = {
            {"merged 2 slots (4 rows), warm tier", two_lane_merged_preset(true), 43.8},
            {"merged 2 slots (4 rows), typical tier", two_lane_merged_preset(false), 57.4},
            {"1 slot (2 rows), warm tier", [] { auto p = two_lane_lane_preset(true); p.lanes = 1; return p; }(), 33.5},
            {"1 slot (2 rows), typical tier", [] { auto p = two_lane_lane_preset(false); p.lanes = 1; return p; }(), 38.5},
            {"solo on the serve path (no batch)", two_lane_serve_solo_preset(), 28.0},
        };
        for (auto& x : c) {
            const TwoLaneResult r = run(x.p, W);
            std::printf("  %-40s model %6.2f ms  measured %6.2f ms  (%+.1f%%)   %6.1f tok/s at %.2f tok/window\n", x.n, r.lane_ms_per_window[0], x.measured,
                        100.0 * (r.lane_ms_per_window[0] / x.measured - 1.0), r.agg_tok_s, r.tokens_per_window);
        }
    }
    double ratio_two[2] = {0, 0}, ratio_ceiling[2] = {0, 0}, abs_two[2] = {0, 0};
    double ratio_e2[2] = {0, 0}, ratio_e2_lean[2] = {0, 0}, ratio_e3[2] = {0, 0}, ratio_e4[2] = {0, 0}, ratio_e4_lean[2] = {0, 0};
    for (int tier = 0; tier < 2; ++tier) {
        const bool warm = tier == 0;
        const TwoLaneParams lane = two_lane_lane_preset(warm);
        const TwoLaneResult merged = run(two_lane_merged_preset(warm), W);
        const TwoLaneResult serve = run(two_lane_serve_solo_preset(), W);
        TwoLaneParams one = lane; one.lanes = 1;
        const TwoLaneResult solo = run(one, W);
        const TwoLaneResult two = run(lane, W);
        std::printf("\n############ %s tier (%s)\n", warm ? "WARM" : "TYPICAL",
                    warm ? "routed entries >= 85% in VRAM; a solo window needs 2.3 CPU experts per layer, the merged one 3.7"
                         : "70-85% in VRAM; ~3.3 CPU experts per layer for a solo window, 6.6 merged");
        // ---- (a)
        std::printf("== a. aggregate tok/s, two concurrent users\n");
        row("merged-row 2-slot batch (today)", merged, merged.agg_tok_s);
        std::printf("  %-52s %6.1f ms/window  %5.2f tok/window  %6.1f tok/s  %5.2fx   (measured solo: 80-85 once warm)\n", "time-sliced solo, serve path (one user at a time)",
                    serve.lane_ms_per_window[0], serve.tokens_per_window, serve.agg_tok_s, serve.agg_tok_s / merged.agg_tok_s);
        row("time-sliced solo, batch path (1 slot, 2 rows)", solo, merged.agg_tok_s);
        {
            TwoLaneResult ceil = solo;
            ceil.agg_tok_s = 2 * solo.agg_tok_s;
            row("two lanes, CEILING (2 x one lane, nothing ever waits)", ceil, merged.agg_tok_s);
        }
        row("two lanes (the pool order is moot for 2 lanes, see b)", two, merged.agg_tok_s);
        ratio_two[tier] = two.agg_tok_s / merged.agg_tok_s;
        ratio_ceiling[tier] = 2 * solo.agg_tok_s / merged.agg_tok_s;
        abs_two[tier] = two.agg_tok_s;
        // ---- (b)
        std::printf("== b. what the pool contention costs (two lanes)\n");
        std::printf("  one lane alone takes %.2f ms/window; in the pair each takes %.2f: +%.2f ms (%.0f%%) = pool wait %.2f + card wait %.2f + host-thread wait %.2f ms\n",
                    solo.lane_ms_per_window[0], two.lane_ms_per_window[0], two.lane_ms_per_window[0] - solo.lane_ms_per_window[0],
                    100.0 * (two.lane_ms_per_window[0] / solo.lane_ms_per_window[0] - 1.0), two.pool_wait_per_window_ms, two.stage_wait_per_window_ms,
                    two.host_wait_per_window_ms);
        std::printf("  the pair reaches %.0f%% of its ceiling; the pool waits %.3f ms a layer; busy: pool %.0f%%, host thread %.0f%%, card 0 %.0f%%, card 1 %.0f%%\n",
                    100.0 * two.agg_tok_s / (2 * solo.agg_tok_s), two.pool_wait_per_layer_ms, 100 * two.pool_util, 100 * two.host_util, 100 * two.stage_util[0],
                    100 * two.stage_util[1]);
        {
            TwoLaneParams q = lane; q.lanes = 3;
            std::string s = "  3 lanes (needs 3 users; the pool order starts to matter):";
            for (StagePolicy pol : {StagePolicy::Fifo, StagePolicy::Pipelined, StagePolicy::Priority, StagePolicy::Quantum}) {
                q.policy = pol;
                char b[64]; std::snprintf(b, sizeof b, " %s %.1f", stage_policy_name(pol), run(q, W).agg_tok_s);
                s += b;
            }
            std::printf("%s tok/s (the cards are the limit: card 0 holds a window %.1f ms, its pool waits included)\n", s.c_str(),
                        lane.gpu_total[0] + 24 * lane.pool_base_ms + 24 * lane.pool_k_ms * lane.experts[0]);
        }
        for (double st : {1.1, 1.25}) {
            TwoLaneParams p = lane; p.overlap_stretch = st;
            char nm[64]; std::snprintf(nm, sizeof nm, "pool x%.2f slower while both lanes run (DRAM, PCIe)", st);
            row(nm, run(p, W), merged.agg_tok_s);
        }
        struct { const char* n; bool excl, blocks; double host, lat; bool host_fifo; } v[] = {
            {"draft holds card 1 and the host thread (engine today)", true, true, 0.9, 2.2, false},
            {"draft does not block the host (draft_begin)", true, false, 0.9, 2.2, false},
            {"draft concurrent on the card, host blocked", false, true, 0.9, 2.2, false},
            {"draft concurrent and async (best case)", false, false, 0.9, 2.2, false},
            {"all of the 4.2 ms loop blocks the host (worst case)", true, true, 4.2 - 1.1, 0.0, false},
            {"host work first come first served with pool layers", true, true, 0.9, 2.2, true},
        };
        for (auto& x : v) {
            TwoLaneParams p = lane; p.draft_exclusive = x.excl; p.draft_blocks_host = x.blocks; p.fixed_host_ms = x.host; p.fixed_lat_ms = x.lat; p.host_fifo = x.host_fifo;
            row(x.n, run(p, W), merged.agg_tok_s);
        }
        // ---- (c)
        std::printf("== c. sensitivities (each ratio against the merged batch under the same what-if)\n");
        std::printf("  MTP rows per lane (1 + proposals; each extra row asks the pool for more experts; MTP acceptance %.2f per row, a constant, unmeasured beyond 1 row):\n", lane.accept);
        for (double rows : {1.0, 1.5, 2.0, 3.0, 4.0}) {
            TwoLaneParams p = lane; p.rows = rows;
            char nm[64]; std::snprintf(nm, sizeof nm, "    rows %.1f", rows);
            row(nm, run(p, W), merged.agg_tok_s);
        }
        std::printf("  CPU experts per layer (x the measured %.2f for a solo window; merged scaled alike):\n", (lane.experts[0] + lane.experts[1]) / 2);
        for (double f : {0.5, 0.75, 1.0, 1.5, 2.0}) {
            TwoLaneParams p = lane; scale_experts(p, f);
            const TwoLaneResult m = run(merged_like(p, warm, f), W);
            char nm[72]; std::snprintf(nm, sizeof nm, "    experts x%.2f (merged %.1f tok/s)", f, m.agg_tok_s);
            row(nm, run(p, W), m.agg_tok_s);
        }
        std::printf("  GPU dense time and the pool kernel (merged gets the same):\n");
        struct { const char* n; double gs, gd, ps; } wi[] = {
            {"GPU x0.8", 0.8, 0, 1}, {"GPU x1.2", 1.2, 0, 1}, {"dense fusion, -3.5 ms a window", 1.0, -3.5, 1},
            {"pool kernel -30%", 1.0, 0, 0.7}, {"dense fusion + pool kernel -30%", 1.0, -3.5, 0.7},
        };
        for (auto& x : wi) {
            TwoLaneParams p = lane; p.gpu_scale = x.gs; p.gpu_delta_ms = x.gd; p.pool_scale = x.ps;
            const TwoLaneResult m = run(merged_like(p, warm, 1.0), W);
            char nm[80]; std::snprintf(nm, sizeof nm, "    %s (merged %.1f tok/s)", x.n, m.agg_tok_s);
            row(nm, run(p, W), m.agg_tok_s);
        }
        // ---- (d)
        std::printf("== d. one user through the lane machinery\n");
        std::printf("  serve path (today, no batch): %.1f ms/window, %.2f tok/window = %.1f tok/s.  One request as a lane (batch path, 1 slot, 2 rows): %.1f ms, %.2f tok = %.1f tok/s: %+.0f%%\n",
                    serve.lane_ms_per_window[0], serve.tokens_per_window, serve.agg_tok_s, solo.lane_ms_per_window[0], solo.tokens_per_window, solo.agg_tok_s,
                    100.0 * (solo.agg_tok_s / serve.agg_tok_s - 1.0));
        std::printf("  => with one active request the engine must stay on the serve path; the lanes are for two active requests only\n");
        // ---- (e)
        std::printf("== e. what would have to be true to reach %.1fx: ratios against today's merged batch, and against a merged batch whose GPU waits are cut the same way\n", kGoRatio);
        {
            TwoLaneParams ml = lane;
            ml.gpu_scale = 15.08 / 18.15;
            const TwoLaneResult merged_lean = run(merged_like(ml, warm, 1.0), W);
            auto erow = [&](const char* name, const TwoLaneParams& q, const TwoLaneResult& ref2) {
                const TwoLaneResult r = run(q, W);
                row(name, r, merged.agg_tok_s, ref2.agg_tok_s);
                return r;
            };
            TwoLaneParams a = lean(lane);
            erow("E1 lane window as lean as the serve path's", a, merged_lean);
            TwoLaneParams b = a; b.draft_exclusive = false; b.draft_blocks_host = false;
            const TwoLaneResult r2 = erow("E2 = E1 + asynchronous, concurrent draft", b, merged_lean);
            ratio_e2[tier] = r2.agg_tok_s / merged.agg_tok_s;
            ratio_e2_lean[tier] = r2.agg_tok_s / merged_lean.agg_tok_s;
            TwoLaneParams c = b; c.rows = 3.0;
            const TwoLaneResult r3 = erow("E3 = E2 + 3 rows per lane at acceptance 0.87 (optimistic)", c, merged_lean);
            ratio_e3[tier] = r3.agg_tok_s / merged.agg_tok_s;
            TwoLaneParams d = b; d.rows = 3.0; d.accept = 0.75;
            const TwoLaneResult r4 = erow("E4 = E2 + 3 rows per lane at acceptance 0.75", d, merged_lean);
            ratio_e4[tier] = r4.agg_tok_s / merged.agg_tok_s;
            ratio_e4_lean[tier] = r4.agg_tok_s / merged_lean.agg_tok_s;
            TwoLaneParams e = d; e.gpu_delta_ms = -3.5; e.pool_scale = 0.7;
            TwoLaneParams ef = e; ef.gpu_scale = 15.08 / 18.15;
            const TwoLaneResult me = run(merged_like(ef, warm, 1.0), W);
            const TwoLaneResult re = run(e, W);
            std::printf("  %-52s %6.1f ms/window  %5.2f tok/window  %6.1f tok/s  %5s   %5.2fx of the lean merged batch with the same fusion and kernel (%.1f tok/s)\n",
                        "E5 = E4 + dense fusion -3.5 ms + pool kernel -30%", re.lane_ms_per_window[0], re.tokens_per_window, re.agg_tok_s, "", re.agg_tok_s / me.agg_tok_s, me.agg_tok_s);
        }
    }
    std::printf("\n== go / no-go: two lanes' aggregate >= %.1fx today's merged batch of the same tier (in-window, with the pool's contention)\n", kGoRatio);
    std::printf("  as the engine can run lanes today (batch-path windows, one proposal a slot, synchronous draft): warm %.2fx (%.1f tok/s), typical %.2fx (%.1f tok/s)\n",
                ratio_two[0], abs_two[0], ratio_two[1], abs_two[1]);
    std::printf("  even with no waiting at all (the ceiling = 2 x one lane): warm %.2fx, typical %.2fx\n", ratio_ceiling[0], ratio_ceiling[1]);
    const bool go = ratio_two[0] >= kGoRatio && ratio_two[1] >= kGoRatio;
    std::printf("  VERDICT: %s\n", go ? "GO" : "NO-GO");
    if (!go)
        std::printf("  It reaches %.1fx only if the lane window is first made as lean as the serve path's AND the draft is asynchronous (E2: warm %.2fx / typical %.2fx\n"
                    "  against today's merged batch, but %.2fx / %.2fx once the merged batch gets the same lean windows), or the lanes carry 3 rows each at an acceptance\n"
                    "  of 0.87 (E3: %.2fx / %.2fx; at 0.75, E4: %.2fx / %.2fx, %.2fx / %.2fx against the lean merged batch).  None of those is measured on lanes.\n",
                    kGoRatio, ratio_e2[0], ratio_e2[1], ratio_e2_lean[0], ratio_e2_lean[1], ratio_e3[0], ratio_e3[1], ratio_e4[0], ratio_e4[1], ratio_e4_lean[0],
                    ratio_e4_lean[1]);
}

}  // namespace

int main(int argc, char** argv) {
    bool rep = false, warm = true;
    int W = 3000;
    double experts_scale = 1.0;
    const char* trace_path = nullptr;
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == "--tier") warm = args[i + 1] != "typical";
    TwoLaneParams p = two_lane_lane_preset(warm);
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string a = args[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= args.size()) { std::fprintf(stderr, "two_lane_sim: %s needs a value\n", a.c_str()); std::exit(2); }
            return args[++i].c_str();
        };
        if (a == "--report") rep = true;
        else if (a == "--tier") (void) next();
        else if (a == "--lanes") p.lanes = std::max(1, std::min(4, std::atoi(next())));
        else if (a == "--policy") {
            const std::string v = next();
            p.policy = v == "stage0" ? StagePolicy::Pipelined : v == "priority" ? StagePolicy::Priority : v == "quantum" ? StagePolicy::Quantum : StagePolicy::Fifo;
        } else if (a == "--rows") p.rows = std::atof(next());
        else if (a == "--accept") p.accept = std::atof(next());
        else if (a == "--experts-scale") experts_scale = std::atof(next());
        else if (a == "--gpu-scale") p.gpu_scale = std::atof(next());
        else if (a == "--gpu-delta") p.gpu_delta_ms = std::atof(next());
        else if (a == "--pool-scale") p.pool_scale = std::atof(next());
        else if (a == "--stretch") p.overlap_stretch = std::max(1.0, std::atof(next()));
        else if (a == "--lat") p.fixed_lat_ms = std::atof(next());
        else if (a == "--host") p.fixed_host_ms = std::atof(next());
        else if (a == "--draft") {
            const std::string v = next();
            p.draft_exclusive = v == "exclusive";
            p.draft_blocks_host = v == "exclusive" || v == "free";
        } else if (a == "--windows") W = std::max(50, std::atoi(next()));
        else if (a == "--seed") p.seed = (uint64_t) std::atoll(next());
        else if (a == "--trace") trace_path = next();
        else {
            std::fprintf(stderr, "usage: two_lane_sim --report [--windows N] | [--tier warm|typical] [--lanes N] [--policy fifo|stage0|priority|quantum] [--rows R] "
                                 "[--accept p] [--experts-scale f] [--gpu-scale f] [--gpu-delta ms] [--pool-scale f] [--stretch f] "
                                 "[--draft exclusive|free|async] [--lat ms] [--host ms] [--windows N] [--seed S] [--trace file.csv]\n");
            return 2;
        }
    }
    if (rep) { report(W); return 0; }
    scale_experts(p, experts_scale);
    if (trace_path != nullptr && !load_trace(trace_path, p.trace)) { std::fprintf(stderr, "two_lane_sim: cannot read a trace from %s\n", trace_path); return 2; }
    p.windows = W;
    const TwoLaneResult r = two_lane_run(p);
    if (!r.error.empty()) { std::fprintf(stderr, "two_lane_sim: %s\n", r.error.c_str()); return 1; }
    const TwoLaneResult m = run(merged_like(p, warm, experts_scale), W);
    std::printf("%d lane(s), %s tier, pool order %s: %.2f ms/window/lane, %.2f tokens/window, aggregate %.1f tok/s (merged batch %.1f: %.2fx)\n", r.lanes,
                warm ? "warm" : "typical", stage_policy_name(p.policy), r.lane_ms_per_window[0], r.tokens_per_window, r.agg_tok_s, m.agg_tok_s, r.agg_tok_s / m.agg_tok_s);
    std::printf("pool %.0f%% busy, host %.0f%%, card0 %.0f%%, card1 %.0f%%; a window waits %.2f ms for the pool, %.2f for a card, %.2f for the host thread\n",
                100 * r.pool_util, 100 * r.host_util, 100 * r.stage_util[0], 100 * r.stage_util[1], r.pool_wait_per_window_ms, r.stage_wait_per_window_ms,
                r.host_wait_per_window_ms);
    return 0;
}
