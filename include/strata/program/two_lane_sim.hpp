#pragma once

#include "strata/program/stage_sim.hpp"   // StagePolicy and its names (the pool orders), shared with the stage-overlap model

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// A trace-driven model of TWO INDEPENDENT STREAMS ("lanes") through the two layer-split stages, with MTP, on one shared
// CPU expert pool (header-only, CPU; no engine code is used).  It answers, before anything is built, what
// `--batch-groups 2` beside `--batch-mtp` could give two concurrent users against today's merged-row 2-slot batch.
//
// The model (ms; the numbers come from the serve log's `strata batch:` and `strata decode GPU stages` lines):
//  * a lane is one conversation's loop: stage 0 (24 layers on the first card) -> stage 1 (24 layers on the second) ->
//    the verdict -> the host's commit and emit -> the draft (the MTP head, on the second card) -> the next window.  A
//    lane's window is solo-size (its own rows; rows are never merged with the other lane's) and every window is real:
//    no speculation, nothing is doomed.
//  * a stage's card runs one lane's window at a time and is HELD for the whole of it, including the layers' waits for the
//    pool.  Lane B's stage 0 can run while lane A is on stage 1.
//  * in a layer the stage's GPU runs `gpu` (the host-visible "GPU-reach wait"), then the layer's experts are computed by the
//    ONE shared pool, one layer at a time and not preemptively, for `pool_base + pool_k * n` where n is the lane's
//    DISTINCT CPU experts of that layer (the pool call is the host thread's work too: it serves nothing else meanwhile),
//    and only when that result is back does the next layer start.  When both lanes wait for the pool, `policy` decides:
//    stage0-first (the pump loop's order today: it polls stage 0 before stage 1), priority (the older window, on stage 1,
//    first), quantum (take turns) or fifo.
//  * the lane's host work after a window (commit and emit) runs on the same host thread, so it also keeps the pool waiting;
//    the draft runs on the second card (it can hold that card against the other lane's stage 1) and, as the engine's
//    `draft()` is synchronous, can hold the host thread too.  The rest of a lane's loop is pure latency.
//  * per-layer costs are not constant: the GPU time follows the layer types (a QSA layer weighs `qsa_weight` GDN layers)
//    plus jitter, and the expert count of a layer is drawn per window (a lognormal-mixed Poisson with the measured mean),
//    from a hash of (seed, lane, window) so every policy sees the same demands.  `trace` replays per-layer windows
//    (gpu ms and experts per layer) recorded or synthesised elsewhere instead.
namespace strata::program {

/// One window's per-layer costs of one lane: the GPU wait (ms) and the pool demand (ms) of each layer of each stage.
struct LaneWindowTrace {
    std::vector<double> gpu[2];
    std::vector<double> cpu[2];
};

struct TwoLaneParams {
    // ---- a lane's window at `rows_ref` rows (a solo window) ----
    int layers[2] = {24, 24};
    double gpu_total[2] = {9.39, 8.76};    ///< host-visible GPU wait per window, ms (the tail below included)
    double tail_ms[2] = {0.0, 1.0};        ///< GPU-only time after the last layer's pool call (stage 1: the head)
    int gdn_per_qsa = 3;                   ///< layer pattern GDN x n, QSA (Qwen3.8: 18 GDN + 6 QSA a stage)
    double qsa_weight = 1.5;               ///< a QSA layer's dense time over a GDN layer's (0.43 / 0.286, the stage profile)
    double experts[2] = {2.64, 1.97};      ///< distinct CPU experts per layer of each stage at rows_ref (mean)
    double pool_base_ms = 0.05;            ///< per pool call: barrier + plan
    double pool_k_ms = 0.0787;             ///< per distinct CPU expert (2 rows)
    double expert_sigma = 0.35;            ///< lognormal sd of the per-layer demand multiplier (mean 1)
    double gpu_jitter = 0.10;              ///< each layer's GPU time x (1 + jitter * u), u in [-1, 1)
    // ---- rows: a lane's window carries `rows` rows (1 + proposals) ----
    double rows = 2.0, rows_ref = 2.0;
    double expert_row_exp = 0.85;          ///< experts ~ (rows / rows_ref)^exp (the rows' experts overlap a little)
    double gpu_row_slope = 0.095;          ///< GPU time per extra row (the merged 4-row window waits 19% longer than a 2-row one)
    double pool_row_slope = 0.05;          ///< pool cost per expert per extra row (0.0787 at 2 rows, 0.087 at 4)
    double accept = 0.87;                  ///< MTP acceptance of each proposal in a row
    double tokens_override = 0;            ///< > 0: tokens per lane window as given (the merged windows)
    // ---- what a lane does besides its two stages ----
    double verdict_ms = 0.1;               ///< last layer to the picks being read
    double hop_ms = 0.0;                   ///< stage 0 -> stage 1 hand-off
    double fixed_host_ms = 0.9;            ///< commit + emit: holds the host thread
    double draft_ms = 1.1;                 ///< the draft: GPU time on stage 1's card
    bool draft_exclusive = true;           ///< true: holds stage 1's card (the other lane's stage 1 waits); false: concurrent, free
    bool draft_blocks_host = true;         ///< the engine's draft() is synchronous: the host thread waits for it
    double fixed_lat_ms = 2.2;             ///< staging, launches, anything else: latency only
    // ---- the pool, the cards ----
    StagePolicy policy = StagePolicy::Fifo;
    bool host_fifo = false;                ///< false: a waiting pool layer goes before commit/emit/draft work; true: first come first served
    double overlap_stretch = 1.0;          ///< pool time x this while the other lane has a stage window in flight
    // ---- what-ifs ----
    double gpu_scale = 1.0;                ///< all GPU time x this
    double gpu_delta_ms = 0.0;             ///< added to the GPU time of a window (e.g. -3.5: the dense fusions), spread by stage totals
    double pool_scale = 1.0;               ///< the pool's per-expert cost x this (e.g. 0.7: a -30% pool kernel); the base stays
    // ---- the run ----
    int lanes = 2;
    int windows = 3000;                    ///< per lane; the rate is measured from 10% of them to the first lane's last
    uint64_t seed = 1;
    bool record = false;                   ///< keep every layer and every hold of a card (for the invariants tests)
    std::vector<LaneWindowTrace> trace;    ///< non-empty: replay (window w of lane l uses trace[(w + l * trace.size() / 2) % size])

    /// mean tokens of one lane window
    double tokens_per_window() const {
        if (tokens_override > 0) return tokens_override;
        const double m = std::max(rows - 1.0, 0.0);
        double t = 1.0, pj = 1.0;
        int j = 1;
        for (; (double) j <= m; ++j) { pj *= accept; t += pj; }
        const double fr = m - std::floor(m);
        if (fr > 0) t += fr * pj * accept;
        return t;
    }
    double row_gpu_factor() const { return std::max(0.05, 1.0 + gpu_row_slope * (rows - rows_ref)); }
    double row_expert_factor() const { return std::pow(std::max(rows, 0.05) / std::max(rows_ref, 0.05), expert_row_exp); }
    double k_eff() const { return pool_k_ms * pool_scale * std::max(0.05, 1.0 + pool_row_slope * (rows - rows_ref)); }
};

struct TwoLaneResult {
    std::string error;                     ///< non-empty: the model made no progress (a bug)
    int lanes = 0;
    double tokens_per_window = 0;
    double span_ms = 0;                    ///< the measured interval
    long windows_measured = 0;             ///< lane windows completed in it (all lanes)
    double agg_tok_s = 0;                  ///< all lanes' tokens per second
    double lane_ms_per_window[4] = {0, 0, 0, 0};
    double lane_tok_s[4] = {0, 0, 0, 0};
    double pool_util = 0;                  ///< share of the interval the pool served layers
    double host_util = 0;                  ///< share of it the host thread was busy (pool + commit/emit + blocking draft)
    double stage_util[2] = {0, 0};         ///< share of it each card was held (windows + the exclusive draft)
    double pool_wait_per_layer_ms = 0;     ///< mean time a layer waited for the pool after its GPU phase
    double pool_wait_per_window_ms = 0;
    double stage_wait_per_window_ms = 0;   ///< mean time a window waited for its card (both stages)
    double host_wait_per_window_ms = 0;    ///< mean time commit/emit/draft waited for the host thread
    double solo_floor_ms = 0;              ///< the lane's window with nothing else running (uncontended mean, deterministic)
    struct Layer { int lane = 0, win = 0, stage = 0, layer = 0; double gpu_start = 0, gpu_end = 0, pool_start = 0, pool_end = 0; };
    struct Hold { int lane = 0, win = 0, card = 0; bool draft = false; double start = 0, end = 0; };
    std::vector<Layer> recorded_layers;
    std::vector<Hold> recorded_holds;
};

namespace two_lane_detail {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double u() { return (double) (next() >> 11) / 9007199254740992.0; }   // [0, 1)
    double normal() {
        const double a = std::max(u(), 1e-12), b = u();
        return std::sqrt(-2.0 * std::log(a)) * std::cos(6.283185307179586 * b);
    }
    int poisson(double mean) {
        if (mean <= 0) return 0;
        if (mean > 30.0) return std::max(0, (int) std::lround(mean + std::sqrt(mean) * normal()));
        const double L = std::exp(-mean);
        double p = 1.0;
        int k = 0;
        do { ++k; p *= u(); } while (p > L);
        return k - 1;
    }
};

/// The GPU weight of layer l of a stage: the pattern GDN x n, QSA (weights 1 and qsa_weight).
inline double layer_weight(const TwoLaneParams& p, int l) { return (l % (p.gdn_per_qsa + 1)) == p.gdn_per_qsa ? p.qsa_weight : 1.0; }

/// The mean (no noise, no what-ifs on rows) per-layer GPU wait of stage s, summing with the tail to gpu_total[s].
inline std::vector<double> base_gpu_layers(const TwoLaneParams& p, int s) {
    std::vector<double> g((size_t) p.layers[s], 0.0);
    double wsum = 0;
    for (int l = 0; l < p.layers[s]; ++l) wsum += layer_weight(p, l);
    const double total = std::max(0.0, p.gpu_total[s] - p.tail_ms[s]);
    for (int l = 0; l < p.layers[s]; ++l) g[(size_t) l] = total * layer_weight(p, l) / std::max(wsum, 1e-12);
    return g;
}

/// What-ifs on the GPU time of a stage: scale, then a delta spread by the stage totals (never below 10%).
inline double gpu_what_if(const TwoLaneParams& p, int s, double v) {
    const double tot = p.gpu_total[0] + p.gpu_total[1];
    const double d = tot > 0 ? p.gpu_delta_ms / tot : 0.0;   // the same share off every layer: a stage loses delta * its total / all
    (void) s;
    return std::max(0.1 * v, v * p.gpu_scale * (1.0 + d));
}

/// One lane window's per-layer costs (gpu wait and pool ms), drawn from the lane's and the window's hash; the tail is in `tail`.
inline void make_window(const TwoLaneParams& p, int lane, int win, LaneWindowTrace& w, double tail[2]) {
    for (int s = 0; s < 2; ++s) { w.gpu[s].clear(); w.cpu[s].clear(); tail[s] = gpu_what_if(p, s, p.tail_ms[s] * p.row_gpu_factor()); }
    if (!p.trace.empty()) {
        const LaneWindowTrace& t = p.trace[(size_t) (((long) win + (long) lane * (long) p.trace.size() / 2) % (long) p.trace.size())];
        for (int s = 0; s < 2; ++s) {
            tail[s] = 0.0;
            for (double v : t.gpu[s]) w.gpu[s].push_back(gpu_what_if(p, s, v));
            for (double v : t.cpu[s]) w.cpu[s].push_back(v);   // the trace's pool demand is in ms already
        }
        return;
    }
    Rng rng(p.seed * 0x2545F4914F6CDD1Dull + (uint64_t) lane * 0x9E3779B97F4A7C15ull + (uint64_t) win * 0xD1B54A32D192ED03ull + 1);
    const double rg = p.row_gpu_factor(), re = p.row_expert_factor(), k = p.k_eff();
    const double sg = p.expert_sigma;
    for (int s = 0; s < 2; ++s) {
        const std::vector<double> g = base_gpu_layers(p, s);
        for (int l = 0; l < p.layers[s]; ++l) {
            const double j = 1.0 + p.gpu_jitter * (2.0 * rng.u() - 1.0);
            w.gpu[s].push_back(gpu_what_if(p, s, g[(size_t) l] * rg * j));
            const double mult = sg > 0 ? std::exp(sg * rng.normal() - 0.5 * sg * sg) : 1.0;
            const int n = rng.poisson(p.experts[s] * re * mult);
            w.cpu[s].push_back(p.pool_base_ms + k * (double) n);
        }
    }
}

}  // namespace two_lane_detail

/// The model's own deterministic mean of one lane window with nothing else running (no noise): S0 + S1 + the lane's loop.
inline double two_lane_solo_mean(const TwoLaneParams& p) {
    double t = p.verdict_ms + p.hop_ms + p.fixed_host_ms + p.draft_ms + p.fixed_lat_ms;
    for (int s = 0; s < 2; ++s) {
        for (double g : two_lane_detail::base_gpu_layers(p, s)) t += two_lane_detail::gpu_what_if(p, s, g * p.row_gpu_factor());
        t += two_lane_detail::gpu_what_if(p, s, p.tail_ms[s] * p.row_gpu_factor());
        const double n = p.experts[s] * p.row_expert_factor();
        t += p.layers[s] * (p.pool_base_ms + p.k_eff() * n);
    }
    return t;
}

inline TwoLaneResult two_lane_run(const TwoLaneParams& P) {
    using namespace two_lane_detail;
    constexpr double INF = 1e300;
    TwoLaneResult res;
    const int NL = std::max(1, std::min(P.lanes, 4));
    res.lanes = NL;
    res.tokens_per_window = P.tokens_per_window();
    res.solo_floor_ms = two_lane_solo_mean(P);
    if (P.windows < 20 || P.layers[0] <= 0 || P.layers[1] <= 0) { res.error = "two_lane_run: windows < 20 or no layers"; return res; }

    enum Ph { S0_WAIT, GPU, POOL_WAIT, POOL_RUN, HOP, S1_WAIT, VERDICT, FIX_WAIT, FIX_RUN, DRAFT_WAIT, DRAFT_RUN, LAT, DONE };
    struct Lane {
        Ph ph = S0_WAIT;
        double t_end = 0, arrive = 0, hold_start = 0, win_start = 0;
        int win = 0, stage = 0, layer = 0, rec = -1;
        LaneWindowTrace w;
        double tail[2] = {0, 0};
        bool in_tail = false;
        long done = 0;
    };
    std::vector<Lane> L((size_t) NL);
    int card_owner[2] = {-1, -1};     // the lane whose stage window (or exclusive draft) holds each card
    int host_owner = -1;              // the lane whose pool layer / commit-emit / blocking draft the host thread serves
    int last_pool_lane = NL - 1;
    double busy_pool = 0, busy_host = 0, busy_card[2] = {0, 0}, pool_wait = 0, stage_wait = 0, host_wait = 0;
    long pool_layers = 0, windows_done_total = 0;
    double now = 0;
    // interval snapshots
    struct Snap { double t = 0, tok = 0, pool = 0, host = 0, card[2] = {0, 0}, pw = 0, sw = 0, hw = 0; long layers = 0, wins = 0, lane_wins[4] = {0, 0, 0, 0}; bool taken = false; };
    Snap s0, s1;
    const long warm = std::max(1, P.windows / 10);
    const double tpw = res.tokens_per_window;

    auto take = [&](Snap& s) {
        s.t = now; s.tok = (double) windows_done_total * tpw; s.pool = busy_pool; s.host = busy_host; s.card[0] = busy_card[0];
        s.card[1] = busy_card[1]; s.pw = pool_wait; s.sw = stage_wait; s.hw = host_wait; s.layers = pool_layers; s.wins = windows_done_total;
        for (int i = 0; i < NL; ++i) s.lane_wins[i] = L[(size_t) i].done;
        s.taken = true;
    };
    auto stretch_for = [&](int lane) {   // the other lane has a stage window in flight
        for (int o = 0; o < NL; ++o)
            if (o != lane && (L[(size_t) o].ph == GPU || L[(size_t) o].ph == POOL_WAIT || L[(size_t) o].ph == POOL_RUN || L[(size_t) o].ph == HOP)) return P.overlap_stretch;
        return 1.0;
    };
    auto begin_layer = [&](int i) {   // the lane's stage `stage` layer `layer` starts its GPU phase at `now`
        Lane& a = L[(size_t) i];
        a.ph = GPU;
        const double g = a.w.gpu[a.stage][(size_t) a.layer];
        a.t_end = now + g;
        if (P.record) {
            TwoLaneResult::Layer r; r.lane = i; r.win = a.win; r.stage = a.stage; r.layer = a.layer; r.gpu_start = now; r.gpu_end = a.t_end;
            r.pool_start = r.pool_end = a.t_end;
            a.rec = (int) res.recorded_layers.size();
            res.recorded_layers.push_back(r);
        }
    };
    auto release_card = [&](int i, int c) {
        Lane& a = L[(size_t) i];
        busy_card[c] += now - a.hold_start;
        if (P.record) { TwoLaneResult::Hold h; h.lane = i; h.win = a.win; h.card = c; h.draft = a.ph == DRAFT_RUN; h.start = a.hold_start; h.end = now; res.recorded_holds.push_back(h); }
        card_owner[c] = -1;
    };
    auto start_window = [&](int i) {   // the lane got stage 0's card
        Lane& a = L[(size_t) i];
        stage_wait += now - a.arrive;
        card_owner[0] = i;
        a.hold_start = now; a.win_start = now;
        make_window(P, i, a.win, a.w, a.tail);
        a.stage = 0; a.layer = 0; a.in_tail = false;
        begin_layer(i);
    };
    auto finish_stage_layers = [&](int i) {   // all layers of the stage are done: its tail, then hand on
        Lane& a = L[(size_t) i];
        if (a.tail[a.stage] > 0 && !a.in_tail) {
            a.in_tail = true; a.ph = GPU; a.t_end = now + a.tail[a.stage];
            if (P.record) a.rec = -1;
            return;
        }
        a.in_tail = false;
        release_card(i, a.stage);
        if (a.stage == 0) { a.ph = HOP; a.t_end = now + P.hop_ms; }
        else { a.ph = VERDICT; a.t_end = now + P.verdict_ms; }
    };
    for (int i = 0; i < NL; ++i) { L[(size_t) i].ph = S0_WAIT; L[(size_t) i].arrive = 1e-9 * i; }

    long guard = 0;
    bool stop = false;
    while (!stop) {
        bool changed;
        do {
            changed = false;
            // ---- 1. completions at `now`
            for (int i = 0; i < NL; ++i) {
                Lane& a = L[(size_t) i];
                if (a.t_end > now + 1e-12) continue;
                switch (a.ph) {
                    case GPU:
                        if (a.in_tail) { finish_stage_layers(i); changed = true; break; }
                        a.ph = POOL_WAIT; a.arrive = a.t_end; changed = true; break;
                    case POOL_RUN:
                        if (P.record && a.rec >= 0) res.recorded_layers[(size_t) a.rec].pool_end = a.t_end;
                        host_owner = -1;
                        ++a.layer;
                        changed = true;
                        if (a.layer >= P.layers[a.stage] || a.layer >= (int) a.w.gpu[a.stage].size()) finish_stage_layers(i); else begin_layer(i);
                        break;
                    case HOP: a.ph = S1_WAIT; a.arrive = a.t_end; changed = true; break;
                    case VERDICT: a.ph = FIX_WAIT; a.arrive = a.t_end; changed = true; break;
                    case FIX_RUN:
                        host_owner = -1;
                        if (P.draft_ms > 0) { a.ph = DRAFT_WAIT; a.arrive = a.t_end; }
                        else { a.ph = LAT; a.t_end = now + P.fixed_lat_ms; }
                        changed = true; break;
                    case DRAFT_RUN:
                        if (P.draft_exclusive) release_card(i, 1);
                        if (P.draft_blocks_host) host_owner = -1;
                        a.ph = LAT; a.t_end = now + P.fixed_lat_ms; changed = true; break;
                    case LAT:
                        ++a.done; ++windows_done_total; ++a.win;
                        if (!s0.taken && windows_done_total >= warm * NL) take(s0);
                        if (a.done >= P.windows) { take(s1); stop = true; a.ph = DONE; a.t_end = INF; }
                        else { a.ph = S0_WAIT; a.arrive = now; }
                        changed = true; break;
                    default: break;
                }
                if (stop) break;
            }
            if (stop) break;
            // ---- 2. card 0: stage 0 windows, first come first served
            if (card_owner[0] < 0) {
                int pick = -1;
                for (int i = 0; i < NL; ++i)
                    if (L[(size_t) i].ph == S0_WAIT && (pick < 0 || L[(size_t) i].arrive < L[(size_t) pick].arrive)) pick = i;
                if (pick >= 0) { start_window(pick); changed = true; }
            }
            // ---- 3. card 1: stage 1 windows, and the draft when it does not need the host thread (first come first served)
            if (card_owner[1] < 0) {
                int pick = -1;
                for (int i = 0; i < NL; ++i) {
                    const Ph ph = L[(size_t) i].ph;
                    const bool want = ph == S1_WAIT || (ph == DRAFT_WAIT && !P.draft_blocks_host && P.draft_exclusive);
                    if (want && (pick < 0 || L[(size_t) i].arrive < L[(size_t) pick].arrive)) pick = i;
                }
                if (pick >= 0) {
                    Lane& a = L[(size_t) pick];
                    if (a.ph == S1_WAIT) {
                        stage_wait += now - a.arrive;
                        card_owner[1] = pick; a.hold_start = now; a.stage = 1; a.layer = 0; a.in_tail = false;
                        begin_layer(pick);
                    } else {
                        host_wait += now - a.arrive;
                        card_owner[1] = pick; a.hold_start = now; a.ph = DRAFT_RUN; a.t_end = now + P.draft_ms;
                    }
                    changed = true;
                }
            }
            // a draft that holds nothing exclusive and does not block the host just runs
            for (int i = 0; i < NL; ++i) {
                Lane& a = L[(size_t) i];
                if (a.ph == DRAFT_WAIT && !P.draft_blocks_host && !P.draft_exclusive) { a.ph = DRAFT_RUN; a.t_end = now + P.draft_ms; changed = true; }
            }
            // ---- 4. the host thread: a pool layer (by policy), or commit/emit, or a blocking draft
            if (host_owner < 0) {
                auto pool_pick = [&]() -> int {
                    int pick = -1;
                    for (int i = 0; i < NL; ++i) {
                        const Lane& a = L[(size_t) i];
                        if (a.ph != POOL_WAIT) continue;
                        if (pick < 0) { pick = i; continue; }
                        const Lane& b = L[(size_t) pick];
                        bool better;
                        switch (P.policy) {
                            case StagePolicy::Pipelined: better = a.stage != b.stage ? a.stage < b.stage : a.arrive < b.arrive; break;
                            case StagePolicy::Priority: better = a.stage != b.stage ? a.stage > b.stage : a.arrive < b.arrive; break;
                            case StagePolicy::Quantum: {
                                const int da = (i - last_pool_lane + NL) % NL, db = (pick - last_pool_lane + NL) % NL;
                                better = da != 0 && (db == 0 || da < db);
                                break;
                            }
                            default: better = a.arrive < b.arrive; break;
                        }
                        if (better) pick = i;
                    }
                    return pick;
                };
                auto fix_pick = [&](double* arr) -> int {   // commit/emit or a blocking draft that can start now
                    int pick = -1;
                    for (int i = 0; i < NL; ++i) {
                        const Lane& a = L[(size_t) i];
                        const bool fx = a.ph == FIX_WAIT;
                        const bool dr = a.ph == DRAFT_WAIT && P.draft_blocks_host && (!P.draft_exclusive || card_owner[1] < 0);
                        if ((fx || dr) && (pick < 0 || a.arrive < *arr)) { pick = i; *arr = a.arrive; }
                    }
                    return pick;
                };
                double fa = INF;
                int fp = fix_pick(&fa);
                int pp = pool_pick();
                bool take_pool = pp >= 0;
                if (pp >= 0 && fp >= 0 && P.host_fifo) take_pool = L[(size_t) pp].arrive <= fa;
                if (take_pool) {
                    Lane& a = L[(size_t) pp];
                    const double c = a.w.cpu[a.stage][(size_t) a.layer] * stretch_for(pp);
                    pool_wait += now - a.arrive;
                    ++pool_layers;
                    a.ph = POOL_RUN; a.t_end = now + c; host_owner = pp; last_pool_lane = pp;
                    busy_pool += c; busy_host += c;
                    if (P.record && a.rec >= 0) res.recorded_layers[(size_t) a.rec].pool_start = now;
                    changed = true;
                } else if (fp >= 0) {
                    Lane& a = L[(size_t) fp];
                    host_wait += now - a.arrive;
                    host_owner = fp;
                    if (a.ph == FIX_WAIT) { a.ph = FIX_RUN; a.t_end = now + P.fixed_host_ms; busy_host += P.fixed_host_ms; }
                    else {
                        a.ph = DRAFT_RUN; a.t_end = now + P.draft_ms; busy_host += P.draft_ms;
                        if (P.draft_exclusive) { card_owner[1] = fp; a.hold_start = now; }
                    }
                    changed = true;
                }
            }
        } while (changed);
        if (stop) break;
        double nt = INF;
        for (int i = 0; i < NL; ++i) {
            const Ph ph = L[(size_t) i].ph;
            if (ph == GPU || ph == POOL_RUN || ph == HOP || ph == VERDICT || ph == FIX_RUN || ph == DRAFT_RUN || ph == LAT) nt = std::min(nt, L[(size_t) i].t_end);
        }
        if (nt >= INF || nt < now - 1e-9 || ++guard > 200000000L) {
            res.error = "the model made no progress at " + std::to_string(now) + " ms";
            return res;
        }
        now = std::max(now, nt);
    }
    if (P.record)   // a pool call still running when the run stops ends at its scheduled time
        for (const Lane& a : L)
            if (a.ph == POOL_RUN && a.rec >= 0) res.recorded_layers[(size_t) a.rec].pool_end = a.t_end;
    if (!s0.taken) take(s0);
    const double span = std::max(s1.t - s0.t, 1e-9);
    res.span_ms = span;
    res.windows_measured = s1.wins - s0.wins;
    res.agg_tok_s = 1000.0 * (s1.tok - s0.tok) / span;
    for (int i = 0; i < NL; ++i) {
        const long w = s1.lane_wins[i] - s0.lane_wins[i];
        res.lane_ms_per_window[i] = w > 0 ? span / (double) w : 0.0;
        res.lane_tok_s[i] = 1000.0 * (double) w * tpw / span;
    }
    res.pool_util = (s1.pool - s0.pool) / span;
    res.host_util = (s1.host - s0.host) / span;
    for (int c = 0; c < 2; ++c) res.stage_util[c] = (s1.card[c] - s0.card[c]) / span;
    const double nl = std::max<long>(s1.layers - s0.layers, 1), nw = std::max<long>(s1.wins - s0.wins, 1);
    res.pool_wait_per_layer_ms = (s1.pw - s0.pw) / nl;
    res.pool_wait_per_window_ms = (s1.pw - s0.pw) / nw;
    res.stage_wait_per_window_ms = (s1.sw - s0.sw) / nw;
    res.host_wait_per_window_ms = (s1.hw - s0.hw) / nw;
    return res;
}

/// The invariants of a finished recorded run: "" when all hold, else the first violation.
inline std::string two_lane_check(const TwoLaneParams& p, const TwoLaneResult& r) {
    const double eps = 1e-7;
    char b[240];
    if (!r.error.empty()) return r.error;
    // 1. the pool serves one layer at a time
    std::vector<std::pair<double, double>> spans;
    for (const auto& L : r.recorded_layers)
        if (L.pool_end > L.pool_start) spans.push_back({L.pool_start, L.pool_end});
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i)
        if (spans[i].first < spans[i - 1].second - eps) {
            std::snprintf(b, sizeof b, "the pool served two layers at once: [%.4f, %.4f) and [%.4f, %.4f)", spans[i - 1].first, spans[i - 1].second, spans[i].first, spans[i].second);
            return b;
        }
    // 2. a layer's pool call starts after its GPU phase; a card is held by one lane at a time
    for (const auto& L : r.recorded_layers)
        if (L.pool_start < L.gpu_end - eps || L.pool_end < L.pool_start - eps) return "a pool call before its layer's GPU phase ended";
    for (int c = 0; c < 2; ++c) {
        std::vector<std::pair<double, double>> h;
        for (const auto& H : r.recorded_holds) if (H.card == c) h.push_back({H.start, H.end});
        std::sort(h.begin(), h.end());
        for (size_t i = 1; i < h.size(); ++i)
            if (h[i].first < h[i - 1].second - eps) {
                std::snprintf(b, sizeof b, "card %d held by two lanes at once at %.4f", c, h[i].first);
                return b;
            }
    }
    // 3. per lane and window: layers in order, a layer starts after the previous one's pool result, stage 1 after stage 0's last layer
    struct Key { int lane, win, stage; bool operator<(const Key& o) const { return lane != o.lane ? lane < o.lane : win != o.win ? win < o.win : stage < o.stage; } };
    std::vector<std::pair<Key, const TwoLaneResult::Layer*>> v;
    for (const auto& L : r.recorded_layers) v.push_back({{L.lane, L.win, L.stage}, &L});
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& c) { return a.first < c.first; });
    for (size_t i = 1; i < v.size(); ++i) {
        const auto &a = v[i - 1], &c = v[i];
        const bool same = !(a.first < c.first) && !(c.first < a.first);
        if (same) {
            if (c.second->layer != a.second->layer + 1) return "a stage ran its layers out of order";
            if (c.second->gpu_start < a.second->pool_end - eps) return "a layer started before the previous layer's pool result";
        } else if (c.first.lane == a.first.lane && c.first.win == a.first.win && c.first.stage == 1 && a.first.stage == 0) {
            if (c.second->gpu_start < a.second->pool_end - eps) return "stage 1 started before stage 0's last layer was done";
        }
    }
    // 4. a lane never has two windows at once: a window's first layer starts after the lane's last one ended
    {
        std::vector<double> last_end(4, -1);
        std::vector<int> last_win(4, -1);
        std::vector<const TwoLaneResult::Layer*> byt;
        for (const auto& L : r.recorded_layers) byt.push_back(&L);
        std::stable_sort(byt.begin(), byt.end(), [](const auto* a, const auto* c) { return a->gpu_start < c->gpu_start; });
        for (const auto* L : byt) {
            if (L->lane < 0 || L->lane >= 4) continue;
            if (L->win != last_win[(size_t) L->lane]) {
                if (L->stage != 0 || L->layer != 0) return "a window did not start at stage 0 layer 0";
                if (L->gpu_start < last_end[(size_t) L->lane] - eps) return "a lane started a window before its last one ended";
                last_win[(size_t) L->lane] = L->win;
            }
            last_end[(size_t) L->lane] = std::max(last_end[(size_t) L->lane], L->pool_end);
        }
    }
    // 5. no rate above the capacity of any one resource (pool, each card): all lanes' windows in the interval
    (void) p;
    return "";
}

// ------------------------------------------------------------------------------------------------------------------
// Presets, from the serve log of the production engine (lm-server /opt/strata-bmtp/strata-q4xl-prod.log, 217 `strata batch:`
// reports; the recordings in /opt/strata-monitor/).  "warm" = the reports whose routed entries were >= 85% in VRAM, "typical" =
// 70-85% (the tier's usual state with an agent's long contexts).  Each line's source is in docs/TWO_LANE_MODEL.md.
// ------------------------------------------------------------------------------------------------------------------

/// A lane's window = today's batch window of ONE slot with one proposal (2 rows):
///   warm: 33.5 ms = stage 0 (wait 9.39 + pool 6.20) + stage 1 (wait 8.76 + pool 4.93) + commit 0.60 + emit 1.42 + 2.2 other, 1.89 tokens
///   typical: 38.5 ms = stage 0 (10.33 + 8.37) + stage 1 (9.42 + 6.21) + commit 0.62 + emit 1.37 + 2.2 other, 1.83 tokens
inline TwoLaneParams two_lane_lane_preset(bool warm) {
    TwoLaneParams p;
    p.rows = p.rows_ref = 2.0;
    p.accept = 0.87;
    if (warm) {
        p.gpu_total[0] = 9.39; p.gpu_total[1] = 8.76;
        p.experts[0] = 2.64; p.experts[1] = 1.97;    // (pool ms per layer - base) / k: 6.20/24, 4.93/24
    } else {
        p.gpu_total[0] = 10.33; p.gpu_total[1] = 9.42;
        p.experts[0] = 3.80; p.experts[1] = 2.70;    // 8.37/24, 6.21/24
        p.accept = 0.83;
    }
    p.fixed_host_ms = 0.9; p.draft_ms = 1.1; p.fixed_lat_ms = 2.2; p.verdict_ms = 0.1;
    return p;
}

/// Today's merged-row batch window of TWO slots (4 rows, one lane, no contention):
///   warm: 43.8 ms = stage 0 (11.25 + 10.29) + stage 1 (10.33 + 7.31) + commit 1.14 + emit 1.84 + 1.6 other, 3.38 tokens
///   typical: 57.4 ms = stage 0 (11.29 + 17.91) + stage 1 (10.40 + 13.08) + commit 1.07 + emit 2.16 + 1.5 other, 3.51 tokens
inline TwoLaneParams two_lane_merged_preset(bool warm) {
    TwoLaneParams p;
    p.lanes = 1;
    p.rows = p.rows_ref = 4.0;
    p.pool_k_ms = 0.087;
    if (warm) {
        p.gpu_total[0] = 11.25; p.gpu_total[1] = 10.33;
        p.experts[0] = 4.42; p.experts[1] = 2.93;    // (10.29/24 - 0.05) / 0.087, (7.31/24 - 0.05) / 0.087
        p.tokens_override = 3.38;
        p.fixed_host_ms = 1.14 + 1.84 - 1.4; p.draft_ms = 1.4; p.fixed_lat_ms = 1.6;
    } else {
        p.gpu_total[0] = 11.29; p.gpu_total[1] = 10.40;
        p.experts[0] = 8.0; p.experts[1] = 5.9;      // 17.91/24, 13.08/24
        p.tokens_override = 3.51;
        p.fixed_host_ms = 1.07 + 2.16 - 1.4; p.draft_ms = 1.4; p.fixed_lat_ms = 1.5;
    }
    p.verdict_ms = 0.1;
    return p;
}

/// Today's solo window on the serve path (no batch): stage waits 7.64 / 7.44 + pool 6.08 / 4.45 (the log's per-stage lines, one
/// user, warm) = 25.6, with ~2.4 of draft, commit and emit around it: 28 ms and 2.08 tokens (2.67 rows).
inline TwoLaneParams two_lane_serve_solo_preset() {
    TwoLaneParams p;
    p.lanes = 1;
    p.rows = p.rows_ref = 2.67;
    p.gpu_total[0] = 7.64; p.gpu_total[1] = 7.44;
    p.experts[0] = (6.08 / 24 - 0.05) / 0.0787; p.experts[1] = (4.45 / 24 - 0.05) / 0.0787;
    p.tokens_override = 2.08;
    p.fixed_host_ms = 0.5; p.draft_ms = 1.9; p.fixed_lat_ms = 0.0; p.verdict_ms = 0.0;
    return p;
}

}  // namespace strata::program
