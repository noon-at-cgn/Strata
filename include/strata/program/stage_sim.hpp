#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// A software model of the two layer-split stages of one decode window and of --pipeline-windows 2, to predict what
// overlapping them can give and to choose the pool's service order (header-only, CPU; no engine code is used).
//
// The model (all times in ms; the numbers come from the serve log's per-stage lines):
//  * a window runs on stage 0 (the first card, `layers[0]` layers) and then on stage 1 (the second card).  In a layer
//    the stage's GPU runs `gpu[stage][l]`, then the layer's experts are computed by the ONE shared CPU pool, which
//    serves ONE layer at a time (`cpu[stage][l]`, non-preemptive), and only when that result is back does the next layer
//    start.  A stage runs one window at a time (its cards have one stream); the two stages run side by side.
//  * serial: window k+1 starts when window k's verdict is in (the stages take turns).  Its time is
//    S0 + S1 + verdict + fixed: with nothing overlapped the pool is never contended.
//  * pipelined: window k+1's stage 0 starts right behind window k's stage 0, on the guess that k is accepted whole
//    (the generate.cpp loop's B window).  A window is, by a deterministic equidistributed draw, ON the path (the guess
//    held: stage 0 saved its time), DOOMED (a wrong guess was run first: when the verdict of k says so, the doomed
//    window's remaining layers are skipped, stage 0 goes back by `undo_ms`, and the real window starts), or fresh (no
//    speculation).  Stage 1 only ever runs a verified window: it starts after the window's stage 0 AND after the
//    verdict of the window before it.  At most two windows are in flight; a speculative one needs the chain of the
//    verdict before the last (`host_ready`).
//  * pool policies, which decide who gets the pool when both stages wait for it:
//      Pipelined: stage 0 first (what the loop does today: it serves stage 0's windows before stage 1's);
//      Priority:  stage 1 first (the verified window is the one the next verdict waits for; WI-4's order);
//      Quantum:   take turns, a layer each;
//      Fifo:      the layer that asked first.
//    Serial has no overlap at all.
namespace strata::program {

enum class StagePolicy { Serial, Pipelined, Priority, Quantum, Fifo };

inline const char* stage_policy_name(StagePolicy p) {
    switch (p) {
        case StagePolicy::Serial: return "serial";
        case StagePolicy::Pipelined: return "stage0-first";
        case StagePolicy::Priority: return "priority";
        case StagePolicy::Quantum: return "quantum";
        case StagePolicy::Fifo: return "fifo";
    }
    return "?";
}

struct StageSimParams {
    std::vector<double> gpu[2];   ///< GPU ms of each layer of stage s (before the layer's pool call)
    std::vector<double> cpu[2];   ///< the shared pool's ms for each layer of stage s
    double fixed_ms = 0.0;        ///< host work after a verdict before the next window can start (commit, emit, drafter chain)
    double verdict_ms = 0.0;      ///< from a window's last stage-1 layer to its verdict being read
    double undo_ms = 0.5;         ///< a wrong guess: restore the GDN snapshot and replay the commit, before the real window
    double skip_ms = 0.0;         ///< a skipped (doomed, killed) layer's remaining GPU time
    double hop_ms = 0.0;          ///< stage 0 to stage 1 hand-off
    int pipeline_windows = 2;     ///< informational (the model has 2 in flight)
    double jitter = 0.0;          ///< each layer's GPU and pool time is scaled by 1 + jitter * u, u uniform in [-1, 1] (mean 1)
    uint64_t seed = 1;            ///< the draws are hashed from (seed, window, stage, layer), the same in every policy
    double overlap_stretch = 1.0; ///< the pool's time for a layer while the other stage has a window in flight (memory
                                  ///< bandwidth, the PCIe copies of the other card: >= 1; to be fitted to a live measurement)

    int layers(int s) const { return (int) gpu[s].size(); }
    double gpu_total(int s) const { double t = 0; for (double v : gpu[s]) t += v; return t; }
    double cpu_total(int s) const { double t = 0; for (double v : cpu[s]) t += v; return t; }
    double stage_total(int s) const { return gpu_total(s) + cpu_total(s); }
    double pool_total() const { return cpu_total(0) + cpu_total(1); }
};

/// Uniform layers: stage s has n[s] layers of gpu ms and cpu ms each, from per-stage totals.
inline StageSimParams stage_sim_uniform(int n0, double gpu0, double cpu0, int n1, double gpu1, double cpu1) {
    StageSimParams p;
    p.gpu[0].assign((size_t) n0, gpu0 / n0);
    p.cpu[0].assign((size_t) n0, cpu0 / n0);
    p.gpu[1].assign((size_t) n1, gpu1 / n1);
    p.cpu[1].assign((size_t) n1, cpu1 / n1);
    return p;
}

/// The report's solo numbers (the brief): window 40.6 = stage 0 22.9 (pool 12.3) + stage 1 17.7 (pool 7.8), 24 + 24 layers.
inline StageSimParams stage_sim_brief() { return stage_sim_uniform(24, 22.9 - 12.3, 12.3, 24, 17.7 - 7.8, 7.8); }

/// The server recording (20261007-021355), one solo request at 190k context: 43.7 ms a window =
/// stage 0 (GPU wait 10.42 + pool 14.07) + stage 1 (9.35 + 9.38) + 0.48 of commit/emit/draft.
inline StageSimParams stage_sim_recording() {
    StageSimParams p = stage_sim_uniform(24, 10.42, 14.07, 24, 9.35, 9.38);
    p.fixed_ms = 43.7 - (10.42 + 14.07 + 9.35 + 9.38);
    return p;
}

/// One layer run of one window on one stage.
struct SimLayer {
    int win = 0, stage = 0, layer = 0;
    double gpu_start = 0, gpu_end = 0;    ///< the layer's GPU phase
    double pool_start = 0, pool_end = 0;  ///< the shared pool's service of the layer (== gpu_end, zero long, when skipped)
    bool skipped = false;
};

struct SimWin {
    int k = 0;                ///< the verified window it is (a doomed one precedes verified window k)
    bool doomed = false;
    bool onpath = false;      ///< a verified window that ran speculatively and whose guess held
    double s0_start = -1, s0_end = -1, s1_start = -1, s1_end = -1, vd = -1;   ///< vd: its verdict (verified windows)
};

struct StageSimResult {
    std::string error;        ///< non-empty: the model deadlocked (a bug)
    int n = 0;                ///< verified windows
    int onpath = 0, doomed = 0, fresh = 0;
    double total_ms = 0;      ///< from 0 to the host being ready for the window after the last
    double ms_per_window = 0;
    double pool_busy_ms = 0;
    double stage_busy_ms[2] = {0, 0};
    std::vector<SimWin> wins;
    std::vector<SimLayer> layers;
};

/// max(S0, S1, P): no schedule goes below it (stage 0 runs all n windows in turn, stage 1 too, the pool serves all)
inline double stage_sim_floor(const StageSimParams& p) { return std::max({p.stage_total(0), p.stage_total(1), p.pool_total()}); }

/// The factor on one layer's GPU (what = 0) or pool (what = 1) time: 1 + jitter * u, u in [-1, 1) hashed (SplitMix64)
/// from the seed and which layer of which window on which stage it is, so every policy draws the same times.
inline double sim_noise(const StageSimParams& p, const SimWin& w, int stage, int layer, int what) {
    if (p.jitter <= 0.0) return 1.0;
    uint64_t z = p.seed + 0x9E3779B97F4A7C15ull * ((uint64_t) w.k * 2 + (w.doomed ? 1 : 0) + 1);
    z += 0xD1B54A32D192ED03ull * ((uint64_t) layer * 4 + (uint64_t) stage * 2 + (uint64_t) what + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const double u = (double) (z >> 11) / 9007199254740992.0;   // [0, 1)
    return 1.0 + p.jitter * (2.0 * u - 1.0);
}

/// Run `n` verified windows.  h: the share that are on the path; d: the share that run a doomed window first (the rest
/// are fresh).  Serial runs no speculation whatever h and d are.
inline StageSimResult stage_sim_run(const StageSimParams& p, StagePolicy pol, double h, double d, int n) {
    constexpr double INF = 1e300;
    StageSimResult res;
    res.n = n;
    if (n <= 0 || p.layers(0) == 0 || p.layers(1) == 0) return res;
    const bool spec = pol != StagePolicy::Serial;
    // ---- the windows and the order stage 0 runs them in: [doomed k,] verified k
    std::vector<int> vwin((size_t) n), dwin((size_t) n, -1), order0;
    for (int k = 0; k < n; ++k) {
        SimWin w;
        w.k = k;
        const double u = std::fmod(0.5 + (double) k * 0.6180339887498949, 1.0);   // equidistributed, reproducible
        const bool on = spec && k > 0 && u < h;
        const bool dm = spec && k > 0 && !on && u < h + d;
        if (dm) {
            SimWin dw;
            dw.k = k;
            dw.doomed = true;
            dwin[(size_t) k] = (int) res.wins.size();
            order0.push_back((int) res.wins.size());
            res.wins.push_back(dw);
            ++res.doomed;
        }
        w.onpath = on;
        (on ? res.onpath : res.fresh) += 1;
        vwin[(size_t) k] = (int) res.wins.size();
        order0.push_back((int) res.wins.size());
        res.wins.push_back(w);
    }
    auto vdv = [&](int k) { return k < 0 ? 0.0 : res.wins[(size_t) vwin[(size_t) k]].vd < 0 ? INF : res.wins[(size_t) vwin[(size_t) k]].vd; };
    auto s0end = [&](int k) { return k < 0 ? 0.0 : res.wins[(size_t) vwin[(size_t) k]].s0_end < 0 ? INF : res.wins[(size_t) vwin[(size_t) k]].s0_end; };
    auto hr = [&](int k) { const double v = vdv(k); return v >= INF ? INF : v + p.fixed_ms; };   // the host can start what k decides
    auto release0 = [&](int wi) -> double {
        const SimWin& w = res.wins[(size_t) wi];
        const int k = w.k;
        if (w.doomed || w.onpath) {   // behind the window before it: its stage 0 done and the verdict before that read
            const double a = s0end(k - 1), b = hr(k - 2);
            return a >= INF || b >= INF ? INF : std::max(a, b);
        }
        if (k == 0) return 0.0;
        double t = hr(k - 1);
        if (dwin[(size_t) k] >= 0) {   // the doomed one first: it has finished, the snapshot is back
            const double e = res.wins[(size_t) dwin[(size_t) k]].s0_end;
            if (e < 0) return INF;
            t = std::max(t, e + p.undo_ms);
        }
        return t;
    };
    auto release1 = [&](int k) -> double {
        const double a = s0end(k), b = vdv(k - 1);
        return a >= INF || b >= INF ? INF : std::max(a, b) + p.hop_ms;
    };
    auto skip_active = [&](int wi, double now) {   // the verdict that kills a doomed window has come
        const SimWin& w = res.wins[(size_t) wi];
        return w.doomed && vdv(w.k - 1) <= now;
    };
    // ---- the two stages' runners
    struct Run { int w = -1, layer = 0, phase = 0; double t_end = 0, arrive = 0; size_t rec = 0; };   // phase 0 idle, 1 gpu, 2 waits for the pool, 3 in the pool
    Run run[2];
    size_t next0 = 0;
    int next1 = 0, last_pool_stage = 1;
    double now = 0;
    auto begin_layer = [&](int s, double t) {
        Run& r = run[s];
        if (r.layer >= p.layers(s)) {   // the window is done
            SimWin& w = res.wins[(size_t) r.w];
            if (s == 0) w.s0_end = t;
            else { w.s1_end = t; w.vd = t + p.verdict_ms; }
            res.stage_busy_ms[s] += t - (s == 0 ? w.s0_start : w.s1_start);
            r.w = -1;
            r.phase = 0;
            return;
        }
        const bool sk = skip_active(r.w, t);
        SimLayer L;
        L.win = r.w; L.stage = s; L.layer = r.layer; L.skipped = false;
        L.gpu_start = t;
        L.gpu_end = t + (sk ? p.skip_ms : p.gpu[s][(size_t) r.layer] * sim_noise(p, res.wins[(size_t) r.w], s, r.layer, 0));
        L.pool_start = L.pool_end = L.gpu_end;
        r.rec = res.layers.size();
        res.layers.push_back(L);
        r.phase = 1;
        r.t_end = L.gpu_end;
    };
    auto start_window = [&](int s, int wi, double t) {
        Run& r = run[s];
        r.w = wi; r.layer = 0;
        SimWin& w = res.wins[(size_t) wi];
        (s == 0 ? w.s0_start : w.s1_start) = t;
        begin_layer(s, t);
    };
    long guard = 0;
    while (true) {
        bool changed;
        do {
            changed = false;
            for (int s = 0; s < 2; ++s) {
                Run& r = run[s];
                if (r.phase == 1 && r.t_end <= now) {
                    SimLayer& L = res.layers[r.rec];
                    if (skip_active(r.w, now)) {   // killed: no pool work for this layer
                        L.skipped = true;
                        ++r.layer;
                        begin_layer(s, now);
                    } else {
                        r.phase = 2;
                        r.arrive = r.t_end;
                    }
                    changed = true;
                } else if (r.phase == 3 && r.t_end <= now) {
                    res.layers[r.rec].pool_end = r.t_end;
                    ++r.layer;
                    begin_layer(s, now);
                    changed = true;
                }
            }
            if (run[0].phase != 3 && run[1].phase != 3) {   // the pool is free
                for (int s = 0; s < 2; ++s) {   // a killed window's waiting layer needs no pool
                    Run& r = run[s];
                    if (r.phase == 2 && skip_active(r.w, now)) {
                        res.layers[r.rec].skipped = true;
                        res.layers[r.rec].pool_start = res.layers[r.rec].pool_end = now;
                        ++r.layer;
                        begin_layer(s, now);
                        changed = true;
                    }
                }
                const bool w0 = run[0].phase == 2, w1 = run[1].phase == 2;
                if (w0 || w1) {
                    int s;
                    if (!(w0 && w1)) s = w0 ? 0 : 1;
                    else if (pol == StagePolicy::Pipelined) s = 0;
                    else if (pol == StagePolicy::Priority) s = 1;
                    else if (pol == StagePolicy::Quantum) s = 1 - last_pool_stage;
                    else s = run[1].arrive < run[0].arrive ? 1 : 0;   // Fifo, Serial
                    Run& r = run[s];
                    SimLayer& L = res.layers[r.rec];
                    L.pool_start = now;
                    const double stretch = run[1 - s].phase != 0 ? p.overlap_stretch : 1.0;   // the other stage has a window in flight
                    const double c = p.cpu[s][(size_t) r.layer] * sim_noise(p, res.wins[(size_t) r.w], s, r.layer, 1) * stretch;
                    r.phase = 3;
                    r.t_end = now + c;
                    res.pool_busy_ms += c;
                    last_pool_stage = s;
                    changed = true;
                }
            }
            if (run[0].phase == 0 && next0 < order0.size()) {
                const double rel = release0(order0[next0]);
                if (rel <= now) { start_window(0, order0[next0++], now); changed = true; }
            }
            if (run[1].phase == 0 && next1 < n) {
                const double rel = release1(next1);
                if (rel <= now) { start_window(1, vwin[(size_t) next1++], now); changed = true; }
            }
        } while (changed);
        if (next1 >= n && run[1].phase == 0) break;
        double nt = INF;
        for (int s = 0; s < 2; ++s)
            if (run[s].phase == 1 || run[s].phase == 3) nt = std::min(nt, run[s].t_end);
        if (run[0].phase == 0 && next0 < order0.size()) { const double r = release0(order0[next0]); if (r < INF) nt = std::min(nt, r); }
        if (run[1].phase == 0 && next1 < n) { const double r = release1(next1); if (r < INF) nt = std::min(nt, r); }
        if (nt >= INF || nt < now || ++guard > 50000000) {
            res.error = "the model made no progress at " + std::to_string(now) + " ms";
            return res;
        }
        now = nt;
    }
    res.total_ms = hr(n - 1);
    res.ms_per_window = res.total_ms / n;
    return res;
}

/// The model's own invariants over a finished run: "" when all hold, else the first violation.
inline std::string stage_sim_check(const StageSimParams& p, const StageSimResult& r) {
    const double eps = 1e-6;
    char b[200];
    if (!r.error.empty()) return r.error;
    // 1. the pool serves one layer at a time
    std::vector<std::pair<double, double>> spans;
    for (const SimLayer& L : r.layers)
        if (!L.skipped && L.pool_end > L.pool_start + 0.0) spans.push_back({L.pool_start, L.pool_end});
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i)
        if (spans[i].first < spans[i - 1].second - eps) {
            std::snprintf(b, sizeof b, "the pool served two layers at once: [%.4f, %.4f) and [%.4f, %.4f)", spans[i - 1].first,
                          spans[i - 1].second, spans[i].first, spans[i].second);
            return b;
        }
    double busy = 0;
    for (const auto& s : spans) busy += s.second - s.first;
    if (std::fabs(busy - r.pool_busy_ms) > 1e-3 + 1e-9 * r.layers.size()) return "the pool's busy time does not add up";
    // 2. a stage's layer l+1 never starts before its layer l's pool result (and a layer's pool call after its GPU phase)
    std::vector<long> last(2 * r.wins.size(), -1);   // per (window, stage): the index of its previous layer record
    for (size_t i = 0; i < r.layers.size(); ++i) {
        const SimLayer& L = r.layers[i];
        if (L.pool_start < L.gpu_end - eps) return "a pool call started before its layer's GPU phase ended";
        if (L.pool_end < L.pool_start - eps) return "a pool call ended before it started";
        long& prev = last[2 * (size_t) L.win + (size_t) L.stage];
        if (prev >= 0) {
            const SimLayer& P = r.layers[(size_t) prev];
            if (L.layer != P.layer + 1) return "a stage ran its layers out of order";
            if (L.gpu_start < P.pool_end - eps) return "a layer started before the previous layer's pool result";
        } else if (L.layer != 0) {
            return "a stage ran its layers out of order";
        }
        prev = (long) i;
    }
    // layers per (window, stage) are complete (all of them, in order) for every window that ran
    std::vector<int> cnt0(r.wins.size(), 0), cnt1(r.wins.size(), 0);
    for (const SimLayer& L : r.layers) (L.stage == 0 ? cnt0 : cnt1)[(size_t) L.win] += 1;
    for (size_t w = 0; w < r.wins.size(); ++w) {
        if (cnt0[w] != p.layers(0)) return "a window did not run all of stage 0's layers";
        if (r.wins[w].doomed ? cnt1[w] != 0 : cnt1[w] != p.layers(1)) return "stage 1 ran a wrong number of layers for a window";
    }
    // 3. stage 1 never runs an unverified window: only verified ones, after their stage 0, after the verdict before them
    std::vector<const SimWin*> v(r.n, nullptr);
    for (const SimWin& w : r.wins)
        if (!w.doomed) v[(size_t) w.k] = &w;
    for (int k = 0; k < r.n; ++k) {
        const SimWin& w = *v[(size_t) k];
        if (w.s1_start < w.s0_end - eps) return "stage 1 started a window before its stage 0 was done";
        if (k > 0 && w.s1_start < v[(size_t) k - 1]->vd - eps) return "stage 1 ran a window before the verdict of the one before it";
        if (k > 0 && w.s1_start < v[(size_t) k - 1]->s1_end - eps) return "stage 1 ran two windows at once";
        if (w.vd < w.s1_end - eps) return "a verdict before its window's stage 1 finished";
        // a speculative window is launched behind its predecessor's stage 0 and no sooner
        if (k > 0 && w.onpath && w.s0_start < v[(size_t) k - 1]->s0_end - eps) return "a window started on stage 0 before the window before it finished there";
        // the stage-0 launch of a non-speculative window waits for the verdict before it
        if (k > 0 && !w.onpath && w.s0_start < v[(size_t) k - 1]->vd - eps) return "a fresh window started before the verdict before it";
    }
    for (const SimWin& w : r.wins)
        if (w.doomed) {
            if (w.s1_start >= 0) return "stage 1 ran a doomed window";
            // it ends before the real window of its k starts there (stage 0 takes one window at a time)
            if (w.s0_end > v[(size_t) w.k]->s0_start + eps) return "a doomed window overlapped the real window on stage 0";
        }
    // a stage runs one window at a time
    std::vector<std::pair<double, double>> st0;
    for (const SimWin& w : r.wins) st0.push_back({w.s0_start, w.s0_end});
    std::sort(st0.begin(), st0.end());
    for (size_t i = 1; i < st0.size(); ++i)
        if (st0[i].first < st0[i - 1].second - eps) return "two windows ran on stage 0 at once";
    // 4. no schedule goes below max(S0, S1, P), of the layer times this run drew: stage 0 runs the verified windows one
    //    after the other, so does stage 1, and the pool serves every layer of every window one at a time
    {
        double s_[2] = {0, 0}, pool = 0;
        for (const SimLayer& L : r.layers) {
            if (!r.wins[(size_t) L.win].doomed) s_[L.stage] += (L.gpu_end - L.gpu_start) + (L.pool_end - L.pool_start);
            pool += L.pool_end - L.pool_start;
        }
        const double floor_ms = std::max({s_[0], s_[1], pool}) / r.n;
        if (r.ms_per_window < floor_ms - 1e-6) {
            std::snprintf(b, sizeof b, "%.3f ms a window is below the floor max(S0, S1, P) = %.3f", r.ms_per_window, floor_ms);
            return b;
        }
    }
    return "";
}

/// Print the predicted window ms (and tok/s, and the gain over serial) for each pool policy and each on-path share h.
/// `tokens` is the tokens per window (0: no tok/s).  The doomed share d is a flat rate of wrong speculative launches.
inline void stage_sim_print_table(std::FILE* f, const StageSimParams& p, double d, double tokens, int n,
                                  const std::vector<double>& hs = {0.3, 0.4, 0.5, 0.7}) {
    const StageSimResult ser = stage_sim_run(p, StagePolicy::Serial, 0, 0, n);
    std::fprintf(f, "stages: S0 %.2f ms (GPU %.2f + pool %.2f), S1 %.2f ms (GPU %.2f + pool %.2f), pool %.2f ms (%.0f%% of the serial "
                    "window), floor max(S0,S1,P) %.2f ms; fixed %.2f ms; doomed share d = %.2f; per-layer jitter +-%.0f%%, pool "
                    "stretch x%.2f while both stages run (model assumptions, not measured)\n",
                 p.stage_total(0), p.gpu_total(0), p.cpu_total(0), p.stage_total(1), p.gpu_total(1), p.cpu_total(1), p.pool_total(),
                 100.0 * p.pool_total() / std::max(ser.ms_per_window, 1e-9), stage_sim_floor(p), p.fixed_ms, d, 100.0 * p.jitter,
                 p.overlap_stretch);
    std::fprintf(f, "serial: %.2f ms/window\n", ser.ms_per_window);
    if (tokens > 0) std::fprintf(f, "        %.1f tok/s at %.2f tokens/window\n", 1000.0 * tokens / ser.ms_per_window, tokens);
    std::fprintf(f, "%-13s", "policy \\ h");
    for (double h : hs) std::fprintf(f, " %14.2f", h);
    std::fprintf(f, "\n");
    for (StagePolicy pol : {StagePolicy::Pipelined, StagePolicy::Priority, StagePolicy::Quantum, StagePolicy::Fifo}) {
        std::fprintf(f, "%-13s", stage_policy_name(pol));
        for (double h : hs) {
            const StageSimResult r = stage_sim_run(p, pol, h, d, n);
            const double gain = ser.ms_per_window / r.ms_per_window - 1.0;
            std::fprintf(f, tokens > 0 ? " %6.2f ms %+4.0f%% %5.1f t/s" : " %6.2f ms %+4.0f%%", r.ms_per_window, 100.0 * gain,
                         1000.0 * tokens / r.ms_per_window);
        }
        std::fprintf(f, "\n");
    }
}

}  // namespace strata::program
