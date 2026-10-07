// --pcie-balance: how many of a layer's missed experts the GPU reads over PCIe, chosen from what each side costs.
//
// A verify window's missed experts are computed either by the CPU pool or by the GPU after it read them over PCIe.
// Both run at the same time, so a layer waits for the slower side:
//
//     layer time(m) = max(m * t_pcie, (nmiss - m) * t_cpu)         m = the experts the GPU reads
//
// The fixed share (--pcie-frac) ignores both costs.  This picks the m that minimises the time, from a running
// estimate of t_cpu (the pool's time per expert, measured on every layer) and t_pcie (one DMA probe per link).
//
// Pure C++, header only: the m choice and the estimator are unit-tested without a GPU (tests/core/pcie_balance_test.cpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::core {

/// The most experts a layer may send over PCIe when `pcie_num`/256 (--pcie-frac) is the upper bound:
/// ceil(nmiss * pcie_num / 256).  0 stays 0 (--pcie-frac 0: never use PCIe).
inline int pcie_balance_cap(int nmiss, int pcie_num) {
    if (nmiss <= 0 || pcie_num <= 0) return 0;
    return std::min(nmiss, (int) (((int64_t) nmiss * pcie_num + 255) >> 8));
}

/// The m in [0, min(cap, nmiss)] with the smallest layer time.  With the GPU reading nothing the pool takes
/// `nmiss * t_cpu`; with it reading m > 0 the layer takes max(m * t_pcie, (nmiss - m) * t_cpu_busy), where t_cpu_busy
/// (default t_cpu) is the pool's per-expert cost while the link reads the same RAM (it can be higher).  A saving under
/// `tie` (fraction) over a smaller m is not taken: the smallest m within `tie` of the best wins, so near-equal choices
/// resolve to fewer PCIe copies, not to whichever the noise favoured.  Costs in any one unit; all must be > 0.
inline int pcie_balance_pick(int nmiss, int cap, double t_cpu, double t_pcie, double tie = 0.03, double t_cpu_busy = -1.0) {
    cap = std::min(cap, nmiss);
    if (t_cpu_busy <= 0.0) t_cpu_busy = t_cpu;
    if (cap <= 0 || !(t_cpu > 0.0) || !(t_pcie > 0.0)) return 0;
    auto cost = [&](int m) {
        return m == 0 ? (double) nmiss * t_cpu : std::max((double) m * t_pcie, (double) (nmiss - m) * t_cpu_busy);
    };
    double best = cost(0);
    for (int m = 1; m <= cap; ++m) best = std::min(best, cost(m));
    for (int m = 0; m <= cap; ++m)
        if (cost(m) <= best * (1.0 + tie)) return m;
    return 0;
}

/// One stage's running costs and its choice.  All costs are ms per MiB of expert blob (a layer's own blob size
/// scales them, so layers with different formats share one estimate).  Single-threaded: the host thread that
/// runs the verify window's pool callback is the only one that touches it.
class PcieBalance {
public:
    bool enabled = false;       ///< this request's switch (--pcie-balance / the request's pcie_balance key)
    double ref_mib = 0.0;       ///< a typical blob of this stage, for reporting ms per expert

    /// The CPU pool's cost of one expert on a layer with `njobs` experts to compute, `ms_per_mib` per MiB of blob.
    /// `pcie_active`: the same layer also had the GPU read experts (the pool and the link share the RAM, so the
    /// pool can be slower then).  Layers with fewer than `kMinJobs` experts are not taken: their per-layer cost
    /// (waking the workers) would read as a per-expert cost.
    void note_cpu(double ms_per_mib, int njobs, bool pcie_active) {
        if (njobs < kMinJobs || !(ms_per_mib > 0.0)) return;
        Track& t = pcie_active ? busy_ : idle_;
        t.add(ms_per_mib);
    }

    /// The link's cost of one MiB (a DMA probe): the held value moves only when the new one differs by `kBand`.
    void set_pcie(double ms_per_mib) {
        if (!(ms_per_mib > 0.0)) return;
        if (pcie_held_ <= 0.0 || std::fabs(ms_per_mib - pcie_held_) > kBand * pcie_held_) pcie_held_ = ms_per_mib;
        pcie_last_ = ms_per_mib;
    }

    /// The CPU side has been measured and the link too: until then the share is 0 (the fixed rule is not used
    /// either: balance on means the estimate decides).
    bool ready() const { return pcie_held_ > 0.0 && idle_.n >= kMinSamples; }

    /// m for a layer with `nmiss` missed experts, at most `cap` (from pcie_balance_cap), blobs of `blob_mib`.
    int choose(int nmiss, int cap, double blob_mib) {
        if (!enabled || !ready() || nmiss <= 0 || cap <= 0) return 0;
        // the pool's cost with nothing read over PCIe (idle), and while the GPU reads (busy: once measured; before
        // that the idle cost)
        const double cpu_idle = idle_.held, cpu_busy = busy_.n >= kMinSamples ? busy_.held : idle_.held;
        int m = pcie_balance_pick(nmiss, cap, cpu_idle * blob_mib, pcie_held_ * blob_mib, kTie, cpu_busy * blob_mib);
        ++layers;
        if (m == 0 && busy_.n >= kMinSamples && nmiss > kMinJobs &&
            pcie_balance_pick(nmiss, cap, cpu_idle * blob_mib, pcie_held_ * blob_mib, kTie) > 0) {
            // the busy cost made the share not worth it: every kExploreEvery-th such layer reads one expert anyway,
            // so the busy cost keeps being measured and can recover (otherwise it would stay at its last, worst value)
            if (++skipped_ >= kExploreEvery) { skipped_ = 0; m = 1; ++explored; }
        }
        moved += m;
        return m;
    }

    double cpu_idle_ms_per_mib() const { return idle_.held; }
    double cpu_busy_ms_per_mib() const { return busy_.n >= kMinSamples ? busy_.held : 0.0; }
    double pcie_ms_per_mib() const { return pcie_held_; }
    double pcie_last_ms_per_mib() const { return pcie_last_; }

    int64_t layers = 0;         ///< layers `choose` decided while ready
    int64_t moved = 0;          ///< experts it chose to send over PCIe
    int64_t explored = 0;       ///< of those, layers that read one expert only to refresh the busy cost

    static constexpr int kMinJobs = 3;          ///< fewest CPU experts of a layer that gives a cost sample
    static constexpr int kMinSamples = 16;      ///< samples before a cost is trusted
    static constexpr int kExploreEvery = 64;
    static constexpr double kBand = 0.08;       ///< a held cost follows its average once they differ by this much
    static constexpr double kAlpha = 1.0 / 16.0;
    static constexpr double kTie = 0.03;        ///< a saving under this fraction does not move a copy to the link

private:
    struct Track {
        double ewma = 0.0, held = 0.0;
        int64_t n = 0;
        void add(double x) {
            if (n == 0) ewma = x;
            else {
                x = std::min(x, 2.0 * ewma);   // one stalled layer (a preempted worker) is not a cost
                ewma += kAlpha * (x - ewma);
            }
            ++n;
            if (held <= 0.0 || std::fabs(ewma - held) > kBand * held) held = ewma;
        }
    };
    Track idle_, busy_;
    double pcie_held_ = 0.0, pcie_last_ = 0.0;
    int skipped_ = 0;
};

}  // namespace strata::core
