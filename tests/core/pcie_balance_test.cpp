// --pcie-balance (include/strata/core/pcie_balance.hpp): the per-layer PCIe share chosen from the pool's and the link's
// costs - the pure choice against a brute force, the --pcie-frac bound, and the estimator's cold start, hysteresis,
// contention handling and exploration.  Header-only, CPU.
#include "strata/core/pcie_balance.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
using strata::core::PcieBalance;
using strata::core::pcie_balance_cap;
using strata::core::pcie_balance_pick;

// the layer time the pick minimises, written out independently of the implementation
double layer_time(int nmiss, int m, double tc, double tp, double tc_busy) {
    return m == 0 ? nmiss * tc : std::max(m * tp, (nmiss - m) * tc_busy);
}

// a stage that has seen `n` layers of `tc` (ms per MiB, per-layer noise +-`noise`) and a link of `tp`
PcieBalance warmed(double tc, double tp, int n = 64, double noise = 0.0, unsigned seed = 1) {
    PcieBalance b;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    b.set_pcie(tp);
    for (int i = 0; i < n; ++i) b.note_cpu(tc * (1.0 + noise * u(rng)), 5, false);
    b.enabled = true;
    return b;
}
}  // namespace

int main() {
    // ---- the --pcie-frac bound: ceil(nmiss * frac), 0 = never, capped at nmiss
    {
        check(pcie_balance_cap(5, 0) == 0, "frac 0: no PCIe");
        check(pcie_balance_cap(0, 26) == 0, "no misses, no share");
        check(pcie_balance_cap(5, 26) == 1, "frac 0.1, 5 misses: 1");
        check(pcie_balance_cap(10, 26) == 2, "frac 0.1, 10 misses: 2 (ceil of 1.02)");
        check(pcie_balance_cap(1, 26) == 1, "frac 0.1, 1 miss: 1 (ceil)");
        check(pcie_balance_cap(10, 256) == 10, "frac 1.0: every miss");
        check(pcie_balance_cap(7, 128) == 4, "frac 0.5, 7 misses: 4");
        check(pcie_balance_cap(3, 300) == 3, "never more than the misses");
    }

    // ---- the pick on the box's numbers: a PCIe expert 0.27 ms, a pool expert 0.075 ms (worked by hand, 3% tie)
    {
        const double tc = 0.075, tp = 0.27;
        const int want[] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4};   // nmiss 0..20
        for (int n = 0; n <= 20; ++n) {
            const int m = pcie_balance_pick(n, n, tc, tp);
            char what[96];
            std::snprintf(what, sizeof what, "pick(nmiss %d) = %d, got %d", n, want[n], m);
            check(m == want[n], what);
        }
        check(pcie_balance_pick(5, 5, tc, tp) == 1, "solo (5 misses): one expert over PCIe");
        check(pcie_balance_pick(10, 10, tc, tp) == 2, "batch (10 misses): two");
        // a flat 0.32 share at 10 misses is 3 copies = 0.81 ms, over the CPU's 0.53: the balance stays at 2
        check(pcie_balance_pick(10, 3, tc, tp) == 2, "a bound of 3 is not used when 2 is faster");
        check(layer_time(10, 3, tc, tp, tc) > layer_time(10, 2, tc, tp, tc), "3 copies are slower than 2");
    }

    // ---- the pick against a brute force, ties to the smaller m; the bound; degenerate costs
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> cost(0.01, 2.0), busy_k(1.0, 2.0);
        for (int it = 0; it < 20000; ++it) {
            const int nmiss = 1 + (int) (rng() % 40), cap = (int) (rng() % 45);
            const double tc = cost(rng), tp = cost(rng), tcb = tc * busy_k(rng);
            const int m = pcie_balance_pick(nmiss, cap, tc, tp, 0.03, tcb);
            const int top = std::min(cap, nmiss);
            check(m >= 0 && m <= top, "m within [0, min(cap, nmiss)]");
            double best = layer_time(nmiss, 0, tc, tp, tcb);
            for (int k = 1; k <= top; ++k) best = std::min(best, layer_time(nmiss, k, tc, tp, tcb));
            check(layer_time(nmiss, m, tc, tp, tcb) <= best * 1.03 + 1e-12, "within the tie of the best");
            // the smallest m inside the tie: nothing smaller is within it
            for (int k = 0; k < m; ++k)
                check(layer_time(nmiss, k, tc, tp, tcb) > best * 1.03, "a smaller m was within the tie");
        }
        check(pcie_balance_pick(5, 0, 0.075, 0.27) == 0, "bound 0 (--pcie-frac 0): never");
        check(pcie_balance_pick(5, 5, 0.0, 0.27) == 0, "unknown pool cost: no share");
        check(pcie_balance_pick(5, 5, 0.075, 0.0) == 0, "unknown link cost: no share");
        check(pcie_balance_pick(0, 5, 0.075, 0.27) == 0, "no misses");
        check(pcie_balance_pick(10, 10, 0.075, 5.0) == 0, "a link slower than the pool: none");
        check(pcie_balance_pick(10, 4, 0.075, 0.0001) == 4, "a very fast link: the bound");
        int prev = 0;   // more misses do not mean fewer copies (the tie rule can step back by one on a plateau)
        for (int n = 1; n <= 64; ++n) {
            const int m = pcie_balance_pick(n, n, 0.075, 0.27);
            check(m + 1 >= prev, "the share does not fall by more than one as the misses grow");
            prev = std::max(prev, m);
        }
        // the pool can be slower while the link reads the same RAM: that can take the benefit away
        check(pcie_balance_pick(5, 5, 0.075, 0.27, 0.03, 0.075) == 1, "no contention: 1");
        check(pcie_balance_pick(5, 5, 0.075, 0.27, 0.03, 0.095) == 0, "the pool 27% slower while the GPU reads: none");
    }

    // ---- the estimator: nothing until measured, then the pick
    {
        const double mib = 3.07;
        PcieBalance b;
        b.enabled = true;
        check(!b.ready(), "cold: not ready");
        check(b.choose(10, 10, mib) == 0, "cold: no share");
        b.set_pcie(0.088);   // ms per MiB: 0.27 ms for a 3.07 MiB expert
        check(!b.ready(), "link only: not ready");
        for (int i = 0; i < 15; ++i) b.note_cpu(0.0245, 5, false);
        check(!b.ready(), "15 pool samples: not ready");
        b.note_cpu(0.0245, 5, false);
        check(b.ready(), "link and 16 pool samples: ready");
        check(b.choose(5, 5, mib) == 1, "5 misses: 1");
        check(b.choose(10, 10, mib) == 2, "10 misses: 2");
        check(b.choose(10, 1, mib) == 1, "bounded by the cap");
        check(b.choose(10, 0, mib) == 0, "cap 0");
        b.enabled = false;
        check(b.choose(10, 10, mib) == 0, "off: no share");
        // too few experts in a layer: no cost sample (its wake-up cost is per layer, not per expert)
        PcieBalance c = warmed(0.0245, 0.088);
        const double before = c.cpu_idle_ms_per_mib();
        for (int i = 0; i < 200; ++i) c.note_cpu(5.0, 2, false);
        check(c.cpu_idle_ms_per_mib() == before, "2-expert layers give no sample");
        for (int i = 0; i < 200; ++i) c.note_cpu(-1.0, 8, false);
        check(c.cpu_idle_ms_per_mib() == before, "a non-positive sample is ignored");
    }

    // ---- hysteresis: noise in the samples does not move the choice
    {
        const double mib = 3.07;
        // the pick flips between 0 and 1 copies at 7 misses when the pool's cost passes 0.2781 / 7 ms an expert (the
        // link's 0.270 ms plus the 3% tie): sit on that edge with +-4% noise on every sample
        const double edge0 = 0.2781 / 7.0 / mib;
        for (double edge : {edge0 * 0.98, edge0, edge0 * 1.02}) {
            PcieBalance b = warmed(edge, 0.088, 64, 0.04, 3);
            std::mt19937 rng(11);
            std::uniform_real_distribution<double> u(-1.0, 1.0);
            int first = -1, flips = 0;
            for (int i = 0; i < 4000; ++i) {
                b.note_cpu(edge * (1.0 + 0.04 * u(rng)), 5, false);
                const int m = b.choose(7, 7, mib);
                if (first < 0) first = m;
                if (m != first) { ++flips; first = m; }
            }
            char what[96];
            std::snprintf(what, sizeof what, "no flapping on the edge (tc %.5f): %d flips", edge, flips);
            check(flips == 0, what);
        }
        // a real change of more than the band is followed
        PcieBalance b = warmed(0.0245, 0.088);
        check(b.choose(10, 10, mib) == 2, "baseline 2");
        for (int i = 0; i < 400; ++i) b.note_cpu(0.0245 * 2.0, 5, false);   // the pool twice as slow
        check(b.choose(10, 10, mib) > 2, "a slower pool takes more copies");
        for (int i = 0; i < 400; ++i) b.note_cpu(0.0245 * 0.5, 5, false);   // and half as slow as at first
        check(b.choose(10, 10, mib) < 2, "a faster pool takes fewer");
        // one stalled layer (a preempted worker) does not move the held cost
        PcieBalance s = warmed(0.0245, 0.088);
        const double held = s.cpu_idle_ms_per_mib();
        s.note_cpu(0.0245 * 50.0, 5, false);
        check(s.cpu_idle_ms_per_mib() == held, "one 50x sample does not move the held cost");
        // a re-probe within the band keeps the held link cost; a larger move takes it
        PcieBalance p = warmed(0.0245, 0.088);
        p.set_pcie(0.088 * 1.05);
        check(p.pcie_ms_per_mib() == 0.088, "5% link change is held");
        p.set_pcie(0.088 * 1.5);
        check(p.pcie_ms_per_mib() > 0.12, "50% link change is taken");
    }

    // ---- contention: the pool measured slower while the GPU reads, and the exploration that lets it recover
    {
        const double mib = 3.07;
        PcieBalance b = warmed(0.0245, 0.088);
        check(b.choose(5, 5, mib) == 1, "idle cost only: 1 copy at 5 misses");
        for (int i = 0; i < 40; ++i) b.note_cpu(0.0245 * 1.6, 4, true);   // 60% slower with a copy in flight
        check(b.cpu_busy_ms_per_mib() > b.cpu_idle_ms_per_mib() * 1.4, "busy cost measured");
        int shared = 0, explored_at = -1;
        for (int i = 1; i <= 200; ++i) {
            const int m = b.choose(5, 5, mib);
            if (m > 0) { ++shared; if (explored_at < 0) explored_at = i; }
        }
        check(b.explored == 3, "the busy cost is re-tested every 64th layer that would otherwise skip the link");
        check(shared == (int) b.explored, "only the exploration reads, otherwise the CPU keeps them");
        check(explored_at == PcieBalance::kExploreEvery, "first exploration at the 64th layer");
        // the contention goes away: the busy samples drop back and the share returns
        for (int i = 0; i < 400; ++i) b.note_cpu(0.0245 * 1.02, 4, true);
        check(b.choose(5, 5, mib) == 1, "recovered");
        // nothing is explored when even an idle pool would not use the link
        PcieBalance q = warmed(0.0245, 2.0);
        for (int i = 0; i < 40; ++i) q.note_cpu(0.0245 * 1.6, 4, true);
        for (int i = 0; i < 300; ++i) (void) q.choose(5, 5, mib);
        check(q.explored == 0, "no exploration against a slow link");
    }

    if (failures == 0) std::printf("pcie_balance_test: OK\n");
    return failures == 0 ? 0 : 1;
}
