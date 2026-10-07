// strata/core/door_latency.hpp - the host's answer time to the GPU's doorbell, as a histogram.
//
// Every layer of a verify window rings the host (the doorbell kernel bumps a mapped counter) and then waits for flag A, the
// plan of the layer's VRAM experts.  The time from the moment the host thread SEES the ring to the moment it raises flag
// A is what the GPU waits for ("waitA", plus the 3-5 us the flag takes to travel).  Physics says ~8 us; a host thread that
// is preempted, or a pool that is slow to start its work, shows up as a long tail.  This counts it per layer in 1 us bins
// (the last bin takes everything slower), cheap enough to stay on: one clock read pair and one increment per layer.
// Counters only grow; a request's numbers are the difference of two snapshots (`since`).
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core {

struct DoorLatency {
    static constexpr int kBins = 2048;   ///< 1 us each; bin kBins-1 holds everything from 2047 us up
    uint64_t n = 0;
    uint64_t sum_ns = 0;
    uint32_t bin[kBins] = {};

    void add(int64_t ns) {
        if (ns < 0) ns = 0;
        const int64_t us = ns / 1000;
        ++bin[us >= kBins ? kBins - 1 : us];
        ++n;
        sum_ns += (uint64_t) ns;
    }
    void merge(const DoorLatency& o) {
        n += o.n;
        sum_ns += o.sum_ns;
        for (int i = 0; i < kBins; ++i) bin[i] += o.bin[i];
    }

    struct Stats {
        uint64_t n = 0;
        double min_us = 0, avg_us = 0, p99_us = 0, max_us = 0;   ///< min/p99/max are bin centres (1 us resolution)
        uint64_t slow = 0;                                        ///< layers in the last bin (>= 2047 us)
    };

    /// The layers counted since `before` (a copy taken earlier of this histogram).
    Stats since(const DoorLatency& before) const {
        Stats s;
        s.n = n - before.n;
        if (s.n == 0) return s;
        s.avg_us = (double) (sum_ns - before.sum_ns) / 1000.0 / (double) s.n;
        const uint64_t need = (s.n * 99 + 99) / 100;   // ceil(0.99 n): the 99th percentile's rank
        uint64_t seen = 0;
        bool have_min = false, have_p99 = false;
        for (int i = 0; i < kBins; ++i) {
            const uint32_t c = bin[i] - before.bin[i];
            if (c == 0) continue;
            if (!have_min) { s.min_us = i + 0.5; have_min = true; }
            seen += c;
            if (!have_p99 && seen >= need) { s.p99_us = i + 0.5; have_p99 = true; }
            s.max_us = i + 0.5;
        }
        s.slow = bin[kBins - 1] - before.bin[kBins - 1];
        return s;
    }

    static std::string describe(const Stats& s) {
        if (s.n == 0) return "no layer served";
        char b[160];
        std::snprintf(b, sizeof b, "min %.1f / avg %.1f / p99 %.1f / max %.1f us over %llu layers%s", s.min_us, s.avg_us,
                      s.p99_us, s.max_us, (unsigned long long) s.n, s.slow ? " (some over 2 ms)" : "");
        return b;
    }
};

}  // namespace strata::core
