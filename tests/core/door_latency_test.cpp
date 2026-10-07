// door_latency_test: the doorbell -> flag A histogram (min / avg / p99 / max over a request's layers), no GPU.
#include "strata/core/door_latency.hpp"

#include <cmath>
#include <cstdio>

using strata::core::DoorLatency;

namespace {
int failures = 0;
#define CHECK(c)                                                                                                  \
    do {                                                                                                          \
        if (!(c)) {                                                                                               \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);                                   \
            ++failures;                                                                                           \
        }                                                                                                         \
    } while (0)
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
}  // namespace

int main() {
    DoorLatency h;
    // nothing counted
    {
        const DoorLatency::Stats s = h.since(DoorLatency{});
        CHECK(s.n == 0);
        CHECK(DoorLatency::describe(s) == "no layer served");
    }
    // 100 layers: 99 of 10 us (10.0-10.999), one of 500 us
    for (int i = 0; i < 99; ++i) h.add(10'500);
    h.add(500'250);
    {
        const DoorLatency::Stats s = h.since(DoorLatency{});
        CHECK(s.n == 100);
        CHECK(near(s.min_us, 10.5));
        CHECK(near(s.p99_us, 10.5));     // the 99th of 100 is still a 10 us layer
        CHECK(near(s.max_us, 500.5));
        CHECK(std::fabs(s.avg_us - (99 * 10.5 + 500.25) / 100.0) < 1e-9);
        CHECK(s.slow == 0);
    }
    // 101 layers: now two slow ones among 101 -> the 99th percentile's rank (ceil(0.99 * 101) = 100) is a slow one
    h.add(900'000);
    {
        const DoorLatency::Stats s = h.since(DoorLatency{});
        CHECK(s.n == 101 && near(s.p99_us, 500.5));
    }
    // a request's numbers are the difference of two snapshots
    const DoorLatency before = h;
    for (int i = 0; i < 10; ++i) h.add(3'000 + i * 100);   // 3.0 .. 3.9 us
    {
        const DoorLatency::Stats s = h.since(before);
        CHECK(s.n == 10);
        CHECK(near(s.min_us, 3.5));
        CHECK(near(s.max_us, 3.5));
        CHECK(std::fabs(s.avg_us - 3.45) < 1e-9);
        CHECK(DoorLatency::describe(s).find("over 10 layers") != std::string::npos);
    }
    // negative (a clock step) counts as 0; the slow bin takes everything from 2047 us up
    DoorLatency g;
    g.add(-5);
    g.add(5'000'000);
    g.add(2'047'000);
    g.add(2'046'999);
    {
        const DoorLatency::Stats s = g.since(DoorLatency{});
        CHECK(s.n == 4 && s.slow == 2 && near(s.min_us, 0.5) && near(s.max_us, DoorLatency::kBins - 0.5));
        CHECK(DoorLatency::describe(s).find("over 2 ms") != std::string::npos);
    }
    // merge adds histograms
    DoorLatency a, b;
    a.add(1'000);
    b.add(2'000);
    b.add(2'000);
    a.merge(b);
    {
        const DoorLatency::Stats s = a.since(DoorLatency{});
        CHECK(s.n == 3 && near(s.min_us, 1.5) && near(s.max_us, 2.5) && std::fabs(s.avg_us - 5.0 / 3.0) < 1e-9);
    }
    if (failures) {
        std::fprintf(stderr, "door_latency_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("door_latency_test: ok\n");
    return 0;
}
