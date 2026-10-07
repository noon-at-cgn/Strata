#include "strata/program/pipeline_gate.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::program;

#define CHECK(c, ...)                                                    \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s  (", __FILE__, __LINE__, #c); \
            std::fprintf(stderr, __VA_ARGS__);                           \
            std::fprintf(stderr, ")\n");                                 \
            return 1;                                                    \
        }                                                                \
    } while (0)

static int64_t roundup(int64_t x, int64_t c) { return (x + c - 1) / c * c; }

int main() {
    int64_t checks = 0;

    // ---- 1. the serial reason: all 2^5 flag combinations x pw x ready, against an independent table ----
    for (int ready = 0; ready < 2; ++ready)
        for (int pw : {0, 1, 2})
            for (int bits = 0; bits < 16; ++bits) {
                PipeRequest r{ready != 0, pw, (bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0, (bits & 8) != 0};
                const char* why = pipeline_serial_reason(r);
                const bool ok = ready && pw >= 2 && bits == 0;
                CHECK((why == nullptr) == ok, "ready=%d pw=%d bits=%d why=%s", ready, pw, bits, why ? why : "null");
                CHECK(pipeline_not_asked(r) == (!ready || pw < 2), "not_asked ready=%d pw=%d", ready, pw);
                if (why != nullptr) CHECK(std::strlen(why) > 0, "empty reason");
                // each hindrance alone gives its own, distinct message; "not asked" ones come first
                if (ready && pw >= 2 && bits != 0) {
                    const char* want = (bits & 1) ? "batch slots decoding beside it"
                                     : (bits & 2) ? "a request waiting for a slot"
                                     : (bits & 4) ? "repetition penalties (penalty_last_n)" : "coupled draft sampling";
                    CHECK(std::strcmp(why, want) == 0, "bits=%d got %s", bits, why);
                }
                ++checks;
            }
    {   // distinct strings for distinct single hindrances
        std::vector<const char*> seen;
        for (int bit = 0; bit < 4; ++bit) {
            PipeRequest r{true, 2, bit == 0, bit == 1, bit == 2, bit == 3};
            const char* why = pipeline_serial_reason(r);
            CHECK(why != nullptr, "single hindrance %d gives no reason", bit);
            for (const char* s : seen) CHECK(std::strcmp(s, why) != 0, "duplicate reason %s", why);
            seen.push_back(why);
        }
    }

    // ---- 2. the KV plan over a grid ----
    const int64_t chunks[] = {1, 7, 64, 4096};
    const int64_t maxctx[] = {1000, 4096, 8192, 100000, 524288};
    for (int64_t chunk : chunks)
        for (int64_t mc : maxctx)
            for (int64_t reserved : {int64_t(0), int64_t(1), chunk, 2 * chunk, int64_t(1000), int64_t(4096), mc / 2, mc, mc + 5})
                for (int64_t window_end = 1; window_end <= mc + 40; window_end += (window_end < 70 ? 1 : window_end < 5000 ? 331 : 4093))
                    for (int fl = 0; fl < 4; ++fl) {
                        const bool inflight = fl & 1, spec = fl & 2;
                        const PipeKv p = pipeline_kv_plan(reserved, window_end, inflight, spec, chunk, mc);
                        ++checks;
                        // never reserve while anything is in flight
                        CHECK(!(inflight && p.reserve_now), "reserve with a window in flight");
                        // never reserve for a speculative window
                        CHECK(!(spec && p.reserve_now), "reserve for a speculative window");
                        // launch only if covered, by what is reserved or by what is reserved now
                        if (p.launch) {
                            if (p.reserve_now) CHECK(window_end <= p.reserve_cells, "launch not covered by the reservation");
                            else CHECK(window_end <= reserved, "launch past the reservation (end %lld reserved %lld)",
                                       (long long) window_end, (long long) reserved);
                        }
                        if (p.reserve_now) {
                            CHECK(p.launch, "reserve without launch");
                            CHECK(p.reserve_cells > reserved, "reservation does not grow");
                            CHECK(p.reserve_cells <= mc, "reservation above max_context");
                            const int64_t want = std::min(mc, roundup(window_end + 8, chunk) + chunk);
                            CHECK(p.reserve_cells == want, "reserve %lld want %lld", (long long) p.reserve_cells, (long long) want);
                            CHECK(p.reserve_cells % chunk == 0 || p.reserve_cells == mc, "reservation not on a chunk");
                        } else {
                            CHECK(p.reserve_cells == 0, "reserve_cells set without reserve_now");
                        }
                        // covered windows always launch (nothing to wait for)
                        if (window_end <= reserved) CHECK(p.launch && !p.reserve_now, "covered window gated");
                        // speculative past the reservation: gated
                        if (spec && window_end > reserved) CHECK(!p.launch, "speculative past the reservation launched");
                        // verified past the reservation: waits while something flies, else reserves (if it can fit)
                        if (!spec && window_end > reserved) {
                            if (inflight) CHECK(!p.launch, "verified window launched uncovered while in flight");
                            else if (window_end <= mc) CHECK(p.launch && p.reserve_now, "drained verified window not reserved");
                            else CHECK(!p.launch && !p.reserve_now, "window beyond max_context launched");
                        }
                    }

    // ---- 3. a decode walk: the pool is grown only at drain points, about once per chunk of tokens ----
    {
        const int64_t chunk = 4096, mc = 524288;
        int64_t reserved = 0, pos = 100, reserves = 0, gated = 0, drains = 0;
        bool inflight = false;
        while (pos < 200000) {
            const int64_t we = pos + 4;                       // a window of 4 rows
            PipeKv p = pipeline_kv_plan(reserved, we, inflight, false, chunk, mc);
            if (!p.launch && inflight) { inflight = false; ++drains; p = pipeline_kv_plan(reserved, we, false, false, chunk, mc); }
            CHECK(p.launch, "stuck at pos %lld", (long long) pos);
            if (p.reserve_now) { CHECK(!inflight, "reserved in flight"); reserved = p.reserve_cells; ++reserves; }
            CHECK(we <= reserved, "launched uncovered");
            inflight = true;
            // the speculative window behind it
            const PipeKv s = pipeline_kv_plan(reserved, we + 4, true, true, chunk, mc);
            if (!s.launch) ++gated;
            CHECK(!s.reserve_now, "speculative reserved");
            pos += 2;
        }
        CHECK(reserves >= 200000 / (2 * chunk) && reserves <= 200000 / chunk + 2, "reserves %lld", (long long) reserves);
        CHECK(drains == reserves - 1 || drains == reserves, "drains %lld reserves %lld", (long long) drains, (long long) reserves);
        // a window of 4 rows that advances 2 gates at most the 2-3 speculative launches whose end falls past the edge
        CHECK(gated <= 3 * reserves, "gated %lld of %lld reservations", (long long) gated, (long long) reserves);
        std::printf("walk: %lld reservations, %lld drains, %lld gated speculative launches over 100k windows\n",
                    (long long) reserves, (long long) drains, (long long) gated);
    }

    std::printf("pipeline_gate_test: %lld checks passed\n", (long long) checks);
    return 0;
}
