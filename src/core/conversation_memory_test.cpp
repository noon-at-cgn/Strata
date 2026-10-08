#include "strata/core/conversation_memory.hpp"

#include "strata/core/conversation_buffer.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = available_host_bytes();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
#if defined(_WIN32)
    // Unknown must fail closed in production, but must not let a broken Windows provider pass this test.
    // Compare with total physical memory, not a second available reading: other processes can allocate
    // between calls. No pressure allocation or fixed free-memory assumption is needed.
    MEMORYSTATUSEX physical{};
    physical.dwLength = sizeof physical;
    check(GlobalMemoryStatusEx(&physical) != 0, "Windows physical-memory API is available");
    check(available.has_value() && *available <= physical.ullTotalPhys,
          "Windows provider returns a known, physically bounded sample");
#endif
#if !defined(_WIN32)
    // STRATA_PARK_FAST is opt-in: unset or 0 touches nothing (no cap, no credit, no note); the mallopt hook arms only on 1.
    {
        std::string note;
        unsetenv("STRATA_PARK_FAST");
        check(!conversation_retain_freed_memory_from_env(note) && note.empty() &&
              conversation_retained_cap_bytes() == 0 && conversation_retained_free_bytes() == 0,
              "park fast: unset env arms nothing");
        setenv("STRATA_PARK_FAST", "0", 1);
        check(!conversation_retain_freed_memory_from_env(note) && note.empty() && conversation_retained_cap_bytes() == 0,
              "park fast: STRATA_PARK_FAST=0 arms nothing");
        std::string refused;
        check(!conversation_retain_freed_memory(0, false, refused) && conversation_retained_cap_bytes() == 0,
              "park fast: a zero retention is refused");
        // a vector this size is mmap'ed and munmap'ed by glibc's defaults: nothing stays behind
        { std::vector<uint8_t> big(96u << 20, 0x11); }
        check(conversation_retained_free_bytes() == 0, "park fast: not armed, no credit");
        setenv("STRATA_PARK_FAST", "1", 1);
        setenv("STRATA_PARK_FAST_RETAIN_MIB", "256", 1);
        const bool armed = conversation_retain_freed_memory_from_env(note);
#if defined(__GLIBC__)
        check(armed && !note.empty() && conversation_retained_cap_bytes() == (256ull << 20), "park fast: STRATA_PARK_FAST=1 arms 256 MiB");
        // freed big blocks now stay in the heap and are what the credit counts, never above the cap
        { std::vector<uint8_t> big(96u << 20, 0x22); }
        const uint64_t kept = conversation_retained_free_bytes();
        check(kept >= (64ull << 20) && kept <= (256ull << 20), "park fast: a freed 96 MiB block is retained and credited, within the cap");
        const auto reading = sample_host_memory();
        check(!reading || reading->available <= reading->limit || reading->limit == 0, "park fast: the credit never exceeds the limit");
        // the copies' bytes do not depend on where the memory came from: a buffer across segment boundaries round-trips exactly
        ConversationBuffer buffer;
        const size_t n = 3 * ConversationBuffer::segment_bytes + 12345;
        buffer.resize(n);
        check(buffer.visit(0, n, [](uint8_t* p, size_t count, size_t at) {
            for (size_t i = 0; i < count; ++i) p[i] = (uint8_t) ((at + i) * 131u + ((at + i) >> 17));
            return true;
        }), "park fast: buffer fill");
        std::vector<uint8_t> back(n);
        check(buffer.read(back.data(), 0, n), "park fast: buffer read back");
        bool same = true;
        for (size_t i = 0; i < n && same; ++i) same = back[i] == (uint8_t) (i * 131u + (i >> 17));
        check(same, "park fast: bytes identical across segment boundaries");
        ConversationBuffer copy = buffer;   // the parked image's copy
        check(copy == buffer, "park fast: copied buffer equal");
        // level 2 also sets M_ARENA_MAX=1, so the other threads' big blocks come from the main arena too
        setenv("STRATA_PARK_FAST", "2", 1);
        check(conversation_retain_freed_memory_from_env(note) && note.find("every thread") != std::string::npos,
              "park fast: STRATA_PARK_FAST=2 arms every thread");
#else
        check(!armed && conversation_retained_cap_bytes() == 0, "park fast: needs glibc, nothing armed elsewhere");
#endif
    }
#endif
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
