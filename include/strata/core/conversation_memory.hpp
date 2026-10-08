#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace strata::core {

// The RAM a runtime allocation (parking a conversation, saving or restoring a session file) may still take.
//
// available_host_bytes() is the smallest of
//   * MemAvailable (Windows: the available physical memory);
//   * the room under every memory limit of this process's cgroup and of each ancestor visible under
//     /sys/fs/cgroup (cgroup v2: memory.max and memory.high, each minus usage; cgroup v1:
//     memory.limit_in_bytes minus memory.usage_in_bytes);
//   * the operator's own cap, when one is set (--memory-limit-mib / STRATA_MEMORY_LIMIT_MIB): that many MiB minus
//     what the container uses now - the top visible cgroup's memory.current, or MemTotal - MemAvailable when the
//     cgroup files cannot be read.  For a container whose real limit sits on a parent cgroup it cannot see.
//
// Usage is memory.current as the kernel reports it, less one reclaimable kind: the clean inactive file cache of the
// same group (memory.stat inactive_file, v1 total_inactive_file, less dirty and writeback pages, never more than
// memory.current), which the kernel gives back at memory.high or the limit before it kills anything.  Nothing else is
// credited: active_file holds mlocked pages, and shmem is the pinned complements and the KV pool.  A memory.stat that
// is missing or unparsable gives no credit (it does not make the sample unknown).
// Pinned, locked and shared memory stays charged in full: a hard-limit kill is worse than a skipped park.
//
// Unknown telemetry is deliberately distinct from a measured zero.  A limit file that exists but cannot be parsed
// makes the sample unknown (fail closed); a file that is absent (no such controller, not mounted) is ignored.
// Not a reservation: other writers can take the room after the sample.

enum class MemorySource { meminfo, cgroup, flag };

const char* memory_source_name(MemorySource source);

/// One sample.  `available` is the smallest term; `source`, `limit` and `current` describe the term that was the
/// smallest (meminfo: MemTotal and MemTotal - MemAvailable; cgroup: that group's limit and memory.current; flag: the
/// operator's cap and the usage it was measured against).  `mem_available` is MemAvailable by itself (0 if unknown).
struct HostMemoryReading {
    uint64_t available = 0;
    MemorySource source = MemorySource::meminfo;
    uint64_t limit = 0;
    uint64_t current = 0;        ///< memory.current (meminfo: MemTotal - MemAvailable), before the credit
    uint64_t credit = 0;         ///< the clean inactive file cache counted as reclaimable (0 for meminfo)
    uint64_t mem_available = 0;
};

/// The files the sampler reads; the defaults are the real ones (tests point them at a fixture directory).
struct MemoryProbePaths {
    std::string meminfo = "/proc/meminfo";
    std::string self_cgroup = "/proc/self/cgroup";
    std::string cgroup_root = "/sys/fs/cgroup";
};

/// The operator's cap in bytes for the container/cgroup this engine runs in (0 = none).  Set once at start.
void set_memory_limit_bytes(uint64_t bytes);
uint64_t memory_limit_bytes();

/// Linux: reads `paths`.  `explicit_limit` 0 = no operator cap.  nullopt when the RAM cannot be determined.
std::optional<HostMemoryReading> sample_host_memory(const MemoryProbePaths& paths, uint64_t explicit_limit);

/// The real files (Windows: GlobalMemoryStatusEx) and the cap set by set_memory_limit_bytes().
std::optional<HostMemoryReading> sample_host_memory();

std::optional<uint64_t> available_host_bytes();

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

// STRATA_PARK_FAST=1|2 (opt-in, default off; glibc only).  Parking a conversation, saving a checkpoint and moving a
// conversation into a batch slot copy a few hundred MB into freshly allocated host vectors; the cost of that copy is the
// first-touch page faults of the new memory (~1.5 GB/s), not the bytes moved.  A freed 237 MB vector goes back to the
// kernel (glibc mmaps big blocks and munmaps them on free), so every copy pays the faults again.  Armed, the process
// takes its big blocks from the brk heap and keeps up to `retain_bytes` of freed memory (M_MMAP_MAX=0, M_TOP_PAD,
// M_TRIM_THRESHOLD), and the next copy lands on pages that are already faulted in.  PROCESS-WIDE: every large
// allocation made after the call is affected, not only the parking ones; call it once, after the model is loaded
// (memory that is cudaHostRegister'ed must keep its own mapping).  Returns false (and changes nothing) when `retain_bytes`
// is 0 or the allocator is not glibc's.  `retain_bytes` is clamped to [64 MiB, 1.5 GiB] (mallopt takes an int).
// glibc gives every other thread its own arena, and an arena that is not the main one ignores all of this (a big block
// there is still mmap'ed and munmap'ed): `all_threads` (STRATA_PARK_FAST=2) also sets M_ARENA_MAX=1 so that the stage
// threads' checkpoint parts and every other thread share the main arena (one malloc lock for the whole process).
bool conversation_retain_freed_memory(uint64_t retain_bytes, bool all_threads, std::string& note);

// The environment form: STRATA_PARK_FAST=1 arms it for the allocations of the main thread (the request loop: parks,
// checkpoints and slot moves), =2 for every thread, with STRATA_PARK_FAST_RETAIN_MIB (default 1024).  Returns whether
// it armed; `note` says what happened (empty when STRATA_PARK_FAST is unset or 0, nothing was touched).
bool conversation_retain_freed_memory_from_env(std::string& note);

// The retention cap armed above, 0 when not armed.
uint64_t conversation_retained_cap_bytes();

// Free bytes the armed heap holds and a park can reuse without a new page (glibc mallinfo2 fordblks, at most the cap);
// 0 when not armed.  sample_host_memory() adds it to `available`: the retained pages are not free for the kernel, but
// they are free for the engine's own next copy, so the park admission must not count them against itself.
uint64_t conversation_retained_free_bytes();

} // namespace strata::core
