#pragma once

#include <algorithm>
#include <cstdint>

// --pipeline-windows 2 beside --batch: two decisions the decode loop makes, kept apart from the engine so they can be
// tested on the CPU (header-only, like prefill_parts.hpp).
//
//  1. pipeline_serial_reason: may THIS request run in the pipelined loop, or does it decode serially (and why)?
//  2. pipeline_kv_plan: the shared KV pool (--kv-pool-tokens) backs the context in chunks.  Backing more cells ends in
//     KvPool::upload, which calls cudaDeviceSynchronize; with a window in flight that spins on a host flag served by
//     the very thread that would be inside the sync, that deadlocks.  So the pool is only grown at a DRAIN POINT, when
//     no window is in flight, and a speculative window never goes past what is already backed.
namespace strata::program {

/// What the pipelined loop needs to know about a request.
struct PipeRequest {
    bool pipe_ready;     ///< the two-stage pipeline exists (two verifiers per stage, the GDN snapshots, the drafter's chain)
    int pw;              ///< the request's --pipeline-windows (0 = off, 2 = two windows in flight)
    bool slots_active;   ///< a --batch slot holds a request that is decoding (the solo window would starve it)
    bool admission;      ///< a request waits for a slot (the loop must give way to its prompt read)
    bool penalties;      ///< repetition penalties (penalty_last_n): the next window's logits depend on this window's tokens
    bool coupled;        ///< coupled draft sampling: the drafter's draw is tied to the verifier's, no teacher-forced chain
};

/// nullptr when the request may run in the pipelined loop; else why it decodes serially.  The first two answers are
/// "not asked for" (the caller keeps quiet about them); every other answer is worth one log line per reason.
inline const char* pipeline_serial_reason(const PipeRequest& r) {
    if (!r.pipe_ready) return "no pipelined stages";
    if (r.pw < 2) return "--pipeline-windows below 2";
    if (r.slots_active) return "batch slots decoding beside it";
    if (r.admission) return "a request waiting for a slot";
    if (r.penalties) return "repetition penalties (penalty_last_n)";
    if (r.coupled) return "coupled draft sampling";
    return nullptr;
}

/// True when pipeline_serial_reason's answer is one of the two "not asked for" ones (nothing to say in the log).
inline bool pipeline_not_asked(const PipeRequest& r) { return !r.pipe_ready || r.pw < 2; }

/// What to do before launching a window that ends at row `window_end` (exclusive: it reads and writes cells < window_end).
struct PipeKv {
    bool launch;           ///< the window may be launched (after the reservation below, when reserve_now)
    bool reserve_now;      ///< back the pool up to `reserve_cells` first (the caller has drained: nothing is in flight)
    int64_t reserve_cells; ///< the pool's new size in cells when reserve_now (always > the old one), else 0
};

/// The pool plan for one launch.
///   reserved      cells the pool backs for this lane now
///   window_end    the window's last position + 1 (its drafts included)
///   any_in_flight any window on any stage still runs (a reservation now could deadlock)
///   speculative   the window is a guess launched behind a window that has not been verified yet
///   chunk         the pool's grow step in cells
///   max_context   the context limit (the pool never backs more)
/// Contract:
///   - launch only if window_end <= reserved, or the reservation reserve_now asks for covers window_end;
///   - a speculative window past the reservation is gated (launch false, no reservation: the loop drains, and the
///     verified window that follows plans the pool at the drain point);
///   - a verified window past the reservation with a window in flight waits (launch false, no reservation): the caller
///     drains and asks again; with nothing in flight it reserves min(max_context, roundup(window_end + 8, chunk) + chunk);
///   - never reserve while anything is in flight; never "reserve" less than the old reservation;
///   - a window that cannot fit in max_context never launches (the caller ends the request).
inline PipeKv pipeline_kv_plan(int64_t reserved, int64_t window_end, bool any_in_flight, bool speculative,
                               int64_t chunk, int64_t max_context) {
    if (window_end <= reserved) return {true, false, 0};
    if (speculative || any_in_flight) return {false, false, 0};
    if (window_end > max_context) return {false, false, 0};
    const int64_t c = std::max<int64_t>(chunk, 1);
    const int64_t target = std::min(max_context, (window_end + 8 + c - 1) / c * c + c);
    return {true, true, target};   // target >= window_end here: window_end <= max_context and the round-up is above it
}

}  // namespace strata::program
