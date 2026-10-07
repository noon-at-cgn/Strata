#pragma once

#include <algorithm>
#include <cstdint>

// A prompt read beside decoding batch slots goes in parts, and the slots decode between two parts.  With a layer
// split a part of k chunks is ONE prompt run, in which the stages overlap their chunks: k chunks take about
// (k + 1) stage times where k runs of one chunk take 2k.  These are the part's arithmetic (header-only, CPU).
namespace strata::program {

/// The end of the part that starts at `q`: k chunks of `chunk` tokens, or what is left of the segment [.., b0).
/// The chunk positions do not depend on k (a part starts where the last ended, k chunks on).
inline int64_t prefill_part_end(int64_t q, int64_t b0, int64_t chunk, int64_t k) {
    return std::min(b0, q + std::max<int64_t>(k, 1) * std::max<int64_t>(chunk, 1));
}

/// The chunks in [q, r).
inline int64_t prefill_part_chunks(int64_t q, int64_t r, int64_t chunk) {
    const int64_t c = std::max<int64_t>(chunk, 1);
    return r > q ? (r - q + c - 1) / c : 0;
}

/// The factor on the decode share for a part of `chunks` chunks: the slots get the share of the time the same
/// chunks would have taken one run each (2 stage times a chunk) instead of the part's (chunks + 1 stage times), so
/// their decode time per prompt token is what one chunk per part gives.  Equal stages assumed.  1 for one chunk.
inline double prefill_part_decode_scale(int64_t chunks) {
    return chunks > 1 ? 2.0 * double(chunks) / double(chunks + 1) : 1.0;
}

/// The chunks per part: the request's (> 0) else the environment's, within 1..16.
inline int64_t prefill_pipe_k(int64_t request_k, int64_t env_k) {
    return std::clamp<int64_t>(request_k > 0 ? request_k : env_k, 1, 16);
}

}  // namespace strata::program
