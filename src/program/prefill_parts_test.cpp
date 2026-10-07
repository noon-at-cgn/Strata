#include "strata/program/prefill_parts.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace strata::program;

// the chunk end positions a segment [a0, b0) is read in with k chunks a part
static std::vector<int64_t> chunk_ends(int64_t a0, int64_t b0, int64_t chunk, int64_t k, int64_t* parts) {
    std::vector<int64_t> ends;
    *parts = 0;
    for (int64_t q = a0; q < b0;) {
        const int64_t r = prefill_part_end(q, b0, chunk, k);
        if (prefill_part_chunks(q, r, chunk) > k) return {};
        for (int64_t c = q; c < r; c += chunk) ends.push_back(std::min(c + chunk, r));
        q = r;
        ++*parts;
    }
    return ends;
}

int main() {
    int checks = 0;
    // 1. every k reads the same chunks (same positions, same sizes, same remainder) as one chunk a part
    for (const int64_t chunk : {1024, 4096, 8192})
        for (const int64_t a0 : {0, 100, 8192, 22016})
            for (const int64_t len : {1, 1000, 8192, 8193, 3 * 8192, 100000, 100001}) {
                int64_t p1 = 0;
                const std::vector<int64_t> ref = chunk_ends(a0, a0 + len, chunk, 1, &p1);
                if (ref.empty() || p1 != (int64_t) ref.size()) return 1;
                for (int64_t k = 2; k <= 16; ++k) {
                    int64_t pk = 0;
                    const std::vector<int64_t> got = chunk_ends(a0, a0 + len, chunk, k, &pk);
                    if (got != ref) return 2;
                    if (pk != ((int64_t) ref.size() + k - 1) / k) return 3;   // k chunks a part, the last has the rest
                    ++checks;
                }
            }
    // 2. the decode scale: 1 chunk -> 1; k chunks -> 2k/(k+1); the per-token decode time k >= 1 gives is k = 1's
    //    when the part takes (k + 1) stage times and one chunk takes 2
    if (prefill_part_decode_scale(0) != 1.0 || prefill_part_decode_scale(1) != 1.0) return 4;
    for (int64_t k = 2; k <= 16; ++k) {
        const double part_time = double(k + 1), one_run_time = 2.0 * double(k);
        if (std::fabs(prefill_part_decode_scale(k) * part_time - one_run_time) > 1e-9) return 5;
        ++checks;
    }
    if (std::fabs(prefill_part_decode_scale(2) - 4.0 / 3.0) > 1e-12 || prefill_part_decode_scale(3) != 1.5) return 6;
    // 3. the knob: the request's value wins over the environment's, both within 1..16, 0 = not asked
    if (prefill_pipe_k(0, 1) != 1 || prefill_pipe_k(0, 3) != 3 || prefill_pipe_k(2, 3) != 2 ||
        prefill_pipe_k(99, 1) != 16 || prefill_pipe_k(0, 99) != 16 || prefill_pipe_k(0, 0) != 1 ||
        prefill_pipe_k(-5, -5) != 1) return 7;
    checks += 7;
    // 4. degenerate inputs never loop forever or go backwards
    if (prefill_part_end(5, 100, 0, 0) != 6 || prefill_part_end(5, 5, 8, 2) != 5 ||
        prefill_part_chunks(7, 7, 8) != 0 || prefill_part_chunks(8, 7, 8) != 0) return 8;
    checks += 4;
    std::printf("prefill_parts_test OK (%d checks)\n", checks);
}
