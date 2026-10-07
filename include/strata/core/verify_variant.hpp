// include/strata/core/verify_variant.hpp - which captured window graph a verify window runs, and what it holds.
//
// Pure C++ (no CUDA): the decisions the Verifier makes before it launches a window, in one place so a test can
// drive them without a GPU (tests/core/verify_variant_test.cpp).
//
// THE PCIe SHARE'S PART OF A WINDOW GRAPH.  After a layer's VRAM groups, the graph by default
//   (1) waits for flag B (the host raises it once the PCIe share's copies have landed),
//   (2) in --pcie-mode 2 stages the share with two kernels (fetch_blobs, rebase_ptrs), and
//   (3) launches the grouped expert kernel over the PCIe groups.
// When the share is 0 these are dead work: the plan has no PCIe group, the launch finds nothing, and the wait passes
// at once.  A window graph is captured once and replayed, so "the share is 0" has to select a different captured
// graph: the *no-PCIe variant*, which holds none of the three, and whose windows the host plans with a share of 0
// (`GpuPlanSink::no_pcie`) and never raises flag B for.
//
// Flag B is also folded into flag A where it guards nothing: in --pcie-mode 1 and 2 nothing is DMA-copied by the
// host, `fetch` raises B at once (n = 0), straight after A, and everything B could guard (the plan's PCIe arrays) is
// written before A.  Only mode 0 (the copy engine) has copies that B must wait for.
#pragma once

#include <cstddef>
#include <vector>

namespace strata::core {

/// The window runs the no-PCIe variant: the request's share is 0 (`pcie_off`) and the stage is not on the
/// all-resident graph (that one has no doorbell, no flag B and no PCIe group at all, whatever the share).
inline bool window_without_pcie(bool pcie_off, bool all_resident_graph) { return pcie_off && !all_resident_graph; }

/// The graph waits for flag B before its PCIe groups.  Never in the no-PCIe variant; otherwise only where B guards
/// DMA copies (--pcie-mode 0), or when `legacy_wait_b` (STRATA_VERIFY_FLAGB=1) keeps the old wait everywhere.
inline bool window_waits_flag_b(bool without_pcie, int pcie_mode, bool legacy_wait_b) {
    return !without_pcie && (pcie_mode == 0 || legacy_wait_b);
}

/// The key of a captured batch window graph: the slot layout (hand-off base, then the slot of each row) and the
/// graph variant (bit 0: the doorbell graph of a stage that is otherwise all-resident, #871; bit 1: no PCIe share).
/// With neither bit set the key is the one the engine used before the variant existed.
inline std::vector<int> batch_graph_key(const int* rows, int S, int hbase, bool ar_off, bool no_pcie) {
    std::vector<int> k;
    k.reserve((size_t) S + 2);
    k.push_back(hbase);
    for (int t = 0; t < S; ++t) k.push_back(rows[t]);
    k.push_back((ar_off ? 1 : 0) | (no_pcie ? 2 : 0));
    return k;
}

/// The batch commit graph holds no PCIe-share work, so both variants of a window graph share one commit graph: its
/// key is the window's with the no-PCIe bit cleared.
inline std::vector<int> commit_graph_key(std::vector<int> window_key) {
    if (!window_key.empty()) window_key.back() &= ~2;
    return window_key;
}

/// True when some window graph in `windows` (any map keyed by window key) other than `leaving` still uses the commit
/// graph `leaving` shares - so evicting `leaving` must keep it.
template <class Map>
bool commit_graph_still_used(const Map& windows, const std::vector<int>& leaving) {
    const std::vector<int> ck = commit_graph_key(leaving);
    for (const auto& kv : windows)
        if (kv.first != leaving && commit_graph_key(kv.first) == ck) return true;
    return false;
}

}  // namespace strata::core
