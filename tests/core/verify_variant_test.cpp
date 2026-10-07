// tests/core/verify_variant_test.cpp - which window graph a verify window runs and what the graph holds, without a GPU.
//
// verify_variant.hpp holds the decisions the Verifier makes at each window's launch about the PCIe share's part of the
// graph (flag B's wait, the fetch kernels and the PCIe group) and the keys under which the batch graphs are kept.  The
// CUDA side only acts on them; the properties that keep a window correct are checked here:
//   * the variants cover the share exactly: a share of 0 selects the graph without the PCIe part, anything else the
//     graph with it (which also serves a share of 0);
//   * the all-resident graph is never replaced by the no-PCIe one;
//   * flag B is waited for exactly where it guards DMA copies (or when the legacy switch asks);
//   * a batch layout's keys differ per variant, the old (variant-free) key is the old key, and the commit graph of a
//     layout is shared by its variants and freed only with the last of them.
#include "strata/core/verify_variant.hpp"

#include <cstdio>
#include <map>
#include <set>
#include <vector>

using namespace strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-96s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("verify_variant_test: the PCIe share's part of the verify window graph\n");

    // ---- which graph: every (share off, all-resident graph) pair
    for (int off = 0; off < 2; ++off)
        for (int ar = 0; ar < 2; ++ar) {
            const bool np = window_without_pcie(off != 0, ar != 0);
            check(np == (off != 0 && ar == 0), "no-PCIe graph exactly when the share is 0 and the stage is not all-resident");
        }

    // ---- flag B: every (variant, pcie mode, legacy switch)
    for (int np = 0; np < 2; ++np)
        for (int mode = 0; mode < 3; ++mode)
            for (int legacy = 0; legacy < 2; ++legacy) {
                const bool w = window_waits_flag_b(np != 0, mode, legacy != 0);
                bool want = false;
                if (np == 0) want = mode == 0 || legacy != 0;   // B guards DMA copies (mode 0); the legacy switch keeps it everywhere
                if (w != want) {
                    std::printf("  FAIL flag B: np=%d mode=%d legacy=%d got %d\n", np, mode, legacy, (int) w);
                    ++g_fail;
                }
            }
    check(!window_waits_flag_b(true, 0, true), "the no-PCIe graph never waits for flag B, legacy switch or not");
    check(window_waits_flag_b(false, 0, false), "with the PCIe share, mode 0 (DMA) waits for the copies");
    check(!window_waits_flag_b(false, 1, false) && !window_waits_flag_b(false, 2, false),
          "with the PCIe share, modes 1 and 2 raise B straight after A: folded into A, no wait");
    check(window_waits_flag_b(false, 1, true) && window_waits_flag_b(false, 2, true), "STRATA_VERIFY_FLAGB=1 restores the wait");

    // ---- batch keys
    const int rows[8] = {0, 1, 1, 2, 0, 0, 0, 0};
    const std::vector<int> k00 = batch_graph_key(rows, 4, 1, false, false), k10 = batch_graph_key(rows, 4, 1, true, false),
                           k01 = batch_graph_key(rows, 4, 1, false, true), k11 = batch_graph_key(rows, 4, 1, true, true);
    check(k00 == std::vector<int>({1, 0, 1, 1, 2, 0}), "variant-free key = hand-off base, the slots, 0 (the key before the variant)");
    check(k10 == std::vector<int>({1, 0, 1, 1, 2, 1}), "doorbell-variant key is unchanged (...,1)");
    check(std::set<std::vector<int>>({k00, k10, k01, k11}).size() == 4, "the four variants of one layout have four keys");
    check(batch_graph_key(rows, 3, 1, false, false) != k00 && batch_graph_key(rows, 4, 0, false, false) != k00,
          "another size or hand-off base is another layout");

    // ---- the commit graph is shared by a layout's PCIe variants (and by nothing else)
    check(commit_graph_key(k00) == commit_graph_key(k01) && commit_graph_key(k10) == commit_graph_key(k11),
          "the commit key ignores the PCIe variant");
    check(commit_graph_key(k00) == k00 && commit_graph_key(k10) == k10, "... and is the old key when there is no PCIe variant bit");
    check(commit_graph_key(k00) != commit_graph_key(k10), "... but keeps the doorbell variant (as before)");
    check(commit_graph_key({}).empty(), "an empty key stays empty");

    // ---- eviction: the commit graph goes with the last window graph that uses it
    {
        std::map<std::vector<int>, int> windows;   // window key -> graph handle
        windows[k00] = 1;
        windows[k01] = 2;
        windows[batch_graph_key(rows, 3, 1, false, false)] = 3;
        check(commit_graph_still_used(windows, k00), "two variants of a layout: evicting one keeps the commit graph");
        check(commit_graph_still_used(windows, k01), "... either one");
        windows.erase(k01);
        check(!commit_graph_still_used(windows, k00), "the only graph of a layout: its commit graph goes with it");
        check(!commit_graph_still_used(windows, batch_graph_key(rows, 3, 1, false, false)),
              "another layout's commit graph is not kept by this one's");
        windows[k10] = 4;   // the doorbell variant of the same slots is a different commit graph (as it always was)
        check(!commit_graph_still_used(windows, k00), "a doorbell-variant graph does not hold the variant-free commit graph");
    }

    // ---- a whole request sequence: A/B pcie_frac 0 / 0.1 on one layout captures each variant once and one commit graph
    {
        std::map<std::vector<int>, int> windows, commits;
        int captured_windows = 0, captured_commits = 0;
        auto launch = [&](bool pcie_off, bool ar_on) {
            const bool np = window_without_pcie(pcie_off, ar_on);
            const std::vector<int> key = batch_graph_key(rows, 4, 1, false, np);
            if (windows.emplace(key, 1).second) ++captured_windows;
            if (commits.emplace(commit_graph_key(key), 1).second) ++captured_commits;
        };
        for (int r = 0; r < 6; ++r) launch(r % 2 == 0, false);   // 0, 0.1, 0, 0.1, ...
        check(captured_windows == 2 && captured_commits == 1, "alternating requests capture 2 window graphs and 1 commit graph, once each");
        launch(true, true);   // an all-resident stage: the (variant-free) graph, whatever the share
        launch(false, true);
        check(captured_windows == 2, "an all-resident stage never needs the no-PCIe graph");
    }

    if (g_fail != 0) {
        std::printf("verify_variant_test: %d FAILURES\n", g_fail);
        return 1;
    }
    std::printf("verify_variant_test: all ok\n");
    return 0;
}
