// lookahead_recall_test: the router lookahead's recall counter (STRATA_LOOKAHEAD_RECALL) on synthetic data, no GPU.
//  1. RecallScorer against a hand-worked window (a fake router whose predictions are known);
//  2. window boundaries, a changed token count and the reset;
//  3. the threaded LookaheadRecall with real BF16 routers: when layer l+d's true routing IS the router applied to
//     x_l, every expert is recalled; with everything resident no miss is counted; the report text.
#include "strata/core/lookahead_recall.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace strata::core;

namespace {
int failures = 0;
#define CHECK(c)                                                                                                  \
    do {                                                                                                          \
        if (!(c)) {                                                                                               \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);                                   \
            ++failures;                                                                                           \
        }                                                                                                         \
    } while (0)

constexpr int64_t kExperts = 16;

// predicts experts {c, c + 1} for target layer L, c = x[t * n_embd] + L: the two highest logits
void fake_router(int64_t layer, const float* x, int nt, float* logits, int64_t n_embd) {
    for (int t = 0; t < nt; ++t) {
        const int c = ((int) x[(size_t) t * (size_t) n_embd] + (int) layer) % (int) kExperts;
        for (int64_t e = 0; e < kExperts; ++e) logits[(size_t) t * (size_t) kExperts + (size_t) e] = 0.0f;
        logits[(size_t) t * (size_t) kExperts + (size_t) c] = 2.0f;
        logits[(size_t) t * (size_t) kExperts + (size_t) ((c + 1) % kExperts)] = 1.0f;
    }
}

RecallJob job(int64_t layer, int n_tok, std::vector<int32_t> ids, int k, std::vector<std::vector<int>> resident_by_dist) {
    RecallJob j;
    j.layer = layer;
    j.n_tok = n_tok;
    j.k = k;
    j.x.assign((size_t) n_tok * 2, 0.0f);   // n_embd = 2, x = 0 everywhere
    j.ids = std::move(ids);
    for (int d = 0; d < 3; ++d) {
        j.res[d].assign(1, 0);
        if ((size_t) d < resident_by_dist.size())
            for (int e : resident_by_dist[(size_t) d]) j.res[d][0] |= 1ull << e;
    }
    return j;
}

void test_scorer() {
    RecallScorer s(6, kExperts, 2, [](int64_t layer, const float* x, int nt, float* logits) { fake_router(layer, x, nt, logits, 2); });
    // one token, k = 2; the predicted set for layer L is {L, L + 1}
    const std::vector<std::vector<int32_t>> truth = {{0, 9}, {1, 5}, {3, 4}, {2, 3}};
    const std::vector<std::vector<int>> res = {{0}, {5}, {}, {2}};   // resident experts of layers 0..3
    for (int l = 0; l < 4; ++l) {
        std::vector<std::vector<int>> r;
        for (int d = 0; d < 3; ++d) r.push_back(l + d < 4 ? res[(size_t) (l + d)] : std::vector<int>{});
        s.step(job(l, 1, truth[(size_t) l], 2, r));
    }
    const RecallCounts& c = s.counts();
    CHECK(c.windows == 1 && c.layers_seen == 4);
    const auto& a = c.d[0];
    CHECK(a.layers == 3);
    CHECK(a.entries == 6 && a.entries_hit == 3);
    CHECK(a.miss_entries == 4 && a.miss_entries_hit == 3);
    CHECK(a.distinct == 6 && a.distinct_hit == 3);
    CHECK(a.miss_distinct == 4 && a.miss_distinct_hit == 3);
    CHECK(a.pred_miss == 6 && a.pred_miss_used == 3);
    const auto& b = c.d[1];
    CHECK(b.layers == 2);
    CHECK(b.entries == 4 && b.entries_hit == 2);
    CHECK(b.miss_entries == 3 && b.miss_entries_hit == 2);
    CHECK(b.distinct == 4 && b.distinct_hit == 2);
    CHECK(b.miss_distinct == 3 && b.miss_distinct_hit == 2);
    CHECK(b.pred_miss == 4 && b.pred_miss_used == 2);
    CHECK(c.layer_miss[0][1] == 1 && c.layer_miss_hit[0][1] == 1 && c.layer_miss[0][2] == 2 && c.layer_miss_hit[0][2] == 1);
    CHECK(c.layer_miss[1][2] == 2 && c.layer_miss_hit[1][2] == 1 && c.layer_miss[1][0] == 0);

    const std::string text = LookaheadRecall::format(c, 2, 2);
    CHECK(text.find("distance 1") != std::string::npos && text.find("distance 2") != std::string::npos);
    CHECK(text.find("NON-RESIDENT experts recalled 75.0% distinct (3/4)") != std::string::npos);
    CHECK(text.find("1:1/1 2:1/2") != std::string::npos);

    // a second window starts when the layer number does not go up: nothing carries over
    s.step(job(0, 1, truth[0], 2, {res[0], res[1], res[2]}));
    CHECK(s.counts().windows == 2 && s.counts().d[0].layers == 3);
    s.step(job(1, 1, truth[1], 2, {res[1], res[2], res[3]}));
    CHECK(s.counts().d[0].layers == 4 && s.counts().d[1].layers == 2);   // layer 1: distance 1 only (layer -1 never predicted)

    // a changed token count at the target layer: that prediction is not scored
    s.step(job(2, 2, {3, 4, 3, 4}, 2, {res[2], res[3], {}}));
    // layer 2 scored nothing: its distance-1 prediction came from a one-token layer 1, its distance-2 one from layer 0
    CHECK(s.counts().d[0].layers == 4 && s.counts().d[1].layers == 2);

    // take() returns the counts and zeroes them; reset() also forgets pending predictions
    const RecallCounts taken = s.take();
    CHECK(taken.d[0].layers == 4);
    CHECK(s.counts().d[0].layers == 0 && s.counts().windows == 0);
    s.reset();
    s.step(job(1, 1, truth[1], 2, {res[1], res[2], res[3]}));
    CHECK(s.counts().d[0].layers == 0);   // layer 1 right after a reset: no prediction was made for it
}

// BF16 routers and x rows that make every layer's true routing equal the router applied to the row
void test_live() {
    constexpr int64_t kLayers = 6, kEmbd = 16, kE = 32;
    constexpr int kK = 4;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::vector<uint16_t>> routers((size_t) kLayers);
    for (auto& r : routers) {
        r.resize((size_t) (kE * kEmbd));
        for (auto& w : r) {
            const float f = nd(rng);
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            w = (uint16_t) (bits >> 16);
        }
    }
    const auto routers_copy = routers;
    LookaheadRecall rec;
    std::string err;
    CHECK(!rec.start({routers_copy[0], routers_copy[1]}, kEmbd, kE, kK, err));   // fewer than three routers
    CHECK(rec.start(routers, kEmbd, kE, kK, err));
    CHECK(rec.running() && rec.n_layers() == kLayers);

    // the truth of layer L for the rows `x` is the router of layer L applied to `x`: the same x in every layer
    const int nt = 3;
    std::vector<float> x((size_t) (nt * kEmbd));
    for (auto& v : x) v = nd(rng);
    std::vector<float> logits((size_t) (nt * kE));
    auto top_ids = [&](int64_t layer) {
        router_logits_host(routers_copy[(size_t) layer].data(), kE, kEmbd, x.data(), nt, logits.data());
        std::vector<int32_t> ids;
        for (int t = 0; t < nt; ++t) {
            std::vector<int> o((size_t) kE);
            for (int e = 0; e < kE; ++e) o[(size_t) e] = e;
            const float* lt = logits.data() + (size_t) t * kE;
            std::partial_sort(o.begin(), o.begin() + kK, o.end(), [&](int a, int b) { return lt[a] > lt[b]; });
            for (int j = 0; j < kK; ++j) ids.push_back(o[(size_t) j]);
        }
        return ids;
    };
    // residency: even experts of every layer resident
    std::vector<int32_t> host_res((size_t) (kLayers * kE), -1);
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kE; e += 2) host_res[(size_t) (l * kE + e)] = (int32_t) e;

    rec.reset();
    for (int w = 0; w < 3; ++w)
        for (int64_t l = 0; l < kLayers; ++l) {
            const std::vector<int32_t> ids = top_ids(l);
            rec.submit(l, x.data(), ids.data(), nt, kK, host_res.data());
        }
    RecallCounts c = rec.take();
    CHECK(c.windows == 3 && c.layers_seen == 3 * kLayers);
    CHECK(c.d[0].layers == 3 * (kLayers - 1) && c.d[1].layers == 3 * (kLayers - 2));
    for (int d = 0; d < 2; ++d) {
        CHECK(c.d[d].entries > 0 && c.d[d].entries == c.d[d].entries_hit);
        CHECK(c.d[d].distinct > 0 && c.d[d].distinct == c.d[d].distinct_hit);
        CHECK(c.d[d].miss_entries > 0 && c.d[d].miss_entries < c.d[d].entries);   // the odd experts are the misses
        CHECK(c.d[d].miss_entries == c.d[d].miss_entries_hit);
        CHECK(c.d[d].miss_distinct == c.d[d].miss_distinct_hit);
        CHECK(c.d[d].pred_miss == c.d[d].pred_miss_used);   // every predicted expert was routed (k tokens' sets coincide)
    }
    // with a null table nothing is resident: every routed expert is a miss
    for (int64_t l = 0; l < kLayers; ++l) {
        const std::vector<int32_t> ids = top_ids(l);
        rec.submit(l, x.data(), ids.data(), nt, kK, nullptr);
    }
    c = rec.take();
    CHECK(c.d[0].miss_distinct == c.d[0].distinct && c.d[0].miss_entries == c.d[0].entries);
    CHECK(rec.take().d[0].layers == 0);   // taken: zero

    // another x at the true layer than the one predicted from: the recall falls (random sets: far from 100%)
    for (int64_t l = 0; l < kLayers; ++l) {
        std::vector<float> other((size_t) (nt * kEmbd));
        for (auto& v : other) v = nd(rng);
        const std::vector<int32_t> ids = top_ids(l);
        rec.submit(l, l == 0 ? x.data() : other.data(), ids.data(), nt, kK, nullptr);
    }
    c = rec.take();
    CHECK(c.d[0].layers == kLayers - 1 && c.d[0].entries_hit < c.d[0].entries);
    // the truth of layer 1 and 2 follow from layer 0's x only when the x is the same; layer 1's prediction (from x) is exact
    // for its own routing, so at least those entries were found
    CHECK(c.d[0].entries_hit >= (int64_t) nt * kK);

    // sampling: one window in 2 is scored, whole (windows 0, 2 of 0..3), and the report says so
    LookaheadRecall sampled;
    CHECK(sampled.start(routers_copy, kEmbd, kE, kK, err, 2));
    for (int w = 0; w < 4; ++w)
        for (int64_t l = 0; l < kLayers; ++l) {
            const std::vector<int32_t> ids = top_ids(l);
            sampled.submit(l, x.data(), ids.data(), nt, kK, host_res.data());
        }
    c = sampled.take();
    CHECK(c.windows == 2 && c.layers_seen == 2 * kLayers && c.every == 2);
    CHECK(c.d[0].layers == 2 * (kLayers - 1) && c.d[0].entries == c.d[0].entries_hit);
    CHECK(LookaheadRecall::format(c, 1, kK).find("sampled windows (one in 2)") != std::string::npos);
    sampled.reset();                                      // the count restarts: window 0 is sampled again
    for (int64_t l = 0; l < kLayers; ++l) {
        const std::vector<int32_t> ids = top_ids(l);
        sampled.submit(l, x.data(), ids.data(), nt, kK, host_res.data());
    }
    CHECK(sampled.take().windows == 1);
}
}  // namespace

int main() {
    test_scorer();
    test_live();
    if (failures) {
        std::fprintf(stderr, "lookahead_recall_test: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("lookahead_recall_test: ok\n");
    return 0;
}
