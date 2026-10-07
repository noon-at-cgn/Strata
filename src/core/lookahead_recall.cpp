// src/core/lookahead_recall.cpp - see the header: the router lookahead's recall, measured.
#include "strata/core/lookahead_recall.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>

namespace strata::core {

void router_logits_host(const uint16_t* w, int64_t rows, int64_t cols, const float* x, int64_t nt, float* out) {
    // The router dot is AVX2 (kq_avx2.cpp); a CPU without AVX2 (the experimental older-CPU builds) takes the AVX1
    // one (kq_avx1.cpp) or, without AVX, the same sums in plain C++.  From the Strata_Dirigo fork (rwkeyes): an
    // AVX-only Xeon E5-2687W died here (vpmovzxwd) on its first request.  An estimate only (which experts to
    // prefetch); the order of the additions differs between the three, the output does not depend on it.
    if (strata::kernels::cpu::cpu_avx2_ok()) {
        strata::kernels::cpu::bf16_rows_dot_multi(w, (int) rows, (int) cols, x, (int) nt, out);
    } else if (strata::kernels::cpu::cpu_avx1_ok()) {
        strata::kernels::cpu::bf16_rows_dot_multi_avx1(w, (int) rows, (int) cols, x, (int) nt, out);
    } else {
        // a mul and an add, not std::fma: without an FMA instruction that is a libm call per element (74x
        // slower on a Xeon E5-2665, measured by the fork)
        for (int64_t r = 0; r < rows; ++r) {
            const uint16_t* wr = w + (size_t) r * (size_t) cols;
            for (int64_t t = 0; t < nt; ++t) {
                const float* xr = x + (size_t) t * (size_t) cols;
                float acc = 0.0f;
                for (int64_t c = 0; c < cols; ++c) {
                    const uint32_t bits = (uint32_t) wr[c] << 16;
                    float wf;
                    std::memcpy(&wf, &bits, sizeof wf);
                    acc += wf * xr[c];
                }
                out[(size_t) t * (size_t) rows + (size_t) r] = acc;
            }
        }
    }
}

namespace {
int popcount64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(v);
#else
    int n = 0;
    for (; v; v &= v - 1) ++n;
    return n;
#endif
}
}  // namespace

RecallScorer::RecallScorer(int64_t n_layers, int64_t n_expert, int k_pred, RouterFn router)
    : n_layers_(n_layers), n_expert_(n_expert), k_pred_(std::clamp<int>(k_pred, 1, (int) n_expert)),
      words_((int) ((n_expert + 63) / 64)), router_(std::move(router)) {
    logits_.resize((size_t) (8 * n_expert_));
    order_.resize((size_t) n_expert_);
    reset();
}

void RecallScorer::reset() {
    counts_ = RecallCounts{};
    for (int d = 0; d < 2; ++d) {
        counts_.layer_miss[d].assign((size_t) n_layers_, 0);
        counts_.layer_miss_hit[d].assign((size_t) n_layers_, 0);
    }
    for (auto& row : pred_)
        for (Pred& p : row) p.valid = false;
    last_layer_ = -1;
}

RecallCounts RecallScorer::take() {
    RecallCounts out = std::move(counts_);
    counts_ = RecallCounts{};
    for (int d = 0; d < 2; ++d) {
        counts_.layer_miss[d].assign((size_t) n_layers_, 0);
        counts_.layer_miss_hit[d].assign((size_t) n_layers_, 0);
    }
    return out;
}

void RecallScorer::score(int dist, const Pred& p, const RecallJob& job) {
    RecallCounts::Distance& c = counts_.d[dist - 1];
    const size_t W = (size_t) words_;
    std::vector<uint64_t> u_true(W, 0), u_miss(W, 0), u_pred(W, 0);
    const std::vector<uint64_t>& res = job.res[0];
    auto resident = [&](int32_t e) { return (size_t) (e >> 6) < res.size() && ((res[(size_t) (e >> 6)] >> (e & 63)) & 1u) != 0; };
    for (int t = 0; t < job.n_tok; ++t) {
        const uint64_t* set = p.sets.data() + (size_t) t * W;
        for (size_t w = 0; w < W; ++w) u_pred[w] |= set[w];
        for (int j = 0; j < job.k; ++j) {
            const int32_t e = job.ids[(size_t) t * (size_t) job.k + (size_t) j];
            if (e < 0 || e >= n_expert_) continue;
            const bool hit = ((set[(size_t) (e >> 6)] >> (e & 63)) & 1u) != 0;
            const bool miss = !resident(e);
            ++c.entries;
            if (hit) ++c.entries_hit;
            if (miss) {
                ++c.miss_entries;
                if (hit) ++c.miss_entries_hit;
                u_miss[(size_t) (e >> 6)] |= 1ull << (e & 63);
            }
            u_true[(size_t) (e >> 6)] |= 1ull << (e & 63);
        }
    }
    int64_t n_true = 0, n_true_hit = 0, n_miss = 0, n_miss_hit = 0, n_pm = 0, n_pm_used = 0;
    for (size_t w = 0; w < W; ++w) {
        n_true += popcount64(u_true[w]);
        n_true_hit += popcount64(u_true[w] & u_pred[w]);
        n_miss += popcount64(u_miss[w]);
        n_miss_hit += popcount64(u_miss[w] & u_pred[w]);
        n_pm += popcount64(u_pred[w] & p.nonres[w]);
        n_pm_used += popcount64(u_pred[w] & p.nonres[w] & u_true[w]);
    }
    ++c.layers;
    c.distinct += n_true;
    c.distinct_hit += n_true_hit;
    c.miss_distinct += n_miss;
    c.miss_distinct_hit += n_miss_hit;
    c.pred_miss += n_pm;
    c.pred_miss_used += n_pm_used;
    if (job.layer >= 0 && job.layer < n_layers_) {
        counts_.layer_miss[dist - 1][(size_t) job.layer] += n_miss;
        counts_.layer_miss_hit[dist - 1][(size_t) job.layer] += n_miss_hit;
    }
}

void RecallScorer::step(const RecallJob& job) {
    if (job.n_tok <= 0 || job.k <= 0 || job.ids.size() < (size_t) job.n_tok * (size_t) job.k) return;
    ++counts_.layers_seen;
    if (job.layer <= last_layer_ || last_layer_ < 0) {       // the layer number did not go up: a new window
        ++counts_.windows;
        for (auto& row : pred_)
            for (Pred& p : row) p.valid = false;
    }
    last_layer_ = job.layer;
    // 1. the predictions made for THIS layer (one layer ago, two layers ago) meet its true routing
    for (int dist = 1; dist <= 2; ++dist) {
        Pred& p = pred_[dist - 1][(size_t) (job.layer & 3)];
        if (p.valid && p.target == job.layer && p.n_tok == job.n_tok) score(dist, p, job);
        p.valid = false;
    }
    // 2. this layer's x through the next two layers' routers
    const size_t W = (size_t) words_;
    for (int dist = 1; dist <= 2; ++dist) {
        const int64_t target = job.layer + dist;
        if (target >= n_layers_ || job.x.size() < (size_t) job.n_tok) continue;
        const int64_t n_embd = (int64_t) (job.x.size() / (size_t) job.n_tok);
        Pred& p = pred_[dist - 1][(size_t) (target & 3)];
        p.valid = false;
        p.target = target;
        p.n_tok = job.n_tok;
        p.sets.assign((size_t) job.n_tok * W, 0);
        p.nonres.assign(W, 0);
        for (int t0 = 0; t0 < job.n_tok; t0 += 8) {
            const int nt = std::min(8, job.n_tok - t0);
            router_(target, job.x.data() + (size_t) t0 * (size_t) n_embd, nt, logits_.data());
            for (int t = 0; t < nt; ++t) {
                const float* lt = logits_.data() + (size_t) t * (size_t) n_expert_;
                for (int64_t e = 0; e < n_expert_; ++e) order_[(size_t) e] = (int32_t) e;
                std::partial_sort(order_.begin(), order_.begin() + k_pred_, order_.end(), [&](int32_t a, int32_t b) {
                    return lt[(size_t) a] > lt[(size_t) b] || (lt[(size_t) a] == lt[(size_t) b] && a < b);
                });
                uint64_t* set = p.sets.data() + (size_t) (t0 + t) * W;
                for (int j = 0; j < k_pred_; ++j) set[(size_t) (order_[(size_t) j] >> 6)] |= 1ull << (order_[(size_t) j] & 63);
            }
        }
        const std::vector<uint64_t>& res = job.res[dist];
        for (int t = 0; t < job.n_tok; ++t)
            for (size_t w = 0; w < W; ++w) p.nonres[w] |= p.sets[(size_t) t * W + w] & ~(w < res.size() ? res[w] : 0ull);
        p.valid = true;
    }
}

// ---------------------------------------------------------------------------------------------------------------

LookaheadRecall::~LookaheadRecall() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
    }
    cv_work_.notify_all();
    if (thread_.joinable()) thread_.join();
}

bool LookaheadRecall::start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k_pred,
                            std::string& err, int every) {
    if (routers.size() < 3) { err = "LookaheadRecall: fewer than three routers"; return false; }
    if (n_embd % 8 != 0 || n_embd <= 0) { err = "LookaheadRecall: n_embd is not a multiple of 8"; return false; }
    if (n_expert < 2 || n_expert > 1024) { err = "LookaheadRecall: n_expert out of range"; return false; }
    for (const auto& r : routers)
        if (r.size() != (size_t) (n_embd * n_expert)) { err = "LookaheadRecall: a router of another shape"; return false; }
    routers_ = std::move(routers);
    n_embd_ = n_embd;
    n_expert_ = n_expert;
    k_pred_ = k_pred;
    every_ = every < 1 ? 1 : every;
    win_ = 0;
    last_in_ = -1;
    scorer_ = std::make_unique<RecallScorer>((int64_t) routers_.size(), n_expert_, k_pred_,
                                             [this](int64_t layer, const float* x, int nt, float* logits) {
                                                 router_logits_host(routers_[(size_t) layer].data(), n_expert_, n_embd_, x, nt, logits);
                                             });
    ring_.assign(kRing, RecallJob{});
    head_ = tail_ = 0;
    running_ = true;
    thread_ = std::thread([this] {
        strata::aux_cpus::pin_current_thread();
        run();
    });
    return true;
}

void LookaheadRecall::submit(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                             const int32_t* host_res) {
    if (!running_ || x == nullptr || ids == nullptr || n_tok <= 0 || k <= 0 || layer < 0 || layer >= n_layers()) return;
    if (layer <= last_in_) ++win_;       // the layer number did not go up: the next window
    last_in_ = layer;
    if (win_ % every_ != 0) return;      // not a sampled window
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_space_.wait(lk, [&] { return tail_ - head_ < kRing; });
    }
    RecallJob& j = ring_[tail_ % kRing];   // the worker reads only [head_, tail_): this slot is ours until tail_ moves
    j.layer = layer;
    j.n_tok = (int) n_tok;
    j.k = (int) k;
    j.x.assign(x, x + (size_t) (n_tok * n_embd_));
    j.ids.assign(ids, ids + (size_t) (n_tok * k));
    const size_t W = (size_t) ((n_expert_ + 63) / 64);
    for (int d = 0; d < 3; ++d) {
        j.res[d].assign(W, 0);
        if (host_res == nullptr || layer + d >= n_layers()) continue;
        const int32_t* row = host_res + (size_t) (layer + d) * (size_t) n_expert_;
        for (int64_t e = 0; e < n_expert_; ++e)
            if (row[e] >= 0) j.res[d][(size_t) (e >> 6)] |= 1ull << (e & 63);
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        ++tail_;
    }
    cv_work_.notify_one();
}

void LookaheadRecall::run() {
    for (;;) {
        size_t slot;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [&] { return quit_ || tail_ > head_; });
            if (tail_ == head_) return;     // quit, nothing queued
            busy_ = true;
            slot = head_ % kRing;
        }
        scorer_->step(ring_[slot]);
        {
            std::lock_guard<std::mutex> lk(mu_);
            ++head_;
            busy_ = false;
        }
        cv_space_.notify_one();
        cv_idle_.notify_all();
    }
}

void LookaheadRecall::drain() {
    if (!running_) return;
    std::unique_lock<std::mutex> lk(mu_);
    cv_idle_.wait(lk, [&] { return head_ == tail_ && !busy_; });
}

void LookaheadRecall::reset() {
    if (!running_) return;
    drain();
    scorer_->reset();
    win_ = 0;
    last_in_ = -1;
}

RecallCounts LookaheadRecall::take() {
    if (!running_) return RecallCounts{};
    drain();
    RecallCounts c = scorer_->take();
    c.every = every_;
    return c;
}

std::string LookaheadRecall::format(const RecallCounts& c, int level, int k_pred) {
    auto pct = [](int64_t a, int64_t b) { return b > 0 ? 100.0 * (double) a / (double) b : 0.0; };
    std::string out;
    char b[1024];
    for (int d = 0; d < 2; ++d) {
        const RecallCounts::Distance& x = c.d[d];
        if (x.layers == 0) continue;
        std::snprintf(b, sizeof b,
                      "strata lookahead recall: distance %d (top-%d of layer l+%d's router on layer l's input), %lld layers "
                      "of %lld sampled windows (one in %d): NON-RESIDENT experts recalled %.1f%% distinct (%lld/%lld), %.1f%% of entries "
                      "(%lld/%lld); ALL routed experts %.1f%% distinct (%lld/%lld), %.1f%% of entries (%lld/%lld); "
                      "predicted non-resident %.2f distinct per layer, %.1f%% of them routed\n",
                      d + 1, k_pred, d + 1, (long long) x.layers, (long long) c.windows, c.every,
                      pct(x.miss_distinct_hit, x.miss_distinct), (long long) x.miss_distinct_hit, (long long) x.miss_distinct,
                      pct(x.miss_entries_hit, x.miss_entries), (long long) x.miss_entries_hit, (long long) x.miss_entries,
                      pct(x.distinct_hit, x.distinct), (long long) x.distinct_hit, (long long) x.distinct,
                      pct(x.entries_hit, x.entries), (long long) x.entries_hit, (long long) x.entries,
                      (double) x.pred_miss / (double) x.layers, pct(x.pred_miss_used, x.pred_miss));
        out += b;
    }
    if (level >= 2) {
        for (int d = 0; d < 2; ++d) {
            if (c.d[d].layers == 0) continue;
            std::string line = "strata lookahead recall: distance " + std::to_string(d + 1) +
                               " per layer, recalled/non-resident distinct (layer: hit/total):";
            for (size_t l = 0; l < c.layer_miss[d].size(); ++l) {
                if (c.layer_miss[d][l] == 0) continue;
                std::snprintf(b, sizeof b, " %zu:%lld/%lld", l, (long long) c.layer_miss_hit[d][l], (long long) c.layer_miss[d][l]);
                line += b;
            }
            out += line + "\n";
        }
    }
    return out;
}

}  // namespace strata::core
