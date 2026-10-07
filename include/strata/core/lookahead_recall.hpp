// strata/core/lookahead_recall.hpp - how well would a router lookahead predict the experts a layer routes to?
//
// A MEASUREMENT, off unless STRATA_LOOKAHEAD_RECALL=1; it prefetches nothing and changes no output.  At layer l of a
// verify window the engine already has the layer's MoE input x_l on the host (the rows the doorbell publishes for the CPU
// pool).  This applies the routers of layers l+1 and l+2 to x_l (the host copies of the BF16 routers, on a thread of its
// own, so the host's layer loop pays a copy of the rows and nothing else), takes each token's top-k set, and when layer
// l+1 / l+2 is then served - its true routing arrives in the same way - scores the prediction:
//   * the recall of the experts that were NOT resident in the GPU cache (the ones a prefetch would have to move: they
//     are what the CPU pool computes or what goes over PCIe), and of all routed experts;
//   * per (token, expert) entry and per DISTINCT expert of the window's rows (a prefetch moves an expert once);
//   * how many non-resident experts the lookahead would have named and how many of those were routed (the cost side).
// The result reads as: "with a lookahead of d layers, a prefetcher that moved the predicted non-resident experts would
// have had X% of the misses of this request on the GPU in time".
//
// The scoring is `RecallScorer`, plain code with the router injected, so a test runs it without a GPU or a model.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

/// The host's router dot, the one RouterLookahead uses: `nt` (<= 8) rows of `cols` floats against `rows` BF16 router
/// rows -> logits[t * rows + r].  AVX2, else AVX1, else plain C++ (an estimate: the order of the additions differs).
void router_logits_host(const uint16_t* w, int64_t rows, int64_t cols, const float* x, int64_t nt, float* out);

/// What was counted.  Everything is a sum over the layer-windows scored since the last `take()`.
struct RecallCounts {
    struct Distance {
        int64_t layers = 0;                 ///< layer-windows scored (a prediction and the true routing both seen)
        int64_t entries = 0, entries_hit = 0;                ///< (token, expert) entries routed; predicted for that token
        int64_t miss_entries = 0, miss_entries_hit = 0;      ///< the ones whose expert was not resident; predicted
        int64_t distinct = 0, distinct_hit = 0;              ///< distinct experts of the window's rows; in the union of the predictions
        int64_t miss_distinct = 0, miss_distinct_hit = 0;    ///< the same for the non-resident ones
        int64_t pred_miss = 0, pred_miss_used = 0;           ///< distinct predicted experts not resident (when predicted); routed
    } d[2];                                 ///< d[0]: predicted one layer ahead, d[1]: two layers ahead
    int64_t windows = 0;                    ///< windows seen (a layer number that did not go up starts one)
    int64_t layers_seen = 0;                ///< layers submitted
    int every = 1;                          ///< one window in `every` was scored (set by LookaheadRecall::take)
    std::vector<int64_t> layer_miss[2], layer_miss_hit[2];   ///< per layer: distinct non-resident experts, and recalled
};

/// One layer's data as the host layer loop has it.
struct RecallJob {
    int64_t layer = 0;
    int n_tok = 0;
    int k = 0;                              ///< routed experts per token
    std::vector<float> x;                   ///< n_tok rows of n_embd
    std::vector<int32_t> ids;               ///< n_tok * k
    std::vector<uint64_t> res[3];           ///< resident-expert bitmasks of layers l, l+1, l+2 (bit e = expert e is in VRAM)
};

class RecallScorer {
public:
    using RouterFn = std::function<void(int64_t layer, const float* x, int nt, float* logits)>;
    RecallScorer(int64_t n_layers, int64_t n_expert, int k_pred, RouterFn router);
    /// Scores the predictions made for `job.layer`, then predicts layers l+1 and l+2 from `job.x`.
    void step(const RecallJob& job);
    /// A new request: the counts and any pending prediction are dropped.
    void reset();
    RecallCounts take();
    const RecallCounts& counts() const { return counts_; }
    int words() const { return words_; }
    int64_t n_layers() const { return n_layers_; }

private:
    struct Pred {
        bool valid = false;
        int64_t target = -1;
        int n_tok = 0;
        std::vector<uint64_t> sets;         ///< n_tok * words_: each token's predicted set
        std::vector<uint64_t> nonres;       ///< words_: the predicted experts not resident when predicted
    };
    void score(int dist, const Pred& p, const RecallJob& job);
    int64_t n_layers_, n_expert_;
    int k_pred_;
    int words_;
    RouterFn router_;
    RecallCounts counts_;
    Pred pred_[2][4];                       ///< [distance - 1][target layer % 4]
    int64_t last_layer_ = -1;
    std::vector<float> logits_;
    std::vector<int32_t> order_;
};

/// The engine's side: a thread that runs the scorer on what the layer loop submits.
class LookaheadRecall {
public:
    LookaheadRecall() = default;
    ~LookaheadRecall();
    LookaheadRecall(const LookaheadRecall&) = delete;
    LookaheadRecall& operator=(const LookaheadRecall&) = delete;
    /// `routers[l]`: layer l's ffn_gate_inp as BF16 bits (n_expert rows of n_embd).  `k_pred`: the experts predicted per token.
    /// `every`: only one window in `every` is scored (the router dots cost the host's helper thread ~1 ms a layer, a window is
    /// ~28 ms: scoring all of them would make the thread fall behind; the sampled windows are whole, so every layer-to-layer
    /// prediction in them is scored).
    bool start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k_pred, std::string& err,
               int every = 1);
    bool running() const { return running_; }
    /// Layer `layer`: this window's x rows and routed ids, the residency table `host_res` (n_layers x n_expert, >= 0 =
    /// resident; null = nothing resident).  Waits only when the worker is more than a window behind.
    void submit(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* host_res);
    /// Waits until everything submitted is scored.
    void drain();
    /// A new request: waits, then drops the counts.
    void reset();
    /// Waits, then returns the counts since the last `take()` / `reset()` and zeroes them.
    RecallCounts take();
    int64_t n_layers() const { return (int64_t) routers_.size(); }

    /// The report: `level` 1 = one line per distance, 2 = also the per-layer recall of the misses.  Empty when nothing was scored.
    static std::string format(const RecallCounts& c, int level, int k_pred);

private:
    void run();
    static constexpr size_t kRing = 256;    ///< layers the worker may be behind (five windows)
    std::vector<std::vector<uint16_t>> routers_;
    int64_t n_embd_ = 0, n_expert_ = 0;
    int k_pred_ = 10;
    bool running_ = false;
    std::unique_ptr<RecallScorer> scorer_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_work_, cv_space_, cv_idle_;
    std::vector<RecallJob> ring_;
    size_t head_ = 0, tail_ = 0;            ///< tail_ - head_ jobs queued
    bool busy_ = false, quit_ = false;
    int every_ = 1;                         ///< one window in `every_` is scored
    int64_t win_ = 0, last_in_ = -1;        ///< the producer's window count and last layer (a layer that did not go up = a new window)
};

}  // namespace strata::core
