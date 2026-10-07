# Optimization knobs on this branch

This branch (`opt-int`) combines four sets of changes made for the production A/B. Every change that has an
on/off switch is listed here with its default, how to set it for the whole engine, how to set it per request
(through the server's `strata_tune` field, so you can compare without a restart), what it changes, and which log
line to read. Nothing here has a measured speed-up attached: the numbers have to come from your own A/B on the
production machine.

All defaults below leave the behaviour of the previous branch (`batch-adapt`) unchanged.

## The knobs

| Knob | Default | Start-up setting | Per request (`strata_tune` key) | What it changes | Log line to read |
| --- | --- | --- | --- | --- | --- |
| AVX2 Q8_K quantizer for the CPU experts | **on** on CPUs with AVX2 | `STRATA_NO_Q8K_AVX2=1` (any value, even `0`, turns it off) | `q8k_avx2`: `1` / `0` | Which code turns the activation into Q8_K for the native i-quant expert layers. Both versions write the same bytes; only the CPU time of that step differs. An AVX-only CPU (or `STRATA_FORCE_ISA=avx`) always uses ggml's. | The `CPU experts` / pool time in the `strata batch:` window line; the pool time of `STRATA_SPLIT_TIMING=1`. Output must be identical either way. |
| `--batch-overlap` | **off** | `STRATA_BATCH_OVERLAP=1` (there is no command-line flag for it on this branch) | `batch_overlap`: `1` / `0` | In a batch window, every stage's commit graph is launched before any is waited for. With `--batch-mtp`, the slots' draft graphs are also launched back to back and collected together (each still waits for the previous drafter's on the GPU). Same work, fewer host waits. | `strata batch: ... + commit X + emit Y` (ms per window) and, at the end of that block, `strata batch: --batch-overlap ON` or `off`. |
| Chunks per prompt part (`prefill_pipe_k`) | **1** | `STRATA_PREFILL_PIPE_K=N` (1 to 16) | `prefill_pipe_k`: whole number 1 to 16 (anything else is not sent) | Only with a layer split and slots decoding while a long prompt is read: a part of the read is N chunks in one pipeline run instead of one chunk. The chunks and their arithmetic are the same; the slots get their turn every N chunks, so a slot waits longer for it. The slots' decode share is scaled with N. | The engine log's `the prompt was read in N parts, the slots decoding X ms between them` (it names the chunks per part), the time the read takes, and the slots' tok/s. |
| `--pcie-balance` | **off** | `--pcie-balance` (with `--serve`) or `STRATA_PCIE_BALANCE=1` | `pcie_balance`: `1` / `true` or `0` / `false` (anything else is not sent). Beats the flag and the variable. | Instead of the fixed floor `nmiss * --pcie-frac`, each layer's PCIe share of the missed experts is chosen from the measured pool cost and the measured link cost (one DMA probe per stage). `--pcie-frac` becomes the upper bound (`ceil`), `--pcie-frac 0` never uses PCIe. No expert goes over PCIe until both costs are measured. The plan used by verify windows (`window_gpu_plan`) makes the choice, so the O(n) planner and the balance work together. | At the end of each request: `strata serve: pcie balance: ...% of the ... routed entries went over PCIe (cap ...)`. The `strata batch:` line adds `pcie balance (ms per expert): ...` while the mode is on. See [PCIE_BALANCE.md](PCIE_BALANCE.md). |
| `--pipeline-windows 2` beside `--batch` / `--batch-mtp` / `--adapt-async 1` | **off** | `--pipeline-windows 2` (with `--serve`, a layer split on exactly two GPUs, `--mtp`). `STRATA_PIPELINE_ADAPT_ASYNC=0` keeps the blocking adaptive tier beside it. | `pipeline_windows`: `0` (serial) / `2` (pipelined) (whole numbers; anything else is not sent). `2` only takes effect when the engine was started with `--pipeline-windows 2`. | A request that decodes alone runs the pipelined loop: the first card starts the next window, guessed from the draft layer, while the last card verifies this one. The batch windows are not pipelined: while a slot is decoding or a request waits for a slot, a request decodes serially. With `--kv-pool-tokens` the pool is backed in 4096-cell steps at moments with no window in flight (see MULTI_GPU.md). A second verifier per card (160 MiB each) and two copies of the first card's recurrent state are kept out of the expert caches. Tokens are the serial loop's (see MULTI_GPU.md for the exactness conditions). While both stages wait on the CPU pool, the verified window's layers are served first (`STRATA_PIPELINE_PRIO=0` with `STRATA_PIPELINE_DEBUG=1` restores the old order, for an A/B). | At start-up: `--pipeline-windows 2: two verifiers per stage ... kept out of the expert caches: CUDA0 N MiB, CUDA1 M MiB; extra pinned RAM X MiB` (and no `--pipeline-windows 2 is off` or `--adapt-async 1 is off`). Per request (needs `STRATA_SPLIT_TIMING=1`): `strata pipeline: N windows in T ms (Y ms/window): S speculative, O on the path, R rolled back, G below the gate` and `strata pipeline classes:`. With a pool: `strata pipeline KV pool: ...`. `a request with ... decodes serially` names why a request was not pipelined. |
| PCIe share 0: window graph without flag B's wait and the PCIe group | **off**: the stage keeps the share `--pcie-frac` gives it (production: `0.1`) | `--pcie-frac 0` (or any value below `1/512`) | `pcie_frac`: a number 0 to 1 (already sent by the server). `0` runs the request on the graph without the PCIe part; any other value, `0.1` included, runs the graph with it. | After a layer's VRAM experts, the window graph normally waits for flag B (the host raises it once the PCIe share's copies have landed), stages the share with two kernels (`--pcie-mode 2`) and launches the grouped expert kernel over the PCIe groups, in every layer, even when the plan has no PCIe expert. With a share of 0 there is no PCIe expert, so a second captured graph leaves all of that out, and the host plans such a window with a share of 0 whatever the balance would choose and never raises flag B for it. Every expert is still computed by the same kernel on the same device as before; the graph just stops launching the empty PCIe work. A window picks its graph when it is launched (solo, batch, `--batch-mtp`, pipelined), so the share can change between requests. Which experts the GPU computes changes with the share (share 0: none over PCIe), so greedy output can differ between 0 and `0.1` as it already does between any two shares; at the same share the output is the same. | At the first window of a share-0 request: `strata verify: layers [a, b) PCIe share 0: windows run the graph without flag B's wait and the PCIe group`, and `captured the N-token window without flag B's wait and the PCIe group (PCIe share 0)` / `captured the batch window over slots ... without flag B's wait ...`, each with `device memory used by the graph X MiB`. With `STRATA_VERIFY_PROFILE=1` the `waitB` and `PCIe grp` columns are missing from that request's GPU stage line, and `wait for the GPU` of `STRATA_SPLIT_TIMING=1` is where a saving shows. |
| Flag B folded into flag A (`--pcie-mode 1` and `2`, so `auto`) | **on** | `STRATA_VERIFY_FLAGB=1` brings the old wait back in every graph (restart) | none | In `--pcie-mode 1` and `2` nothing is copied by the host: it raised flag B right after flag A with no work in between, and everything B could guard (the plan's PCIe arrays) is written before A. So those graphs no longer wait for B, and the host no longer raises it. `--pcie-mode 0` (the copy engine) still waits: there B guards the copies. Same experts, same kernels, same output. It applies to the graph with the share, so with `pcie_frac` 0.1 the `waitB` column of the profile now holds only the two staging kernels. | `waitB` in the `STRATA_VERIFY_PROFILE=1` line is smaller than with `STRATA_VERIFY_FLAGB=1`. |

How the per-request keys behave:

- They are read when a request starts and hold for the windows after it, so with several streams decoding the most
  recently started request's value is the one in force. Compare with one client at a time, or tag every request the
  same way.
- A request without the key goes back to the start-up value.
- The keys go in the request body like the existing ones, for example
  `"strata_tune": {"pcie_balance": 1, "pcie_frac": 0.5, "batch_overlap": 1, "q8k_avx2": 1, "prefill_pipe_k": 2,
  "pipeline_windows": 2}`.
- `pcie_frac`, `spec_min_p`, `q8k_avx2` and `batch_overlap` take a number from 0 to 1 (use `0` or `1` for the last
  two); `prefill_pipe_k` must be a whole number; `pipeline_windows` must be the whole number `0` or `2`; `pcie_balance`
  must be `0`, `1`, `false` or `true`.

An A/B of `--pcie-balance` with the production flags (`--pcie-frac 0.1`) without a restart:

```
A: strata_tune {"pcie_balance": 0}
B: strata_tune {"pcie_balance": 1, "pcie_frac": 0.5}
```

`pcie_frac` 0.5 is the bound in B, not a target; `{"pcie_balance": 1}` alone bounds the share at
`ceil(nmiss * 0.1)`.

### A/B of the PCIe share's graph, no restart

```
A: strata_tune {"pcie_frac": 0.1}    (production's value; no key does the same)
B: strata_tune {"pcie_frac": 0}
```

What to check before reading speeds: B's first request logs the `PCIe share 0` line and the captures (they take time and
VRAM once: the first request of each variant pays for its graphs, the later ones reuse them); after that, alternate A
and B on one client at a time, discard the first requests of each (the first minutes after a start run slower), and
compare `STRATA_SPLIT_TIMING=1`'s `wait for the GPU` per stage. In B the `waitB` and `PCIe grp` columns of
`STRATA_VERIFY_PROFILE=1` are gone. How large the share was at `0.1` is in A's own lines: the server's `[strata] done: ...`
line adds `(+X% of the routed experts over PCIe)` (the `pcie_share` of `/metrics`); B has none, those entries go to the
CPU pool instead, so A's X is what the share carried. Both variants stay in VRAM once captured (see "Memory" below), so
switching back and forth costs nothing after the first round.

### The PCIe share's graphs: what is kept, and the limits

- **Which graph.** Per stage, per window: the graph without the PCIe part when that stage's share is 0 (`--pcie-frac 0`
  or the request's `pcie_frac` is 0, which also covers anything below `1/512`), otherwise the graph with it. A stage that
  holds every expert in VRAM already has a graph without a doorbell, flag B or PCIe group; the share never changes that
  one. With a layer split each stage follows its own share: a request that sets `pcie_frac` (different from the start-up
  one) sets every stage's, otherwise each stage keeps the one the start-up gave it.
- **The host cannot disagree with the graph.** The planner takes the verifier's flag, so a window launched on the graph
  without the PCIe part gets a plan with no PCIe group even if the engine's share, `--pcie-balance` or the balance's
  exploration would have asked for one (`expert_window_plan_test` compares that plan with a share-0 plan on thousands of
  windows, with and without a balance). If a plan with a PCIe group ever reached such a graph, the window stops with
  `the pool planned a PCIe share for a window whose graph has none` instead of leaving the entry uncomputed.
- **`--pcie-balance`** needs the graph with the PCIe part whenever it can choose a share above 0, that is whenever the
  stage's `pcie_frac` is above 0. With `pcie_frac` 0 the balance never gets to choose, as before.
- **Memory.** A variant is its own set of captured graphs, captured the first time a window of it runs (pipelined
  windows: when the request starts, with nothing in flight), so a second variant takes about the VRAM of the first
  graph set again. The log line of each capture prints the device memory it used. The batch graphs of both variants
  count against the `--batch-mtp` limit of 64 layouts (least recently used goes first); the commit graph of a layout is
  shared by its variants. When there is no VRAM for a new graph, older batch layouts are freed, then the solo graphs of the
  variant not in use (`no VRAM for a new window graph: ... freed the solo graphs of the other variants`); a larger
  `--vram-reserve-mib` keeps more.
- **Pipelined windows** (`--pipeline-windows 2`) launch only graphs that were captured at the request's start. If the
  graph without the PCIe part is missing, the window runs on the one with it, which serves any share.
- A window picks its graph at its launch, and a window in flight keeps the graph it launched with: a request that
  changes the share takes effect from the next window of each stage.

## Changes without a switch

These are always on; each replaces code that did the same job.

- **Per-slot image positions with `--vision` and `--batch`** (#1242): each slot keeps its own image-position table
  on every GPU stage (and its own drafter's with `--batch-mtp`). The tables take VRAM before the expert cache is
  sized; the start-up line `--batch N --vision: N per-slot image-position tables ...` says how much. With `--vision`
  a batch window runs the per-head norm and the rope as separate launches instead of one fused launch (details in
  [BATCHING.md](BATCHING.md)). One launch per run of a slot's rows applies the private-position rope.
- **Re-probe of stale lookup window costs** (#1316): speculative-decoding policy.
- **O(n) plan of the verify window** (#1181): `detail::window_gpu_plan` groups a window's experts in one pass over
  the routing ids instead of a quadratic search. `expert_window_plan_test` compares it with the old planner on
  fixed and random windows, with and without `--pcie-balance`.
- **A long read gives way by what is left to read, not by prompt length** (#656, #1288): the server's `BYIELD`
  decision counts only the tokens the engine still has to read ([BATCHING.md](BATCHING.md)).

## What this file does not claim

No knob above has a measured effect on tokens per second, latency or hit rate in this document. `--batch-overlap` is
marked "not measured yet on GPUs" in BATCHING.md; `--pcie-balance`'s link cost is a DMA probe, while the default
`--pcie-mode auto` stages the share with a copy kernel inside the window's graph, so the probe can differ from the
in-window cost (bound the share with `pcie_frac` if so, see the Limits section of [PCIE_BALANCE.md](PCIE_BALANCE.md)).
Greedy output can differ between different PCIe shares (which experts the GPU computes changes their rounding), as it
already does with `--pcie-frac`; the AVX2 quantizer, `--batch-overlap` and `prefill_pipe_k` are meant to leave the
output unchanged.
