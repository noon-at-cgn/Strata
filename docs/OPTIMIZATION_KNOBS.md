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
| Dense Q8_0 prompt projections on int8 tensor cores | **off** | `STRATA_PREFILL_DENSE_MMQ=1` (restart; the HIP spelling `STRATA_DENSE_MMQ=1` also works). `STRATA_PREFILL_DENSE_MMQ_MIN_N=n` (default 2048) keeps products with fewer than n output rows on cuBLAS. | none (restart only) | The prompt path's GGUF dense projections (GDN qkv / z / out, QSA q / k / v / out, shared expert) are by default dequantized to FP16 and multiplied by cuBLAS. With this on, the ones whose K is a multiple of 256 and whose N is at least the minimum go through llama.cpp's MMQ: the weights stay Q8_0, the FP16 activations are rounded to q8_1 (a scale per 32 values, int8 codes) and the product runs on the int8 tensor cores. **This changes the arithmetic**: in the unit test's engine-shaped data the result is about 1% (relative L2) away from the exact product, where the default path is 0.02% away, so greedy text can diverge from the default after a while. The routed experts already work this way. Costs ~10 MiB of VRAM per stage (the quantized-activation buffer), allocated at the first product. | At start of the first prompt: `prefill gemm: dense MMQ on (STRATA_PREFILL_DENSE_MMQ=1)`. With `STRATA_PREFILL_TIMING=1` each prompt run prints `strata prefill timing: layers A-B dense products ...` with the GPU time of the GGUF projections (`native`) and the BF16 ones and `dense MMQ ON (n products through MMQ, m through dequantize + cuBLAS since start)`. Unit tests: `dense_mmq_numerics_test` (CPU), `prefill_dense_mmq_test [--bench]` (GPU). |
| Idle card's share of a one-chunk read, per request (`prefill_help_frac`) | the rule of `STRATA_PREFILL_HELP_FRAC` | needs `STRATA_PREFILL_HELP=1` at start-up; `STRATA_PREFILL_HELP_FRAC=f` fixes the share for every read | `prefill_help_frac`: number 0 to 1 (0 = no help for this request; anything else is not sent) | With a layer split and `STRATA_PREFILL_HELP=1`, a prompt that fits one chunk lets the other card stream and compute a share of the experts. The built-in rule gives 0.41 of them at 1.5K tokens and none above ~3.3K tokens (measured on PCIe 4.0 cards; not re-measured on this PCIe 3.0 machine). The key sets the share for this request's read, so 4K / 6K / 8K single-chunk reads can be timed with 0, 0.2 and 0.3 without a restart. The last read's request value is the one in force. | The read's time (`strata serve: prompt N tokens = ... read in X ms`); with `STRATA_PREFILL_TIMING=1` the peer lines `peer-streamed experts ...`. |

How the per-request keys behave:

- They are read when a request starts and hold for the windows after it, so with several streams decoding the most
  recently started request's value is the one in force. Compare with one client at a time, or tag every request the
  same way.
- A request without the key goes back to the start-up value.
- The keys go in the request body like the existing ones, for example
  `"strata_tune": {"pcie_balance": 1, "pcie_frac": 0.5, "batch_overlap": 1, "q8k_avx2": 1, "prefill_pipe_k": 2,
  "pipeline_windows": 2}`.
- `pcie_frac`, `spec_min_p`, `q8k_avx2`, `batch_overlap` and `prefill_help_frac` take a number from 0 to 1 (use `0` or `1` for
  `q8k_avx2` and `batch_overlap`); `prefill_pipe_k` must be a whole number; `pipeline_windows` must be the whole number `0` or `2`; `pcie_balance`
  must be `0`, `1`, `false` or `true`.

An A/B of `--pcie-balance` with the production flags (`--pcie-frac 0.1`) without a restart:

```
A: strata_tune {"pcie_balance": 0}
B: strata_tune {"pcie_balance": 1, "pcie_frac": 0.5}
```

`pcie_frac` 0.5 is the bound in B, not a target; `{"pcie_balance": 1}` alone bounds the share at
`ceil(nmiss * 0.1)`.

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

## What one prompt chunk's GEMMs cost (micro-benchmark, RTX 3080 20 GB, T = 8192)

`prefill_dense_mmq_test --bench` times the matrix products of one chunk on one card with synthetic weights of the
engine's shapes (one stage = 18 GDN + 6 QSA layers). It does not time attention, the GDN recurrence, the expert
copies or the host's grouping, so it is not a split of the whole chunk.

| Work per chunk and stage | ms |
| --- | --- |
| Dense GGUF projections, dequantize + cuBLAS (the default) | 472 |
| The same through MMQ with every shape forced (`STRATA_PREFILL_DENSE_MMQ_MIN_N=0`) | 366 |
| The same through MMQ with the default minimum N (2048): the N = 512 / 640 products stay on cuBLAS | 346 |
| Routed experts through MMQ (32 groups of 16 experts x 160 rows per layer, 24 layers) | 418 |
| Hyper-connection BF16 products (cuBLAS, two reads per layer) | 148 |

Per product, MMQ runs the large ones (N 2560 to 12288) 1.26 to 1.55 times faster than dequantize + cuBLAS (cuBLAS 51
TFLOP/s, MMQ 64 to 80) and the N = 512 / 640 ones about 2 times slower, which is why they stay on cuBLAS.
