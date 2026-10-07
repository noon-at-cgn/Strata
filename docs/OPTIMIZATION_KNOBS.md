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
| CPU expert row kernel (`--cpu-kernel`) for Unsloth UD-Q4_K_XL experts (Q4_K gate/up, Q5_1 down) | **ggml** (ggml-cpu's one-token dot per row and token, as before) | `--cpu-kernel ggml\|kq256\|fast` or `STRATA_KQ_KERNEL=ggml\|kq256\|fast` (`STRATA_KQ256=1` still means `kq256`; `STRATA_KQ_KERNEL` beats it; the flag beats both) | `cpu_kernel`: `"ggml"` / `"kq256"` / `"fast"` (exact words; anything else is not sent) | Which code computes the expert rows on the CPU. `kq256` is the older multi-token AVX2 kernel (groups of two or more tokens only). `fast` runs four rows at a time for one token (two for two tokens, one for three or four; groups of five or more tokens go to the `kq256` code), unpacks the Q4_K scales with vector instructions, prefetches the next rows, and converts four Q5_1 scale pairs at once. All three use the same integer sums and the same float operations in the same order, so every row and token comes out **bit for bit** the same: `kq_fast_parity` compares each against ggml-cpu's own `vec_dot` with `memcmp` (random and extreme blocks, 1 to 8 tokens, whole and odd partial row ranges, rows computed in chunks). A CPU without AVX2 always runs `ggml`. Other expert formats (i-quants, Q8_0 down with one token) are not touched. | `strata decode timing:` (needs `STRATA_DECODE_TIMING=1`) prints `CPU pool X ms per distinct CPU expert ...; CPU kernel fast, pool drain ...` next to the `CPU` figure of the window line; `STRATA_SPLIT_TIMING=1`'s `pool + plan` per window. On the server's own CPU, before a restart: `kq_fast_parity --bench` (below). |
| Expert pool layer drain (`--pool-drain`) | **barriered** (the pool's three phases per layer: gate/up rows, the host quantizes the intermediates, down rows; each published, drained, waited for and re-parked) | `--pool-drain barriered\|counters` or `STRATA_POOL_DRAIN=barriered\|counters` (the flag beats the variable) | `pool_drain`: `"barriered"` / `"counters"` (exact words; anything else is not sent) | `counters` publishes the layer once. Every pool thread, and the host thread when it drains, claims gate/up row chunks of any expert from per-expert atomic counters; whoever finishes an expert's last gate/up chunk quantizes that expert's intermediate and opens its down chunks; whoever finishes the layer's last down chunk raises the completion flag the host waits on. No barrier and no serial quantize step in between. The rows are computed by the same kernels (a row's result does not depend on the row range it is in), so the outputs are bit-identical (`pool_layer_test` switches drain and kernel at random between layers and compares every output with a single-thread reference). The pool is read once per call and every worker is parked again before a call returns, so a request can switch it. Chunk sizes (rounded down to a multiple of 4, at least 4): `STRATA_POOL_GU_ROWS` (default 40 gate/up rows per chunk) and `STRATA_POOL_DOWN_ROWS` (default 160), start-up only. | The same `strata decode timing:` line (`pool drain counters`) and its `CPU` / `CPU pool ... ms per distinct CPU expert` figures; `pool_layer_test --bench` on the server's CPU. A stalled layer is reported by the pool's existing watchdog (`strata: the CPU expert pool stalled ...`). |

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
  must be `0`, `1`, `false` or `true`; `cpu_kernel` one of the words `ggml`, `kq256`, `fast`; `pool_drain` one of
  `barriered`, `counters`.
- A request's `cpu_kernel` and `pool_drain` take effect for every pool call after it starts, like the other keys; because
  every setting gives the same bits, two requests with different values running together can only differ in speed.
- The server in front of the engine (`serve/server.py`) must be the one from this branch, or it drops the two new keys
  (it forwards only the keys it knows).

An A/B of `--pcie-balance` with the production flags (`--pcie-frac 0.1`) without a restart:

```
A: strata_tune {"pcie_balance": 0}
B: strata_tune {"pcie_balance": 1, "pcie_frac": 0.5}
```

`pcie_frac` 0.5 is the bound in B, not a target; `{"pcie_balance": 1}` alone bounds the share at
`ceil(nmiss * 0.1)`.

An A/B of the CPU pool changes without a restart (the engine started with the production flags; the server in front of it
must be this branch's `serve/server.py`, or it drops the two keys):

```
A: strata_tune {"cpu_kernel": "ggml", "pool_drain": "barriered"}
B: strata_tune {"cpu_kernel": "fast", "pool_drain": "barriered"}
C: strata_tune {"cpu_kernel": "fast", "pool_drain": "counters"}
```

### Measuring the CPU pool on the server's own CPU, before deploying

Both programs are built with the tests (CPU only, no GPU, no model) and pin their threads like the engine does.
Run them with the engine stopped, or at least with the pool idle:

```
build/kq_fast_parity                      # the parity checks (must print "0 differences")
build/kq_fast_parity --bench --nt 1,2 --threads 1 --cpu0 2          # GB/s of weight bytes per core, one thread
build/kq_fast_parity --bench --nt 1 --cpus 2,4,6,8,10,11,13,14,16,17,18,19,20,21,23,29,34   # the engine's worker CPUs
build/pool_layer_test                     # drains and kernels against the single-thread reference
build/pool_layer_test --bench --epl 3.3 --nt 1     # ms per layer and per distinct expert, each drain x kernel
```

`kq_fast_parity --bench` streams a pool of random expert blobs larger than the L3 (`--mb`, default 512), so the
weights come from DRAM as in decode; it prints, for the three kernels, GB/s per core for the gate/up rows, the down
rows and the whole expert, and the aggregate over `--threads` pinned threads. `pool_layer_test --bench` runs the real
`ExpertPool` (workers on the physical cores minus the first, the host draining) for `--epl` experts per layer and `--nt`
tokens each, for `--drain barriered,counters` x `--kernel ggml,fast`, and prints ms per layer (mean, p50, p90, p99),
ms per distinct expert and GB/s; `--gu-rows` / `--down-rows` set the Counters chunk sizes, `--workers` / `--no-host`
the pool shape.

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
