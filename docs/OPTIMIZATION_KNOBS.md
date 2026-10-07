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

## Dense chain: the hyper-connection read (branch `w2-dense`)

Two restart-time knobs for the GPU side of a decode window (the part before and after the experts: `hc-read0`,
`hc-read1+router` in the profile). Both are read when the verify window's graph is captured, so **neither can be switched
per request** (`strata_tune` has no key for them): changing one means a restart.

| Knob | Default | Start-up setting | Per request | What it changes | Log line to read |
| --- | --- | --- | --- | --- | --- |
| `STRATA_HC_Q8` (existed, this branch audited it) | **off** | `STRATA_HC_Q8=1` | none (restart) | The verify window's hyper-connection read takes the GGUF's own Q8_0 `hc_*_down` / `hc_*_up` projections (and the final mixer's `output_hc_*`) instead of the pack's BF16 copy of them. **Output changes**: the pack's copy is each Q8_0 value (fp16 scale times int8, exact in fp32) rounded to BF16, which holds only about 1.6% of those values exactly and is off by up to 2^-8 on the rest (`hc_q8_numeric_test` prints the numbers), and the Q8_0 kernels sum a row in 640-column chunks, another order than the BF16 kernels. The Q8_0 read is the closer one to the file. Halves the bytes of this read (the file's 0.65 GiB against the pack's 1.20 GiB for all layers). Costs VRAM: the Q8_0 copy is kept beside the BF16 one (the prompt path still reads that): about 0.65 GiB for the whole model, half of it on each card of a layer split, taken out of the expert cache (the cache is sized after the dense weights are loaded). The inject rows stay BF16 (they are exact there, `STRATA_HC_Q8_INJECT=1` reads them as Q8_0 too). | start-up: `STRATA_HC_Q8=1: N GiB of Q8_0 hyper-connection projections for the verify read` |
| `STRATA_HC_FUSED` | **off** | `STRATA_HC_FUSED=1` (with or without `STRATA_HC_Q8=1`) | none (restart) | Removes redundant work from the hyper-connection read and starts its last launch's weight loads earlier. **Bit for bit the same outputs** as the read it replaces (every task runs the same operations in the same order; `hc_q8_emu_test` proves it on the CPU for 1..8 rows, `hc_q8_parity` on the GPU, and every card re-checks it at start). Q8_0 read (two launches as before): the down launch's last chunk block of each row group reduces that group's 16 partial dots into `lo` / inject / `rs` (the old up kernel redid this in each of its 160 blocks), and the up launch requests its weights and inputs first and then reads the finished `lo`. BF16 read (three launches as before): the up launch requests all eight rows' weights and the epilogue inputs first (the old one asked for each row's weights when it got to the row). Neither waits for another block. The counters this needs (`FusedGrArgs::hc_sync`, 16 words per verifier, zero between launches) cost 64 bytes. At start every card runs `fused_gr_fused_check` (the BF16 form, and the Q8_0 form with `STRATA_HC_Q8=1`; random weights, 1..8 rows, with and without the pending write, each launched twice) and uses the form only if every output equals the plain read's bit for bit; the log says why when not. `STRATA_HC_FUSED_ONE_LAUNCH=1` (with `STRATA_HC_FUSED=1`) instead runs each read as **one launch** whose blocks wait for each other through the counters (a block takes its task from an atomic ticket as it starts, so tasks start in ascending order whatever order the GPU dispatches blocks in and no block waits for one that has not started; a wait over about 2 s traps instead of hanging; the BF16 form carries windows of up to 4 rows). **That one-launch form measured slower than the plain reads** on an RTX 3080 (T = 3 rows, a graph of 96 reads over 6 layers' weights: BF16 38 us per read plain, 57 us one launch; Q8_0 32 us plain, 40 us one launch; the blocks' dependent phases cost about 3 us per hand-off, and 128 registers per thread leave two blocks per SM so the 336 blocks run in waves), it is kept as an experiment. `STRATA_GR_V3=1`, `STRATA_NO_MULTI_GR=1` and `STRATA_HC_SPLIT=0|1` (the plain / split reads) keep the plain reads. Needs a compute capability 7.0 card or newer; HIP builds ignore it. | `strata hc: CUDA0: the STRATA_HC_FUSED Q8_0 read equals the plain read bit for bit on this card ...` then `STRATA_HC_FUSED=1: the hyper-connection read (Q8_0 projections) runs with its reduction folded into the down launch ...`; in the profile (`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`) the `hc-read0` column of the GDN layers (shown split as `hc0 norm` / `hc0 down` / `hc0 up` for the multi-launch read) and `hc-read1+router` |

A restart A/B of the four combinations is `STRATA_HC_Q8` x `STRATA_HC_FUSED`; the two cells with `STRATA_HC_Q8=0`
(fused or not) have the same outputs, so tokens can be compared exactly; the cells with `STRATA_HC_Q8=1` have the
Q8_0 read's outputs with or without the fusion (those two also agree bit for bit).

Other switches that already exist in this tree and touch the same chain, found while auditing (not changed here, off by
default on CUDA; their authors report them bit-identical, measured on an AMD Strix Halo only): `STRATA_QFUSE=1` (the
`q8_1` activation images written by the hyper-connection read and the GDN output norm instead of separate quantize
launches), `STRATA_Q8_PACKED=1` (a second, repacked copy of the Q8_0 attention / GDN projections for the decode
GEMVs: it costs about as much VRAM as those projections take), `STRATA_GDN_SPLIT=1`, `STRATA_TSUM=1`,
`STRATA_MMVF_ROWS=1`. `gr_parity --selftest` checks `STRATA_QFUSE`'s hyper-connection part, `mmvq_multi_parity`
the packed layout.

Tests (none needs a model):

- `hc_q8_numeric_test` (CPU): the Q8_0 versus BF16 weights, over every (fp16 scale, int8) pair and on real-shaped blocks.
- `hc_q8_emu_test` (CPU, about 4 minutes): the production device code of both reads run on the CPU by
  `tests/cuda_emu/cuda_emu.hpp` (every CUDA thread a fiber, blocks dispatched in a chosen order with a chosen number
  resident). Checks the existing Q8_0 read against a double-precision reference built from the raw Q8_0 bytes, and both
  one-launch reads bit for bit against the multi-launch reads, for 1..8 rows, with and without the pending write, with
  the `q8_1` image, under ascending, descending and shuffled dispatch down to one resident block; and that a ticketless
  version of the Q8_0 kernel does hang there. It cannot show timing, memory-ordering or the GPU's `exp`.
- `hc_q8_parity` (GPU, a few seconds, about 25 MiB of buffers plus the CUDA context): the same checks on the card, through
  CUDA graphs replayed with new inputs, with another kernel holding 56 KiB of every SM; `hc_q8_parity --bench [rows]`
  times the BF16 read, the Q8_0 read and the one-launch Q8_0 read in a graph of 96 reads the way a window runs them
  (about 150 MiB).

## What this file does not claim

No knob above (the dense-chain ones included: nothing was timed on a GPU for them) has a measured effect on tokens per second, latency or hit rate in this document. `--batch-overlap` is
marked "not measured yet on GPUs" in BATCHING.md; `--pcie-balance`'s link cost is a DMA probe, while the default
`--pcie-mode auto` stages the share with a copy kernel inside the window's graph, so the probe can differ from the
in-window cost (bound the share with `pcie_frac` if so, see the Limits section of [PCIE_BALANCE.md](PCIE_BALANCE.md)).
Greedy output can differ between different PCIe shares (which experts the GPU computes changes their rounding), as it
already does with `--pcie-frac`; the AVX2 quantizer, `--batch-overlap` and `prefill_pipe_k` are meant to leave the
output unchanged.
