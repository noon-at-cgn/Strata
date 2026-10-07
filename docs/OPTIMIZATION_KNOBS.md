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
| `--aux-cpus LIST\|auto\|off` | **off** | `--aux-cpus auto` or `--aux-cpus 24,26`, or `STRATA_AUX_CPUS=auto\|LIST\|off` (Linux) | `aux_cpus`: `1` / `0` (`true` / `false`). A start with it off plans the CPU set anyway, so `1` takes `auto`. | Where the threads that are neither pool workers nor the host thread run: the adaptive tier's job thread, the prefill helpers and Stager copy threads, the PLE and file readers, the CUDA driver's threads (moved by a sweep). Default off = wherever they were created. Moves threads only, never a result; the pool's workers and the host thread are never touched. | `strata aux cpus: ...` at start; `strata serve: aux cpus ON/off for this request`; the `doorbell -> flag A` line; `waitA` in `STRATA_VERIFY_PROFILE`. See the section below. |
| Doorbell -> flag A histogram | always collected | printed with `STRATA_SPLIT_TIMING=1` | none | The host's answer time to each layer's doorbell, per stage (min / avg / p99 / max, 1 us bins). Pure measurement: one clock read per layer. | `strata serve: stage S: doorbell -> flag A, this request: ...` (batch: `strata batch: stage S doorbell -> flag A: ...`). |
| Router-lookahead recall counter | **off** | `STRATA_LOOKAHEAD_RECALL=1` (2: per layer too), `STRATA_LOOKAHEAD_RECALL_K`, `STRATA_LOOKAHEAD_RECALL_EVERY` (restart only) | none | A measurement for the PCIe-prefetch decision: how many of the experts a layer routes to (and of the non-resident ones) the routers of the next one and two layers name from this layer's input. Prefetches nothing, changes no output; every layer publishes its rows to the host and a helper thread scores one window in 3. | `strata lookahead recall: distance 1 ...` and `distance 2 ...` after each request. See the section below. |

How the per-request keys behave:

- They are read when a request starts and hold for the windows after it, so with several streams decoding the most
  recently started request's value is the one in force. Compare with one client at a time, or tag every request the
  same way.
- A request without the key goes back to the start-up value.
- The keys go in the request body like the existing ones, for example
  `"strata_tune": {"pcie_balance": 1, "pcie_frac": 0.5, "batch_overlap": 1, "q8k_avx2": 1, "prefill_pipe_k": 2,
  "pipeline_windows": 2, "aux_cpus": 1}`.
- `pcie_frac`, `spec_min_p`, `q8k_avx2` and `batch_overlap` take a number from 0 to 1 (use `0` or `1` for the last
  two); `prefill_pipe_k` must be a whole number; `pipeline_windows` must be the whole number `0` or `2`; `pcie_balance`
  must be `0`, `1`, `false` or `true`.
- `aux_cpus` takes `1`, `0`, `true` or `false`.

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

## Thread placement: `--aux-cpus` (Linux, off by default)

The CPU pool pins its workers (one logical CPU per physical core) and the host thread pins itself to the pool's reserved
core. Every thread the host creates after that inherits the host's one CPU: the adaptive tier's job thread (the bounce
copies and RAM moves of `--adapt-async`), the prefill helpers, the router lookahead. The threads the CUDA driver creates
float over every CPU, the workers' included. On the production box the architect review measured the host thread being
preempted about five times as often while the adaptive tier was busy (1055 against 159-196 involuntary context switches
per four requests) and the host's answer to a layer's doorbell taking 60-155 us instead of about 8; moving those threads
by hand (`taskset` after the start, to the host core's SMT sibling and CPUs 24 and 26) gave +26% decode speed in the
first requests after a start and nothing once the engine was warm. `--aux-cpus` does the same inside the engine, for
every thread that is neither a pool worker nor the host thread.

- **Which CPUs.** `auto`: the SMT siblings of the host's core first, then every CPU of a physical core that has no pool
  worker (the pool takes one logical CPU of each core it uses, so the other hardware threads of those cores are never
  taken). Nothing spare (a 16-worker pool on 16 physical cores leaves only the host
  core's sibling) is fine: the set is just that one CPU. No CPU at all spare: the feature does nothing and the start-up
  line says so. A list (`--aux-cpus 24,26`, `24-27`) is used as given, minus the host's CPU and the workers' CPUs, and only
  CPUs the process may run on. Taking a worker's SMT sibling by hand, as the manual test above did with CPU 26, shares
  that worker's core with the moved threads: only an explicit list can do that.
- **How.** Threads the engine creates pin themselves first thing (the adaptive tier's job thread and its per-round
  threads, the prefill Stager copy threads, the prefill helper threads, the router lookahead and the recall counter, the
  PLE reader and the file-reader threads). The threads the driver (or a library) created are moved by a sweep over
  `/proc/self/task` once the pool exists and again at every request start; the pool's workers and the host thread are
  never touched, and neither is a thread somebody pinned to one worker CPU. `STRATA_ADAPT_JOB_CPU` (an explicit CPU for
  the adaptive tier's job thread, or `-1` for none) still wins over this for that thread. `STRATA_AUX_STAGER=0` leaves the
  prefill copy threads where they were created: with a one-CPU spare set up to 32 of them (`STRATA_STAGER_THREADS`)
  would share that CPU, which can limit a long prompt's read.
- **Not covered.** Other processes (the Python launcher, dockerd) and the engine's own host thread; their placement is
  the operating system's or the container's.
- **Start-up line.** `strata aux cpus: threads that are neither pool workers nor the host go to CPUs 24 (on; N threads
  placed so far)`, or `(off at start; a request's aux_cpus=1 turns it on)`. At every request that changes the state
  there is a `strata serve: aux cpus ON/off for this request` line.
- **Per request.** `strata_tune {"aux_cpus": 1}` moves the threads to the set (the sweep) and `0` puts every moved thread
  back to the placement it had before it was moved. A request without the key goes back to the start-up value. With
  batch slots the most recently admitted request's value holds, as for the other keys.
- **What to compare.** The per-request `doorbell -> flag A` line below (the host's answer time, `STRATA_SPLIT_TIMING=1`),
  the `waitA` column of `STRATA_VERIFY_PROFILE=1`, and decode tok/s in two arms; the host thread's involuntary context
  switches (`grep nonvoluntary /proc/<engine pid>/task/<host tid>/status`, before and after a set of requests). The
  difference showed in the first requests after a start or on new content and not once warm, so discard the first 5-10
  requests after each start and compare cold rounds with cold rounds. A restart A/B (`STRATA_AUX_CPUS=auto` against
  unset) and the per-request toggle in one process (`aux_cpus` 1 / 0 alternating, ABBA) both work; the first needs a
  restart for each arm, the second compares arms on the same warm engine, where the review found no difference with the
  manual placement.
- **Deploying it.** The `aux_cpus` key is added to `serve/server.py`'s `sampling_keys`: copy this branch's `server.py` next
  to the binary, or the server drops the key.

## Doorbell -> flag A latency (always collected, printed with `STRATA_SPLIT_TIMING=1`)

Each layer of a verify window rings the host (the GPU's doorbell) and then waits for flag A (the layer's VRAM plan). The
engine now times, per stage and per layer served, the host thread's own part: from the moment it sees the ring to the
moment flag A is raised. The histogram has 1 us bins (the last one takes everything from 2.047 ms). Its cost is one clock
read per layer. `STRATA_SPLIT_TIMING=1` prints, after each request,
`strata serve: stage S: doorbell -> flag A, this request: min A / avg B / p99 C / max D us over N layers` (the `strata
batch:` burst line prints `stage S doorbell -> flag A: ...` the same way for batch windows; min, p99 and max are bin
centres). The GPU's `waitA` also holds the 3-5 us the flag takes to travel, so the two do not add up; a long p99 or max
with a small average is a host thread that was delayed (preempted, or a pool still busy with the previous layer's
work). Layers a stage serves without a doorbell (every expert resident) are not counted.

## Router-lookahead recall (a measurement; `STRATA_LOOKAHEAD_RECALL=1|2`, off by default)

Whether a PCIe prefetch of predicted misses (candidate 7 of the architect review) is worth building depends on how well
the routers of layers l+1 and l+2, applied to layer l's MoE input, name the experts those layers then route to. With
`STRATA_LOOKAHEAD_RECALL=1` (2 adds a per-layer line) the engine takes layer l's MoE input rows, which the doorbell
already publishes to the host for the CPU pool, applies the host copies of the routers of layers l+1 and l+2 on a helper
thread, takes each token's top-k set (`STRATA_LOOKAHEAD_RECALL_K`, default the routed count, 10), and when layer l+1 and l+2
are served scores the prediction against the true routing and against the GPU cache's residency at that moment. It
prefetches nothing and changes no output; it is a restart setting (the routers are copied from the stages' weights at
start-up: about 125 MiB of host RAM), not a per-request key.

- **What it changes while on.** Every layer publishes its x rows to the host (the doorbell's `publish_res` form copies
  them only when a miss needs them; this is the form the pipelined windows use), so each layer's doorbell copies 27 KB more
  over PCIe; a helper thread spends about 1 ms of CPU per scored layer (two router dots of 0.54 ms each, from
  `router_dot_parity` on this box's CPU), so it scores one window in `STRATA_LOOKAHEAD_RECALL_EVERY` (default 3; whole
  windows) and the layers of the others are not copied. Run it with `--aux-cpus` on so the thread stays off the pool's cores.
  It is refused with `--pipeline-windows 2`. Timings of a run with it on are not the production timings.
- **The line** (one per distance, per request for `--serve`, per burst for batch windows):
  `strata lookahead recall: distance 2 (top-10 of layer l+2's router on layer l's input), N layers of M sampled windows (one
  in 3): NON-RESIDENT experts recalled X% distinct (a/b), Y% of entries (c/d); ALL routed experts Z% distinct (...), W% of
  entries (...); predicted non-resident P distinct per layer, Q% of them routed`.
  *Non-resident* is the expert not being in the GPU cache when the layer was served (the ones the CPU pool computes or
  PCIe carries). *Distinct* counts each expert once per layer-window, which is what a prefetch moves; *entries* count every
  (token, expert) pair. *Predicted non-resident* is what a prefetcher would have to move per layer, and the share of
  those that were routed is its precision: moving an expert over PCIe takes about 0.27 ms on the review's measurement
  (11.4 GB/s, one expert 1.38 MB), against about 0.08 ms of pool time per expert.
- **How to read it.** Distance 2 is the lead the review needs (about 0.7 ms ahead). A high recall of the non-resident
  experts at a moderate predicted-non-resident count per layer says the prefetch has material; a low recall, or a
  precision so low that most of the moved experts go unused, says it does not. Run one profiled request set (the same
  requests that make the production profile) and read the lines per request; `STRATA_LOOKAHEAD_RECALL=2` adds the recall
  per layer (`layer:hit/total`) if it varies along the stack.
- **Tests.** `lookahead_recall_test` (CPU only) scores a hand-worked window, window boundaries, a changed token count,
  the sampling and the report text, and runs the threaded counter against synthetic BF16 routers.

## Draft head vocabulary and the 4-bit draft head (configuration only)

The MTP drafter reads its head (the main head's rows for a token subset) four times per window. Two settings make that
read smaller; neither needs engine changes, both are bind-time settings (a restart), and both change only which tokens the
drafter can propose and how exactly it scores them: the verify window decides every token.

- **`--mtp-draft-vocab FILE`** (engine flag; `mtp.cpp` reads `FILE`, else `<mtp>/draft_vocab.bin`, as a list of int32 token
  ids and keeps those rows of the main head). The shipped subsets in `data/`: `draft_vocab.bin` (the default, with the
  Chinese, Japanese and Korean tokens; 425,196 bytes = 106,299 ids), `draft_vocab_en.bin` (English and code; 162,100 bytes
  = 40,525 ids), `draft_vocab_cyrillic.bin` (58,963) and `draft_vocab_fr.bin` (46,211). `setup --draft-vocab en` does the
  same by copying the file over `mtp/rt/draft_vocab.bin`; the engine flag leaves the pack alone. With the main head's Q8_0
  rows (2,720 bytes) the head takes 275.7 MiB for the default subset and 105.1 MiB for `en`; the VRAM it frees is on the
  drafter's card (the last stage) and goes to that card's expert cache, since the cache is sized after the drafter's
  reservation.
- **`--mtp-q4 head`** (also `proj`: the draft layer's projections, `all`: both) keeps a Q4_0 copy of the draft head made at
  load from the main head's format (`N / 32 * 18` bytes a row: 1,440 bytes, so 55.7 MiB for `en`, 145.9 MiB for the
  default subset). Needs a build with native experts (the production build has it). `--mtp-q4` is not the same as the
  subset: it keeps every token the subset has and reads fewer bytes per row.
- **The two together** add only in bytes. Cost of `en`: a non-English answer's tokens are mostly outside the subset, so
  the drafter cannot propose them (fewer drafts accepted, not wrong output). Cost of `--mtp-q4`: the draft head's logits
  are those of a 4-bit copy, so the drafter's proposals can differ a little.

Exact lines for the production config (`configs/dual-3080-ud-q4_k_xl.json`, `"args"`, after `"--mtp", "mtp/rt"`; the architect
review says the file is in `/opt/strata-bmtp/data/` on lm-server; this tree's `data/draft_vocab_en.bin` is 162,100 bytes, sha256 `369151522226...`):

```
arm B (English subset):      "--mtp-draft-vocab", "/opt/strata-bmtp/data/draft_vocab_en.bin",
arm C (4-bit head only):     "--mtp-q4", "head",
arm D (both):                "--mtp-draft-vocab", "/opt/strata-bmtp/data/draft_vocab_en.bin", "--mtp-q4", "head",
arm A (today):               none of these
```

The start-up log shows what took effect: `strata mtp: draft head over 40525 tokens (105.1 MiB)` (106299 and 275.7 MiB for the
default), and with `--mtp-q4 head` `strata mtp: --mtp-q4: draft head over 40525 tokens as Q4_0 (55.7 MiB read per step, was
105.1)`. A path that cannot be read does not stop the start: `bind` then drafts over the whole vocabulary (the main head
itself, far more rows than any subset, read four times a window) and neither line appears, so look for them first.

A/B protocol (a restart per arm, so interleave the restarts and the request sets):

1. Start the arm. Discard the first 10 requests (the adaptive tier is still learning; the review measured the first 5-10
   requests after a start 15-25% slower than steady state).
2. Run the same request set in every arm: English prose, code, and the languages the server really sees (a few Chinese or
   French requests, where `en` is expected to lose drafts). Solo requests, `temperature` 0, 300 tokens, the same prompts in
   the same order.
3. Per request read `strata serve: prompt ... drafts accepted X of Y` (or `strata batch: MTP proposals accepted a of b`) for
   tokens per window, the `draft` column of `STRATA_DECODE_TIMING=1` (ms per window) for what the head read costs, and the
   decode tok/s of the line.
4. Order the arms A B C D, then D C B A, and compare per language: a smaller subset is worth it only if the draft time it
   saves per window is more than the accepted drafts it loses cost on the content that matters.

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
