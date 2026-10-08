# up-merge: upstream/main (v0.1.40.3) merged into our `w2-dense-fix` stack

Branch `up-merge` = merge commit of `upstream/main` (d5ea713, v0.1.40.3, 265 commits past the old merge base 82f46a8)
into `w2-dense-fix` (c45d854: opt-w2 and every optimisation of this fork). History is kept (merge, not rebase).
Engine binary built from it for deployment: `strata-w5`.

Labels below follow the search-first rule: **known** = upstream already had the same fix, upstream's version taken;
**adapted** = both sides changed the same code, semantics of both kept; **new** = ours only, kept as it was.

## Conflicts and how each was resolved

| File | Conflict | Resolution | Label |
|---|---|---|---|
| `CMakeLists.txt` (2 blocks) | tests added next to each other (upstream `expert_profile_version_test`; ours `expert_window_plan_test`, `verify_variant_test`, `kq_fast_parity`, `pool_layer_test`) | all targets kept (an `endif()` between the two `if(EXISTS ...)` blocks) | adapted |
| `include/strata/spec/draft_policy.hpp`, `src/spec/draft_policy.cpp`, `src/spec/draft_policy_test.cpp` | our "re-probe stale lookup costs" (878f335, `cost_age_`/`kReprobeRounds`) against upstream #1252 (`stale()`/`last_[]`/`kStaleRounds`) | upstream's files taken whole, our `observe_cost` + age counters and our extra test cases dropped | **known** |
| `src/core/verify.cpp` destructor | `hcsync_` (ours) / `d_spec_` (upstream SPEC_PROB) frees | both frees | adapted |
| `src/core/verify.cpp` `norm_rope` | upstream factors the plain norm+rope into `norm_rope_on(..., stream)` (used by the PDL/branch paths); ours has per-slot private position tables (`private_positions`, mrope in `--batch`) with one rope launch per run of a slot's rows | `norm_rope(data, norm, rows, cols, pos, first, count)` keeps our signature; the no-private-position case calls upstream's `norm_rope_on`, the private-position case runs our loop. `STRATA_DF_BRANCH`'s side-stream query path (`qbr`, upstream, opt-in) is switched off when `private_positions(tb, n)` because its rope has no per-slot tables | adapted |
| `src/core/verify.cpp` K/V projections | upstream `mm(wk/wv,...)` (the interleaved-activation MMVQ, `STRATA_MMVQ_IL`) against our direct `native_mmvq` | upstream's `mm(...)`, our `norm_rope(..., tb, n)` arguments | known (+ adapted call args) |
| `src/core/verify.cpp` K8V4 batched append, q/gate split | upstream #1188 comment text (same code as our cherry-pick fda484f); upstream `copy_rows_strided` for the q/gate split (PDL chain) | upstream's versions | known |
| `src/core/verify.cpp` capture / eviction (4 blocks) | our `vram0`/`vram1` "device memory used by the graph" log and PCIe-variant graph eviction (`commit_graph_still_used`, `evict_idle_variants`) against upstream #1185/#997 eviction without them | ours (superset: upstream's `freed` loop is inside it, same log line) | new |
| `src/core/verify.cpp` `ar_on()` comments | #646/#871 per-window doorbell skip, same code both sides | upstream's comments | known |
| `src/kernels/cpu/native_expert.cpp` | both sides added the AVX2 Q8_K activation quantizer (upstream: static `q8k_avx2()` from #851; ours: the same plus `native_set_q8k_avx2` runtime switch and the `STRATA_KQ_KERNEL` / `--cpu-kernel` mode) | ours (superset, `STRATA_NO_Q8K_AVX2` still honoured) | adapted |
| `src/program/generate.cpp` (7 blocks) | `--adapt-async` gating: upstream "not with `--batch` slots", ours runs it between batch windows (not with `--batch-groups`, not `--pipeline-windows 1`); empty-round guard in `AState::CopyBack`; pipelined loop: pool priority (`serve1`, `STRATA_PIPELINE_PRIO`) and KV-pool drain points (`pl_kv`) | ours for all five | new |
| `src/program/generate.cpp` per-request log | upstream added the file-tier I/O line (inside a now braced `if (srcp == &src) { ... }`), ours the `--pcie-balance` request line right after it | upstream's block, the closing brace, then ours | adapted |
| `serve/server.py` ERR handling | upstream #1059 (`_ctl_erred`, "an engine ERR ends the request, no 300 s wait") against our same fix via `_ctl_result = ("err", None)` that also maps the KV pool's refusal to `PoolFull` (503) | `_ctl_erred = True` plus our `raise engine_error(...)`; our `_ctl_result[0] == "err"` check in the `finally` removed (superseded) | known (+ our PoolFull) |
| `serve/server.py` `reused0` | upstream tracks the first read's `reused` count for a continued leg; ours passes `prompt=` to `_take_control` | both lines | adapted |
| `serve/test_parallel.py`, `src/core/conversation_cache_test.cpp` | additive (upstream `message()` helper beside ours `chat_messages()`; upstream pin=N test beside our retained-identity test) | both kept | adapted |
| `docs/DETAILS.md`, `docs/MULTI_GPU.md` | text about `--adapt-async` with `--batch`, and our `--pipeline-windows` with `--batch` / KV pool paragraphs | ours | new |

Everything else merged without conflict, including upstream's PDL / interleaved-MMVQ / PLE one-pass / Gumbel / pool-tasks /
expert-cache-per-layer / file-tier I/O / Stager changes, `--batch-groups` slot spreading (#1249) and the serve changes
(prometheus, request accounting, ready-timeouts).

## Our features, still wired (checked in the merged tree)

Every `"--flag"`, `STRATA_*` name and `strata_tune` key that our stack added over the old base is still in the merged
sources (compared programmatically), and the built engine's `--help` lists `--memory-limit-mib`,
`--conversation-cache-mib/-slots/-min-free-mib`, `--pcie-frac`, `--pcie-balance`, `--adapt-async`, `--batch-mtp`,
`--aux-cpus`, `--pool-drain`, `--cpu-kernel`, `--pipeline-windows`, `--kv-pool-tokens`. `--batch-overlap` and
`STRATA_PREFILL_PIPE_K` are env / `strata_tune` knobs (`batch_overlap`, `prefill_pipe_k`) exactly as before.
`serve/server.py` still takes every `strata_tune` key (`pipeline_windows`, `pcie_frac`, `pcie_balance`, `batch_overlap`,
`prefill_pipe_k`, `aux_cpus`, `q8k_avx2`, `cpu_kernel`, `pool_drain`).

Changed or lost relative to `w2-dense-fix`:
- `DraftPolicy` is upstream's #1252 re-probe (every 300 rounds a stale size, instead of ours: every 64 timed rounds a size
  whose cost is older than 64 other rounds). Both fix the same starvation; probe cadence differs.
- `--adapt-async` gating text and logic are ours (beside `--batch` slots); upstream has it refused there.
- Upstream defaults that now apply without any flag (all output-neutral by their tests, but they move timing):
  `STRATA_MMVQ_IL` on (interleaved q8_1 for 2-4 row dense projections), `STRATA_STAGER_SLEEP` on Linux (prompt staging
  threads sleep instead of yield-spin), `STRATA_PIN_GUARD` on (the pinned complement is registered in steps when the RAM
  free would have to come from reclaimed cache; `STRATA_PIN_RESERVE_GIB`), `STRATA_ENGINE_READY_S` 900 s in the serve
  layer (below), `STRATA_VISION_READY_S` / `STRATA_VISION_ENCODE_S` 300 s.
- Nothing of ours was removed except the superseded draft-policy re-probe and our `_ctl_result` "err" shortcut.

## Required config change for serve-w5

**`STRATA_ENGINE_READY_S`**: upstream's serve layer now gives the engine 900 s to print READY (#1317, `serve/server.py`
`ENGINE_READY_S`); `0` waits for ever as before. Our production start pins a 52-55 GiB complement, which the prod log
shows took 520-730 s on its own, before the PLE table and the cache fill, so a start can take longer than 900 s. Put
`"STRATA_ENGINE_READY_S": "0"` (or `"3600"`) into the config's `"env"` block of any run that uses `serve-w5`.

## New upstream knobs and what they mean for 2x RTX 3080 20 GB (sm_86) + Xeon E5-2696 v4 (AVX2, 16 pool workers + host) + 100 GiB + UD-Q4_K_XL

Baseline for every A/B below: `strata-w5` with the production args and env of `strata-q4xl-prod.json` (plus
`STRATA_ENGINE_READY_S=0`). Run it first against `strata-w2` to see whether the merge itself moved anything.

| Knob | What it is (from upstream docs / code) | Verdict for this box |
|---|---|---|
| `--batch-groups G` (with `--batch N`, explicit layer split) | slots in G pipelined groups: GPU k runs one group while GPU k+1 runs another; #1249 (now merged) sends a second request to the other group and stops idle slots costing rows | Our two-lane model (`w3-twolane`, `TWO_LANE_MODEL.md`) said NO-GO for 2 lanes with MTP at 1.25-1.33x of the merged batch (needs 1.5x). Upstream also makes it **exclusive with `--batch-mtp`** (engine turns MTP off), `--adapt-async` (blocking tier), `--pipeline-windows` (serial), and a pipelined slot is not kept as a conversation cache (parking is lost). So it is a different trade: 2 lanes without MTP, one token per lane per window. [INFERENCE] aggregate would be about 2 lanes x 1 token per ~28 ms stage cycle, below our 44-51 tok/s with batch-MTP; only worth measuring as a closing arm. Needs `--batch 2 --batch-groups 2` and dropping `--batch-mtp` / `--adapt-async`. |
| `--pool-tasks N` | row tasks per batched CPU expert phase (0 = automatic, 3 per participating thread = 51 here; capped by row count) | **Config change, cheap.** More tasks = better balance between cores, more scheduling overhead; the CPU part of a window is ~15 ms of 56. |
| `--expert-cache-per-layer` | every layer gets the same number of slots, each the size of that layer's own blob (native packs) instead of one shared pool filled in profile order | **Config change, risky.** Our tier is profile-preloaded and adaptive (`--expert-profile`, `--adapt-every 2`, `--adapt-async 1`), a per-layer quota fixes how many slots each layer may hold, which fights the learned global popularity. Upstream's own number (54% hits at 4096 slots) is against a profile-less fill. Try last, watch the `decode expert cache hit rate` line. |
| `STRATA_PREFILL_CPU_SHARE=auto\|0.4` | a prompt chunk below 1024 tokens hands the idle decode pool the experts few tokens route to instead of streaming them over PCIe | **Does not apply**: `set_cpu_pool` is only called when `!multi_gpu && batch <= 0`; we run a layer split with `--batch 2`. |
| PDL (`STRATA_DF_PDL=1\|2`, `pdl.hpp`) | programmatic dependent launch inside the verify graph | **Does not apply**: `pdl_supported()` needs sm_90+; the RTX 3080 is sm_86 and the build is `-DCMAKE_CUDA_ARCHITECTURES=86` (`pdl_parity` checks "no programmatic edges elsewhere"). |
| `STRATA_MMVQ_IL` (default **on**) | 2-4 token dense projections read an interleaved copy of the q8_1 activations, bitwise the same outputs | Already active in w5 (not in w2). Our MTP windows are 2-4 rows per slot and the 2-slot batch window is 4 rows, so it applies. A/B arm: `STRATA_MMVQ_IL=0`. |
| `STRATA_DF_PLE=1` | the PLE post-projection steps of a verify window in one pass, bitwise the per-token loop | Solo path only (`!batch_rec_`): candidate for the single-user decode (the 50-73 tok/s numbers). Env only. |
| `STRATA_STAGER_SLEEP` (Linux default now sleep) | prompt-staging threads sleep instead of yield-spinning | Prefill (835-2150 tok/s) may move either way; A/B arm `STRATA_STAGER_SLEEP=0` against the new default on 5k / 27k / 96k prompts. |
| `STRATA_SPEC_GUMBEL=1` (with `STRATA_SPEC_COUPLED=1`) | Gumbel-max coupled picks for sampled requests: upstream measured accepted drafts 66.9% -> 68.0% on an RTX 3060, decode within noise | Only matters for requests with temperature > 0 (greedy rows ignore it). Env only; check what the clients send before spending a restart. |
| `STRATA_SPEC_PROB=1` | probabilistic draft acceptance for sampled drafts | same condition as above. |
| `STRATA_DF_BRANCH=1` | side-stream branches inside the verify graph | Not in split or batch windows, and upstream reports a graph stop on Linux after an NVML query (nvidia-smi). Skip. |
| `STRATA_IO_PREFETCH=1`, `STRATA_IO_PF_*` | file-tier readers into the page cache | Skip: we serve with the resident RAM copy and no disk reads. |
| `STRATA_PIN_GUARD`, `STRATA_PIN_RESERVE_GIB` (new, default guard on) | pin the complement in steps when the free RAM would have to come from reclaimed cache; warns "only N of M GiB could be page-locked" | Check the start log: it must say the full 52-55 GiB pinned. `STRATA_PIN_GUARD=0` restores the one-shot pin. Overlaps in purpose with our `--memory-limit-mib 101376` guard; both stay. |
| layer-split `auto` fixes (#1238, #1094/#1160, tie -> balanced) | pricing and refusal rules of `--layer-split auto` | No effect: production uses the explicit `"layer_split": "24"`. |
| serve: `STRATA_ENGINE_READY_S`, `STRATA_VISION_READY_S`, `STRATA_VISION_ENCODE_S`, prometheus, request accounting | read timeouts and metrics | `STRATA_ENGINE_READY_S=0` required (above). The CPU vision encoder starts well inside 300 s. |

Existing knobs of our own stack that production does not set yet and that this binary still has (unchanged semantics):
`--cpu-kernel fast` (also `STRATA_KQ_KERNEL=fast`), `--pool-drain counters`, `--aux-cpus auto`, `--pcie-balance`,
`--batch-overlap` (`STRATA_BATCH_OVERLAP` / `strata_tune batch_overlap=1`), `STRATA_PREFILL_PIPE_K`, `STRATA_HC_FUSED=1`,
`STRATA_HC_Q8=1`.

## Recommended A/B order (each arm = one restart with `strata-w5`, same prompts, interleave with the baseline)

0. **Baseline**: `strata-w5`, production args, `STRATA_ENGINE_READY_S=0`. Compare against `strata-w2` (same prompts):
   the delta is the merge itself (MMVQ_IL, Stager sleep, pin guard, upstream CPU Q8_K, sampler).
1. `STRATA_MMVQ_IL=0` (env, no arg change): isolates the one new default most likely to touch the 4-row batch window.
2. `STRATA_STAGER_SLEEP=0` (env): prefill-only arm.
3. `--pool-tasks 68` then `--pool-tasks 102` (args: add `"--pool-tasks", "68"`); 0 is the baseline. 16 workers + host = 17
   participants: 51 is the default, 68 = 4 per thread, 102 = 6 per thread.
4. `STRATA_DF_PLE=1` (env): solo-path decode only; do not expect any change in the 2-stream numbers.
5. Own-stack knobs not yet in prod, in this order: `--cpu-kernel fast`, `--pool-drain counters`, `--aux-cpus auto`,
   `STRATA_HC_FUSED=1` (+ `STRATA_HC_Q8=1`), `strata_tune batch_overlap=1`.
6. `--expert-cache-per-layer` (needs a restart and a cold hit-rate warm-up; compare the hit rate and the tok/s after the
   same number of requests).
7. Only if the clients sample (temperature > 0): `STRATA_SPEC_COUPLED=1`, then `STRATA_SPEC_COUPLED=1 STRATA_SPEC_GUMBEL=1`.
8. Closing arm, expected to lose against the batch-MTP baseline: `--batch 2 --batch-groups 2` with `--batch-mtp` and
   `--adapt-async 1` removed.

Needs a config (args) change: 3, 5 (`--cpu-kernel`, `--pool-drain`, `--aux-cpus`), 6, 8. Env-only (the config's `"env"`
block): 0 (`STRATA_ENGINE_READY_S`), 1, 2, 4, 5 (`STRATA_HC_*`), 7. Per request, no restart: `strata_tune` keys
(`batch_overlap`, `pcie_frac`, `pcie_balance`, `pipeline_windows`, `aux_cpus`, `cpu_kernel`, `pool_drain`, `q8k_avx2`).

## Tests (codebox, `-j2` Release build, `CMAKE_CUDA_ARCHITECTURES=86`, ctest commands run through a runner; GPU ones behind `flock ~/gpu.lock`)

Before = the `w2-dense-fix` build (`strata-w2/build`, 115 registered tests); after = this branch (124 registered: nine new
upstream tests).

- Before: 108 pass. After: 115 pass (all of the 108 plus `expert_profile_version_test`, `mmvq_il_parity`, `pdl_parity`,
  `spec_prob_test`, `spec_verify_parity`, `verify_batch_parity`, `pool_tasks_test`). No test that passed before fails now.
- Same failures before and after, all from this machine, not from code: `expert_multi_test` (the scalar test wants
  AVX512-VNNI/VBMI), `native_expert_parity_refuses_q6_K` (`STRATA_Q6K_EXPERTS` off), `platform_memory_test`
  (`ulimit -l`), `ple_parity` (no Q2_0 fixture), `prefill_mmq_kquant_test` (Q6_K not built), `q2_bitplane_parity`
  (needs `STRATA_Q2_BITPLANE=1`), and `prefill_fused_moe_test` / `prefill_fused_iq_test` which hit `cudaMalloc: out of
  memory` on the shared 20 GB cards and flip between pass and fail on both builds depending on what else holds VRAM.
- Named in the task: `hc_q8_parity` pass (the `w2-dense-fix` build failed it once with "STRATA_HC_FUSED BF16 read is NOT
  used: lo differs for 1 token" and passed on the repeat), `gr_parity` pass, `kq_fast_parity` (and its four env
  variants) pass, `pool_layer_test` (and its two env variants) pass, `draft_policy_test` (upstream's), `pool_tasks_test`,
  `memory_guard_test`, `verify_variant_test`, `expert_window_plan_test` pass.
- Built with `-DSTRATA_BUILD_CONVERSATION_TESTS=ON` (off by default; `conversation_cache_test` has a merged conflict):
  `conversation_cache_test` (4729 checks), `_split_failure`, `_memory`, `_file`, `_validation`, `_transfer`, `_snapshot`
  (needs a GPU) all pass.
- `serve/` unit tests (`python -m unittest discover -s serve`, jinja2 + regex + Pillow): 591 tests OK, 11 skipped.

## Deployment layout (lm-server, nothing production touched)

- `/opt/strata-bmtp/build/strata-w5`: the engine.
- `/opt/strata-bmtp/build/serve-w5/`: `serve/` of this branch (without `test_*.py`) incl. `web/`.
  `server.py` resolves `ROOT = parents[1]` of its own path for `tools/` and `serve/web`, so to run it from there use
  a root directory that has `serve/` and `tools/` beside each other: `/opt/strata-bmtp/build/w5root/` is that layout
  (`serve/` = a copy, `tools/` = the merged `tools/strata_tokenizer.py` and friends). Start it as the production launcher
  does (`runpy.run_module('serve.server')`) with `PYTHONPATH=/opt/strata-bmtp/build/w5root` and `"exe"` set to
  `build/strata-w5`.
