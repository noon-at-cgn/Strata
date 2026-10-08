# up-141: the production patch stack on upstream 0.1.41

`up-141` = upstream/main fb58e0d (engine 0.1.41) + the patches production (`strata-w7` = `up-merge` 9098e89 + `opt/04`) needs.
Built as `strata-w10`. Written 2026-10-08. Everything below was observed unless marked [INFERENCE] or listed under "Not verified".

## Commits stacked (36 on top of fb58e0d, in order)

Included:

| group | commits (source ids) | note |
|---|---|---|
| #1190 resident split lend regions (Evan, open) | 39708ae 3ba11b1 6b6277a | conflict in expert_source.cpp, see below |
| shared KV pool (`--kv-pool-tokens`) | 3273bd0 bf6395e 7dd65f1 455689b f5397b8 d786779 041b4a2 85694b5 2540ce6 f47cec6 72b91fb (as origin/pr/kv-pool 2c7985e..3aefa11) | minus 3965498 (= upstream ef6c18a) and 1d80930 (= upstream 35435e3) |
| batch-mtp on a split | 63be998 f4852c6 (as 2a67d7b 484b49c) | minus 08ef4a3 (redundant, see F05) |
| batch adaptive tier | 347420d 3e9cecb | 3e9cecb conflicts with upstream 13eee45, see below |
| #1599 split-mtp-batch | 69fa861 5e76e1f | same tree as the reissued origin/pr/split-mtp-batch 3d57d0c |
| #1598 aux-cpus | 91f6238 b79f197 | same tree as origin/pr/aux-cpus 1e80ce9 |
| #1601 memory guard | 290cf3a a59ce5e | same tree as origin/pr/memory-guard dcb9fdd |
| parking | eab374a (#1163), 460b0a7 | 460b0a7 needs the docs commit below (it touches docs/DUAL_3080_Q4XL.md) |
| #1242 per-slot vision | d8d707e, 73f6950, 88b007d | d8d707e/73f6950 hand-merged with upstream's verify.cpp refactor |
| #1288 serve give-way | ffddd39 | |
| docs/config | 61bb01c 859bec4 e23f395 9c533de 384d67e | DUAL_3080_Q4XL.md and configs/dual-3080-ud-q4_k_xl.json |
| batch-groups fix | db925c1 | new |

Left out:

- Same content already in main (take upstream's): fda484f (35435e3), a7a59e6 (3bcfd8f), c4e0b52 (7ef55e1), 7db79e8 (5208b32), 878f335 (48d2e6c), 939af26 (ef6c18a), 1244892 (13eee45).
- f197c62: the type line is in main (569c094); the vocab line is dead in main.
- d4234dc (#1181 O(n) window plan): conflicts with upstream's Foresight `fs->take/note_miss` in the window-plan loop (the planner would need a Foresight hook in `WindowGpuPlanInput`); not needed, measured decode unchanged.
- 530afec d68cf5e 1485da0 (flag-B fold): depend on #1181 and pcie-balance, do not apply; unmeasured. w10 keeps flag B's wait as upstream does (STRATA_VERIFY_FLAGB has no meaning here).
- Not needed for the production config: pcie-balance (674a0af 7254d27 ad574a3 903af0d), pipeline-windows beside batch (acf426b d84da29 94cfdf1 5f645b0), batch-overlap/prefill_pipe_k (7bb8540 eb8c003 6de760f), q8k switch (b5a48ec), kq fast (c21c44e), counters drain (067b4d8 7689e8a 4bfd2ff 7e42aea), HC fused (00d791e 9588cc2 4e7ad11 77fea7e c46d0bb c45d854), doorbell histogram (f97fe9b 1dfa3dd 546d98c), OPTIMIZATION_KNOBS and merge notes docs (14e66a4 c6ac365 1d32fe1 999e78c 4098153 9098e89).
- #1600 lend-sizing: not needed (sets STRATA_PREFILL_RING, which is what production uses, not STRATA_SPLIT_RING).

## Conflicts and resolutions

- `src/core/expert_source.cpp` (3ba11b1, #1190 vs upstream #1389 warning 14d9d2e): kept both blocks (the `complement_lent_slots_` warning, then the `stage_lend_regions_` line); the conditions are exclusive.
- `serve/server.py` (kv pool 2c7985e): upstream `self._ctl_erred = True` plus our `raise engine_error(...)`.
- `src/program/generate.cpp` + `docs/DETAILS.md` (3e9cecb vs upstream 13eee45): gate comment = up-merge's text; the `--adapt-async` gate line is `o.batch > 0 && o.batch_groups > 1 ? "not with --batch-groups (the pipelined slot groups)"` replacing upstream's "not with --batch slots"; CopyBack: our `aswaps.empty()` early exit followed by upstream's pipelined `a_stage_in` / `res_upload` body; DETAILS.md paragraph = up-merge's.
- `src/core/conversation_cache_test.cpp` (460b0a7): both test blocks kept.
- `src/core/verify.cpp` (d8d707e vs upstream's `norm_rope_on`/`mm`/`copy_rows_strided`): `norm_rope` takes (first, count); when `private_positions` is false it calls upstream's `norm_rope_on`, otherwise norm over all rows then one rotate per slot under its own `MropeScope`; upstream's `mm(...)`, `copy_rows_strided`, the `qbr` branch (off when `batch_rec_`) are unchanged. 73f6950: the rotate loop is one launch per run of rows that share a slot's table.
- `serve/server.py`, `serve/test_parallel.py` (ffddd39): kept `reused0 = None` and the new `_take_control(..., prompt=prompt)`; kept both `message` and `chat_messages` test helpers.

## The batch-groups fix (commit db925c1)

Problem (AUDIT 3.3): 0.1.41 makes `--batch-groups auto` the default for a layer split with `--batch` >= 2, and the `--batch-mtp` gate ran before it was resolved, so batch-MTP stayed on while the groups went to 2 (pipelined groups run neither the slots' MTP drafts, #1413, nor the batch adaptive tier).

Fix: `include/strata/program/batch_groups.hpp` (pure `resolve()`), `src/program/batch_groups_test.cpp` (CPU test, registered in CMake), wired into `generate.cpp`: when `--batch-mtp` (after its other gates) or `--adapt-async 1` is requested and `--batch-groups` was not given, auto = 1 group and one line says why (`--batch-groups auto: 1 group of N slots - --batch-mtp does not run in pipelined groups; --batch-groups G would pipeline them and turn it off`); an explicit number is honoured; an explicit `auto` (or N > 1) keeps the pipeline and `--batch-mtp` is turned off by the existing gate (it now sees the resolved number). `docs/BATCHING.md` updated. Not done: the `--pipeline-windows` gate (not in this stack).

## Build

Release, CUDA arch 86 (nvcc 13.0), tests ON, conversation tests ON, `ninja -j2`: all 359 targets built. `strata --version` prints `strata 0.1.41`.

## Tests

- CPU only (`CUDA_VISIBLE_DEVICES=-1`, ctest -j6): 49 of 111 pass; the 56 failures are the missing GPU (the "no CUDA-capable device" family), plus the native_expert_parity_* "width invariance" and `ple_parity` (missing PLE table), which also fail in ~/strata-up. Baseline (~/strata-up/build, 131 tests): 59 failed. Failing in both: 54. Only in baseline: expert_multi_test, hc_q8_parity, prefill_mmq_kquant_test. Only in new: qsa_select_bench, qsa_select_bench_fp32_tiled (baseline: "Not Run", binaries not built there; here they fail with "no CUDA-capable device").
- Passing CPU tests of the stacked features: batch_groups_test, kv_chunks_test, batch_rows_test, vision_records_test, aux_cpus_test, memory_guard_test, conversation_cache_test and the other conversation tests, draft_kv_plan_test, split_lend_complement_test, file_expert_source_test, conv_cache_test.
- GPU 0 under `~/gpu.lock` (the codebox GPUs have only ~1 GiB free, another process holds ~19 GiB each): 106 of 111 pass. Failing: platform_memory_test and ple_parity (fail in the baseline too), prefill_fused_moe_test and prefill_fused_iq_test (cudaMalloc OOM, same in the baseline), gdn_rec_parity (cudaMalloc OOM after its T=1..3 checks; passes in the baseline; upstream added ~500 lines to that test after d5ea713 and our stack does not touch it; not confirmed on an empty GPU). mrope_slot_graph_test, verify_parity, verify_batch_parity, rope_parity, kv_stream_parity, kv_hybrid_parity, qsa_parity pass.
- `python -m unittest discover -s serve` (/tmp/prvenv): 615 tests OK, 11 skipped.

## Deployed (lm-server)

- `/opt/strata-bmtp/build/strata-w10`, sha256 `84f677f95f2c25affa484e597972c741638343e536d7eb37c7673b7c903a0d9f` (57277536 bytes).
- `/opt/strata-bmtp/build/w10root/{serve,tools}` (85 files, `test_*` and `__pycache__` excluded, as w5root). `import serve.server` works with /opt/strata-swift/.venv; psutil 7.2.2 present.
- The production config, service and the running w7 engine were not touched.

## Production config changes

- `exe`: `/opt/strata-bmtp/build/strata-w10`; the serve root of the launcher: `w10root`.
- `--batch-groups 1` is not required (the fix resolves auto to 1 for `--batch-mtp` / `--adapt-async 1` with no `--batch-groups`); adding it is harmless and makes it explicit.
- Env: keep `STRATA_ENGINE_READY_S=0`, `STRATA_SPLIT_MTP_BATCH=1`, `STRATA_PREFILL_RING=384`, `STRATA_PREFILL_EQUAL=1`, `STRATA_SPLIT_TIMING=1`, `STRATA_LOOKAHEAD=0`. Nothing to remove. (`STRATA_VERIFY_FLAGB`, HC_FUSED and other opt-w2 knobs are not in this stack; the production config sets none.)
- 0.1.41 defaults that act on this configuration (from `git diff d5ea713 upstream/main -- src/program/generate.cpp` and serve/): `--batch-groups` default auto (above); a warning when pipelined groups meet `--batch-mtp` (#1413); a pipelined slot is a conversation cache again (only with groups > 1); watchdog IO allowance `STRATA_WATCHDOG_IO_S` default 10 x `STRATA_WATCHDOG_S` (600 s) while the file tier is still read; the server's `STRATA_ENGINE_STALL_S` (90 s, new) ends an engine that is silent with no CPU and no disk use (needs psutil: present); parking admission now evicts the oldest parked conversation until a snapshot fits (new log lines); `STRATA_STAGE_PIN` is opt-in; CPU prefill share does not apply with a split plus batch; per-stage link scaling in `--layer-split auto` (production gives an explicit split, so [INFERENCE] no effect); `STRATA_ENGINE_READY_S` default 900 is unchanged from d5ea713.

## Author identity check

`git log --format='%an <%ae> | %cn <%ce>' upstream/main..up-141 | sort | uniq -c` (before this file's commit): 26 x `noon-at-cgn <75719527+noon-at-cgn@users.noreply.github.com>` (author and committer); Evan x3, Danny x4, Alex, BlueKingMuch, Jackwwg83 with their own author and committer `noon-at-cgn <75719527+...>`. No commit carries `noon@users.noreply.github.com`. The old email was rewritten with `git filter-branch --env-filter` limited to that exact address (author and committer) before the first push; trees unchanged. Note: `~/strata-fork/.git/config` still has a local `user.email = noon@users.noreply.github.com` that overrides the global one for the shared repo; commit with `-c user.email=75719527+noon-at-cgn@users.noreply.github.com` or remove that line.

## Not verified

- No engine was started (never load the model on codebox): the batch-groups log line, the serial resolution, `--batch-mtp` on the split, `--adapt-async` between batch windows, #1190 lend regions, the KV pool and #1242 with images are untested end to end on w10. The hand-merged verify.cpp rope path is covered only by the build, mrope_slot_graph_test and verify_batch_parity, not by a multimodal batch run.
- No performance measurement; w10 may differ from w7 (0.1.41 base, flag-B fold and #1181 absent).
- gdn_rec_parity failure is attributed to GPU memory pressure, not shown on a free GPU.
- The deployed serve layer was imported but not served; `w10root` is not wired into any service.
