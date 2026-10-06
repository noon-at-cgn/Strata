# UD-Q4_K_XL on 2x RTX 3080 20 GB: everything in VRAM or RAM, 512k shared KV pool

This is the setup the `kv-shared-pool` branch of this fork was built for: Unsloth **UD-Q4_K_XL** (111 GB) on
two RTX 3080 20 GB (PCIe 3.0 x16, no NVLink, no P2P, one NUMA node) with about 100 GiB of usable pinned RAM
and a Xeon E5-2696 v4 (AVX2 only), a **524,288-token KV pool shared by four lanes of up to 262,144 tokens each**,
and **no SSD or HDD read while serving**. The model sits on a spinning-disk pool, so any page-fault read of an
expert or PLE row is a failure of the goal, not a slowdown.

Branch contents (on top of upstream `main` 82f46a8): PR #1190 (the resident RAM copy keeps each stage's lend
region), PR #1011 (one pinned KV pool shared by the sessions, `--kv-pool-tokens`) ported from its pre-rewrite
history **and extended to a layer split** (one chunk table per lane, a device copy per stage GPU; #1011 refused a
split), a `k8v4` host-mirror fix in `verify.cpp`, `KvChunkMap` + `kv_pool_policy.hpp` with CPU tests.

## Memory budget (measured, this box)

| | |
|---|---|
| Experts in the two VRAM caches | 8,772 of 24,576 (35.7%): CUDA0 4,606 slots (13.45 GiB), CUDA1 4,166 (12.15 GiB) |
| Experts in locked RAM (the complement, no overlap with VRAM) | 53.7 GiB (includes the prompt path's lend regions, 7.6 GiB) |
| PLE table (`--ple-io ram`, mlocked) | 26.8 GiB |
| KV pool, pinned | 6.24 GiB for 524,288 cells (int8, both stages) |
| `MemAvailable` once serving | about 27 GiB |

The container is OOM-killed near 100 GiB of pinned memory (the host's other guests share the physical RAM), so
the arena mode (every expert in RAM, 71.7 + 26.8 GiB) cannot fit and the resident mode is the one that does.

## Config

`configs/dual-3080-ud-q4_k_xl.json` (paths are this box's; change them):

```
--pack <pack> --native <shard 1 of the 4> --expert-profile data/expert-profile.bin --expert-cache auto
--prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtp/rt>
--max-context 262144 --kv int8 --kv-resident 32768 --kv-pool-tokens 524288
--resident-experts --ple-io ram --trim-stage-weights --vram-reserve-mib 1100
"parallel": 4, "layer_split": "24", "gpu": [0, 1]
```

- **Explicit `layer_split`**, not `auto`: auto can pick a split the prompt path cannot start at long contexts (#1094).
- **`--resident-experts` + #1190**: the RAM copy is every expert no stage's cache holds plus each stage's lend
  region, so neither decode nor prefill reads the GGUF.
- **`--ple-io ram` needs a memlock limit above the table (26.8 GiB)**. A login shell is capped at 8 MiB; run under
  systemd with `LimitMEMLOCK=infinity` (or at least 32 GiB). Without it the table silently stays evictable and is
  re-read from disk.
- **`--vram-reserve-mib 1100`**: at 700 the MTP draft head did not fit (276 MiB needed, 236 free) after the cache
  took the rest; `--pipeline-windows` would want ~200 MiB more per card.
- **`--max-context 262144` per lane with a 524,288 pool** gives two full-length conversations of headroom and keeps
  the model inside its trained length (no YaRN). A single 512k conversation needs `--rope-scaling yarn --rope-scale 2`
  (untested on this quant).

## Results (one run each; MTP on)

| | |
|---|---|
| Decode, short context | 42-44 tok/s |
| Decode after a 27k context | 55-58 tok/s |
| Prefill, 5k / 27k / 113k / 232k tokens | 870-936 / 1,571-1,647 / 2,098-2,199 / 2,068 tok/s |
| 4 concurrent streams | 36.6 tok/s aggregate (about 10 each; batch slots decode without MTP on a split) |
| Disk read by the engine, all phases | 0-8 MiB per phase (140 MiB in a benchmark whose startup read 133 GB) |

Measured with `/proc/<engine pid>/io` `read_bytes` around each phase (mmap page-fault reads are counted there).

Pool stress: four concurrent 133,865-token prompts (556k tokens against a 524k pool, with a 233k conversation
already held) all answered 200 and each lane returned **its own** needle; the "KV pool full: slot N gives back its
cached conversation" eviction path ran; moving a 134k-token conversation between the main session and a slot took
about 0.2 s (chunk-table swap, no copy).

## KV in RAM or in VRAM

Same config, one lane, `--max-context 262144`, only `--kv-resident` differs:

| | KV streamed to pinned RAM (32768 cells/layer resident) | whole KV in VRAM |
|---|---|---|
| experts in VRAM caches | **10,009 (40.7%)** | 8,997 (36.6%) |
| experts in locked RAM | 50.1 GiB | 52.6 GiB |
| decode short / at 27k | **43.5** / 56.1 tok/s | 40.9 / 58.0 |
| prefill 27k / 110k | 1,647 / 2,199 | 1,619 / 2,157 |
| container memory | **94.4 GiB** | 103.2 GiB |

Streaming costs nothing measurable here (the streamed layers hit VRAM ~96%, misses are 2-4 KB contiguous runs
fetched by a UVA kernel), buys ~1,000 cached experts, and uses *less* total RAM because the VRAM it frees shrinks the
RAM complement. Keep KV in RAM.

## Not verified

- Greedy output is not reproducible run to run (solo or slot): the adaptive expert cache moves experts between GPU
  and CPU compute (DETAILS.md documents `--adapt-every 100000` for reproducible output). The solo-vs-slot exactness
  of the pool was therefore not testable; cross-lane isolation was (distinct needles).
- `--pipeline-windows 2`, `STRATA_PREFILL_HELP`, chunk size and `--pcie-frac` tuning; `k8v4` with the pool; parking and
  session files with the pool; YaRN at 512k; images (this file has no vision wiring in setup).
- Only CUDA was compiled (the HIP `block()` path of the pool is untested).
