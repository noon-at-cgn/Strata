# Two independent lanes with MTP: what a model says before anything is built

Question: on a layer split with one shared CPU expert pool, would `--batch-groups 2` beside `--batch-mtp` (today refused,
`generate.cpp` "does not combine with --batch-groups", `DUAL_3080_Q4XL.md`) give two concurrent users more total speed than
today's merged-row 2-slot batch? Each user would be a *lane*: its own solo-size windows (its own rows, never merged with the other
lane's), lane A on stage 1 while lane B is on stage 0, and the pool serving both. No speculation: both lanes' windows are real.

The model is `include/strata/program/two_lane_sim.hpp` (header-only, CPU), its test `src/program/two_lane_sim_test.cpp`
(`two_lane_sim_test`, 142 checks) and a CLI, `two_lane_sim` (`src/program/two_lane_sim_cli.cpp`). It builds on the stage model of
`stage_sim.hpp` (same pool orders, same cross-check: one lane without a loop equals `stage_sim`'s serial window).
No engine code is used and no knob is added.

```
two_lane_sim --report                       # the whole study below, ~10 s
two_lane_sim --tier typical --lanes 2 --rows 3 --accept 0.75 --draft async --lat 0.8 --host 0.5
two_lane_sim --trace layers.csv             # replay per-layer costs: window,stage,layer,gpu_ms,cpu_ms
```

## What the model contains

- Two cards, each running **one lane's window at a time and holding it for the whole window, pool waits included**; one pool that
  serves **one layer at a time**, not preemptively, and is the host thread's work too.
- A lane's loop: stage 0 (24 layers) -> stage 1 (24) -> verdict -> commit and emit (host thread) -> the draft (second card, host
  waits for it as the engine's `draft()` is synchronous) -> a latency remainder -> the next window.
- Per layer: GPU wait (the host-visible "GPU-reach wait", split over layers by type: a QSA layer weighs 1.5 GDN layers, from the
  stage profile's 0.43 / 0.286 ms) with +-10% jitter; pool time `0.05 ms + 0.0787 ms x n` with `n` the distinct CPU experts of the
  layer, drawn per window (Poisson on a lognormal-mixed mean). Pool order for 2+ lanes: fifo, stage0-first, priority (stage 1 first),
  quantum.
- Rows per lane: experts grow with `rows^0.85`, GPU wait by 9.5% per extra row, pool cost by 5% per extra row per expert (the last
  two from the 4-row merged window against the 2-row one); MTP acceptance is one constant per row.

## Where the numbers come from (production log, lm-server, `/opt/strata-bmtp/strata-q4xl-prod.log`, 217 `strata batch:` reports)

"warm" = reports with at least 85% of routed entries in VRAM (80 two-slot reports, 19,380 windows); "typical" = 70-85% (41
reports, 18,153 windows; the recordings' median is 80%).

| window | stage 0 wait + pool | stage 1 wait + pool | commit + emit | window | tokens/window | CPU experts/layer |
|---|---|---|---|---|---|---|
| 2 slots merged (4 rows), warm | 11.25 + 10.29 | 10.33 + 7.31 | 1.14 + 1.84 | 43.8 ms | 3.38 | 3.67 |
| 2 slots merged, typical | 11.29 + 17.91 | 10.40 + 13.08 | 1.07 + 2.16 | 57.4 ms | 3.51 | 6.63 |
| 1 slot (2 rows), warm (594 windows) | 9.39 + 6.20 | 8.76 + 4.93 | 0.60 + 1.42 | 33.5 ms | 1.89 | 2.31 |
| 1 slot, typical (130 windows, one report) | 10.33 + 8.37 | 9.42 + 6.21 | 0.62 + 1.37 | 38.5 ms | 1.83 | 3.40 |
| solo serve path, warm (`strata serve: stage N`) | 7.64 + 6.08 | 7.44 + 4.45 | 0.46 | ~28 ms | 2.08 | ~2.3 |

Tokens per window: the report's "avg rows" is tokens kept per window (a 2-slot window with 87% acceptance keeps 3.4). The pool
cost fit is 0.0787 ms/expert solo (0.087 for 4 rows) plus 0.05 ms a call, which reproduces the measured stage pool times of all
four batch rows. The window remainder (33.5 - 29.3 = 4.2 ms for a lane) is split into commit/emit + draft on the host (2.0), and
2.2 ms of latency. `two_lane_sim --report` section 0 shows each preset reproducing its measured window within 1% (the test
asserts 2%).

## Result (in-window rates; ratios against today's merged batch of the same tier)

| | warm | typical |
|---|---|---|
| merged-row 2-slot batch (today) | 76.9 tok/s | 60.6 tok/s |
| time-sliced solo, serve path (2.08 tok in 28 ms; measured 80-85 once warm) | 74.2 (0.97x) | 74.2 (1.22x) |
| time-sliced solo, batch path (1 slot) | 55.7 (0.72x) | 47.3 (0.78x) |
| **two lanes** | **95.9 (1.25x)** | **80.4 (1.33x)** |
| two lanes' ceiling, nothing ever waits (2 x one lane) | 111.4 (1.45x) | 94.6 (1.56x) |

The merged batch's 57-67 tok/s at the clients matches its in-window 61 (typical) to 77 (warm).

**(b) Pool contention costs 16-18% of a lane's window** (5.4 ms warm, 6.8 ms typical) and the pair reaches 85-86% of its ceiling:
pool wait 3.6-4.9 ms + card wait 0.8-0.9 + host-thread wait 1.0-1.1 ms per window. The pool is busy 57-64% and the host
thread 67-73%, but the cards are the limit: card 0 is held 90-91% of the time (a stage-0 window, pool waits included, is 15.6 ms
warm / 18.7 typical, twice a cycle). Consequences:
- **With two lanes the pool order cannot matter**: when one lane is served the other is the only one waiting. fifo, stage0-first,
  priority and quantum give identical results (a test asserts it). With 3 lanes they differ by 2% (98.7 vs 96.6 tok/s warm), and
  3 lanes give 97.9 tok/s, no more than 2: the cards saturate.
- A pool 10% / 25% slower while both lanes run (DRAM or PCIe interference, unmeasured) costs 4% / 9%.
- The lane's loop matters more: an asynchronous draft that neither holds the host thread nor the second card gives 1.32x (warm)
  and 1.39x (typical); if all of the 4.2 ms blocks the host thread, 1.11x / 1.19x.

**(c) Sensitivities** (ratio against the merged batch under the same change):
- rows per lane 1 / 1.5 / 2 / 3 / 4 (acceptance 0.87 a row, constant, unmeasured beyond 1): warm 0.84x / 1.07x / 1.25x / 1.45x / 1.52x,
  typical 0.95x / 1.17x / 1.33x / 1.45x / 1.44x.
- CPU experts per layer x0.5 ... x2: 1.25x-1.22x (warm), 1.26x-1.33x (typical): the ratio does not move; the absolute rate does
  (116 -> 69 tok/s warm).
- GPU dense time x0.8 / x1.2: 1.22x / 1.28x (warm). Fable's -3.5 ms dense fusion: 1.24x / 1.32x; -30% pool kernel: 1.26x / 1.30x;
  both: 1.25x / 1.30x (and 117.8 / 99.8 tok/s absolute, against 94.2 / 76.8 for the merged batch given the same two changes).
  The improvements help the merged batch as much as the lanes.

**(d) A single user:** one request through the lane machinery is a batch-path window of one slot: 33.6 ms and 1.87 tokens = 55.7 tok/s
(warm; 47.3 typical) against 74 for the serve path today (measured 80-85): -25% to -36%. The engine must stay on the serve path with one
active request and take the lanes only when two are active.

## Go / no-go

Go if two lanes' in-window aggregate is **at least 1.5x today's merged batch of the same tier, with the pool's contention** (about
90 tok/s in the owner's 60 tok/s frame).

**NO-GO as the engine can run lanes today** (batch-path windows, one proposal per slot, synchronous draft): **1.25x warm, 1.33x
typical**, and even with no waiting at all the ceiling is 1.45x warm. The reason is structural, not the pool: a lane's own window
(33.6 ms for 1.87 tokens) is already long, the merged window shares its dense work and its loop between 3.4 tokens, and two lanes
can only double a lane.

It would reach 1.5x only with changes that are **not measured**, all of them single-lane improvements:
- E1: a lane window as lean as the serve path's (the batch path's GPU waits are 18.15 ms a window against 15.08 for the serve path
  in the same tier, and its loop 4.2 ms against 2.4): 1.38x warm / 1.42x typical.
- E2 = E1 + an asynchronous draft: 1.51x / 1.56x against today's merged batch, but 1.39x / 1.46x once the merged batch gets the same
  lean windows (E1's GPU saving applies to it as well).
- E3 = E2 + 3 rows per lane at acceptance 0.87: 1.72x / 1.69x; at 0.75 (E4) 1.52x / 1.55x (1.39x / 1.45x against the lean merged
  batch); E5 = E4 + fusion + pool kernel: 1.51x of its lean merged batch.

So the justification for two lanes would have to come from first measuring *why a 1-slot batch window is slower than a serve-path
window* (waits +20%) and whether the draft/commit loop can be made asynchronous; both help the merged batch too.

## Not in the model (each would lower the lane numbers further)

- `--adapt-async 1` is refused with `--batch-groups` (BATCHING.md "does not adapt"), so lanes would also freeze the VRAM tier;
  the cold-tier lines of the log (52-61% in VRAM) show a 1-slot window of 38-44 ms with 6.5-8 CPU experts per layer.
- KV-pool memory and slot sessions per lane, the pump's polling cost, the commit-with-prefix hazard (the window scratch of the next
  window on a stage overwrites what the earlier window's commit reads). Upstream #1253's author prototyped overlapped speculative slot
  groups and reports intermittent exact-token mismatches with unresolved cause (QFUSE #1139/#1209 ruled out on their setup).

## Search-first record

- `stage_sim.hpp` and its CLI: **known**, cherry-picked from `opt-gate` with `-x` (e168d46); `two_lane_sim.hpp` reuses its `StagePolicy`.
- `two_lane_sim*`: **new** (two independent streams with their own windows, a host thread shared by the pool and the loop, a
  draft that can hold the second card, per-window per-layer expert draws, a trace input).
- Upstream: #1253 (serial multi-GPU batch MTP; its text and blange48's 4-stage numbers: batch MTP 120 vs 123 tok/s serial, groups 4
  pipelined without MTP 378 vs 202 at 8 clients; pipelined groups with MTP "unrun"), #1249 / #1179 (landed as e21dc57: group
  windows only up to the last active slot, slot spreading over groups; needed by any lane implementation, not in this branch),
  #1413 (open: `--batch-groups` with `--batch-mtp` is silently inert), docs/BATCHING.md's measured pipelined-groups table. None
  has a two-lane-with-MTP measurement.
