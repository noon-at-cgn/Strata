# `--pcie-balance`: the PCIe share chosen per layer from measured costs

A verify window (decode, speculative, batch) finds, per layer, the experts that are not in VRAM. Each one is computed
either by the CPU pool, or by the GPU after it read the expert over PCIe (the expert blobs sit in page-locked RAM).
Both happen at the same time, so a layer waits for the slower side:

```
layer time(m) = max(m * t_pcie, (nmiss - m) * t_cpu)        m = the missed experts the GPU reads
```

`t_pcie` is the time to move one expert over the link, `t_cpu` the time the pool needs for one expert.

## What changes

**Fixed share (default, unchanged).** `--pcie-frac F` sends `floor(nmiss * F)` of a layer's missed experts over PCIe.
With `--pcie-frac 0.1` that is 0 for every layer with fewer than 10 misses, so almost everything goes to the pool.

**Balanced share (`--pcie-balance`).** For every layer it picks the `m` between 0 and `nmiss` with the smallest layer
time above, from the two costs measured on this machine:

- `t_cpu`: the pool's time per expert, measured on every layer that has at least 3 CPU experts (a running average,
  per stage, scaled by the blob size). It is kept twice: with nothing read over PCIe, and while the GPU reads some
  (the pool and the link read the same RAM, so the pool can be slower then; the second cost is used for the
  choice once it has 16 samples, and every 64th layer that would otherwise skip the link reads one expert so that
  cost stays current).
- `t_pcie`: one DMA probe per stage and link, a few pinned expert blobs of that stage copied into the stage's own
  staging area between windows (about 100 MB moved, a few ms; no new pinned RAM, no new VRAM). It is measured at the
  start of the first request that turns the mode on and again when the last reading is older than a minute. The two
  cards have separate probes. If a link cannot be measured (for example no pinned blobs, or a window in flight on a
  batch-groups pipeline), that stage keeps the fixed share and the log says so.

Until both costs are known (the link probed and 16 layers timed) no expert goes over PCIe.

**Hysteresis.** A cost used for a choice only follows its running average when the two differ by more than 8%, and
among choices within 3% of the best layer time the one with the fewest PCIe copies wins. So noise does not make the
share flap.

**`--pcie-frac` as the bound.** While the balance is on, `--pcie-frac F` is the upper bound: at most
`ceil(nmiss * F)` experts per layer, and `--pcie-frac 0` never uses PCIe. With the production value 0.1 that allows 1
expert for 1-10 misses and 2 for 11-20; give a larger `pcie_frac` (for example 0.5) to let the balance use its whole
range. The staging area also bounds it (8 experts per layer when a window is cut into two token groups, 16 otherwise).

## Switches

| | |
|---|---|
| `--pcie-balance` | on for every request (the engine's `--serve`) |
| `STRATA_PCIE_BALANCE=1` | the same, from the environment |
| `strata_tune: {"pcie_balance": 1}` (or `true`) in a request | on for that request; `0` / `false` off for that request. Beats the flag and the variable. |

The request key works like `pcie_frac` in `strata_tune`: it is read when the request starts and then holds for every
verify window until the next request sets it, so with several streams decoding the most recent request's setting is
the one in force. Everything is off by default.

A/B without a restart, with the production flags (`--pcie-frac 0.1`):

```
A: strata_tune {"pcie_balance": 0}
B: strata_tune {"pcie_balance": 1, "pcie_frac": 0.5}
```

`pcie_frac` 0.5 is the bound for B; it is not a target. `{"pcie_balance": 1}` alone bounds the share at
`ceil(nmiss * 0.1)`.

## What the log shows

With the mode on, the end of each request prints one line (the numbers below only show the format):

```
strata serve: pcie balance: 3.10% of the 41200 routed entries went over PCIe (cap 0.50), 118 experts chosen over
2140 layers; ms per expert: CUDA0 t_cpu 0.071 (0.078 while the GPU reads) / t_pcie 0.268; CUDA1 t_cpu ... / t_pcie ...
```

The share is the realised one (the same number as the `+N% of the routed experts over PCIe` in the existing
`decode expert cache hit rate` line, which is printed with the mode off too). The ms values are per expert of the
stage's first layer's blob size. The `strata batch:` line ends its adaptive-tier text with the same per-stage costs
while the mode is on. `STRATA_SPLIT_TIMING=1` still gives the host wait and pool time per window and stage.

## Limits

- `t_pcie` is a DMA measurement. The default `--pcie-mode auto` stages the PCIe share with a copy kernel inside the
  window's graph, which reads the same pinned memory over the same link; its time was not measured separately and
  the engine has no GPU-side timing for it. If the probe is far from the in-window cost, bound the share with
  `pcie_frac`.
- The copy overlaps the pool: the plan is published before the pool starts, and in `auto`/`kernel` mode the GPU runs
  the copy kernel (and the PCIe experts) while the host thread computes the CPU share; in `dma` mode the host only
  queues the copies before the pool starts. The graph waits for both the copies and the CPU rows before the combine,
  so the layer takes the longer of the two - what the formula above models. A copy per layer is also traffic on the
  RAM the pool is reading, which is why the pool's cost while the GPU reads is tracked separately.
- Decode and speculative windows only; the prompt path is not touched.
- Greedy output can differ between runs of different shares (which experts the GPU computes changes their rounding),
  as it already does with `--pcie-frac` and the adaptive tier.

CPU tests: `ctest -R pcie_balance` (the choice against a brute force, the bound, cold start, hysteresis,
contention and exploration).
