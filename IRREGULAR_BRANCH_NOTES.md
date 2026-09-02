# Irregular-sampling branch — design notes

Replaces fixed 15-minute binning + interpolation with raw, irregularly
sampled AIS pings and explicit time handling. Files here replace their
same-named counterparts on `main`; `mesh.py`, `coastline.py` and
`losses.py` are unchanged and carry over as-is.

## Why

Measured on one real DMA day: per-vessel median ping intervals span
**10s (p10) to 284s (p90)** — a 28x range. Under fixed binning the model
never sees this; it assumes every step is 15 minutes. That assumption is
false in live operation, where pings arrive whenever they arrive.

## The four changes

**1. Ego-anchored snapshots (`irregular_ingest.py`)**
A snapshot is anchored to one ego-vessel ping. The ego row holds its
exact reported state; every other vessel holds its most recent report
*before* that instant, with a staleness feature. This mirrors what a
live system actually knows — you never have a neighbour's current
position, only its last transmission. Neighbours are resolved by a
single ordered pass carrying a running last-known-state table (O(n) in
nearby pings; a per-neighbour asof-join would be intractable at ~29M
pings/day).

**2. Δt as a feature (14-dim vessel vector)**
`[lon, lat, sog, cog_sin, cog_cos, dt_norm, staleness_norm, *type_onehot(7)]`
`dt_norm` is log-scaled time since that vessel's own previous report;
`staleness_norm` is log-scaled age of a neighbour's last report. Without
these, "moved 2km in 40 seconds" and "moved 2km in 6 minutes" are
indistinguishable to the network.

**3. Continuous-time attention (`cached_attention.py`)**
The learned step-index positional embedding is replaced by a sinusoidal
encoding of *real elapsed seconds*. Under fixed binning this would be a
no-op (steps genuinely are uniform); with raw pings it's what lets
attention tell a tightly-sampled burst from a sparse one.
`step()` now takes an elapsed time rather than an integer position.

**4. Δt-conditioned FGN head (`model.py`)**
Each target's elapsed time is encoded and fed to the sampling head, so
the horizon is an explicit input rather than something inferred. One
noise draw is shared across all horizons of a given sample, keeping each
sample a coherent trajectory. At inference the model can be queried at
arbitrary future times, not only those seen in training.

## Normalization — the piece that makes this work

`main` normalizes displacement by a single global std, which is valid
only because every target is exactly 15 minutes out. Here targets are
real pings at whatever times they arrived, so one displacement scale
would average 40-second moves together with 19-minute moves —
reintroducing exactly the entanglement normalization was meant to
remove.

Targets are normalized as **velocity** instead:

```
velocity   = (target_pos - anchor_pos) / dt     # deg/sec
normalized = velocity / vel_scale               # dimensionless
```

A vessel holding constant speed yields the same normalized target
regardless of horizon, so the model only learns *departures* from
constant velocity. This does not flatten the uncertainty structure:
because the head is separately conditioned on dt, it can still widen its
spread at longer horizons.

Recovering a position: `pred = anchor + output * vel_scale * dt`.

## Verified

- Ingest preserves exact irregular ping times (context spans e.g.
  -831s…0 with uneven gaps); targets are real observations, never
  interpolated
- Ego always row 0; ego staleness 0, neighbours carry varied staleness;
  graph edge counts correct
- Δt-conditioning demonstrably changes head output for identical context
  and noise seed
- Continuous-time encoding distinguishes different real spacings
- **Streaming `rollout_step` matches full recompute to 6.3e-7 with
  genuinely irregular timestamps** — the property that makes live
  deployment viable
- Full train → checkpoint → resume → held-out validation cycle runs
  clean; validation reports the true median horizon from actual ping
  times

Accuracy numbers from the sandbox run are meaningless (4 epochs, 338
windows, synthetic vessels) — only the machinery is validated. Real
numbers require a run on Rivanna.

## Usage

```bash
python3 train_irregular.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-*.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 150 --n-stationary 150 \
    --min-ping-gap-sec 60 \
    --seq-len 12 --future-len 4 \
    --hidden 128 --n-layers 3 \
    --n-epochs 50 \
    --checkpoint-path checkpoints/irregular.pt \
    --resume
```

Knobs that don't exist on `main`:

- `--min-ping-gap-sec` (default 60) — thins ego pings closer than this.
  Some vessels report every 2-3s; consecutive reports that close carry
  almost no new information but would dominate window counts and make
  "next 4 pings" a sub-minute horizon. **This is the most important knob
  to tune** — it directly sets your effective forecast horizon.
- `--staleness-cutoff-sec` (1800) — drop neighbours whose last report is
  older than this
- `--neighbor-radius-deg` (0.5) / `--max-neighbors` (150) — spatial
  neighbour limits; the second bounds per-snapshot cost
- `--max-window-span-sec` — reject windows spanning longer than this,
  guarding against a window straddling a multi-hour AIS blackout

## Not done / open questions

- **No fair comparison to `main` yet.** The two branches now predict at
  different horizons (fixed 60 min vs. whatever the pings give), so
  headline numbers aren't directly comparable. A fair test needs either
  matching horizons or evaluating both at a common set of query times.
- **Multi-step rollout (`inference.py`) not ported.** The Δt-conditioned
  head can already emit a whole trajectory in one pass, so autoregressive
  rollout may not even be needed — worth deciding before porting.
- **Cost.** Per-snapshot neighbour resolution is heavier than binned
  lookup, and thinning at 60s yields more snapshots per vessel than 96
  daily bins. Watch dataset-build time and GPU memory on the first real
  run.
- `--min-ping-gap-sec` interacts with `--future-len` to set the horizon,
  and the right pairing hasn't been tuned against real data.
