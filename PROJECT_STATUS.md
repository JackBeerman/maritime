# Maritime Vessel Trajectory Prediction — Project Status & Handoff

Repo: https://github.com/JackBeerman/maritime
Environment: UVA Rivanna HPC, account `sds_baek_energetic`, user `jtb3sud`

**Two git worktrees, one repo:**
- `~/maritime` — always on `main` (fixed 15-minute interval sampling)
- `~/maritime-irregular` — always on `irregular-sampling` (raw irregular pings)

Worktrees rather than `git checkout` because SLURM jobs read code from
disk at launch; switching branches in a shared directory can hand a
queued job the wrong branch's files. Each worktree has its own files but
shares history and remote. Bulk AIS data lives at
`/scratch/jtb3sud/maritime/ais/` (12 days, 2026-08-18 .. 2026-08-29,
~236M records total); `venv` and `data/` are symlinked into the
irregular worktree and gitignored.

## What this project is

Predicts a single "ego" vessel's future positions from its recent AIS
track, surrounding traffic, and coastline/port geometry — outputting a
**distribution of plausible futures** via sampling, not a point estimate.

Architecture and key decisions follow DeepMind's weather-forecasting
lineage:
- **GraphCast** — mesh + GNN spatial structure, and predicting a
  **residual** from the last known state rather than an absolute value.
- **WeatherNext FGN** — a sampling head producing many trajectories per
  forward pass, trained with an energy score loss.

Domain: Danish waters, using the Danish Maritime Authority's public AIS
archive (`http://aisdata.ais.dk/aisdk-YYYY-MM-DD.zip`).

**Naming note:** the model class is still `GCVTP` ("Goal-Conditioned
Vessel Trajectory Predictor") from a pre-rebuild design.
Goal-conditioning was fully removed; the name is legacy.

## Current status

### `main` — fixed 15-minute intervals

Best completed run (1 day, 190 train / 47 val vessels, 50 epochs,
held-out validation):
- spread-vs-displacement correlation **0.719**
- beat "assume no movement" baseline on **78.6%** of moving windows
  (6.97 km vs 11.56 km mean error)

**A 12-day run is in progress** (480 train / 120 val vessels, 113,785
train windows, `--seq-len 12`, `hidden 128`). Confirms the dedup fix
works — 1,152 shared world snapshots on GPU, no OOM. But it runs at
**~2 hours/epoch**, so it will hit the 12-hour wall around epoch 6.
`--resume` works, so nothing is lost; results at the wall should still
be informative (~6 epochs × 113k windows is roughly 75 epochs' worth of
gradient updates at the previous data scale).

### `irregular-sampling` — raw irregular pings

Built and unit-tested; **no completed training run yet.** A smoke test
(40 vessels, 18,246 windows) trained cleanly (loss 1.21 → 0.61 in one
epoch) but timed out. A larger run is in progress and has not completed
an epoch — almost certainly just slow (~7 h/epoch projected at 150+150
vessels), not hung.

## THE BINDING CONSTRAINT: `batch_size=1`

Both branches now produce 100k+ training windows and **neither can
complete epochs at a usable rate.** Every optimization so far
(KD-tree neighbour search, world-snapshot dedup, mesh sharing) addressed
*memory* and *setup* cost; none touched per-step training throughput.

Each step is one forward/backward over a full GNN (~5,600 mesh nodes +
thousands of vessels) for a single window, so the A100 is largely idle
on Python and graph overhead.

**Next work, in priority order:**

1. **Batching.** PyG supports batching heterogeneous graphs of variable
   size; plausibly 5-10x throughput. Invasive — touches the training
   loop, collate function, and ego-vessel indexing, which has produced
   two real bugs already — so it should be its own isolated change with
   its own tests. This is now what blocks using the collected data.
2. **Window subsampling.** A stride knob in `VesselSequenceDataset`
   so window count is something you control rather than a function of
   data volume. ~30 minutes of work; gets both branches running today at
   reduced data.
3. **Epoch-count sanity.** With 12x the data, 50 epochs was never the
   right target — gradient updates matter, not epochs. 10-15 epochs at
   the new scale ≈ 100-150 at the old.
4. **CPU RAM on multi-day loads.** Concatenating 12 daily CSVs into one
   DataFrame (~236M records) OOM'd a 64 GB job. Structural fix: load and
   resample each day separately, keep only the (much smaller) resampled
   snapshots, free each DataFrame before the next. Would cut peak CPU
   memory ~12x. Currently worked around with `--mem`.

## Architecture

1. **Spatial mesh** (`mesh.py`) — Delaunay triangulation over the
   domain, denser near coastlines and ports. Built once from Natural
   Earth coastline + Danish ports CSV. Node features
   `[lon, lat, is_land, is_port]` (raw coordinates).
2. **Graph per timestep** (`graph_data.py`) — mesh nodes + vessel nodes;
   edges `mesh-to-mesh` (triangulation), `vessel-near-mesh` (3 nearest),
   `vessel-near-vessel` (6 nearest). KD-tree neighbour search.
3. **GNN encoder** (`model.py: MeshVesselGNN`) — two `SAGEConv`
   heterogeneous layers, mean aggregation. Normalizes lon/lat (to domain
   bounds) and SOG internally; stored tensors keep raw values because
   k-NN, targets and plotting all read columns 0-1 as real coordinates.
4. **Cached causal transformer** (`cached_attention.py`) — full-window
   pass for training, incremental KV-cached `.step()` for streaming
   rollout. Verified exact against full recompute (~1e-6 CPU).
5. **FGN sampling head** — context + noise → many trajectory samples.
   Output is a **normalized residual**, not an absolute coordinate.

### Feature layouts (differ by branch)

`main` (12 dims):
`[lon, lat, sog, cog_sin, cog_cos, *type_onehot(7)]`

`irregular` (14 dims):
`[lon, lat, sog, cog_sin, cog_cos, dt_norm, staleness_norm, *type_onehot(7)]`

COG is sin/cos encoded (a raw 0-359 scalar implies 359° and 1° are
nearly opposite); vessel type is one-hot (a raw id implies false
ordering between categories).

### How the branches differ

| | `main` | `irregular-sampling` |
|---|---|---|
| Time handling | resample to 15-min bins, interpolate single gaps | raw pings, no interpolation |
| Snapshot | one per timestamp, **shared by all ego vessels** | ego-anchored; neighbours = last report before that instant + staleness |
| Positional encoding | learned, indexed by integer step | sinusoidal over **real elapsed seconds** |
| Head conditioning | context + noise | context + noise + **target Δt** |
| Target | position at fixed +15/30/45/60 min | actual observed pings at true elapsed times |
| Normalization | displacement / global std | **velocity** (displacement ÷ Δt) / global std |
| Dedup possible? | yes (world state is shared) | no (snapshots are per-ego-vessel); only mesh is shareable |

Motivation for the irregular branch: measured per-vessel median ping
intervals span **10s (p10) to 284s (p90)** — a 28x range. Fixed binning
hides this and assumes uniform spacing, which is false in live
operation.

## Key findings from diagnostics

**Failure mode is turning, decisively.** Comparing worst-20% to best-20%
of held-out windows by error ratio:

| factor | worst 20% | best 20% | ratio |
|---|---|---|---|
| turn angle (deg) | 40.87 | 10.09 | **4.05** |
| speed variability | 0.01 | 0.00 | 3.12 |
| distance to port (km) | 83.65 | 87.04 | 0.96 |
| vessels in snapshot | 3362 | 3228 | 1.04 |
| SOG (knots) | 5.73 | 7.53 | 0.76 |

Nothing but turning (and speed variability, likely correlated)
distinguishes failures. Not congestion, not port proximity. Prompted the
move to `--seq-len 12` (3 h lookback) so the model can see turn *rate*,
not just current heading. **AIS Rate-of-Turn (ROT) is still dropped
during ingest** — adding it is the obvious untried next feature.

**Error grows linearly with horizon**, not exponentially:
+15 min 1.59 km, +30 min 3.43 km, +45 min 5.43 km, +60 min 7.58 km
(~1.9 km per step). Longer forecasts degrade predictably.

**Longer windows cost less data than expected**: 2 h → 4 h total span
drops windows only 22% (1177 → 919), though vessel count falls faster
(32 → 22), which matters more for generalization.

## Key bugs found and fixed

1. **KV cache concatenation** — concatenated along the heads dimension
   instead of sequence. Silent corruption, not a crash.
2. **`pyg-lib` dependency** — PyG's `knn_graph()` needs an optional
   package with version-pinned wheels. Replaced with in-repo KD-tree.
3. **NaN from blank SOG/COG** — real AIS legitimately has these; the
   resampler only checked `lon`. Now checks all of lon/lat/sog/cog.
4. **Absolute-coordinate targets (the big one)** — loss was dominated by
   a task-irrelevant ~55°N offset. Switched to residual displacement,
   matching GraphCast. Correlation ~0 → 0.19.
5. **Missing output normalization** — targets normalized to unit
   variance took correlation 0.19 → **0.68**. Single biggest lever in
   the project.
6. **`HeteroData.to(device)` mutates in place** — combined with sliding
   windows sharing snapshot objects, calling it per-batch silently
   corrupted other windows mid-epoch. Fixed by moving to device once at
   dataset-build time.
7. **`inference.py` not updated for delta targets** — treated raw output
   as absolute position during rollout.
8. **World-snapshot duplication → CUDA OOM** — a graph was built per
   *(ego vessel, timestamp)*, but the world at a timestamp is identical
   regardless of which vessel you're forecasting. Measured **856x**
   redundancy (3.6 GB → 4.3 MB on a 300-vessel benchmark); at full scale
   178-710 GB vs 1.2 GB needed. Fixed on `main` by building one shared
   graph per timestamp with the ego row index carried alongside; the
   irregular branch can only share the mesh (~32%), since its snapshots
   are genuinely per-ego-vessel.
9. **CPU RAM OOM on 12-day load** — unrelated to the GPU fix; see next
   steps item 4.
10. **GPU non-determinism (not a bug)** — a ~2% divergence in a
    cache-vs-recompute check is ordinary CUDA scatter nondeterminism,
    reproducible with two plain GNN calls on identical input.

## Design decisions and reasoning

- **Stratified (not exclusive) vessel sampling.** Stationary vessels are
  sampled, not excluded — a deployed predictor must handle "stays put."
  The original failure was an *unbalanced* distribution plus a decoder
  that couldn't express "it depends"; both are addressed.
- **Single ego vessel, conditioned on others' past/current state only.**
  Conditioning on neighbours' *future* state would be an oracle leak.
  Joint multi-vessel prediction is a legitimate later step.
- **Train/val split by vessel, not window.** Windows from one vessel are
  highly correlated; a window-level split would leak and overstate
  generalization.
- **Always segment AIS statistics by movement regime.** A naive
  full-population displacement check (mixing stationary and underway)
  gave a misleading near-zero median that would have led to a wrong
  interval choice.

## Environment

```bash
cd ~/maritime
python3 -m venv venv && source venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Validated: torch 2.9.1+cu128, torch_geometric 2.8.0.post1,
geopandas 1.1.4, shapely 2.1.2. Runs on A100 and RTX A6000. No
container — the venv is the tested environment.

**Rivanna access:** `rivanna.hpc.virginia.edu` is dead; use
`login.hpc.virginia.edu`. Off-campus SSH needs UVA VPN. Open OnDemand
(`https://ood.hpc.virginia.edu`) works from anywhere without VPN.

**Storage:** code in `~/maritime` (200 GB quota, snapshotted); AIS in
`/scratch/jtb3sud/maritime/ais/` (10 TB, no backup, 90-day inactivity
purge).

## Running

```bash
cd ~/maritime           && sbatch train_gcvtp.slurm       # fixed-interval
cd ~/maritime-irregular && sbatch train_irregular.slurm    # irregular
```
Always submit from the matching worktree. Both scripts use `--resume`,
which is safe whether or not a checkpoint exists.

Irregular-only knobs: `--min-ping-gap-sec` (thins dense pings; **this
sets your effective forecast horizon** and is the most important one to
tune), `--staleness-cutoff-sec`, `--neighbor-radius-deg`,
`--max-neighbors`, `--max-window-span-sec`.

## Open questions

- **The branches aren't comparable yet.** `main` predicts a fixed 60 min
  ahead; `irregular` predicts whatever the pings give (median ~14 min at
  60 s thinning). A fair test needs matched horizons or evaluation at
  common query times.
- **`inference.py` not ported to the irregular branch.** The
  Δt-conditioned head emits a whole trajectory in one pass, so
  autoregressive rollout may not be needed at all — decide before
  porting.
- **Notebook cells are stale on `main`** — `VesselSequenceDataset` now
  returns `(ctx, ego_rows, target)` (three values), and shared snapshots
  carry no `ego_mask`. The prediction plots, GIF animation, and
  failure-analysis cells all need updating to use `ego_rows`.
- **Haversine vs Euclidean neighbours.** k-NN uses raw-degree Euclidean
  distance; at higher latitudes a degree of longitude is shorter than a
  degree of latitude. Denmark's narrow band (53.5-58.5°N) limits the
  impact. Deferred as it means touching well-tested topology code.