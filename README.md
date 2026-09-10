# Vessel Trajectory Prediction from Irregular AIS

Forecasts where a vessel will be — as a **distribution of plausible
futures**, from raw AIS reports at the times they actually arrive, the
traffic around it, and the surrounding coastline.

![Prediction example](assets/prediction.gif)

*Black: observed reports fed to the model. Blue: sampled predicted
futures. Red: what actually happened. Dashed/dotted: sample mean and
medoid.*

> **This is the primary branch.** A fixed-interval variant, which
> resamples AIS onto a uniform 15-minute grid, is kept on the
> [`main`](../../tree/main) branch as a comparison arm. See
> [Two approaches](#two-approaches).

---

## Why irregular sampling

AIS transmitters report on their own schedule — faster when a vessel is
manoeuvring, slower at anchor, and not at all when out of receiver
range. Measured on one day of Danish traffic (26.5M reports, 6,640
vessels), **per-vessel median reporting intervals span 10 s to 284 s, a
28× range.**

The usual response is to resample onto a fixed grid. That is convenient
offline but bakes in an assumption that is false in operation: that
observations arrive uniformly. It also discards bursts of reports and
*invents* positions by interpolation where none were transmitted.

This branch keeps observations where they are and makes time explicit:

- elapsed time since a vessel's own last report is a model input
- neighbours carry a **staleness** — a live system never has anyone's
  current position, only their last transmission
- attention is encoded over **real elapsed seconds**, not step index
- the sampling head is conditioned on each target's Δt, so it can be
  queried at arbitrary future times

---

## Benchmark

Other researchers can evaluate their own models against ours without
adopting this codebase. A benchmark set is a single `.npz` of observed
track segments and their true futures; entering a model takes one
method.

```python
from vtp.benchmark import Predictor, BenchmarkSet, default_baselines, compare_predictors

class MyModel(Predictor):
    name = "my-model"
    probabilistic = True
    def predict(self, window, n_samples=32):
        # -> (n_samples, len(window.target_times), 2) absolute lon/lat
        ...

bench = BenchmarkSet.load('benchmarks/danish_60min.npz')
df, results, _ = compare_predictors(default_baselines() + [MyModel()], bench)
```

Metrics use the standard names — ADE, FDE, minADE, minFDE — plus energy
score and spread–error correlation, reported over all windows and over
moving windows separately, and broken out by horizon.

**On baselines.** Results here are reported against *constant velocity*
(dead reckoning), not persistence. Persistence ("the vessel stays put")
has error equal to however far the ship moved, so beating it says only
that the model noticed motion. On the reference set below persistence is
about 2× worse than constant velocity; on easier data the gap is far
larger. Read `beats_const_velocity_pct`, not `beats_persistence_pct`.

See [BENCHMARK.md](BENCHMARK.md) for the format, metrics and how to build
a set.

### Reference set: `danish_60min`

1,898 windows, 193 vessels, one day of Danish AIS, ~62 min median
horizon, 953 moving / 945 stationary.

| method | ADE (moving) | FDE (moving) | skill vs CV |
|---|---|---|---|
| persistence | 7.94 km | 12.55 km | −1.05 |
| **constant velocity** | **3.60 km** | **6.13 km** | 0.00 |
| CTRV | 3.59 km | 6.46 km | −0.05 |

CTRV does not beat constant velocity here. Turn rate estimated from
three noisy AIS points is unreliable, and extrapolating an arc an hour
ahead amplifies that noise — which is itself a useful result: manoeuvring
needs to be *learned*, not extrapolated.

Model rows are pending a full training run scored on this set.

---

## Architecture

Following DeepMind's weather-forecasting lineage:

- **GraphCast** — a triangulated spatial mesh with graph-neural-network
  message passing, and predicting a **residual** from the last known
  state rather than an absolute position.
- **WeatherNext FGN** — a sampling head emitting many trajectories per
  forward pass, trained with an energy score loss.

| stage | module |
|---|---|
| triangulated mesh over the domain, denser near coasts and ports | `vtp/data/mesh.py` |
| ego-anchored graph per observation | `vtp/data/graphs.py` |
| heterogeneous GNN encoder | `vtp/models/irregular_vtp.py` |
| time-aware cached causal transformer | `vtp/models/attention.py` |
| Δt-conditioned FGN sampling head | `vtp/models/irregular_vtp.py` |

**Vessel features (14):**
`[lon, lat, sog, cog_sin, cog_cos, dt_norm, staleness_norm, type_onehot(7)]`

COG is sin/cos encoded — a raw 0–359 scalar would imply 359° and 1° are
nearly opposite. Vessel type is one-hot, not an ordinal id.

**Targets are normalized as velocity**, not displacement:

```
velocity   = (target_position - anchor) / Δt
normalized = velocity / velocity_scale
```

With horizons varying per window, a single displacement scale would
average 40-second moves together with 19-minute ones. Velocity
normalization makes them comparable; because the head is separately
conditioned on Δt, it can still widen its spread at longer horizons.

Recovering a position: `output * velocity_scale * Δt + anchor`.

---

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install -e .
```

`torch` is deliberately not a declared dependency — it must be installed
against a specific CUDA build, and pinning it here would pull a CPU wheel
and silently break GPU runs.

### Data

```bash
# coastline
mkdir -p data/ne_10m_land && cd data/ne_10m_land
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip && unzip ne_10m_land.zip

# ports: data/ports_denmark.csv with columns name,lon,lat

# AIS (~2 GB/day) -- keep on scratch, not home
wget http://aisdata.ais.dk/aisdk-YYYY-MM-DD.zip && unzip aisdk-YYYY-MM-DD.zip
```

---

## Usage

**Train** (single GPU):

```bash
python3 -m vtp.training.train \
    --ais-glob "/scratch/$USER/maritime/ais/aisdk-*.csv" \
    --n-underway 300 --n-stationary 300 \
    --min-ping-gap-sec 900 \
    --seq-len 16 --future-len 4 \
    --hidden 256 --n-layers 6 \
    --batch-size 32 \
    --n-epochs 60 \
    --val-every 5 \
    --checkpoint-path checkpoints/model.pt \
    --resume
```

**Multi-GPU** (DistributedDataParallel):

```bash
torchrun --standalone --nproc_per_node=4 -m vtp.training.train --distributed [same args]
```

`--min-ping-gap-sec` thins dense reporting and is the main horizon
control: with `future_len` targets the horizon is roughly
`future_len × min_ping_gap_sec`. Set it to the horizon you intend to
predict at, and report it.

**Validation runs every `--val-every` epochs** and saves a separate
`_best.pt` whenever held-out error improves. Training loss falling while
validation flattens is the overfitting signal; without periodic checks
you only discover it at the end.

```bash
python3 scripts/plot_history.py checkpoints/model_history.json
```

**Evaluate and visualize:**

```python
import vtp.viz.plots as viz

ctx = viz.setup(min_ping_gap_sec=900, n_underway=300, n_stationary=300, seq_len=16)
model, vel_scale, meta = viz.load_checkpoint_model('checkpoints/model_best.pt')

viz.evaluate(model, ctx.combined_val, vel_scale)
viz.per_horizon_errors(model, ctx.combined_val, vel_scale)
viz.plot_grid(model, ctx.combined_val, vel_scale, ctx)
viz.animate_prediction(model, ctx.combined_val, 0, vel_scale, ctx)
```

Pass the same selection arguments the training run used — `setup`
reproduces the train/val split, and mismatched parameters silently mix in
vessels the model trained on.

---

## Two approaches

| | this branch (`irregular-sampling`) | [`main`](../../tree/main) |
|---|---|---|
| input timing | raw reports at true times | resampled to a 15-min grid |
| missing data | represented as a longer Δt | interpolated |
| snapshot | one per ego-vessel report; neighbours carry staleness | one per timestamp, shared across ego vessels |
| positional encoding | sinusoidal over real seconds | learned, by step index |
| output | position at each of the next N observations | position at +15/30/45/60 min |
| runs on a live stream? | yes | not really |

The fixed-interval branch is ahead on accuracy so far. Regularizing the
input genuinely makes the learning problem easier. The trade is
deployability: a fixed-interval model must wait for bin boundaries,
interpolate positions it never observed, and has no way to express that a
neighbour's report is twenty minutes stale.

`notebooks/compare_approaches.ipynb` walks through the difference with
figures.

---

## Repo layout

```
vtp/
  data/         mesh, coastline, AIS ingest, caching, graph construction
  models/       GNN, time-aware transformer, FGN head, losses, inference
  training/     training loop, batching, distributed
  benchmark/    protocol, baselines, metrics, adapters, set builder
  viz/          evaluation, plots, animations, failure analysis
benchmarks/     shareable .npz benchmark sets
notebooks/      comparison walkthrough, analysis
scripts/        metrics export, sweeps, plotting
slurm/          batch scripts
```

---

## Known issues

- **Intermittent CUDA device-side assert** in batched training. Setting
  `CUDA_LAUNCH_BLOCKING=1` avoids it at a throughput cost; `--resume`
  makes an interrupted run cheap to continue. Cause not established.
- **Rate-of-Turn is dropped at ingest.** Failure analysis found turn
  angle separates the worst-predicted windows from the best by a factor
  of 4, so feeding ROT in is the obvious untried experiment.
- **k-NN uses degree-space Euclidean distance**, not haversine. Denmark's
  narrow latitude band (53.5–58.5° N) limits the distortion.
- **The benchmark's model adapter has not been run against trained
  weights** — its logic mirrors the evaluation path but is unverified.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for full history, every bug
found and why it mattered, and open questions.
