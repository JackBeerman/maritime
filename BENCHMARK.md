# Vessel Trajectory Prediction Benchmark

A portable evaluation protocol, so a baseline table can be reproduced
without adopting this codebase.

## Why it exists

Two problems this fixes:

**Baselines were too weak.** Results were reported against *persistence*
("the vessel stays put"), whose error is just how far the ship moved.
Measured on a synthetic set, persistence scores about **100x worse** than
constant velocity — so "beats the baseline 89% of the time" was a claim
about noticing motion, not about predicting it. The benchmark reports
persistence, constant velocity, and CTRV together, with skill scores
against constant velocity.

**Comparisons were not like-for-like.** Different models were evaluated
on different vessels at different horizons. A benchmark set fixes the
windows, the horizons and the ground truth in one file.

## Format

A benchmark set is a single `.npz` holding `Window` records. Each window
carries only what is observable at prediction time:

- ego vessel history: positions, **real elapsed times**, SOG, COG
- nearby traffic as of the anchor instant, each with a **staleness**
  (a live system never has a neighbour's current position)
- the times at which predictions are requested
- ground truth, used only for scoring

Times are seconds relative to the anchor, so context times are ≤ 0 and
target times > 0. Nothing about meshes, graphs or model internals appears
— the format cannot smuggle in an advantage for the model that produced
it. Windows come from **raw pings**, never resampled, so it does not
privilege either sampling approach; a fixed-interval model may resample
internally, and that information loss is part of what is being measured.

## Entering a model

Implement one method:

```python
from vtp.benchmark import Predictor

class MyModel(Predictor):
    name = "my-model"
    probabilistic = True          # set False for deterministic models

    def predict(self, window, n_samples=32):
        # -> (n_samples, len(window.target_times), 2) absolute lon/lat
        ...
```

Deterministic models return identical samples; the metrics handle that
and simply omit them from calibration statistics.

```python
from vtp.benchmark import BenchmarkSet, default_baselines, compare_predictors

bench = BenchmarkSet.load('benchmarks/danish_60min.npz')
df, results, per_window = compare_predictors(
    default_baselines() + [MyModel()], bench)
print(df.round(3))
```

## Metrics

Standard trajectory-prediction names, so results drop into a paper table:

| metric | meaning |
|---|---|
| **ADE** | average displacement error over predicted steps (from the sample mean) |
| **FDE** | final displacement error |
| **minADE / minFDE** | best of *k* samples — the usual way to score multi-modal predictors |
| **energy score** | proper scoring rule over the whole distribution; cannot be gamed by widening spread |
| **spread–error correlation** | does self-reported uncertainty track actual error? |
| **skill vs. reference** | `1 - error / reference_error`; positive beats it, negative is worse |

Metrics are reported over all windows and over **moving windows only**
(displacement > 1 km). Aggregates over everything are dominated by
stationary vessels where every method scores near zero; the moving subset
is where methods separate. Errors are also broken out **by horizon
bucket**, which is essential when windows have different horizons — a
single mean silently averages 5-minute and 90-minute problems.

A large gap between ADE and minADE is a signal to check calibration
rather than a win: a model can improve minADE simply by predicting
widely.

## Baselines

| name | description |
|---|---|
| `persistence` | vessel stays put. Weak; included to show how weak. |
| `constant-velocity` | dead reckoning from the last two reports, using real elapsed time. The honest reference. |
| `ctrv` | constant turn rate and velocity — the standard manoeuvring model. The right reference when turning matters, which failure analysis shows it does. |
| `cv+noise` | constant velocity with horizon-growing Gaussian spread. A probabilistic reference: a learned model claiming useful uncertainty should beat this on calibration, not just on point accuracy. |

## Building a set

```bash
python3 -m vtp.benchmark.build \
    --ais-path /scratch/$USER/maritime/ais/aisdk-2026-08-25.csv \
    --out benchmarks/danish_60min.npz \
    --seq-len 12 --future-len 4 \
    --min-ping-gap-sec 900 \
    --n-underway 100 --n-stationary 100
```

`--min-ping-gap-sec` is the horizon control: with `future_len` targets the
horizon is roughly `future_len × min_ping_gap_sec`. Set it to the horizon
you intend to benchmark at, and report it — comparing methods at
different horizons is meaningless.

Publish the resulting `.npz` alongside a paper and the table becomes
reproducible by anyone.
