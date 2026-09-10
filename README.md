# Vessel Trajectory Prediction — Fixed-Interval Branch

> **This is the comparison arm, not the primary model.** Active
> development is on the
> [`irregular-sampling`](../../tree/irregular-sampling) branch, which
> consumes raw AIS reports at their true arrival times. This branch is
> kept because it is a genuine, and currently stronger, alternative — and
> because a fair comparison needs both.

Forecasts a vessel's position as a **distribution of plausible futures**,
from its recent track, nearby traffic, and the surrounding coastline.
AIS is resampled onto a uniform 15-minute grid, so every forecast is
exactly 60 minutes ahead.

---

## What this branch does differently

| | this branch | [`irregular-sampling`](../../tree/irregular-sampling) |
|---|---|---|
| input timing | resampled to a 15-min grid, gaps interpolated | raw reports at true times |
| snapshot | one per timestamp, **shared across all ego vessels** | one per ego-vessel report |
| positional encoding | learned, indexed by step | sinusoidal over real elapsed seconds |
| output | position at +15/30/45/60 min | position at each of the next N observations |
| normalization | displacement ÷ global std | velocity ÷ global std |

The shared-snapshot design is the notable one. The world at a timestamp
does not depend on which vessel you are forecasting — only the choice of
ego row does. Building one graph per *(vessel, timestamp)* pair
duplicated the same graph hundreds of times; sharing it measured an
**856× memory reduction** and is what made multi-day training possible.
That optimization does not transfer to the irregular branch, where
snapshots are anchored to each vessel's own reports.

---

## Results

Held-out vessels, 12 days of AIS, 60-minute forecast, 5 epochs:

| metric | value |
|---|---|
| spread ↔ displacement correlation | 0.753 |
| mean error (moving vessels) | 5.37 km |
| beats persistence | 82.4% |

**Read these with care.** The "beats baseline" figure is against
*persistence* ("the vessel stays put"), whose error is just how far the
ship moved — a weak bar. The shared benchmark on the primary branch
reports constant velocity and CTRV instead, which is what belongs in a
comparison table. On the reference benchmark set, constant velocity
achieves 6.13 km FDE on moving vessels; this model has not yet been
scored on that set.

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install -r requirements.txt
```

Data staging (coastline, ports CSV, AIS) is identical to the primary
branch — see its README.

## Usage

```bash
python3 train.py \
    --ais-glob "/scratch/$USER/maritime/ais/aisdk-*.csv" \
    --n-underway 300 --n-stationary 300 \
    --seq-len 12 --future-len 4 \
    --hidden 128 --n-layers 4 \
    --batch-size 32 \
    --n-epochs 60 \
    --val-every 5 \
    --checkpoint-path checkpoints/model.pt \
    --resume
```

Multi-GPU:
```bash
torchrun --standalone --nproc_per_node=4 train.py --distributed [same args]
```

Evaluation and figures:
```python
import viz
ctx = viz.setup(n_underway=300, n_stationary=300, seq_len=12)
model, norm_scale, meta = viz.load_checkpoint_model('checkpoints/model_best.pt')
viz.evaluate(model, ctx.combined_val, norm_scale)   # reports both baselines
viz.plot_grid(model, ctx.combined_val, norm_scale, ctx)
```

## Scaling study

`sweep.py` measures mesh density, input context length and forecast
horizon on one axis at a time:

```bash
python3 sweep.py --axis mesh    --out-dir results/sweep_mesh
python3 sweep.py --axis context --out-dir results/sweep_context
python3 sweep.py --axis horizon --out-dir results/sweep_horizon
```

Findings at reduced scale (one day, 15 epochs — trends, not final
numbers):

- **Mesh density does essentially nothing.** Error spans 4.45–4.80 km
  across a 12× range of mesh nodes, while cost more than doubles. Worth
  an ablation to check whether the mesh contributes at all.
- **Longer context helps calibration more than accuracy** — spread
  correlation rises 0.43 → 0.67 from 1 h to 4 h lookback, best error at
  `seq_len 16`.
- **Error grows roughly linearly with horizon**, so longer forecasts
  degrade predictably rather than collapsing.

`analyze_sweeps.ipynb` plots them.

## Layout

This branch keeps a flat module layout (the primary branch is packaged
as `vtp/`). Files: `mesh.py`, `coastline.py`, `ais_ingest.py`,
`ais_cache.py`, `graph_data.py`, `cached_attention.py`, `model.py`,
`losses.py`, `batching.py`, `distributed.py`, `train.py`, `viz.py`,
`sweep.py`.

## Known issues

- Intermittent CUDA device-side assert in batched training;
  `CUDA_LAUNCH_BLOCKING=1` avoids it, `--resume` limits the cost.
- k-NN uses degree-space Euclidean distance rather than haversine.
- Notebooks may lag the current dataset API — `VesselSequenceDataset`
  returns `(ctx, ego_rows, target)`, and shared snapshots carry no
  `ego_mask`.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for full history.
