# Maritime Vessel Trajectory Prediction

Predicts a single "ego" vessel's future positions from its recent AIS
track, the traffic around it, and the surrounding coastline/port
geometry — outputting a distribution of plausible future paths rather
than a single point estimate.

The architecture and key design choices are directly inspired by
DeepMind's weather forecasting lineage: a triangulated spatial mesh with
graph-neural-network message passing (GraphCast), and a functional
generative sampling head that produces many trajectory samples per
forward pass instead of one deterministic output (WeatherNext's FGN).
Most notably, the model predicts a **normalized residual displacement
from the last known position**, not an absolute coordinate — the same
residual-prediction approach GraphCast uses for atmospheric state, and
the single fix that took this model's predicted uncertainty from
uncorrelated-with-reality to a real, meaningful signal (see Validation).

Currently configured for Danish waters, using the Danish Maritime
Authority's public AIS archive.

## Architecture

1. **Spatial mesh** (`mesh.py`) — a triangulated graph over the maritime
   domain, denser near coastlines and ports, built once from Natural
   Earth coastline data and a port list.
2. **Heterogeneous graph per timestep** (`graph_data.py`) — mesh nodes
   plus every vessel present at that timestamp, connected via
   vessel-to-mesh and vessel-to-vessel (k-NN) edges.
3. **GNN encoder** (`model.py: MeshVesselGNN`) — two layers of
   heterogeneous graph convolution, producing a spatial embedding per
   vessel per timestep.
4. **Cached causal transformer** (`cached_attention.py`) — attends
   across the context window's timesteps. Supports both a full-window
   forward pass (training) and an incremental, KV-cached `.step()` mode
   (efficient multi-step rollout at inference).
5. **FGN sampling head** (`model.py: FGNDecoderHead`) — decodes many
   independent future trajectory samples per window, trained with an
   energy-score loss that rewards both accuracy and appropriate sample
   diversity.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Data

Requires three inputs, none committed to this repo:

1. **Coastline** — Natural Earth 10m land polygons:
```bash
   mkdir -p data/ne_10m_land && cd data/ne_10m_land
   wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip
   unzip ne_10m_land.zip
```
2. **Ports** — `data/ports_denmark.csv`, columns `name,lon,lat`.
3. **AIS data** — Danish Maritime Authority daily archives:
```bash
   wget http://aisdata.ais.dk/aisdk-YYYY-MM-DD.zip
   unzip aisdk-YYYY-MM-DD.zip
```
   Recommended to store on `/scratch/$USER/` rather than `/home`, given
   file sizes (~2GB/day) and `/home`'s smaller quota.

## Usage

**Interactive exploration**: `test_pipeline_denmark.ipynb` — mesh
construction, coastline/port visualization, a single training step, and
KV-cache correctness checks.

**Training** (interactive or via SLURM):
```bash
python3 train.py \
    --ais-glob "/scratch/$USER/maritime/ais/aisdk-*.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 150 --n-stationary 150 \
    --n-epochs 50 \
    --checkpoint-path checkpoints/checkpoint.pt \
    --resume
```
`--resume` continues from `checkpoint-path` if it exists, otherwise
starts fresh — safe to always include.

**SLURM**: see `train_gcvtp.slurm`.

## Key design decisions

- **Delta targets + unit-variance normalization** — predicting
  normalized displacement rather than absolute coordinates, matching
  GraphCast's documented residual-prediction approach. Empirically
  improved the correlation between predicted uncertainty and actual
  displacement from ~0 to a real, meaningful positive relationship (see
  Validation).
- **Stratified vessel sampling** (`ais_ingest.select_ego_vessels_stratified`)
  — training deliberately balances underway and stationary/anchored
  vessels, rather than excluding either. Excluding stationary vessels
  would leave the model unable to correctly predict "stays put," which
  is a common and valid real-world outcome.
- **NaN-safe resampling** — real AIS data has legitimately blank
  SOG/COG on some pings; `resample_to_snapshots` checks all of
  lon/lat/SOG/COG (not just position) before accepting a timestep.

## Validation

On a real single-day training run (27 vessels, 903 windows, 20 epochs):
- Predicted-uncertainty-vs-actual-displacement correlation: 0.68
  (vs. ~0 with absolute-coordinate targets).
- On genuinely moving vessels (>1km displacement), the model beat a
  trivial "assume no movement" baseline on 88% of windows (mean error
  3.23km vs. 9.08km baseline).

This was an early, small-scale run intended to validate the pipeline,
not a final result — see `train_gcvtp.slurm` for the full-scale
configuration.

## Repo structure
mesh.py -- spatial mesh construction
coastline.py -- coastline/port loading, domain bounds
ais_ingest.py -- AIS parsing, resampling, vessel selection
graph_data.py -- HeteroData graph construction, windowed dataset
cached_attention.py -- KV-cached causal transformer
model.py -- GNN + transformer + FGN head, full model
losses.py -- energy score loss
train.py -- standalone training script (SLURM-ready)
inference.py -- multi-step rollout / forecasting
train_gcvtp.slurm -- SLURM batch script
test_pipeline_denmark.ipynb -- interactive exploration notebook
