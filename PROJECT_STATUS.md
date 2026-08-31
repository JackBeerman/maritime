# Maritime Vessel Trajectory Prediction — Project Status & Handoff

Repo: https://github.com/JackBeerman/maritime
Environment: UVA Rivanna HPC, account `sds_baek_energetic`, user `jtb3sud`
Code lives at `~/maritime` (home dir); bulk AIS data lives at `/scratch/jtb3sud/maritime/ais/`

## What this project is

Predicts a single "ego" vessel's future positions from its recent AIS
track, the traffic around it, and the surrounding coastline/port
geometry — outputting a **distribution of plausible future paths**
(via sampling), not a single point estimate.

The architecture and key design choices are deliberately modeled on
DeepMind's weather forecasting lineage:
- **GraphCast**: mesh + graph-neural-network spatial structure, and
  predicting a **residual/delta** from the last known state rather than
  an absolute value (this was the single most impactful fix made so
  far — see "Key bugs found" below).
- **WeatherNext's FGN (Functional Generative Network)**: a sampling
  head that decodes many independent trajectory samples per forward
  pass instead of one deterministic output, trained with an energy
  score loss.

Domain: Danish waters (`DMA_BOUNDS` in `coastline.py`), using the
Danish Maritime Authority's free public AIS archive
(`http://aisdata.ais.dk/aisdk-YYYY-MM-DD.zip`). An earlier South China
Sea configuration was explored and abandoned early in the rebuild; no
trace of it remains in the current codebase.

**Note on the name**: The repo/model class is still called `GCVTP`
("Goal-Conditioned Vessel Trajectory Predictor") from an earlier,
pre-rebuild design. Goal-conditioning was fully removed from the code
during this rebuild (see below) — the name is legacy and doesn't
reflect current functionality. Consider renaming the repo/class if it
becomes confusing.

## Current status (as of last session)

1. Full pipeline validated end-to-end on real Danish AIS data, on
   Rivanna, via both interactive notebook and SLURM batch jobs.
2. Completed one full 50-epoch SLURM training run (before the latest
   round of fixes below) on 190 train / 47 val vessels (one day of
   data, 2026-08-25), with **held-out validation** (not in-sample):
   - Spread-vs-actual-displacement correlation: **0.68**
   - On moving vessels (>1km): beat a trivial "assume no movement"
     baseline on **80%** of windows (6.98km vs 11.56km mean error)
   - Loss curve had **plateaued** by ~epoch 20-25 (0.745-0.760 band,
     no further improvement through epoch 49) — more epochs on this
     single day of data were not helping further.
3. **Just applied three more fixes (untested at full scale — this is
   the immediate next step)**: vessel-type one-hot encoding, COG
   sin/cos encoding, and input feature normalization (see below).
   Verified correct via unit tests and a full small-scale local
   `train.py` run, but **not yet run at full scale on Rivanna**.
4. This change **breaks checkpoint compatibility** (vessel feature
   width changed 8 -> 12) — any old `checkpoints/checkpoint.pt` must
   be deleted/renamed before the next run; `--resume` will crash
   trying to load an incompatible shape into the new model otherwise.

## Immediate next steps (in likely priority order)

1. **Delete/rename the old checkpoint**, then run the updated
   `train.py` on Rivanna via SLURM (smoke test first at small scale,
   e.g. `--n-underway 30 --n-stationary 30 --n-epochs 5`, then the
   full run) to confirm the new features/normalization actually help
   held-out validation numbers, not just that the code runs.
2. **Get more days of AIS data.** Training has so far only ever used
   a single day (`aisdk-2026-08-25.csv`). Given the loss plateau
   observed with more epochs on one day, additional days (different
   traffic patterns, days of week) are likely the biggest remaining
   lever for real improvement — more so than further epoch count or
   architecture tweaks on the same single day.
   Download via: `wget http://aisdata.ais.dk/aisdk-YYYY-MM-DD.zip`
   into `/scratch/jtb3sud/maritime/ais/`, then unzip. `train.py`'s
   `--ais-glob` picks up all matching files automatically.
3. Once more data is in the mix, consider increasing model capacity
   (`--hidden`, `--n-layers`) — current `hidden=64, n_layers=3` is
   fairly small, and the loss plateau on one day's data may partly
   reflect limited data more than limited capacity, so scale data
   before scaling the model to avoid overfitting.
4. **Known limitation, not yet addressed**: vessel-to-mesh and
   vessel-to-vessel k-NN graph construction
   (`assign_vessels_to_mesh`, `knn_graph_manual` in `graph_data.py`)
   uses raw-degree Euclidean distance, not true haversine distance —
   at higher latitudes a degree of longitude covers less real distance
   than a degree of latitude, so neighbor-finding is subtly distorted.
   Not fixed yet because it means touching well-tested graph-topology
   code; worth revisiting once more data/capacity changes are settled.
   Denmark's narrow latitude band (53.5-58.5N) limits how much this
   actually matters in practice.
5. Longer-term / not yet started: multi-vessel joint prediction
   (currently strictly single-ego-vessel-at-a-time, conditioned on
   others' past/current state only — this was a deliberate early
   scoping decision, see "Design decisions" below).

## Architecture

1. **Spatial mesh** (`mesh.py`) — a triangulated graph over the
   maritime domain, denser near coastlines and ports (built via
   `sample_domain_points` + Delaunay triangulation in `build_mesh`).
   Built once from Natural Earth 10m coastline polygons + a hand-built
   Danish ports CSV. Node features: `[lon, lat, is_land, is_port]`
   (raw real-world coordinates).

2. **Heterogeneous graph per timestep** (`graph_data.py`) — for each
   AIS snapshot, builds a `HeteroData` object with `mesh` nodes (fixed,
   from step 1) and `vessel` nodes (every vessel present at that
   timestamp). Edge types: `mesh-to-mesh` (triangulation),
   `vessel-near-mesh` (each vessel to its 3 nearest mesh nodes),
   `vessel-near-vessel` (k=6 nearest other vessels). One vessel per
   snapshot is flagged `ego_mask=True`.

   **Vessel feature layout (12 dims, as of the latest fix):**
   `[lon, lat, sog, cog_sin, cog_cos, *type_onehot(7)]` — lon/lat/sog
   stay as raw real-world values in storage (normalized only inside
   the model at consumption time — see point 3); COG is sin/cos
   encoded rather than a raw 0-359 scalar (raw would falsely tell the
   network that 359 deg and 1 deg are nearly maximally different, when
   they're 2 deg apart); vessel type is one-hot encoded rather than a
   raw ordinal-looking integer.

3. **GNN encoder** (`model.py: MeshVesselGNN`) — two layers of
   heterogeneous graph convolution (`SAGEConv`, mean aggregation)
   across all four edge types. **Normalizes lon/lat (centered/scaled
   to domain bounds) and SOG (scaled by a typical max speed) internally
   right before the linear projection** — this only affects what the
   network sees, not the stored raw coordinates, since k-NN
   construction, target/displacement computation, and plotting
   elsewhere in the pipeline all depend on reading those columns as
   real coordinates.

4. **Cached causal transformer** (`cached_attention.py`) — attends
   across the context window's timesteps (default 4 x 15-min steps =
   1 hour of lookback). Two interchangeable modes, verified
   mathematically exact matches of each other (checked to ~1e-6 on
   CPU; GPU shows expected CuBLAS/scatter nondeterminism on the order
   of 1e-3, not a bug — see "Key bugs found" for the full story):
   - `.forward()` — full-window pass (training)
   - `.step()` — incremental, KV-cached pass (efficient multi-step
     rollout at inference, avoids recomputing attention over the whole
     history every step)

5. **FGN sampling head** (`model.py: FGNDecoderHead`) — takes the
   context summary embedding, concatenates with random noise, decodes
   to a batch of independent trajectory samples (default 16 during
   training, more at inference). **Raw output is a normalized
   residual displacement (delta) from the last known position, NOT an
   absolute coordinate** — see "Key bugs found" for why this matters
   enormously. To recover a real position:
   `sample * norm_scale + last_known_position`.
   `norm_scale` is computed once from the training set's actual delta
   standard deviation and saved in the model checkpoint.

**Note on removed goal-conditioning**: An earlier `GoalEncoder` module
(accepting an optional known destination) was part of the original
pre-rebuild design but was never exercised (always fed `NaN`) and has
been **fully removed** from `model.py`/`train.py`/`inference.py` as of
this session. The model's forward signature no longer takes a
`goal_xy` argument.

## Data pipeline, raw file to training example

1. **Raw AIS CSV** (`aisdk-YYYY-MM-DD.csv`, ~2GB/day, ~29M rows) —
   one row per AIS ping: MMSI, timestamp, lat/lon, SOG, COG, ship
   type, etc. Some files have a `#`-prefixed first header column
   (handled).

2. **Cleaning** (`ais_ingest.load_dma_ais_csv`) — strips header
   prefix if present, drops `Base Station` rows (fixed shore
   transmitters, not vessels), renames columns to internal schema,
   parses day-first timestamps.

3. **Resampling** (`ais_ingest.resample_to_snapshots`) — bins each
   vessel's pings into fixed 15-min windows, linearly interpolating
   across single missed intervals. **Drops a timestep entirely if ANY
   of lon/lat/sog/cog is still missing after interpolation** (see "Key
   bugs found" — checking only `lon` here was a real, silent-NaN bug).

4. **Vessel selection** (`ais_ingest.select_ego_vessels_stratified`)
   — filters to vessels with enough pings and a sane ping rate (not
   an AIS repeater/buoy), then **deliberately balances underway and
   stationary/anchored vessels** (roughly 50/50 by default) as ego
   training targets. See "Design decisions" for why this matters.

5. **Per-vessel windowing** (`graph_data.VesselSequenceDataset`) —
   slides a window over one vessel's snapshot sequence: 4 timesteps
   context (1hr lookback) -> next 4 timesteps target (1hr forecast).

6. **Train/val split** (`train.py: split_vessels_train_val`) — splits
   by VESSEL (not window), stratified by regime, default 80/20. This
   matters because windows from the same vessel are highly correlated
   (overlapping, similar behavior) — a window-level split would leak
   information and overstate generalization.

## Key bugs found and fixed (chronological — useful context for why
things are built the way they are)

1. **KV cache concatenation bug** — cache was concatenating new
   keys/values along the attention-heads dimension instead of the
   sequence dimension. Silent corruption, not a crash. Found via
   direct numerical comparison against full recomputation.

2. **`pyg-lib` dependency risk** — PyTorch Geometric's built-in
   `knn_graph()` needs the optional `pyg-lib` package, which has
   version-pinned wheels not guaranteed to match any given
   torch/CUDA build. Replaced with a manual `torch.cdist`-based k-NN
   (`knn_graph_manual` in `graph_data.py`) to remove the dependency
   entirely.

3. **NaN-from-blank-SOG/COG bug** — real AIS data legitimately has
   blank SOG/COG on some pings even when position is present.
   `resample_to_snapshots` originally only checked `lon` for NaN
   before accepting a timestep; fixed to check all of
   lon/lat/sog/cog.

4. **Absolute-coordinate targets (the big one)** — the model
   originally predicted absolute lon/lat directly. This meant the
   loss was dominated by a huge, task-irrelevant offset (~55degN),
   which swamped the much smaller, more important signal of how
   confident to be. Fixed by predicting a **residual displacement
   from the last known position** instead — directly matching
   GraphCast's documented approach (confirmed via research, not
   assumed). This alone took spread-vs-displacement correlation from
   ~0 (uncorrelated) to ~0.19 on real data.

5. **Missing output normalization** — even with delta targets,
   correlation was weak (0.19) on data spanning a wide range of
   vessel speeds, because the network had to learn "typical speed
   scale" and "confidence" as one entangled thing. Adding
   **unit-variance normalization** of the delta target (dividing by
   the training set's empirical std, matching GraphCast's documented
   per-variable normalization) took correlation to 0.68 on real
   held-out data (and 0.937 on an idealized synthetic test). This was
   the single biggest lever found in this project.

6. **HeteroData.to(device) mutates in place** — PyTorch Geometric's
   `.to(device)` returns the same object with tensors reassigned in
   place, not a new copy. Combined with `VesselSequenceDataset`'s
   sliding windows sharing the same underlying snapshot objects
   across overlapping windows, calling `.to(device)` inside the
   per-batch training loop silently corrupted other windows'
   snapshots mid-epoch (mixed-CPU/CUDA tensor crashes, order-dependent
   on shuffle). Fixed by moving each vessel's full snapshot list to
   device ONCE at dataset-build time, never again during training.

7. **`inference.py` never updated for the delta-target change** —
   caught during this session's cleanup: the multi-step rollout script
   still treated raw model output as an absolute position and fed it
   back into the next snapshot uncorrected. Fixed to unnormalize
   (`* norm_scale`) and add back the last known position at every
   step.

8. **GPU non-determinism (not a bug, but worth knowing)** — a ~2%
   relative divergence in a "cache vs. full-recompute" correctness
   check turned out to be ordinary CUDA/CuBLAS scatter-reduction
   nondeterminism (confirmed by showing the same divergence exists
   calling the plain GNN twice on identical input, no caching
   involved). The underlying cache logic itself is exact (verified on
   CPU to ~1e-6 at real data scale). Not something to "fix" — just
   don't expect bit-identical repeated runs on GPU.

## Design decisions worth knowing the reasoning behind

- **Stratified (not exclusive) vessel sampling by movement regime.**
  Early instinct was to exclude stationary/anchored vessels from
  training as "boring" targets. Corrected: a real deployed predictor
  needs to correctly predict "stays put" as a valid outcome too —
  excluding stationary vessels would leave the model unable to handle
  that case at inference. The ORIGINAL pre-rebuild GC-VTP's failure
  mode wasn't "trained on stationary vessels" per se, it was an
  UNBALANCED distribution (mostly stationary) combined with a decoder
  that couldn't express "it depends" — both are now addressed (energy
  score loss + balanced sampling).
- **Single ego vessel at a time, conditioned on others' past/current
  state only** (not their future state, and not joint multi-vessel
  prediction). Deliberate scoping decision: conditioning on others'
  FUTURE state would be an oracle leak (undeployable, since real
  inference never has others' future positions); joint multi-vessel
  prediction is a reasonable phase-two extension but adds meaningful
  architectural complexity that wasn't worth taking on before the
  single-vessel case was validated.
- **15-minute resampling interval** — chosen after checking real
  displacement-per-step against real mesh edge spacing. For
  genuinely underway vessels, median displacement (~2.5km/15min) is
  reasonably close to open-water mesh spacing (~6km) and well inside
  near-port mesh spacing (~4.5km) — good enough alignment, not
  revisited further.
- **Interval choice deliberately decoupled from the earlier finding**
  that a naive full-population displacement check (mixing stationary
  and underway vessels) gave a misleading near-zero median — always
  segment by SOG/movement regime before drawing conclusions from
  aggregate AIS statistics.

## Environment / how to run things

**Local dev/testing venv** (not a container — Apptainer was set up but
deliberately not used for the real runs, since the Jupyter-validated
`pip install --user` / venv environment was the only one actually
proven to work):
```bash
cd ~/maritime
python3 -m venv venv
source venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Training** (interactive or via SLURM):
```bash
python3 train.py \
    --ais-glob "/scratch/jtb3sud/maritime/ais/aisdk-*.csv" \
    --land-shp data/ne_10m_land/ne_10m_land.shp \
    --ports-csv data/ports_denmark.csv \
    --n-underway 150 --n-stationary 150 \
    --n-epochs 50 \
    --checkpoint-path checkpoints/checkpoint.pt \
    --resume
```

**SLURM** (`train_gcvtp.slurm` in repo): account `sds_baek_energetic`,
partition `gpu`, tested working on both A100 and RTX A6000 (no need to
pin to a specific GPU type).

**Rivanna access notes**: `rivanna.hpc.virginia.edu` is a dead/removed
hostname — use `login.hpc.virginia.edu`. Off-campus SSH requires UVA's
VPN (Cisco AnyConnect, install via the UVA ITS service portal). Open
OnDemand (`https://ood.hpc.virginia.edu`) works from anywhere without
VPN and provides a browser-based shell — useful fallback.

**Storage**: code in `~/maritime` (home, 200GB quota, weekly
snapshots). Bulk AIS data in `/scratch/jtb3sud/maritime/ais/`
(10TB quota, no backups, 90-day inactivity purge — fine since active
training counts as access).

## Repo structure

```
mesh.py                -- spatial mesh construction (Delaunay triangulation)
coastline.py            -- coastline/port loading, DMA_BOUNDS domain bounds
ais_ingest.py            -- AIS parsing, resampling, vessel selection, feature encoding
graph_data.py            -- HeteroData graph construction, windowed dataset
cached_attention.py      -- KV-cached causal transformer
model.py                 -- GNN + transformer + FGN head, full GCVTP model
losses.py                -- energy score loss
train.py                 -- standalone training script (SLURM-ready, train/val split)
inference.py             -- multi-step rollout / forecasting (delta-aware)
train_gcvtp.slurm        -- SLURM batch script
test_pipeline_denmark.ipynb  -- interactive exploration notebook
requirements.txt
.gitignore
README.md
```

## What NOT to re-litigate (settled, tested questions)

- KV-cache correctness: exact, verified multiple times at multiple
  scales. GPU numerical noise in comparisons is expected CUDA
  behavior, not a bug.
- Delta-target + normalization approach: strongly validated on both
  synthetic and real held-out data, directly matches documented
  GraphCast practice. Don't revert to absolute-coordinate targets.
- Stratified vessel sampling: deliberate, reasoned design choice, not
  an oversight — don't "simplify" back to excluding stationary
  vessels.
- Single-vessel-at-a-time scope: deliberate, not a limitation to
  rush to fix — joint prediction is a real but separate future step.