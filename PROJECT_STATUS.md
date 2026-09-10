# Project Status & Handoff

Repo: https://github.com/JackBeerman/maritime
Environment: UVA Rivanna HPC, account `sds_baek_energetic`, user `jtb3sud`

**Two branches, two worktrees:**

| branch | worktree | role |
|---|---|---|
| `irregular-sampling` | `~/maritime-irregular` | **primary model** — raw AIS reports at true times, packaged as `vtp/` |
| `main` | `~/maritime` | comparison arm — fixed 15-minute resampling, flat layout |

Worktrees rather than `git checkout`, because SLURM jobs read code from
disk at launch and switching branches in a shared directory can hand a
queued job the wrong files. Bulk AIS lives at
`/scratch/jtb3sud/maritime/ais/` (12 days, 2026-08-18 … 2026-08-29,
~236M records); `venv` and `data/` are symlinked into the irregular
worktree and gitignored.

> The GitHub **default branch** should be set to `irregular-sampling`
> (Settings → Branches) so visitors land on the primary model. Renaming
> the branches themselves was deliberately avoided — with two worktrees
> checked out it invites a history-rewrite mess for no real gain.

## What this is

Vessel trajectory prediction as a **distribution of plausible futures**,
following DeepMind's weather-forecasting lineage: GraphCast's mesh + GNN
structure and residual prediction, and WeatherNext's FGN sampling head
trained with an energy score loss.

Domain: Danish waters, using the Danish Maritime Authority's public AIS
archive.

## Current state

### Primary: `irregular-sampling`

- Reorganized into a `vtp/` package (`data`, `models`, `training`,
  `benchmark`, `viz`), installed with `pip install -e .`.
- **Portable benchmark harness** — the piece intended for the paper. A
  benchmark set is a single `.npz`; any model enters by implementing one
  `predict` method with no dependency on this codebase.
- Best measured run so far (one day, 20 epochs, ~62 min horizon):
  spread correlation 0.436, 9.77 km error vs 10.45 km persistence.
  Not yet scored on the benchmark.

### Comparison: `main`

- Best run (12 days, 5 epochs): correlation 0.753, 5.37 km error on
  moving vessels, 82.4% beats persistence.
- A 60-epoch run reached epoch 53 with training loss still declining
  (0.5016). Loss shown during training is **training** loss; validation
  only ran at the end in that job.

### Benchmark reference: `danish_60min`

1,898 windows, 193 vessels, ~62 min median horizon, 953 moving / 945
stationary.

| method | ADE (moving) | FDE (moving) | skill vs CV |
|---|---|---|---|
| persistence | 7.94 km | 12.55 km | −1.05 |
| **constant velocity** | **3.60 km** | **6.13 km** | 0.00 |
| CTRV | 3.59 km | 6.46 km | −0.05 |

**Constant velocity at 6.13 km FDE is the bar.** CTRV does not beat it —
turn rate from three noisy AIS points is unreliable, and extrapolating an
arc an hour ahead amplifies the noise. That is a useful negative result:
manoeuvring has to be learned, not extrapolated.

## Next steps

1. **Score both models on `danish_60min`.** Until that happens there is
   no like-for-like comparison, and no defensible table row. The
   benchmark adapter (`vtp/benchmark/adapters.py`) has never been run
   against trained weights — expect to debug it.
2. **Full training run with periodic validation** on the irregular
   branch. `--val-every 5` now saves a separate `_best.pt`; earlier runs
   could silently overwrite their own best weights.
3. **Add Rate-of-Turn.** Failure analysis found turn angle separates the
   worst-predicted windows from the best by a factor of 4, and AIS
   carries an ROT field that ingest currently discards. Highest-value
   untried experiment.
4. **Mesh ablation.** The scaling sweep found error varies only
   4.45–4.80 km across a 12× range of mesh density while cost doubles.
   Worth training with the mesh removed entirely — if nothing changes, a
   large part of the architecture is dead weight.
5. **Resolve the CUDA assert.** Intermittent, always inside
   `encode_windows_batched`. `CUDA_LAUNCH_BLOCKING=1` avoids it. Two
   suspected causes were ruled out (mesh sharing, index bounds).

## Key findings

**Turning dominates failures.** Worst-20% vs best-20% of held-out
windows:

| factor | worst | best | ratio |
|---|---|---|---|
| turn angle (deg) | 40.87 | 10.09 | **4.05** |
| speed variability | 0.01 | 0.00 | 3.12 |
| distance to port (km) | 83.65 | 87.04 | 0.96 |
| vessels in snapshot | 3362 | 3228 | 1.04 |

Nothing but turning (and speed variability, likely correlated)
distinguishes them. Not congestion, not port proximity.

**Error grows linearly with horizon**, not exponentially: 1.59 / 3.43 /
5.43 / 7.58 km at +15/30/45/60 min.

**Reporting intervals span 28×** across vessels (10 s to 284 s per-vessel
median) — the measurement that motivated the irregular branch.

**Scaling sweeps** (one day, 15 epochs; trends only): mesh density has
no measurable effect; longer context improves calibration (0.43 → 0.67)
with best error at `seq_len 16`; the model is near parity with dead
reckoning across most configurations.

## Bugs found and fixed

1. **KV cache concatenation** along the heads dimension instead of
   sequence. Silent corruption, not a crash.
2. **`pyg-lib` dependency** — replaced PyG's `knn_graph` with an in-repo
   KD-tree implementation to drop a version-pinned optional package.
3. **NaN from blank SOG/COG** — real AIS legitimately has these; the
   resampler only checked `lon`.
4. **Absolute-coordinate targets** — loss dominated by a
   task-irrelevant ~55° N offset. Switching to residual displacement
   (matching GraphCast) took spread correlation from ~0 to 0.19.
5. **Missing output normalization** — unit-variance normalization took
   it to **0.68**. Single largest improvement in the project.
6. **`HeteroData.to()` mutates in place** — combined with sliding
   windows sharing snapshot objects, per-batch device moves silently
   corrupted other windows mid-epoch.
7. **World-snapshot duplication → CUDA OOM** — a graph per *(vessel,
   timestamp)* rather than per timestamp. **856×** redundancy measured.
8. **Mesh duplication → 50 GB OOM** on the irregular branch, where
   snapshots cannot be shared. Fixed by keeping snapshots on CPU and
   moving one chunk at a time.
9. **Device mismatches** after the CPU-storage change, in three places
   (training targets, validation, viz).
10. **Persistence reported as "the baseline"** — not a code bug, but the
    most consequential error in how results were being read. Every
    "beats baseline 89%" figure was against a bar that a straight line
    clears trivially.

## Design decisions

- **Stationary vessels are sampled, not excluded.** A deployed predictor
  must handle "stays put" as a valid outcome. The original failure was
  an *unbalanced* distribution plus a decoder that could not express
  "it depends".
- **Single ego vessel, conditioned on others' past state only.**
  Conditioning on neighbours' futures would be an oracle leak.
- **Train/val splits by vessel, not window.** Windows from one vessel
  overlap heavily; a window-level split leaks.
- **Benchmark windows come from raw pings**, never resampled, so the
  format does not privilege either branch. A fixed-interval model may
  resample internally — that information loss is part of what is
  measured.
- **Always segment AIS statistics by movement regime.** A naive
  full-population displacement check gave a misleading near-zero median
  that would have led to a wrong interval choice.

## Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install -e .          # irregular branch; main uses requirements.txt
```

Validated: torch 2.9.1+cu128, torch_geometric 2.8.0.post1,
geopandas 1.1.4, shapely 2.1.2. Runs on A100 and RTX A6000.

**Rivanna:** `rivanna.hpc.virginia.edu` is dead — use
`login.hpc.virginia.edu`. Off-campus SSH needs UVA VPN; Open OnDemand
(`ood.hpc.virginia.edu`) works without it. Code in `~` (200 GB,
snapshotted); AIS in `/scratch` (10 TB, no backup, 90-day purge).

**Caching:** load + resample of 12 days takes ~46 minutes and is cached
to `cache/` keyed by input files and interval, so it happens once.

## Running

```bash
cd ~/maritime-irregular && sbatch slurm/train_irregular.slurm     # primary
cd ~/maritime           && sbatch slurm/train.slurm                # comparison
```

Always submit from the matching worktree. Both use `--resume`, safe
whether or not a checkpoint exists.
