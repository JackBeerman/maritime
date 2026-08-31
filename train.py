"""
Full training script for the GC-VTP maritime rebuild, meant to run via
SLURM (or interactively for testing with small arguments).

Usage:
    python3 train.py \
        --ais-glob "data/ais/aisdk-*.csv" \
        --land-shp data/ne_10m_land/ne_10m_land.shp \
        --ports-csv data/ports_denmark.csv \
        --n-underway 100 --n-stationary 100 \
        --n-epochs 50 \
        --checkpoint-path checkpoints/checkpoint.pt

Resume an interrupted run with --resume (loads checkpoint-path if present,
continues from the next epoch; falls back to a fresh start if the file
doesn't exist).

Vessels are split into train/val sets BEFORE windowing (not at the window
level) -- windows from the same vessel are highly correlated (overlapping,
similar behavior), so a window-level split would leak information and
overstate generalization. At the end of training, validation diagnostics
(uncertainty-calibration correlation and a trivial-baseline comparison)
are computed on held-out vessels the model never trained on.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from coastline import load_land_polygons, load_ports, DMA_BOUNDS
from mesh import sample_domain_points, build_mesh
from ais_ingest import (
    load_dma_ais_csv, resample_to_snapshots, build_snapshot_sequence,
    select_ego_vessels_stratified,
)
from graph_data import VesselSequenceDataset
from model import GCVTP
from losses import energy_score_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ais-glob', type=str, default='data/ais/aisdk-*.csv',
                   help='glob pattern matching one or more daily AIS CSVs to train on')
    p.add_argument('--land-shp', type=str, default='data/ne_10m_land/ne_10m_land.shp')
    p.add_argument('--ports-csv', type=str, default='data/ports_denmark.csv')
    p.add_argument('--n-underway', type=int, default=100)
    p.add_argument('--n-stationary', type=int, default=100)
    p.add_argument('--min-pings', type=int, default=40)
    p.add_argument('--sog-threshold-knots', type=float, default=2.0)
    p.add_argument('--interval-minutes', type=int, default=15)
    p.add_argument('--seq-len', type=int, default=4)
    p.add_argument('--future-len', type=int, default=4)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--n-samples', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-epochs', type=int, default=50)
    p.add_argument('--val-fraction', type=float, default=0.2,
                   help='fraction of selected vessels held out for validation (split by vessel, not window)')
    p.add_argument('--val-seed', type=int, default=42)
    p.add_argument('--checkpoint-path', type=str, default='checkpoints/checkpoint.pt')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def load_multi_day_ais(ais_paths, bounds):
    """
    Loads and concatenates multiple daily DMA AIS CSVs into one DataFrame.
    """
    dfs = []
    for path in ais_paths:
        df = load_dma_ais_csv(path, bounds, pd.Timestamp.min, pd.Timestamp.max)
        dfs.append(df)
        print(f"  loaded {path}: {len(df)} records")
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp')


def split_vessels_train_val(good_vessels, val_fraction, seed):
    """
    Splits selected vessels into train/val sets, stratified by regime
    (underway/stationary) so both splits contain a mix of each, and split
    BY VESSEL so no vessel's windows appear in both sets.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for regime, group in good_vessels.groupby('regime'):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx)))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return train_idx, val_idx


def build_vessel_datasets(vessel_ids, ais_snapshots, mesh_node_features, mesh_edge_index,
                            device, seq_len, future_len):
    datasets = []
    for ego_mmsi in vessel_ids:
        snaps, ego_steps = build_snapshot_sequence(
            ais_snapshots, mesh_node_features, mesh_edge_index, ego_mmsi
        )
        snaps = [d.to(device) for d in snaps]
        ds = VesselSequenceDataset(snaps, ego_steps, seq_len=seq_len, future_len=future_len)
        if len(ds) > 0:
            datasets.append(ds)
    return datasets


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@torch.no_grad()
def run_validation(model, val_dataset, norm_scale, n_samples=32):
    """
    Computes, on held-out vessels the model never trained on:
      - correlation between predicted spread and actual displacement
        (uncertainty calibration)
      - mean error vs. a trivial "assume no movement" baseline, restricted
        to genuinely moving windows (>1km actual displacement)
    """
    model.eval()
    actual_displacements, predicted_spreads = [], []
    model_errors, trivial_errors = [], []

    for i in range(len(val_dataset)):
        ctx, target = val_dataset[i]
        ego_idx_this_window = [d['vessel'].ego_mask.nonzero()[0].item() for d in ctx]
        last_ctx_pos = ctx[-1]['vessel'].x[ego_idx_this_window[-1], :2]

        samples = model(ctx, ego_idx_this_window, n_samples=n_samples, training=False)
        samples_real = samples * norm_scale + last_ctx_pos

        final_step = samples_real[0, :, -1, :].cpu().numpy()
        true_final = target[-1].cpu().numpy()
        last_pos_np = last_ctx_pos.cpu().numpy()

        disp_km = haversine_km(last_pos_np[0], last_pos_np[1], true_final[0], true_final[1])
        center = final_step.mean(axis=0)
        spread_km = np.mean([haversine_km(center[0], center[1], p[0], p[1]) for p in final_step])

        actual_displacements.append(disp_km)
        predicted_spreads.append(spread_km)

        if disp_km > 1.0:
            model_err = haversine_km(center[0], center[1], true_final[0], true_final[1])
            model_errors.append(model_err)
            trivial_errors.append(disp_km)

    actual_displacements = np.array(actual_displacements)
    predicted_spreads = np.array(predicted_spreads)
    corr = np.corrcoef(actual_displacements, predicted_spreads)[0, 1] if len(actual_displacements) > 1 else float('nan')

    results = {'n_val_windows': len(val_dataset), 'spread_correlation': corr}
    if model_errors:
        model_errors = np.array(model_errors)
        trivial_errors = np.array(trivial_errors)
        results['n_moving_windows'] = len(model_errors)
        results['model_mean_error_km'] = model_errors.mean()
        results['trivial_baseline_mean_error_km'] = trivial_errors.mean()
        results['beats_baseline_pct'] = 100 * (model_errors < trivial_errors).mean()
    return results


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)

    bounds = DMA_BOUNDS
    print("building mesh...")
    land_polygons = load_land_polygons(bounds, natural_earth_path=args.land_shp)
    ports = load_ports(args.ports_csv, bounds)
    pts, land_flags = sample_domain_points(bounds, land_polygons, ports)
    mesh_node_features, mesh_edge_index, _ = build_mesh(pts, land_flags, ports)
    print(f"  mesh nodes: {mesh_node_features.shape[0]}, edges: {mesh_edge_index.shape[1]}")

    ais_paths = sorted(glob.glob(args.ais_glob))
    if not ais_paths:
        raise FileNotFoundError(f"no AIS files matched {args.ais_glob}")
    print(f"loading {len(ais_paths)} AIS file(s)...")
    ais_df = load_multi_day_ais(ais_paths, bounds)
    print(f"  total records: {len(ais_df)}, unique vessels: {ais_df['mmsi'].nunique()}")

    print("resampling to snapshots...")
    ais_snapshots = resample_to_snapshots(ais_df, interval_minutes=args.interval_minutes)
    print(f"  timestamps: {len(ais_snapshots)}")

    print("selecting ego vessels...")
    good_vessels = select_ego_vessels_stratified(
        ais_df, min_pings=args.min_pings, sog_threshold_knots=args.sog_threshold_knots,
        n_underway=args.n_underway, n_stationary=args.n_stationary,
    )
    print(f"  {good_vessels['regime'].value_counts().to_dict()}")

    train_vessel_ids, val_vessel_ids = split_vessels_train_val(
        good_vessels, args.val_fraction, args.val_seed
    )
    print(f"  train vessels: {len(train_vessel_ids)}, val vessels (held out): {len(val_vessel_ids)}")

    print("building per-vessel datasets...")
    train_datasets = build_vessel_datasets(
        train_vessel_ids, ais_snapshots, mesh_node_features, mesh_edge_index,
        device, args.seq_len, args.future_len,
    )
    val_datasets = build_vessel_datasets(
        val_vessel_ids, ais_snapshots, mesh_node_features, mesh_edge_index,
        device, args.seq_len, args.future_len,
    )
    train_combined = ConcatDataset(train_datasets)
    val_combined = ConcatDataset(val_datasets) if val_datasets else None
    print(f"  train windows: {len(train_combined)} (across {len(train_datasets)} vessels)")
    print(f"  val windows: {len(val_combined) if val_combined else 0} (across {len(val_datasets)} vessels)")
    if len(train_combined) == 0:
        raise RuntimeError("no training windows -- loosen vessel selection or check data")

    def collate_single(batch):
        return batch[0]

    loader = DataLoader(train_combined, batch_size=1, shuffle=True, collate_fn=collate_single)

    model = GCVTP(mesh_in=4, vessel_in=12, hidden=args.hidden, future_len=args.future_len,
                  n_heads=args.n_heads, n_layers=args.n_layers, max_cache_len=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    norm_scale = None
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.checkpoint_path):
            ckpt = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            norm_scale = ckpt['norm_scale']
            start_epoch = ckpt['epoch'] + 1
            print(f"resumed from {args.checkpoint_path} at epoch {start_epoch}")
        else:
            print(f"--resume given but no checkpoint at {args.checkpoint_path}, starting fresh")

    if norm_scale is None:
        print("computing delta normalization scale (from TRAIN vessels only)...")
        all_deltas = []
        for ctx, target in train_combined:
            ego_idx_this_window = [d['vessel'].ego_mask.nonzero()[0].item() for d in ctx]
            last_ctx_pos = ctx[-1]['vessel'].x[ego_idx_this_window[-1], :2]
            all_deltas.append((target - last_ctx_pos.unsqueeze(0)).cpu())
        norm_scale = torch.cat(all_deltas, dim=0).std().item()
    print(f"  norm_scale: {norm_scale:.5f}")

    print("training...")
    model.train()
    for epoch in range(start_epoch, args.n_epochs):
        total_loss = 0.0
        for ctx, target in loader:
            ego_idx_this_window = [d['vessel'].ego_mask.nonzero()[0].item() for d in ctx]
            last_ctx_pos = ctx[-1]['vessel'].x[ego_idx_this_window[-1], :2]
            target_delta = (target - last_ctx_pos.unsqueeze(0)) / norm_scale

            samples = model(ctx, ego_idx_this_window, n_samples=args.n_samples, training=True)
            loss = energy_score_loss(samples, target_delta.unsqueeze(0))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_combined)
        print(f"epoch {epoch}: avg loss = {avg_loss:.4f}", flush=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'norm_scale': norm_scale,
            'epoch': epoch,
            'avg_loss': avg_loss,
        }, args.checkpoint_path)
        model.train()

    print("training complete.")

    if val_combined is not None and len(val_combined) > 0:
        print("\nrunning validation on held-out vessels...")
        results = run_validation(model, val_combined, norm_scale)
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print("no validation vessels available -- skipping validation diagnostics")


if __name__ == '__main__':
    main()
