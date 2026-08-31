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
    p.add_argument('--ais-glob', type=str, default="/scratch/jtb3sud/maritime/ais/aisdk-*.csv",
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
    p.add_argument('--checkpoint-path', type=str, default='checkpoints/checkpoint.pt')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def load_multi_day_ais(ais_paths, bounds):
    """
    Loads and concatenates multiple daily DMA AIS CSVs into one DataFrame.
    Each file covers its own calendar day, so a wide time window is used
    per-file load (lon/lat bounds filtering still applies inside
    load_dma_ais_csv) -- the actual per-vessel windowing later respects
    real timestamps regardless of which file a record came from.
    """
    dfs = []
    for path in ais_paths:
        df = load_dma_ais_csv(path, bounds, pd.Timestamp.min, pd.Timestamp.max)
        dfs.append(df)
        print(f"  loaded {path}: {len(df)} records")
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp')


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

    print("building per-vessel datasets...")
    datasets = []
    for ego_mmsi in good_vessels.index:
        snaps, ego_steps = build_snapshot_sequence(
            ais_snapshots, mesh_node_features, mesh_edge_index, ego_mmsi
        )
        snaps = [d.to(device) for d in snaps]
        ds = VesselSequenceDataset(snaps, ego_steps, seq_len=args.seq_len, future_len=args.future_len)
        if len(ds) > 0:
            datasets.append(ds)
    combined = ConcatDataset(datasets)
    print(f"  total training windows across {len(datasets)} vessels: {len(combined)}")
    if len(combined) == 0:
        raise RuntimeError("no training windows -- loosen vessel selection or check data")

    def collate_single(batch):
        return batch[0]

    loader = DataLoader(combined, batch_size=1, shuffle=True, collate_fn=collate_single)

    model = GCVTP(mesh_in=4, vessel_in=8, hidden=args.hidden, future_len=args.future_len,
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
        print("computing delta normalization scale...")
        all_deltas = []
        for ctx, target in combined:
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

        avg_loss = total_loss / len(combined)
        print(f"epoch {epoch}: avg loss = {avg_loss:.4f}", flush=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'norm_scale': norm_scale,
            'epoch': epoch,
            'avg_loss': avg_loss,
        }, args.checkpoint_path)

    print("training complete.")


if __name__ == '__main__':
    main()
