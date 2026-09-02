"""
Training for the irregular-sampling branch.

NORMALIZATION (the piece that makes irregular horizons comparable)
-----------------------------------------------------------------
The fixed-interval branch normalized displacement by a single global
std, which works only because every target was exactly 15 minutes out.
Here targets are real pings at whatever times they happened to arrive,
so a single displacement scale would be averaging 40-second moves
together with 19-minute moves -- reintroducing exactly the entanglement
that normalization was introduced to remove.

Instead targets are normalized as VELOCITY:

    velocity   = (target_pos - anchor_pos) / dt          # deg/sec
    normalized = velocity / vel_scale                    # dimensionless

with vel_scale the training-set std of velocity. A vessel holding
constant speed produces the same normalized target regardless of how far
ahead the target ping is, so the model only has to learn departures from
constant velocity -- the actually-interesting part.

This does NOT flatten the uncertainty structure: because the head is
separately conditioned on dt, it can still learn to widen its spread in
normalized units at longer horizons. Recovering a position:

    pred_pos = anchor_pos + output * vel_scale * dt
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import ConcatDataset, DataLoader

from coastline import load_land_polygons, load_ports, DMA_BOUNDS
from mesh import sample_domain_points, build_mesh
from irregular_ingest import (
    load_dma_ais_csv, build_ego_anchored_snapshots, select_ego_vessels_stratified,
)
from graph_data import IrregularVesselDataset, share_mesh_on_device
from model import IrregularVTP
from losses import energy_score_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ais-glob', type=str, default='data/ais/aisdk-*.csv')
    p.add_argument('--land-shp', type=str, default='data/ne_10m_land/ne_10m_land.shp')
    p.add_argument('--ports-csv', type=str, default='data/ports_denmark.csv')
    p.add_argument('--n-underway', type=int, default=100)
    p.add_argument('--n-stationary', type=int, default=100)
    p.add_argument('--min-pings', type=int, default=60)
    p.add_argument('--min-ping-gap-sec', type=float, default=60.0,
                   help='thin ego pings closer than this; 0 disables thinning')
    p.add_argument('--staleness-cutoff-sec', type=float, default=1800.0)
    p.add_argument('--neighbor-radius-deg', type=float, default=0.5)
    p.add_argument('--max-neighbors', type=int, default=150)
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--future-len', type=int, default=4)
    p.add_argument('--max-window-span-sec', type=float, default=None,
                   help='reject windows spanning longer than this (guards against '
                        'a window straddling a multi-hour AIS blackout)')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--n-samples', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-epochs', type=int, default=50)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--val-seed', type=int, default=42)
    p.add_argument('--checkpoint-path', type=str, default='checkpoints/irregular.pt')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def split_vessels_train_val(good_vessels, val_fraction, seed):
    """Split BY VESSEL (not window), stratified by regime."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for _, group in good_vessels.groupby('regime'):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx)))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return train_idx, val_idx


def build_datasets(vessel_ids, ais_df, mesh_node_features, mesh_edge_index, device, args,
                    mesh_x_tensor=None, mesh_edge_tensor=None, mesh_tree=None):
    datasets = []
    for mmsi in vessel_ids:
        snaps, ego_idx, times, positions = build_ego_anchored_snapshots(
            ais_df, mmsi, mesh_node_features, mesh_edge_index,
            staleness_cutoff_sec=args.staleness_cutoff_sec,
            neighbor_radius_deg=args.neighbor_radius_deg,
            max_neighbors=args.max_neighbors,
            min_ping_gap_sec=args.min_ping_gap_sec,
            mesh_x_tensor=mesh_x_tensor, mesh_edge_tensor=mesh_edge_tensor,
            mesh_tree=mesh_tree,
        )
        if len(snaps) < args.seq_len + args.future_len:
            continue
        snaps = share_mesh_on_device(snaps, device)
        ds = IrregularVesselDataset(snaps, ego_idx, times, positions,
                                     seq_len=args.seq_len, future_len=args.future_len,
                                     max_window_span_sec=args.max_window_span_sec)
        if len(ds) > 0:
            datasets.append(ds)
    return datasets


def compute_velocity_scale(dataset, eps=1e-6):
    """
    Training-set std of per-target velocity (deg/sec). See module
    docstring for why velocity rather than displacement.
    """
    vels = []
    for i in range(len(dataset)):
        ctx, _, target_pos, target_dts = dataset[i]
        ego_i = ctx[-1]['vessel'].ego_mask.nonzero()[0].item()
        anchor = ctx[-1]['vessel'].x[ego_i, :2].cpu()
        dt = target_dts.clamp(min=eps).unsqueeze(-1)
        vels.append(((target_pos.cpu() - anchor) / dt))
    return torch.cat(vels, dim=0).std().item()


@torch.no_grad()
def run_validation(model, dataset, vel_scale, n_samples=32, min_move_km=1.0, eps=1e-6):
    model.eval()
    disps, spreads, errs, base_errs, dts = [], [], [], [], []

    for i in range(len(dataset)):
        ctx, ctx_times, target_pos, target_dts = dataset[i]
        ego_i = [d['vessel'].ego_mask.nonzero()[0].item() for d in ctx]
        anchor = ctx[-1]['vessel'].x[ego_i[-1], :2]

        out = model(ctx, ego_i, ctx_times, target_dts.to(anchor.device), n_samples=n_samples)
        dt = target_dts.to(anchor.device).clamp(min=eps).view(1, 1, -1, 1)
        pred = out * vel_scale * dt + anchor           # (1, S, F, 2)

        final = pred[0, :, -1, :].cpu().numpy()
        truth = target_pos[-1].cpu().numpy()
        a = anchor.cpu().numpy()

        moved = haversine_km(a[0], a[1], truth[0], truth[1])
        center = final.mean(axis=0)
        spread = np.mean([haversine_km(center[0], center[1], p[0], p[1]) for p in final])

        disps.append(moved)
        spreads.append(spread)
        dts.append(float(target_dts[-1]))
        if moved > min_move_km:
            errs.append(haversine_km(center[0], center[1], truth[0], truth[1]))
            base_errs.append(moved)

    res = {'n_val_windows': len(dataset),
           'median_target_horizon_min': float(np.median(dts)) / 60.0,
           'spread_correlation': float(np.corrcoef(disps, spreads)[0, 1]) if len(disps) > 1 else float('nan')}
    if errs:
        errs, base_errs = np.array(errs), np.array(base_errs)
        res.update({'n_moving_windows': len(errs),
                    'model_mean_error_km': float(errs.mean()),
                    'trivial_baseline_mean_error_km': float(base_errs.mean()),
                    'beats_baseline_pct': float(100 * (errs < base_errs).mean())})
    return res


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
    mesh_x_tensor = torch.as_tensor(mesh_node_features, dtype=torch.float, device='cpu')
    mesh_edge_tensor = torch.as_tensor(mesh_edge_index, dtype=torch.long, device='cpu')
    mesh_tree = cKDTree(mesh_node_features[:, :2])
    print(f"  mesh nodes: {mesh_node_features.shape[0]}, edges: {mesh_edge_index.shape[1]}")

    paths = sorted(glob.glob(args.ais_glob))
    if not paths:
        raise FileNotFoundError(f"no AIS files matched {args.ais_glob}")
    print(f"loading {len(paths)} AIS file(s)...")
    dfs = []
    for p in paths:
        d = load_dma_ais_csv(p, bounds)
        print(f"  {p}: {len(d)} records")
        dfs.append(d)
    ais_df = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
    print(f"  total: {len(ais_df)} records, {ais_df['mmsi'].nunique()} vessels")

    print("selecting ego vessels...")
    good = select_ego_vessels_stratified(
        ais_df, min_pings=args.min_pings, n_underway=args.n_underway,
        n_stationary=args.n_stationary)
    print(f"  {good['regime'].value_counts().to_dict()}")

    train_ids, val_ids = split_vessels_train_val(good, args.val_fraction, args.val_seed)
    print(f"  train vessels: {len(train_ids)}, val vessels: {len(val_ids)}")

    print("building ego-anchored datasets (no resampling)...")
    train_ds = build_datasets(train_ids, ais_df, mesh_node_features, mesh_edge_index,
                               device, args, mesh_x_tensor, mesh_edge_tensor, mesh_tree)
    val_ds = build_datasets(val_ids, ais_df, mesh_node_features, mesh_edge_index,
                             device, args, mesh_x_tensor, mesh_edge_tensor, mesh_tree)
    train_combined = ConcatDataset(train_ds)
    val_combined = ConcatDataset(val_ds) if val_ds else None
    print(f"  train windows: {len(train_combined)} across {len(train_ds)} vessels")
    print(f"  val windows: {len(val_combined) if val_combined else 0} across {len(val_ds)} vessels")
    if len(train_combined) == 0:
        raise RuntimeError("no training windows -- loosen selection or lower --min-ping-gap-sec")

    loader = DataLoader(train_combined, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

    model = IrregularVTP(hidden=args.hidden, n_heads=args.n_heads,
                          n_layers=args.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    vel_scale, start_epoch = None, 0
    if args.resume and os.path.exists(args.checkpoint_path):
        ck = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimizer_state_dict'])
        vel_scale = ck['vel_scale']
        start_epoch = ck['epoch'] + 1
        print(f"resumed at epoch {start_epoch}")
    elif args.resume:
        print(f"--resume given but no checkpoint at {args.checkpoint_path}, starting fresh")

    if vel_scale is None:
        print("computing velocity normalization scale (train vessels only)...")
        vel_scale = compute_velocity_scale(train_combined)
    print(f"  vel_scale: {vel_scale:.3e} deg/sec")

    print("training...")
    eps = 1e-6
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        total = 0.0
        for ctx, ctx_times, target_pos, target_dts in loader:
            ego_i = [d['vessel'].ego_mask.nonzero()[0].item() for d in ctx]
            anchor = ctx[-1]['vessel'].x[ego_i[-1], :2]
            dts = target_dts.to(anchor.device)

            # velocity-normalized target (see module docstring)
            dt_col = dts.clamp(min=eps).unsqueeze(-1)
            tgt = ((target_pos.to(anchor.device) - anchor) / dt_col) / vel_scale

            out = model(ctx, ego_i, ctx_times, dts, n_samples=args.n_samples)
            loss = energy_score_loss(out, tgt.unsqueeze(0))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(train_combined)
        print(f"epoch {epoch}: avg loss = {avg:.4f}", flush=True)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'vel_scale': vel_scale, 'epoch': epoch, 'avg_loss': avg,
                    'args': vars(args)}, args.checkpoint_path)

    print("training complete.")
    if val_combined is not None and len(val_combined) > 0:
        print("\nvalidation on held-out vessels...")
        for k, v in run_validation(model, val_combined, vel_scale).items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()