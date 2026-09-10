"""
Scaling study: measure how mesh density, input context length, and
forecast horizon each affect held-out performance.

Each configuration is a short training run, so a sweep is expensive.
Three things keep it tractable, and all three are deliberate trade-offs
worth understanding before reading the results:

  * ONE DAY of AIS by default, not twelve. The point is the *shape* of
    each curve, not the best achievable number.
  * A REDUCED vessel count and epoch budget. Absolute metrics will be
    worse than a full run; only the trend across configurations is
    meaningful.
  * The load+resample CACHE (ais_cache.py) is shared across every
    configuration, so the ~25-minute ingest happens once instead of once
    per config. Mesh construction and world-snapshot building still
    repeat, since those depend on the swept parameters.

Results are written one JSON per configuration, so a sweep that dies
partway can be resumed and partial results are still analysable.

    python3 sweep.py --axis mesh     --out-dir results/sweep_mesh
    python3 sweep.py --axis context  --out-dir results/sweep_context
    python3 sweep.py --axis horizon  --out-dir results/sweep_horizon
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import ConcatDataset, DataLoader

from coastline import load_land_polygons, load_ports, DMA_BOUNDS
from mesh import sample_domain_points, build_mesh
from ais_ingest import load_dma_ais_csv, resample_to_snapshots, vessel_presence, VESSEL_FEATURE_DIM
from ais_cache import load_and_resample_cached, select_ego_vessels_from_stats
from graph_data import VesselSequenceDataset
from model import GCVTP
from losses import energy_score_loss
from batching import collate_windows, encode_windows_batched, batch_anchors_and_targets


# Each axis varies ONE thing and holds the rest fixed. Mesh density is
# swept as (n_open_water, n_coastal) pairs keeping their ratio roughly
# constant, so "density" means one number rather than two.
AXES = {
    'mesh': [
        {'n_open_water': 750, 'n_coastal': 1000},
        {'n_open_water': 1500, 'n_coastal': 2000},
        {'n_open_water': 3000, 'n_coastal': 4000},     # current default
        {'n_open_water': 6000, 'n_coastal': 8000},
        {'n_open_water': 12000, 'n_coastal': 16000},
    ],
    'context': [
        {'seq_len': 4},      # 1 hr lookback
        {'seq_len': 8},      # 2 hr
        {'seq_len': 12},     # 3 hr -- current default
        {'seq_len': 16},     # 4 hr
        {'seq_len': 24},     # 6 hr
    ],
    'horizon': [
        {'future_len': 2},   # 30 min
        {'future_len': 4},   # 1 hr -- current default
        {'future_len': 8},   # 2 hr
        {'future_len': 12},  # 3 hr
        {'future_len': 16},  # 4 hr
    ],
}


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def constant_velocity(context_xy, n_steps):
    """Dead-reckoning baseline -- the bar that actually matters."""
    if len(context_xy) < 2:
        return np.repeat(context_xy[-1:], n_steps, axis=0)
    v = context_xy[-1] - context_xy[-2]
    return np.stack([context_xy[-1] + v * (k + 1) for k in range(n_steps)])


def split_vessels(good, val_fraction, seed):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for _, g in good.groupby('regime'):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx)))) if len(idx) > 1 else 0
        va.extend(idx[:n_val]); tr.extend(idx[n_val:])
    return tr, va


@torch.no_grad()
def evaluate(model, dataset, norm_scale, n_samples=32, min_move_km=1.0):
    model.eval()
    disps, spreads, errs, base, cv_errs = [], [], [], [], []
    for i in range(len(dataset)):
        ctx, ego_rows, target = dataset[i]
        anchor = ctx[-1]['vessel'].x[ego_rows[-1], :2]
        s = model(ctx, ego_rows, n_samples=n_samples, training=False) * norm_scale + anchor
        final = s[0, :, -1, :].cpu().numpy()
        truth = target[-1].cpu().numpy(); a = anchor.cpu().numpy()

        moved = haversine_km(a[0], a[1], truth[0], truth[1])
        c = final.mean(axis=0)
        disps.append(moved)
        spreads.append(np.mean([haversine_km(c[0], c[1], p[0], p[1]) for p in final]))
        if moved > min_move_km:
            errs.append(haversine_km(c[0], c[1], truth[0], truth[1]))
            base.append(moved)
            ctx_xy = np.array([d['vessel'].x[ego_rows[t], :2].cpu().numpy()
                                for t, d in enumerate(ctx)])
            cv = constant_velocity(ctx_xy, target.shape[0])
            cv_errs.append(haversine_km(cv[-1, 0], cv[-1, 1], truth[0], truth[1]))

    out = {'n_windows': len(dataset),
           'spread_correlation': float(np.corrcoef(disps, spreads)[0, 1]) if len(disps) > 1 else float('nan')}
    if errs:
        errs, base, cv_errs = np.array(errs), np.array(base), np.array(cv_errs)
        out.update({'n_moving': len(errs),
                    'model_error_km': float(errs.mean()),
                    'persistence_error_km': float(base.mean()),
                    'beats_persistence_pct': float(100 * (errs < base).mean()),
                    'const_velocity_error_km': float(cv_errs.mean()),
                    'beats_const_velocity_pct': float(100 * (errs < cv_errs).mean())})
    return out


def run_config(cfg, args, ais_snapshots, vessel_stats, land_polygons, ports, device):
    """Train and evaluate one configuration; returns a metrics dict."""
    t_start = time.time()
    n_open = cfg.get('n_open_water', args.n_open_water)
    n_coast = cfg.get('n_coastal', args.n_coastal)
    seq_len = cfg.get('seq_len', args.seq_len)
    future_len = cfg.get('future_len', args.future_len)

    pts, land_flags = sample_domain_points(DMA_BOUNDS, land_polygons, ports,
                                            n_open_water=n_open, n_coastal=n_coast)
    mnf, mei, _ = build_mesh(pts, land_flags, ports)
    mesh_x = torch.as_tensor(mnf, dtype=torch.float)
    mesh_e = torch.as_tensor(mei, dtype=torch.long)
    mtree = cKDTree(mnf[:, :2])

    from ais_ingest import build_world_snapshots
    world, ts, row_maps = build_world_snapshots(ais_snapshots, mnf, mei, mesh_x, mesh_e, mtree)
    # each snapshot keeps its own mesh copy -- sharing one tensor caused a
    # CUDA fault once training ran at scale
    world = [d.to(device) for d in world]

    good = select_ego_vessels_from_stats(
        vessel_stats, min_pings=args.min_pings,
        n_underway=args.n_underway, n_stationary=args.n_stationary)
    train_ids, val_ids = split_vessels(good, args.val_fraction, args.val_seed)

    def build(ids):
        out = []
        for m in ids:
            widx, erow = vessel_presence(row_maps, m)
            if len(widx) < seq_len + future_len:
                continue
            ds = VesselSequenceDataset(world, widx, erow, seq_len, future_len,
                                        stride=args.window_stride)
            if len(ds) > 0:
                out.append(ds)
        return out

    train_ds, val_ds = build(train_ids), build(val_ids)
    if not train_ds or not val_ds:
        return {'error': 'no windows at this configuration',
                'mesh_nodes': int(mnf.shape[0]), 'seq_len': seq_len,
                'future_len': future_len}
    train_c, val_c = ConcatDataset(train_ds), ConcatDataset(val_ds)

    loader = DataLoader(train_c, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_windows)
    model = GCVTP(mesh_in=4, vessel_in=VESSEL_FEATURE_DIM, hidden=args.hidden,
                  future_len=future_len, n_layers=args.n_layers,
                  max_cache_len=max(64, seq_len + 8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    deltas = []
    with torch.no_grad():
        for ctx, ego_rows, target in train_c:
            anchor = ctx[-1]['vessel'].x[ego_rows[-1], :2]
            deltas.append((target - anchor.unsqueeze(0)).cpu())
    norm_scale = torch.cat(deltas, dim=0).std().item()

    model.train()
    losses = []
    for epoch in range(args.n_epochs):
        tot, n = 0.0, 0
        for windows in loader:
            context = encode_windows_batched(model, windows, args.gnn_chunk_size)
            anchors, targets = batch_anchors_and_targets(windows)
            tgt = (targets - anchors.unsqueeze(1)) / norm_scale
            samples = model.head.sample(context, n_samples=args.n_samples)
            loss = energy_score_loss(samples, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(windows); n += len(windows)
        losses.append(tot / max(n, 1))

    metrics = evaluate(model, val_c, norm_scale, n_samples=args.n_samples)
    metrics.update({
        'n_open_water': n_open, 'n_coastal': n_coast,
        'mesh_nodes': int(mnf.shape[0]), 'mesh_edges': int(mei.shape[1]),
        'seq_len': seq_len, 'future_len': future_len,
        'lookback_min': seq_len * args.interval_minutes,
        'horizon_min': future_len * args.interval_minutes,
        'train_windows': len(train_c), 'val_windows': len(val_c),
        'train_vessels': len(train_ds), 'val_vessels': len(val_ds),
        'final_loss': losses[-1] if losses else None,
        'loss_curve': losses,
        'norm_scale': norm_scale,
        'wall_sec': time.time() - t_start,
    })
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--axis', required=True, choices=list(AXES.keys()))
    p.add_argument('--out-dir', default='results/sweep')
    p.add_argument('--ais-glob', default='/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv')
    p.add_argument('--land-shp', default='data/ne_10m_land/ne_10m_land.shp')
    p.add_argument('--ports-csv', default='data/ports_denmark.csv')
    p.add_argument('--cache-dir', default='cache')
    p.add_argument('--interval-minutes', type=int, default=15)
    p.add_argument('--n-open-water', type=int, default=3000)
    p.add_argument('--n-coastal', type=int, default=4000)
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--future-len', type=int, default=4)
    p.add_argument('--n-underway', type=int, default=60)
    p.add_argument('--n-stationary', type=int, default=60)
    p.add_argument('--min-pings', type=int, default=40)
    p.add_argument('--window-stride', type=int, default=1)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--n-samples', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--gnn-chunk-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-epochs', type=int, default=15)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--val-seed', type=int, default=42)
    p.add_argument('--skip-existing', action='store_true',
                   help='resume a partial sweep instead of redoing finished configs')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {device} | axis: {args.axis}", flush=True)

    land_polygons = load_land_polygons(DMA_BOUNDS, natural_earth_path=args.land_shp)
    ports = load_ports(args.ports_csv, DMA_BOUNDS)

    import glob as _glob
    paths = sorted(_glob.glob(args.ais_glob))
    if not paths:
        raise FileNotFoundError(f"no AIS files matched {args.ais_glob}")

    # loaded ONCE and reused by every configuration in the sweep
    ais_snapshots, vessel_stats = load_and_resample_cached(
        paths, DMA_BOUNDS, args.interval_minutes,
        loader_fn=lambda pth, b: load_dma_ais_csv(pth, b, pd.Timestamp.min, pd.Timestamp.max),
        resampler_fn=lambda df, iv: resample_to_snapshots(df, interval_minutes=iv),
        cache_dir=args.cache_dir)
    print(f"  {len(ais_snapshots)} timestamps, {len(vessel_stats)} vessels\n", flush=True)

    for i, cfg in enumerate(AXES[args.axis]):
        tag = "_".join(f"{k}{v}" for k, v in cfg.items())
        out_path = os.path.join(args.out_dir, f"{args.axis}_{tag}.json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{i+1}/{len(AXES[args.axis])}] {tag}: already done, skipping", flush=True)
            continue

        print(f"[{i+1}/{len(AXES[args.axis])}] {tag} ...", flush=True)
        try:
            m = run_config(cfg, args, ais_snapshots, vessel_stats, land_polygons, ports, device)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            m = {'error': f"{type(e).__name__}: {e}", **cfg}
        m['axis'] = args.axis
        m['config'] = cfg
        with open(out_path, 'w') as f:
            json.dump(m, f, indent=2)

        if 'error' not in m:
            print(f"    mesh {m['mesh_nodes']} nodes | windows {m['train_windows']} | "
                  f"err {m.get('model_error_km', float('nan')):.2f} km | "
                  f"vs CV {m.get('const_velocity_error_km', float('nan')):.2f} km | "
                  f"corr {m.get('spread_correlation', float('nan')):.3f} | "
                  f"{m['wall_sec']/60:.1f} min", flush=True)

    print(f"\nsweep complete -> {args.out_dir}")


if __name__ == '__main__':
    main()