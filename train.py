"""
Training script (fixed-interval branch) using SHARED world snapshots.

The previous version built one graph per (ego vessel, timestamp). Since
the world at a timestamp is identical regardless of which vessel you're
forecasting, that duplicated the same graph once per ego vessel --
measured at ~856x redundancy in a 300-vessel test, and the direct cause
of CUDA OOM on multi-day runs. Here each timestamp's graph is built once
and shared; each vessel's dataset holds index pairs into it.
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
from ais_ingest import (
    load_dma_ais_csv, resample_to_snapshots, select_ego_vessels_stratified,
    build_world_snapshots, vessel_presence, VESSEL_FEATURE_DIM,
)
from graph_data import VesselSequenceDataset
from ais_cache import load_and_resample_cached, select_ego_vessels_from_stats
from distributed import (
    setup_distributed, cleanup_distributed, wrap_model, unwrap, make_loader,
    scaled_lr, all_reduce_mean, is_main_process, rank0_print, barrier, get_world_size,
)
from batching import (
    collate_windows, encode_windows_batched, batch_anchors_and_targets, dedup_ratio,
)
from model import GCVTP
from losses import energy_score_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ais-glob', type=str, default='data/ais/aisdk-*.csv')
    p.add_argument('--land-shp', type=str, default='data/ne_10m_land/ne_10m_land.shp')
    p.add_argument('--ports-csv', type=str, default='data/ports_denmark.csv')
    p.add_argument('--n-underway', type=int, default=150)
    p.add_argument('--n-stationary', type=int, default=150)
    p.add_argument('--min-pings', type=int, default=40)
    p.add_argument('--sog-threshold-knots', type=float, default=2.0)
    p.add_argument('--interval-minutes', type=int, default=15)
    p.add_argument('--n-open-water', type=int, default=3000,
                   help='mesh sample points in open water; controls mesh density')
    p.add_argument('--n-coastal', type=int, default=4000,
                   help='mesh sample points along coastlines/ports')
    p.add_argument('--cache-dir', type=str, default='cache',
                   help='disk cache for the load+resample stage')
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--future-len', type=int, default=4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=3)
    p.add_argument('--n-samples', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32,
                   help='windows per gradient step; at 1 the GPU is mostly idle on '
                        'launch overhead, which is what made 100k-window epochs unusable')
    p.add_argument('--gnn-chunk-size', type=int, default=32,
                   help='graphs pushed through the GNN at once; bounds transient memory '
                        'independently of batch-size (each graph replicates the mesh)')
    p.add_argument('--window-stride', type=int, default=1,
                   help='keep every Nth window per vessel; lets window count be a choice '
                        'rather than a function of how much data was loaded')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-epochs', type=int, default=50)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--val-seed', type=int, default=42)
    p.add_argument('--checkpoint-path', type=str, default='checkpoints/checkpoint.pt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--distributed', action='store_true',
                   help='multi-GPU via torchrun --nproc_per_node=N')
    p.add_argument('--lr-scale-rule', choices=['sqrt', 'linear', 'none'], default='sqrt',
                   help='how to adjust lr for the larger effective batch under DDP')
    return p.parse_args()


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def split_vessels_train_val(good_vessels, val_fraction, seed):
    """Split BY VESSEL (not window), stratified by regime -- windows from
    one vessel are highly correlated, so a window-level split would leak."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for _, g in good_vessels.groupby('regime'):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx)))) if len(idx) > 1 else 0
        va.extend(idx[:n_val]); tr.extend(idx[n_val:])
    return tr, va


def build_datasets(vessel_ids, world, row_maps, seq_len, future_len, stride=1):
    out = []
    for mmsi in vessel_ids:
        widx, erow = vessel_presence(row_maps, mmsi)
        if len(widx) < seq_len + future_len:
            continue
        ds = VesselSequenceDataset(world, widx, erow, seq_len, future_len, stride=stride)
        if len(ds) > 0:
            out.append(ds)
    return out


@torch.no_grad()
def run_validation(model, dataset, norm_scale, n_samples=32, min_move_km=1.0):
    model.eval()
    disps, spreads, errs, base = [], [], [], []
    for i in range(len(dataset)):
        ctx, ego_rows, target = dataset[i]
        dev = next(model.parameters()).device
        ctx = [d.clone().to(dev) for d in ctx]          # clone: leave CPU originals intact
        target = target.to(dev)
        anchor = ctx[-1]['vessel'].x[ego_rows[-1], :2]
        samples = model(ctx, ego_rows, n_samples=n_samples, training=False)
        pred = samples * norm_scale + anchor

        final = pred[0, :, -1, :].cpu().numpy()
        truth = target[-1].cpu().numpy()
        a = anchor.cpu().numpy()
        moved = haversine_km(a[0], a[1], truth[0], truth[1])
        c = final.mean(axis=0)
        spread = np.mean([haversine_km(c[0], c[1], p[0], p[1]) for p in final])
        disps.append(moved); spreads.append(spread)
        if moved > min_move_km:
            errs.append(haversine_km(c[0], c[1], truth[0], truth[1]))
            base.append(moved)

    res = {'n_val_windows': len(dataset),
           'spread_correlation': float(np.corrcoef(disps, spreads)[0, 1]) if len(disps) > 1 else float('nan')}
    if errs:
        errs, base = np.array(errs), np.array(base)
        res.update({'n_moving_windows': len(errs),
                    'model_mean_error_km': float(errs.mean()),
                    'trivial_baseline_mean_error_km': float(base.mean()),
                    'beats_baseline_pct': float(100 * (errs < base).mean())})
    return res


def main():
    args = parse_args()
    device, rank, world_size = setup_distributed()
    rank0_print(f"device: {device} | world size: {world_size}")
    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)

    bounds = DMA_BOUNDS
    print("building mesh...")
    land_polygons = load_land_polygons(bounds, natural_earth_path=args.land_shp)
    ports = load_ports(args.ports_csv, bounds)
    pts, land_flags = sample_domain_points(bounds, land_polygons, ports,
                                            n_open_water=args.n_open_water,
                                            n_coastal=args.n_coastal)
    mesh_node_features, mesh_edge_index, _ = build_mesh(pts, land_flags, ports)
    mesh_x_tensor = torch.as_tensor(mesh_node_features, dtype=torch.float, device='cpu')
    mesh_edge_tensor = torch.as_tensor(mesh_edge_index, dtype=torch.long, device='cpu')
    mesh_tree = cKDTree(mesh_node_features[:, :2])
    rank0_print(f"  mesh nodes: {mesh_node_features.shape[0]}, edges: {mesh_edge_index.shape[1]}")

    paths = sorted(glob.glob(args.ais_glob))
    if not paths:
        raise FileNotFoundError(f"no AIS files matched {args.ais_glob}")
    rank0_print(f"loading {len(paths)} AIS file(s)...")

    # Load + resample is expensive (~25 min for 12 days) and identical on
    # every job, so it is cached to disk keyed by the input files and the
    # resampling interval. This matters most for sweeps, which would
    # otherwise repeat the same work once per configuration.
    ais_snapshots, vessel_stats = load_and_resample_cached(
        paths, bounds, args.interval_minutes,
        loader_fn=lambda pth, bnds: load_dma_ais_csv(pth, bnds, pd.Timestamp.min, pd.Timestamp.max),
        resampler_fn=lambda df, iv: resample_to_snapshots(df, interval_minutes=iv),
        cache_dir=args.cache_dir, force_rebuild=args.no_cache)
    rank0_print(f"  timestamps: {len(ais_snapshots)}")

    rank0_print("selecting ego vessels...")
    good = select_ego_vessels_from_stats(
        vessel_stats, min_pings=args.min_pings,
        sog_threshold_knots=args.sog_threshold_knots,
        n_underway=args.n_underway, n_stationary=args.n_stationary)
    print(f"  {good['regime'].value_counts().to_dict()}")
    train_ids, val_ids = split_vessels_train_val(good, args.val_fraction, args.val_seed)
    print(f"  train vessels: {len(train_ids)}, val vessels: {len(val_ids)}")

    # ONE graph per timestamp, shared across every ego vessel
    print("building shared world snapshots (one per timestamp)...")
    world, timestamps, row_maps = build_world_snapshots(
        ais_snapshots, mesh_node_features, mesh_edge_index,
        mesh_x_tensor, mesh_edge_tensor, mesh_tree,
        progress_every=max(1, len(ais_snapshots)//10))
    # Snapshots stay on CPU; batching moves one chunk at a time (see
    # batching.py). Holding them all on the GPU duplicated the mesh per
    # snapshot and did not scale.
    print(f"  {len(world)} world snapshots on cpu "
          f"(chunks moved to {device} during training)", flush=True)

    train_ds = build_datasets(train_ids, world, row_maps, args.seq_len, args.future_len,
                               stride=args.window_stride)
    val_ds = build_datasets(val_ids, world, row_maps, args.seq_len, args.future_len,
                             stride=args.window_stride)
    train_combined = ConcatDataset(train_ds)
    val_combined = ConcatDataset(val_ds) if val_ds else None
    print(f"  train windows: {len(train_combined)} across {len(train_ds)} vessels")
    print(f"  val windows: {len(val_combined) if val_combined else 0} across {len(val_ds)} vessels")
    if len(train_combined) == 0:
        raise RuntimeError("no training windows -- loosen vessel selection or check data")

    loader, sampler = make_loader(train_combined, args.batch_size, collate_windows,
                                   shuffle=True, seed=args.val_seed)

    model = GCVTP(mesh_in=4, vessel_in=VESSEL_FEATURE_DIM, hidden=args.hidden,
                  future_len=args.future_len, n_heads=args.n_heads,
                  n_layers=args.n_layers, max_cache_len=max(64, args.seq_len + 8)).to(device)
    model = wrap_model(model, device)
    lr = scaled_lr(args.lr, world_size, args.lr_scale_rule)
    if world_size > 1:
        rank0_print(f"  lr {args.lr} -> {lr:.2e} ({args.lr_scale_rule} rule, "
                    f"effective batch {args.batch_size * world_size})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    norm_scale, start_epoch = None, 0
    if args.resume and os.path.exists(args.checkpoint_path):
        ck = torch.load(args.checkpoint_path, map_location=device)
        unwrap(model).load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimizer_state_dict'])
        norm_scale = ck['norm_scale']; start_epoch = ck['epoch'] + 1
        print(f"resumed at epoch {start_epoch}")
    elif args.resume:
        print(f"--resume given but no checkpoint at {args.checkpoint_path}, starting fresh")

    if norm_scale is None:
        print("computing delta normalization scale (train vessels only)...")
        deltas = []
        # no_grad: this loop touches every training window and never
        # backprops, so building autograd state for it is pure waste
        with torch.no_grad():
          for ctx, ego_rows, target in train_combined:
            anchor = ctx[-1]['vessel'].x[ego_rows[-1], :2]
            deltas.append((target - anchor.unsqueeze(0)).cpu())
        norm_scale = torch.cat(deltas, dim=0).std().item()
    rank0_print(f"  norm_scale: {norm_scale:.5f}")

    print(f"training (batch_size={args.batch_size}, gnn_chunk={args.gnn_chunk_size})...")
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        total, n_win = 0.0, 0
        if sampler is not None:
            sampler.set_epoch(epoch)   # without this every rank repeats one order
        for windows in loader:
            context = encode_windows_batched(unwrap(model), windows,
                                              args.gnn_chunk_size, device)
            anchors, targets = batch_anchors_and_targets(windows, device)
            tgt = (targets - anchors.unsqueeze(1)) / norm_scale
            samples = unwrap(model).head.sample(context, n_samples=args.n_samples)
            loss = energy_score_loss(samples, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(windows)
            n_win += len(windows)

        avg = all_reduce_mean(total / max(n_win, 1), device)
        rank0_print(f"epoch {epoch}: avg loss = {avg:.4f}")
        if is_main_process():
            torch.save({'model_state_dict': unwrap(model).state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                        'norm_scale': norm_scale, 'epoch': epoch, 'avg_loss': avg,
                        'args': vars(args)}, args.checkpoint_path)
        barrier()

    rank0_print("training complete.")
    # validation on rank 0 only -- it is cheap next to training, and
    # sharding it would just complicate aggregation
    if is_main_process() and val_combined is not None and len(val_combined) > 0:
        print("\nvalidation on held-out vessels...", flush=True)
        for k, v in run_validation(unwrap(model), val_combined, norm_scale).items():
            print(f"  {k}: {v}")
    cleanup_distributed()


if __name__ == '__main__':
    main()