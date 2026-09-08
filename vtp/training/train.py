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
import json
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
from graph_data import IrregularVesselDataset
from batching import collate_windows, encode_windows_batched, batch_timing_and_targets
from distributed import (
    setup_distributed, cleanup_distributed, wrap_model, unwrap, make_loader,
    scaled_lr, all_reduce_mean, is_main_process, rank0_print, barrier,
)
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
    p.add_argument('--batch-size', type=int, default=32,
                   help='windows per gradient step')
    p.add_argument('--gnn-chunk-size', type=int, default=16,
                   help='graphs through the GNN at once; bounds transient memory. '
                        'Note snapshots here are per-ego-vessel, so unlike the '
                        'fixed-interval branch there is no reuse to exploit')
    p.add_argument('--window-stride', type=int, default=1,
                   help='keep every Nth window per vessel')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-epochs', type=int, default=50)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--val-seed', type=int, default=42)
    p.add_argument('--val-every', type=int, default=5,
                   help='run held-out validation every N epochs (0 disables). '
                        'Training loss falling while validation stalls is the '
                        'overfitting signal; without this you only find out at the end')
    p.add_argument('--val-subsample', type=int, default=400,
                   help='windows scored in the periodic check; the final run uses all')
    p.add_argument('--early-stop-patience', type=int, default=0,
                   help='stop if validation FDE has not improved for N checks (0 = never)')
    p.add_argument('--checkpoint-path', type=str, default='checkpoints/irregular.pt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--distributed', action='store_true',
                   help='multi-GPU via torchrun --nproc_per_node=N')
    p.add_argument('--lr-scale-rule', choices=['sqrt', 'linear', 'none'], default='sqrt')
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
        # Snapshots stay on CPU. Batching moves one chunk at a time
        # (see batching.py) -- holding a 12-day dataset on the GPU meant
        # ~50 GB of duplicated mesh and OOM'd an 80 GB A100.
        ds = IrregularVesselDataset(snaps, ego_idx, times, positions,
                                     seq_len=args.seq_len, future_len=args.future_len,
                                     max_window_span_sec=args.max_window_span_sec,
                                     stride=args.window_stride)
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
def run_validation(model, dataset, vel_scale, n_samples=32, min_move_km=1.0, eps=1e-6,
                    max_windows=None, seed=0):
    """
    Held-out metrics.

    `max_windows` subsamples for the periodic in-training check: scoring
    every window each epoch would cost more than the epoch itself, and a
    few hundred windows is plenty to see whether validation is tracking
    training or diverging from it. The final run uses the full set.
    """
    model.eval()
    device = next(model.parameters()).device
    disps, spreads, errs, base_errs, dts = [], [], [], [], []

    idxs = range(len(dataset))
    if max_windows is not None and max_windows < len(dataset):
        idxs = np.random.default_rng(seed).choice(len(dataset), max_windows, replace=False)

    for i in idxs:
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

    res = {'n_val_windows': len(disps),
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
    device, rank, world_size = setup_distributed()
    rank0_print(f"device: {device} | world size: {world_size}")
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

    loader, sampler = make_loader(train_combined, args.batch_size, collate_windows,
                                   shuffle=True, seed=args.val_seed)

    model = IrregularVTP(hidden=args.hidden, n_heads=args.n_heads,
                          n_layers=args.n_layers).to(device)
    model = wrap_model(model, device)
    lr = scaled_lr(args.lr, world_size, args.lr_scale_rule)
    if world_size > 1:
        rank0_print(f"  lr {args.lr} -> {lr:.2e} ({args.lr_scale_rule} rule, "
                    f"effective batch {args.batch_size * world_size})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    vel_scale, start_epoch = None, 0
    if args.resume and os.path.exists(args.checkpoint_path):
        ck = torch.load(args.checkpoint_path, map_location=device)
        unwrap(model).load_state_dict(ck['model_state_dict'])
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

    # best-checkpoint tracking: the last epoch is not necessarily the
    # best one, and without this an overfitting run silently overwrites
    # its own best weights
    best_path = args.checkpoint_path.replace('.pt', '_best.pt')
    history_path = args.checkpoint_path.replace('.pt', '_history.json')
    history, best_val, best_epoch, stale_checks = [], None, None, 0
    if args.resume and os.path.exists(args.checkpoint_path):
        _ck = torch.load(args.checkpoint_path, map_location='cpu')
        history = _ck.get('history', [])
        best_val = _ck.get('best_val_error_km')
        best_epoch = _ck.get('best_epoch')

    rank0_print(f"training... (validation every {args.val_every} epochs on "
                f"{args.val_subsample} windows)")
    eps = 1e-6
    for epoch in range(start_epoch, args.n_epochs):
        model.train()
        total = 0.0
        n_win = 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        for windows in loader:
            context = encode_windows_batched(unwrap(model), windows,
                                              args.gnn_chunk_size, device)
            anchors, targets, dts = batch_timing_and_targets(windows, context.device)

            # velocity-normalized target (see module docstring)
            dt_col = dts.clamp(min=eps).unsqueeze(-1)
            tgt = ((targets - anchors.unsqueeze(1)) / dt_col) / vel_scale

            out = unwrap(model).head.sample(context, dts, n_samples=args.n_samples)
            loss = energy_score_loss(out, tgt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(windows)
            n_win += len(windows)

        avg = all_reduce_mean(total / max(n_win, 1), device)

        val_line, val_err = "", None
        do_val = (args.val_every and val_combined is not None
                  and len(val_combined) > 0
                  and ((epoch + 1) % args.val_every == 0 or epoch == args.n_epochs - 1))
        if do_val and is_main_process():
            vres = run_validation(unwrap(model), val_combined, vel_scale,
                                   max_windows=args.val_subsample, seed=args.val_seed)
            val_err = vres.get('model_mean_error_km')
            history.append({'epoch': epoch, 'train_loss': avg, **vres})
            val_line = (f" | val err {val_err:.3f} km"
                        f" (baseline {vres.get('trivial_baseline_mean_error_km', float('nan')):.3f})"
                        f" corr {vres.get('spread_correlation', float('nan')):.3f}")
            model.train()          # run_validation put it in eval mode

        rank0_print(f"epoch {epoch}: avg loss = {avg:.4f}{val_line}")

        if val_err is not None:
            if best_val is None or val_err < best_val - 1e-6:
                best_val, best_epoch, stale_checks = val_err, epoch, 0
                if is_main_process():
                    torch.save({'model_state_dict': unwrap(model).state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'vel_scale': vel_scale, 'epoch': epoch,
                                'avg_loss': avg, 'val_error_km': val_err,
                                'args': vars(args)}, best_path)
            else:
                stale_checks += 1

        if is_main_process():
            torch.save({'model_state_dict': unwrap(model).state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                        'vel_scale': vel_scale, 'epoch': epoch, 'avg_loss': avg,
                        'history': history, 'best_val_error_km': best_val,
                        'best_epoch': best_epoch,
                        'args': vars(args)}, args.checkpoint_path)
            if history:
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
        barrier()

        if (args.early_stop_patience and stale_checks >= args.early_stop_patience):
            rank0_print(f"early stop: no validation improvement for "
                        f"{stale_checks} checks (best {best_val:.3f} km @ epoch {best_epoch})")
            break

    rank0_print("training complete.")
    if is_main_process() and val_combined is not None and len(val_combined) > 0:
        if best_val is not None:
            print(f"\nbest validation: {best_val:.3f} km @ epoch {best_epoch} "
                  f"-> {best_path}", flush=True)
        print("\nfinal validation on ALL held-out windows...", flush=True)
        for k, v in run_validation(unwrap(model), val_combined, vel_scale).items():
            print(f"  {k}: {v}")
    cleanup_distributed()


if __name__ == '__main__':
    main()