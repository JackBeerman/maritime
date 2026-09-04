"""
Reusable evaluation and visualization for the IRREGULAR-SAMPLING branch.

    from viz_irregular import (setup, load_checkpoint_model, evaluate,
                               plot_prediction, plot_grid, animate_prediction,
                               per_horizon_errors, analyze_failures,
                               summarize_failures)

Differences from the fixed-interval `viz.py` that matter when reading
results:

* A window is `(ctx, ctx_times, target_positions, target_dts)`. Targets
  are REAL observed pings at their true elapsed times, never interpolated
  onto a grid -- so every window has its own forecast horizon rather than
  a shared +15/30/45/60 min.
* Model output is a normalized VELOCITY residual. Recovering a position
  is `output * vel_scale * dt + anchor` -- the dt factor is essential and
  easy to forget, so it lives in exactly one place (`sample_positions`).
* Because horizons vary, a single "mean error" mixes 5-minute and
  90-minute forecasts. `per_horizon_errors` buckets by actual elapsed
  time, which is the honest way to read accuracy here and the only way to
  compare against the fixed-interval branch.
* Ego row is read from `ego_mask` (snapshots are per-ego-vessel on this
  branch, so the mask is meaningful, unlike the shared world snapshots on
  `main`).
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import cKDTree
from torch.utils.data import ConcatDataset

from coastline import load_land_polygons, load_ports, DMA_BOUNDS
from mesh import sample_domain_points, build_mesh
from irregular_ingest import (
    load_dma_ais_csv, build_ego_anchored_snapshots, select_ego_vessels_stratified,
)
from graph_data import IrregularVesselDataset
from model import IrregularVTP


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class Context:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        n = len(self.combined_val) if self.combined_val else 0
        return (f"Context(mesh_nodes={self.mesh_node_features.shape[0]}, "
                f"val_windows={n})")


def setup(ais_path="/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv",
          land_shp="data/ne_10m_land/ne_10m_land.shp",
          ports_csv="data/ports_denmark.csv",
          seq_len=12, future_len=4,
          min_ping_gap_sec=900, staleness_cutoff_sec=1800,
          neighbor_radius_deg=0.5, max_neighbors=150,
          n_underway=300, n_stationary=300,
          val_fraction=0.2, val_seed=42, held_out_only=True,
          device=None, verbose=True):
    """
    Build mesh, load raw AIS, and assemble ego-anchored windows.

    Pass the SAME min_ping_gap_sec / n_underway / n_stationary / val_seed
    the training run used. min_ping_gap_sec in particular determines the
    forecast horizon, so a mismatch silently evaluates a different task
    than the one that was trained.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    bounds = DMA_BOUNDS

    land_polygons = load_land_polygons(bounds, natural_earth_path=land_shp)
    ports = load_ports(ports_csv, bounds)
    pts, land_flags = sample_domain_points(bounds, land_polygons, ports)
    mnf, mei, _ = build_mesh(pts, land_flags, ports)
    mesh_x = torch.as_tensor(mnf, dtype=torch.float)
    mesh_e = torch.as_tensor(mei, dtype=torch.long)
    mesh_tree = cKDTree(mnf[:, :2])
    if verbose:
        print(f"mesh: {mnf.shape[0]} nodes, {mei.shape[1]} edges")

    ais_df = load_dma_ais_csv(ais_path, bounds)
    if verbose:
        print(f"AIS: {len(ais_df)} records, {ais_df['mmsi'].nunique()} vessels")

    good = select_ego_vessels_stratified(
        ais_df, min_pings=60, n_underway=n_underway, n_stationary=n_stationary)

    rng = np.random.default_rng(val_seed)
    train_ids, val_ids = [], []
    for _, g in good.groupby('regime'):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx)))) if len(idx) > 1 else 0
        val_ids.extend(idx[:n_val]); train_ids.extend(idx[n_val:])
    if verbose:
        print(f"vessels: {len(train_ids)} train, {len(val_ids)} held-out")

    use_ids = val_ids if held_out_only else list(train_ids) + list(val_ids)
    datasets = []
    for mmsi in use_ids:
        snaps, ego_idx, times, positions = build_ego_anchored_snapshots(
            ais_df, mmsi, mnf, mei,
            staleness_cutoff_sec=staleness_cutoff_sec,
            neighbor_radius_deg=neighbor_radius_deg,
            max_neighbors=max_neighbors, min_ping_gap_sec=min_ping_gap_sec,
            mesh_x_tensor=mesh_x, mesh_edge_tensor=mesh_e, mesh_tree=mesh_tree)
        if len(snaps) < seq_len + future_len:
            continue
        # each snapshot keeps its own mesh copy -- sharing one tensor
        # caused a CUDA device-side assert once training ran at scale
        snaps = [d.to(device) for d in snaps]
        ds = IrregularVesselDataset(snaps, ego_idx, times, positions,
                                     seq_len=seq_len, future_len=future_len)
        if len(ds) > 0:
            datasets.append(ds)
    combined = ConcatDataset(datasets) if datasets else None
    if verbose:
        label = "held-out" if held_out_only else "all"
        print(f"{label} windows: {len(combined) if combined else 0} across {len(datasets)} vessels")

    return Context(device=device, bounds=bounds, land_polygons=land_polygons,
                   ports=ports, mesh_node_features=mnf, mesh_edge_index=mei,
                   mesh_tree=mesh_tree, ais_df=ais_df,
                   train_ids=train_ids, val_ids=val_ids,
                   combined_val=combined, seq_len=seq_len, future_len=future_len)


def load_checkpoint_model(checkpoint_path, device=None):
    """Loads a checkpoint, inferring architecture from the state dict.
    Returns (model in eval mode, vel_scale, metadata)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt['model_state_dict']

    vessel_in = sd['gnn.vessel_proj.weight'].shape[1]
    hidden = sd['gnn.vessel_proj.weight'].shape[0]
    mesh_in = sd['gnn.mesh_proj.weight'].shape[1]
    n_layers = len([k for k in sd if k.startswith('temporal.layers.')
                     and k.endswith('.attn.qkv.weight')])

    model = IrregularVTP(mesh_in=mesh_in, vessel_in=vessel_in, hidden=hidden,
                          n_layers=n_layers).to(device)
    model.load_state_dict(sd)
    model.eval()

    meta = {'epoch': ckpt.get('epoch'), 'avg_loss': ckpt.get('avg_loss'),
            'hidden': hidden, 'n_layers': n_layers, 'vessel_in': vessel_in}
    if 'args' in ckpt:
        for k in ('min_ping_gap_sec', 'seq_len', 'future_len'):
            if k in ckpt['args']:
                meta[k] = ckpt['args'][k]
    return model, ckpt['vel_scale'], meta


@torch.no_grad()
def sample_positions(model, dataset, idx, vel_scale, n_samples=32, eps=1e-6):
    """
    The single place normalized velocity residuals become real
    coordinates: `output * vel_scale * dt + anchor`.
    """
    ctx, ctx_times, target_pos, target_dts = dataset[idx]
    ego_rows = [int(d['vessel'].ego_mask.nonzero()[0].item()) for d in ctx]
    anchor = ctx[-1]['vessel'].x[ego_rows[-1], :2]
    dts = target_dts.to(anchor.device)

    out = model(ctx, ego_rows, ctx_times, dts, n_samples=n_samples)
    dt_col = dts.clamp(min=eps).view(1, 1, -1, 1)
    samples_real = out * vel_scale * dt_col + anchor

    context_xy = np.array([d['vessel'].x[ego_rows[t], :2].cpu().numpy()
                            for t, d in enumerate(ctx)])
    samples_np = samples_real[0].cpu().numpy()
    truth_np = target_pos.cpu().numpy()
    a = anchor.cpu().numpy()
    center = samples_np[:, -1, :].mean(axis=0)

    return {
        'context_xy': context_xy, 'context_times': ctx_times.cpu().numpy(),
        'samples': samples_np, 'truth': truth_np, 'anchor': a,
        'target_dts': dts.cpu().numpy(),
        'mean_path': samples_np.mean(axis=0),
        'medoid_path': samples_np[medoid_index(samples_np)],
        'moved_km': haversine_km(a[0], a[1], truth_np[-1, 0], truth_np[-1, 1]),
        'err_km': haversine_km(center[0], center[1], truth_np[-1, 0], truth_np[-1, 1]),
        'spread_km': float(np.mean([haversine_km(center[0], center[1], p[0], p[1])
                                      for p in samples_np[:, -1, :]])),
        'horizon_min': float(dts[-1].cpu()) / 60.0,
    }


def constant_velocity_prediction(context_xy, context_times, target_dts):
    """
    Dead-reckoning baseline for irregular sampling.

    Velocity is computed from the last two observed pings using their
    REAL elapsed time (not an assumed step), then extrapolated to each
    target's own dt. On this branch both the input spacing and the
    target horizons vary, so using elapsed seconds rather than step
    counts is essential -- otherwise the baseline is wrong in a way that
    flatters the model.

    This is a far stronger bar than persistence ("vessel stays put"),
    which is what `persistence_error_km` reports. Ships mostly travel
    straight, so constant velocity is the comparison that actually says
    whether the model has learned anything beyond extrapolation.
    """
    if len(context_xy) < 2:
        return np.repeat(context_xy[-1:], len(target_dts), axis=0)
    dt_ctx = context_times[-1] - context_times[-2]
    if abs(dt_ctx) < 1e-6:
        return np.repeat(context_xy[-1:], len(target_dts), axis=0)
    v = (context_xy[-1] - context_xy[-2]) / dt_ctx      # degrees per second
    return np.stack([context_xy[-1] + v * float(dt) for dt in target_dts])


def medoid_index(samples_np):
    """Sample closest to all others -- an ACTUAL path, unlike the mean."""
    S = samples_np.shape[0]
    flat = samples_np.reshape(S, -1)
    d = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(d.sum(axis=1).argmin())


def evaluate(model, dataset, vel_scale, n_samples=32, min_move_km=1.0, verbose=True):
    """Headline metrics, plus the horizon distribution -- without which
    the error number can't be interpreted or compared across branches."""
    disps, spreads, errs, base, hors, cv_errs = [], [], [], [], [], []
    for i in range(len(dataset)):
        r = sample_positions(model, dataset, i, vel_scale, n_samples)
        disps.append(r['moved_km']); spreads.append(r['spread_km'])
        hors.append(r['horizon_min'])
        if r['moved_km'] > min_move_km:
            errs.append(r['err_km']); base.append(r['moved_km'])
            cv = constant_velocity_prediction(r['context_xy'], r['context_times'],
                                               r['target_dts'])
            cv_errs.append(haversine_km(cv[-1, 0], cv[-1, 1],
                                         r['truth'][-1, 0], r['truth'][-1, 1]))

    out = {'n_windows': len(dataset),
           'median_horizon_min': float(np.median(hors)),
           'horizon_p10_min': float(np.percentile(hors, 10)),
           'horizon_p90_min': float(np.percentile(hors, 90)),
           'spread_correlation': float(np.corrcoef(disps, spreads)[0, 1]) if len(disps) > 1 else float('nan')}
    if errs:
        errs, base, cv_errs = np.array(errs), np.array(base), np.array(cv_errs)
        out.update({'n_moving': len(errs), 'model_mean_error_km': float(errs.mean()),
                    # persistence: error equals how far the vessel actually moved
                    'persistence_error_km': float(base.mean()),
                    'beats_persistence_pct': float(100 * (errs < base).mean()),
                    # constant velocity (dead reckoning): the meaningful bar
                    'const_velocity_error_km': float(cv_errs.mean()),
                    'beats_const_velocity_pct': float(100 * (errs < cv_errs).mean())})
    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return out


def per_horizon_errors(model, dataset, vel_scale, n_samples=32,
                        bins_min=(0, 15, 30, 45, 60, 90, 120, 1e9),
                        min_move_km=1.0, verbose=True):
    """
    Error bucketed by ACTUAL elapsed time to the target.

    On this branch every window has its own horizon, so a single mean
    error silently averages 5-minute and 90-minute forecasts. Bucketing
    is also the only way to compare fairly against the fixed-interval
    branch, which always predicts exactly 60 minutes ahead.
    """
    rows = []
    for i in range(len(dataset)):
        r = sample_positions(model, dataset, i, vel_scale, n_samples)
        if r['moved_km'] > min_move_km:
            rows.append((r['horizon_min'], r['err_km'], r['moved_km']))
    if not rows:
        print("no moving windows")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['horizon_min', 'err_km', 'moved_km'])
    df['bucket'] = pd.cut(df['horizon_min'], bins=list(bins_min))
    g = df.groupby('bucket', observed=True).agg(
        n=('err_km', 'size'), mean_err_km=('err_km', 'mean'),
        mean_baseline_km=('moved_km', 'mean'),
        beats_pct=('err_km', lambda s: 100 * (s < df.loc[s.index, 'moved_km']).mean()))
    if verbose:
        print(g.to_string())
    return g


def _basemap(ax, land_polygons, ports):
    ax.set_facecolor('#cfe8f3')
    for poly in land_polygons:
        xs, ys = poly.exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), facecolor='#d8c39a',
                                 edgecolor='#8a7350', linewidth=0.5, zorder=1))
    ax.scatter([p[0] for p in ports], [p[1] for p in ports], s=30, c='black',
               marker='^', zorder=2, label='port')


def _fit_view(ax, all_xy, min_window_deg=0.05):
    """Minimum extent, so a near-stationary window doesn't auto-scale to a
    tiny box and make ordinary noise look like wild disagreement."""
    lon_r = max(all_xy[:, 0].max() - all_xy[:, 0].min(), min_window_deg)
    lat_r = max(all_xy[:, 1].max() - all_xy[:, 1].min(), min_window_deg)
    cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(cx - lon_r/2 - 0.01, cx + lon_r/2 + 0.01)
    ax.set_ylim(cy - lat_r/2 - 0.01, cy + lat_r/2 + 0.01)
    ax.set_aspect('equal')


def plot_prediction(model, dataset, idx, vel_scale, ctx_obj, n_samples=32,
                     ax=None, show_summaries=True, figsize=(9, 8)):
    r = sample_positions(model, dataset, idx, vel_scale, n_samples)
    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    _basemap(ax, ctx_obj.land_polygons, ctx_obj.ports)

    ax.plot(r['context_xy'][:, 0], r['context_xy'][:, 1], c='black', marker='o',
            markersize=4, linewidth=1.5, label='observed pings', zorder=5)
    start = r['context_xy'][-1:]
    for s in range(r['samples'].shape[0]):
        p = np.vstack([start, r['samples'][s]])
        ax.plot(p[:, 0], p[:, 1], c='steelblue', alpha=0.22, linewidth=1, zorder=2,
                label='predicted samples' if s == 0 else None)
    if show_summaries:
        mp = np.vstack([start, r['mean_path']])
        ax.plot(mp[:, 0], mp[:, 1], c='darkblue', ls='--', lw=2, label='sample mean', zorder=6)
        dp = np.vstack([start, r['medoid_path']])
        ax.plot(dp[:, 0], dp[:, 1], c='purple', ls=':', lw=2, label='medoid', zorder=6)
    tp = np.vstack([start, r['truth']])
    ax.plot(tp[:, 0], tp[:, 1], c='red', marker='*', markersize=9, lw=2.2,
            label='actual pings', zorder=7)

    _fit_view(ax, np.vstack([r['context_xy'], r['samples'].reshape(-1, 2), r['truth']]))
    ax.set_title(f"horizon {r['horizon_min']:.0f} min | moved {r['moved_km']:.2f}km | "
                  f"err {r['err_km']:.2f}km | spread {r['spread_km']:.2f}km", fontsize=10)
    return fig, r


def classify_windows(model, dataset, vel_scale, move_threshold_km=3.0):
    moving, stationary = [], []
    for i in range(len(dataset)):
        ctx, _, target_pos, _ = dataset[i]
        ego = int(ctx[-1]['vessel'].ego_mask.nonzero()[0].item())
        a = ctx[-1]['vessel'].x[ego, :2].cpu().numpy()
        t = target_pos[-1].cpu().numpy()
        (moving if haversine_km(a[0], a[1], t[0], t[1]) > move_threshold_km
         else stationary).append(i)
    return moving, stationary


def plot_grid(model, dataset, vel_scale, ctx_obj, moving=None, stationary=None,
               n_each=3, seed=0, n_samples=32, figsize=(16, 10)):
    """2x3 grid, sampled at RANDOM -- taking the first few windows tends to
    draw them all from one vessel in one place."""
    if moving is None or stationary is None:
        moving, stationary = classify_windows(model, dataset, vel_scale)
    rng = np.random.default_rng(seed)
    pm = rng.choice(moving, size=min(n_each, len(moving)), replace=False) if moving else []
    ps = rng.choice(stationary, size=min(n_each, len(stationary)), replace=False) if stationary else []

    fig, axes = plt.subplots(2, n_each, figsize=figsize)
    for ax, i in zip(axes.flat, list(pm) + list(ps)):
        plot_prediction(model, dataset, int(i), vel_scale, ctx_obj, n_samples=n_samples, ax=ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Top: moving | Bottom: near-stationary  (irregular sampling)\n'
                  'black = observed pings | blue = samples | red = actual', fontsize=12)
    fig.tight_layout()
    return fig


def animate_prediction(model, dataset, idx, vel_scale, ctx_obj, n_samples=32,
                        interval_ms=700, out_path='prediction_irregular.gif',
                        figsize=(10, 8)):
    """
    Reveals the forecast one observed ping at a time. Unlike the
    fixed-interval branch, the title shows each target's REAL elapsed
    time, which varies within a single window.
    """
    r = sample_positions(model, dataset, idx, vel_scale, n_samples)
    F = r['truth'].shape[0]
    start = r['context_xy'][-1:]

    fig, ax = plt.subplots(figsize=figsize)
    _basemap(ax, ctx_obj.land_polygons, ctx_obj.ports)
    ax.plot(r['context_xy'][:, 0], r['context_xy'][:, 1], c='black', marker='o',
            markersize=5, linewidth=2, label='observed pings (input)', zorder=5)

    lines = [ax.plot([], [], c='steelblue', alpha=0.22, lw=1, zorder=2,
                      label='predicted samples' if s == 0 else None)[0]
              for s in range(r['samples'].shape[0])]
    mean_l, = ax.plot([], [], c='darkblue', ls='--', lw=2, label='sample mean', zorder=6)
    med_l, = ax.plot([], [], c='purple', ls=':', lw=2, label='medoid', zorder=6)
    truth_l, = ax.plot([], [], c='red', marker='*', markersize=10, lw=2.5,
                        label='actual pings', zorder=7)

    _fit_view(ax, np.vstack([r['context_xy'], r['samples'].reshape(-1, 2), r['truth']]))
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.legend(loc='upper left', fontsize=9)
    title = ax.set_title('')

    def update(frame):
        step = frame + 1
        for s, ln in enumerate(lines):
            p = np.vstack([start, r['samples'][s, :step]])
            ln.set_data(p[:, 0], p[:, 1])
        mp = np.vstack([start, r['mean_path'][:step]]); mean_l.set_data(mp[:, 0], mp[:, 1])
        dp = np.vstack([start, r['medoid_path'][:step]]); med_l.set_data(dp[:, 0], dp[:, 1])
        tp = np.vstack([start, r['truth'][:step]]); truth_l.set_data(tp[:, 0], tp[:, 1])

        c = r['samples'][:, step-1, :].mean(axis=0)
        spread = np.mean([haversine_km(c[0], c[1], p[0], p[1])
                           for p in r['samples'][:, step-1, :]])
        err = haversine_km(c[0], c[1], r['truth'][step-1, 0], r['truth'][step-1, 1])
        moved = haversine_km(start[0, 0], start[0, 1],
                              r['truth'][step-1, 0], r['truth'][step-1, 1])
        mins = r['target_dts'][step-1] / 60.0
        title.set_text(f'ping {step}/{F}  |  +{mins:.1f} min  |  moved {moved:.2f}km  |  '
                        f'err {err:.2f}km  |  spread {spread:.2f}km')
        return lines + [mean_l, med_l, truth_l, title]

    anim = animation.FuncAnimation(fig, update, frames=F, interval=interval_ms, blit=False)
    anim.save(out_path, writer='pillow', fps=1000/interval_ms)
    plt.close(fig)
    return out_path


def turn_angle_deg(context_xy):
    if len(context_xy) < 3:
        return 0.0
    v1, v2 = context_xy[-2] - context_xy[-3], context_xy[-1] - context_xy[-2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))


def analyze_failures(model, dataset, vel_scale, ctx_obj, n_samples=32, min_move_km=1.0):
    """
    Per-window metrics plus candidate explanatory factors. Includes
    horizon and mean context ping gap, which are specific to this branch
    -- irregular sampling adds "how stale is the input" as a possible
    failure driver that the fixed-interval branch cannot express.
    """
    port_arr = np.array(ctx_obj.ports)
    records = []
    for i in range(len(dataset)):
        r = sample_positions(model, dataset, i, vel_scale, n_samples)
        if r['moved_km'] < min_move_km:
            continue
        a = r['anchor']
        d_port = float(np.min(np.sqrt((port_arr[:, 0]-a[0])**2 +
                                       (port_arr[:, 1]-a[1])**2)) * 111.0)
        ctx, ctx_times, _, _ = dataset[i]
        ego = int(ctx[-1]['vessel'].ego_mask.nonzero()[0].item())
        gaps = np.diff(r['context_times'])
        records.append({
            'idx': i, 'moved_km': r['moved_km'], 'err_km': r['err_km'],
            'spread_km': r['spread_km'], 'beats_baseline': r['err_km'] < r['moved_km'],
            'horizon_min': r['horizon_min'],
            'mean_ctx_gap_sec': float(np.mean(gaps)) if len(gaps) else 0.0,
            'turn_deg': turn_angle_deg(r['context_xy']),
            'dist_to_port_km': d_port,
            'n_vessels_in_snapshot': int(ctx[-1]['vessel'].x.shape[0]),
            'sog_knots': float(ctx[-1]['vessel'].x[ego, 2].cpu()),
        })
    return records


def summarize_failures(records, top_frac=0.2, verbose=True):
    """Worst vs best windows, factor by factor. Ratio near 1.0 means the
    factor doesn't distinguish failures."""
    df = pd.DataFrame(records)
    if df.empty:
        print("no moving windows to analyze")
        return df
    df['err_ratio'] = df['err_km'] / df['moved_km']
    df = df.sort_values('err_ratio', ascending=False)
    n = max(10, int(len(df) * top_frac))
    worst, best = df.head(n), df.tail(n)

    if verbose:
        print(f"moving windows: {len(df)} | beats baseline: {100*df['beats_baseline'].mean():.1f}%")
        print(f"mean error {df['err_km'].mean():.2f}km vs mean movement {df['moved_km'].mean():.2f}km")
        print(f"median horizon {df['horizon_min'].median():.1f} min\n")
        print(f"{'factor':<26}{'worst 20%':>12}{'best 20%':>12}{'ratio':>9}")
        print('-' * 59)
        for col in ['turn_deg', 'horizon_min', 'mean_ctx_gap_sec', 'dist_to_port_km',
                     'n_vessels_in_snapshot', 'sog_knots', 'moved_km', 'spread_km']:
            w, b = worst[col].mean(), best[col].mean()
            print(f"{col:<26}{w:>12.2f}{b:>12.2f}{(w/b if abs(b) > 1e-9 else float('nan')):>9.2f}")
    return df