"""
Reusable evaluation and visualization helpers.

Import in a notebook instead of re-pasting cells:

    from viz import (setup, load_checkpoint_model, evaluate,
                     plot_prediction, plot_grid, animate_prediction,
                     analyze_failures, summarize_failures,
                     per_step_errors, plot_mesh_with_reference_map)

Everything that reads a window uses the current dataset API --
`(ctx, ego_rows, target)`, with the ego row index carried alongside the
graph rather than baked in as an `ego_mask`, since world snapshots are
shared across ego vessels.

All model output is treated as a NORMALIZED RESIDUAL: a real position is
`sample * norm_scale + anchor`. Getting that conversion wrong silently
produces nonsense, so it happens in exactly one place here
(`sample_positions`).
"""
import math
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
from ais_ingest import (
    load_dma_ais_csv, resample_to_snapshots, select_ego_vessels_stratified,
    build_world_snapshots, vessel_presence,
)
from graph_data import VesselSequenceDataset, share_mesh_on_device
from model import GCVTP


# --------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------
# setup
# --------------------------------------------------------------------

class Context:
    """Everything a notebook session needs, in one object."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return (f"Context(mesh_nodes={self.mesh_node_features.shape[0]}, "
                f"timestamps={len(self.world)}, "
                f"val_windows={len(self.combined_val) if self.combined_val else 0})")


def setup(ais_path="/scratch/jtb3sud/maritime/ais/aisdk-2026-08-25.csv",
          land_shp="data/ne_10m_land/ne_10m_land.shp",
          ports_csv="data/ports_denmark.csv",
          seq_len=12, future_len=4,
          n_underway=300, n_stationary=300,
          val_fraction=0.2, val_seed=42,
          held_out_only=True, device=None, verbose=True):
    """
    Builds mesh, loads AIS, and assembles a dataset for inspection.

    held_out_only=True reproduces the training run's vessel split and
    keeps ONLY validation vessels. That matters: a dataset built with
    different selection parameters silently mixes in vessels the model
    trained on, which flatters the numbers. Pass the same n_underway /
    n_stationary / val_seed the training run used.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    bounds = DMA_BOUNDS

    land_polygons = load_land_polygons(bounds, natural_earth_path=land_shp)
    ports = load_ports(ports_csv, bounds)
    pts, land_flags = sample_domain_points(bounds, land_polygons, ports)
    mesh_node_features, mesh_edge_index, _ = build_mesh(pts, land_flags, ports)
    mesh_x = torch.as_tensor(mesh_node_features, dtype=torch.float)
    mesh_e = torch.as_tensor(mesh_edge_index, dtype=torch.long)
    mesh_tree = cKDTree(mesh_node_features[:, :2])
    if verbose:
        print(f"mesh: {mesh_node_features.shape[0]} nodes, {mesh_edge_index.shape[1]} edges")

    ais_df = load_dma_ais_csv(ais_path, bounds, pd.Timestamp.min, pd.Timestamp.max)
    ais_snapshots = resample_to_snapshots(ais_df, interval_minutes=15)
    if verbose:
        print(f"AIS: {len(ais_df)} records, {len(ais_snapshots)} timestamps")

    good = select_ego_vessels_stratified(
        ais_df, min_pings=40, sog_threshold_knots=2.0,
        n_underway=n_underway, n_stationary=n_stationary)

    rng = np.random.default_rng(val_seed)
    train_ids, val_ids = [], []
    for _, g in good.groupby('regime'):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx)))) if len(idx) > 1 else 0
        val_ids.extend(idx[:n_val]); train_ids.extend(idx[n_val:])
    if verbose:
        print(f"vessels: {len(train_ids)} train, {len(val_ids)} held-out")

    world, timestamps, row_maps = build_world_snapshots(
        ais_snapshots, mesh_node_features, mesh_edge_index, mesh_x, mesh_e, mesh_tree)
    world = share_mesh_on_device(world, device)

    use_ids = val_ids if held_out_only else list(train_ids) + list(val_ids)
    datasets = []
    for mmsi in use_ids:
        widx, erow = vessel_presence(row_maps, mmsi)
        if len(widx) < seq_len + future_len:
            continue
        ds = VesselSequenceDataset(world, widx, erow, seq_len, future_len)
        if len(ds) > 0:
            datasets.append(ds)
    combined = ConcatDataset(datasets) if datasets else None
    if verbose:
        label = "held-out" if held_out_only else "all"
        print(f"{label} windows: {len(combined) if combined else 0} across {len(datasets)} vessels")

    return Context(device=device, bounds=bounds, land_polygons=land_polygons, ports=ports,
                   mesh_node_features=mesh_node_features, mesh_edge_index=mesh_edge_index,
                   mesh_tree=mesh_tree, ais_df=ais_df, ais_snapshots=ais_snapshots,
                   world=world, timestamps=timestamps, row_maps=row_maps,
                   train_ids=train_ids, val_ids=val_ids,
                   combined_val=combined, seq_len=seq_len, future_len=future_len)


def load_checkpoint_model(checkpoint_path, device=None):
    """
    Loads a checkpoint, inferring architecture from the saved state dict
    so you don't have to remember which --hidden / --n-layers a run used.
    Returns (model in eval mode, norm_scale, metadata).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt['model_state_dict']

    vessel_in = sd['gnn.vessel_proj.weight'].shape[1]
    hidden = sd['gnn.vessel_proj.weight'].shape[0]
    mesh_in = sd['gnn.mesh_proj.weight'].shape[1]
    n_layers = len([k for k in sd if k.startswith('temporal.layers.')
                     and k.endswith('.attn.qkv.weight')])
    future_len = sd['head.net.4.weight'].shape[0] // 2
    max_cache_len = sd['temporal.pos_embed'].shape[1]

    model = GCVTP(mesh_in=mesh_in, vessel_in=vessel_in, hidden=hidden,
                  future_len=future_len, n_layers=n_layers,
                  max_cache_len=max_cache_len).to(device)
    model.load_state_dict(sd)
    model.eval()

    meta = {'epoch': ckpt.get('epoch'), 'avg_loss': ckpt.get('avg_loss'),
            'hidden': hidden, 'n_layers': n_layers, 'vessel_in': vessel_in,
            'future_len': future_len}
    return model, ckpt['norm_scale'], meta


# --------------------------------------------------------------------
# core inference
# --------------------------------------------------------------------

@torch.no_grad()
def sample_positions(model, dataset, idx, norm_scale, n_samples=32):
    """
    The single place normalized residuals become real coordinates.
    Returns a dict with context path, samples, truth, and per-window
    metrics -- all in absolute lon/lat.
    """
    ctx, ego_rows, target = dataset[idx]
    anchor = ctx[-1]['vessel'].x[ego_rows[-1], :2]
    samples = model(ctx, ego_rows, n_samples=n_samples, training=False)
    samples_real = samples * norm_scale + anchor

    context_xy = np.array([d['vessel'].x[ego_rows[t], :2].cpu().numpy()
                            for t, d in enumerate(ctx)])
    samples_np = samples_real[0].cpu().numpy()
    truth_np = target.cpu().numpy()
    a = anchor.cpu().numpy()

    center = samples_np[:, -1, :].mean(axis=0)
    return {
        'context_xy': context_xy, 'samples': samples_np, 'truth': truth_np,
        'anchor': a, 'mean_path': samples_np.mean(axis=0),
        'medoid_path': samples_np[medoid_index(samples_np)],
        'moved_km': haversine_km(a[0], a[1], truth_np[-1, 0], truth_np[-1, 1]),
        'err_km': haversine_km(center[0], center[1], truth_np[-1, 0], truth_np[-1, 1]),
        'spread_km': float(np.mean([haversine_km(center[0], center[1], p[0], p[1])
                                      for p in samples_np[:, -1, :]])),
    }


def constant_velocity_prediction(context_xy, n_steps):
    """
    Dead-reckoning baseline: take the last observed step as a velocity and
    extrapolate linearly.

    This is a much stronger and more honest baseline than persistence
    ("assume no movement"), which is what `baseline_mean_error_km`
    reports. Ships mostly travel in straight lines, so constant velocity
    is genuinely hard to beat -- and since failure analysis showed
    turning dominates the model's errors, the model is likely doing
    something close to this already. Beating persistence by 89% means
    little if constant velocity also beats it by 89%.

    Assumes evenly spaced context steps, which holds on the
    fixed-interval branch.
    """
    if len(context_xy) < 2:
        return np.repeat(context_xy[-1:], n_steps, axis=0)
    v = context_xy[-1] - context_xy[-2]
    return np.stack([context_xy[-1] + v * (k + 1) for k in range(n_steps)])


def medoid_index(samples_np):
    """
    Sample with minimum total distance to all others. Unlike the mean,
    the medoid is an ACTUAL sampled path, so it is always physically
    plausible -- the mean can land where no sample went if the
    distribution is genuinely bimodal.
    """
    S = samples_np.shape[0]
    flat = samples_np.reshape(S, -1)
    d = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return int(d.sum(axis=1).argmin())


def evaluate(model, dataset, norm_scale, n_samples=32, min_move_km=1.0, verbose=True):
    """
    Headline metrics: uncertainty calibration and accuracy vs a trivial
    'assume no movement' baseline, restricted to genuinely moving windows.
    """
    disps, spreads, errs, base, cv_errs = [], [], [], [], []
    for i in range(len(dataset)):
        r = sample_positions(model, dataset, i, norm_scale, n_samples)
        disps.append(r['moved_km']); spreads.append(r['spread_km'])
        if r['moved_km'] > min_move_km:
            errs.append(r['err_km']); base.append(r['moved_km'])
            cv = constant_velocity_prediction(r['context_xy'], r['truth'].shape[0])
            cv_errs.append(haversine_km(cv[-1, 0], cv[-1, 1],
                                         r['truth'][-1, 0], r['truth'][-1, 1]))

    out = {'n_windows': len(dataset),
           'spread_correlation': float(np.corrcoef(disps, spreads)[0, 1]) if len(disps) > 1 else float('nan')}
    if errs:
        errs, base, cv_errs = np.array(errs), np.array(base), np.array(cv_errs)
        out.update({'n_moving': len(errs), 'model_mean_error_km': float(errs.mean()),
                    # persistence: "vessel stays put" -- error equals actual displacement
                    'persistence_error_km': float(base.mean()),
                    'beats_persistence_pct': float(100 * (errs < base).mean()),
                    # constant velocity (dead reckoning): the meaningful bar
                    'const_velocity_error_km': float(cv_errs.mean()),
                    'beats_const_velocity_pct': float(100 * (errs < cv_errs).mean())})
    if verbose:
        for k, v in out.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return out


def per_step_errors(model, dataset, norm_scale, indices=None, n_samples=32,
                     interval_minutes=15, verbose=True):
    """Mean error at each forecast step -- shows how error grows with horizon."""
    indices = indices if indices is not None else range(len(dataset))
    steps = None
    with torch.no_grad():
        for i in indices:
            ctx, ego_rows, target = dataset[i]
            anchor = ctx[-1]['vessel'].x[ego_rows[-1], :2]
            s = model(ctx, ego_rows, n_samples=n_samples, training=False) * norm_scale + anchor
            pred = s[0].mean(dim=0).cpu().numpy()
            truth = target.cpu().numpy()
            if steps is None:
                steps = [[] for _ in range(truth.shape[0])]
            for k in range(truth.shape[0]):
                steps[k].append(haversine_km(pred[k, 0], pred[k, 1], truth[k, 0], truth[k, 1]))
    means = [float(np.mean(s)) for s in steps]
    if verbose:
        for k, m in enumerate(means):
            print(f"  +{(k+1)*interval_minutes:>3} min: {m:.2f} km")
    return means


# --------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------

def _basemap(ax, land_polygons, ports):
    ax.set_facecolor('#cfe8f3')
    for poly in land_polygons:
        xs, ys = poly.exterior.xy
        ax.add_patch(MplPolygon(list(zip(xs, ys)), facecolor='#d8c39a',
                                 edgecolor='#8a7350', linewidth=0.5, zorder=1))
    ax.scatter([p[0] for p in ports], [p[1] for p in ports], s=30, c='black',
               marker='^', zorder=2, label='port')


def _fit_view(ax, all_xy, min_window_deg=0.05):
    """
    Enforce a minimum extent. Without this a near-stationary window
    auto-scales to a tiny box, making ordinary small-scale noise look
    like wild disagreement.
    """
    lon_r = max(all_xy[:, 0].max() - all_xy[:, 0].min(), min_window_deg)
    lat_r = max(all_xy[:, 1].max() - all_xy[:, 1].min(), min_window_deg)
    cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(cx - lon_r/2 - 0.01, cx + lon_r/2 + 0.01)
    ax.set_ylim(cy - lat_r/2 - 0.01, cy + lat_r/2 + 0.01)
    ax.set_aspect('equal')


def plot_prediction(model, dataset, idx, norm_scale, ctx_obj, n_samples=32,
                     ax=None, show_summaries=True, figsize=(9, 8)):
    """One window: context (black), samples (blue), truth (red)."""
    r = sample_positions(model, dataset, idx, norm_scale, n_samples)
    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    _basemap(ax, ctx_obj.land_polygons, ctx_obj.ports)

    ax.plot(r['context_xy'][:, 0], r['context_xy'][:, 1], c='black', marker='o',
            markersize=4, linewidth=1.5, label='observed context', zorder=5)
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
            label='actual future', zorder=7)

    _fit_view(ax, np.vstack([r['context_xy'], r['samples'].reshape(-1, 2), r['truth']]))
    ax.set_title(f"moved {r['moved_km']:.2f}km | err {r['err_km']:.2f}km | "
                  f"spread {r['spread_km']:.2f}km", fontsize=10)
    return fig, r


def classify_windows(model, dataset, norm_scale, move_threshold_km=3.0, n_samples=8):
    """Split window indices into moving / near-stationary."""
    moving, stationary = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            ctx, ego_rows, target = dataset[i]
            a = ctx[-1]['vessel'].x[ego_rows[-1], :2].cpu().numpy()
            t = target[-1].cpu().numpy()
            (moving if haversine_km(a[0], a[1], t[0], t[1]) > move_threshold_km
             else stationary).append(i)
    return moving, stationary


def plot_grid(model, dataset, norm_scale, ctx_obj, moving=None, stationary=None,
               n_each=3, seed=0, n_samples=32, figsize=(16, 10)):
    """
    2x3 grid: moving vessels on top, near-stationary below, sampled at
    RANDOM. Taking the first few windows instead tends to draw them all
    from one vessel in one place, which is not representative.
    """
    if moving is None or stationary is None:
        moving, stationary = classify_windows(model, dataset, norm_scale)
    rng = np.random.default_rng(seed)
    pick_m = rng.choice(moving, size=min(n_each, len(moving)), replace=False) if moving else []
    pick_s = rng.choice(stationary, size=min(n_each, len(stationary)), replace=False) if stationary else []

    fig, axes = plt.subplots(2, n_each, figsize=figsize)
    for ax, i in zip(axes.flat, list(pick_m) + list(pick_s)):
        plot_prediction(model, dataset, int(i), norm_scale, ctx_obj, n_samples=n_samples, ax=ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Top: moving vessels | Bottom: near-stationary\n'
                  'black = observed context | blue = samples | red = actual', fontsize=12)
    fig.tight_layout()
    return fig


def animate_prediction(model, dataset, idx, norm_scale, ctx_obj, n_samples=32,
                        interval_ms=700, out_path='prediction.gif', figsize=(10, 8),
                        interval_minutes=15):
    """Reveals the forecast one step at a time; returns the gif path."""
    r = sample_positions(model, dataset, idx, norm_scale, n_samples)
    F = r['truth'].shape[0]
    start = r['context_xy'][-1:]

    fig, ax = plt.subplots(figsize=figsize)
    _basemap(ax, ctx_obj.land_polygons, ctx_obj.ports)
    ax.plot(r['context_xy'][:, 0], r['context_xy'][:, 1], c='black', marker='o',
            markersize=5, linewidth=2, label='observed context (input)', zorder=5)

    lines = [ax.plot([], [], c='steelblue', alpha=0.22, lw=1, zorder=2,
                      label='predicted samples' if s == 0 else None)[0]
              for s in range(r['samples'].shape[0])]
    mean_l, = ax.plot([], [], c='darkblue', ls='--', lw=2, label='sample mean', zorder=6)
    med_l, = ax.plot([], [], c='purple', ls=':', lw=2, label='medoid', zorder=6)
    truth_l, = ax.plot([], [], c='red', marker='*', markersize=10, lw=2.5,
                        label='actual future', zorder=7)

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
        title.set_text(f'+{step*interval_minutes} min  |  moved {moved:.2f}km  |  '
                        f'err {err:.2f}km  |  spread {spread:.2f}km')
        return lines + [mean_l, med_l, truth_l, title]

    anim = animation.FuncAnimation(fig, update, frames=F, interval=interval_ms, blit=False)
    anim.save(out_path, writer='pillow', fps=1000/interval_ms)
    plt.close(fig)
    return out_path


def plot_mesh_with_reference_map(ctx_obj, max_edges=15000, figsize=(16, 7)):
    """Side-by-side: filled coastline vs the mesh built from it."""
    nf, ei = ctx_obj.mesh_node_features, ctx_obj.mesh_edge_index
    lon, lat = nf[:, 0], nf[:, 1]
    is_land, is_port = nf[:, 2].astype(bool), nf[:, 3].astype(bool)
    min_lon, min_lat, max_lon, max_lat = ctx_obj.bounds

    fig, (ax_map, ax_mesh) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    _basemap(ax_map, ctx_obj.land_polygons, ctx_obj.ports)
    ax_map.set_title('Reference map (coastline + ports)')

    n_edges = ei.shape[1]
    if n_edges > max_edges:
        sel = np.random.default_rng(0).choice(n_edges, size=max_edges, replace=False)
        e = ei[:, sel]
    else:
        e = ei
    ax_mesh.plot(np.stack([lon[e[0]], lon[e[1]]]), np.stack([lat[e[0]], lat[e[1]]]),
                  color='steelblue', linewidth=0.15, alpha=0.35, zorder=1)
    water = ~is_land & ~is_port
    ax_mesh.scatter(lon[water], lat[water], s=3, c='deepskyblue', label='water', zorder=2)
    ax_mesh.scatter(lon[is_land], lat[is_land], s=4, c='saddlebrown', label='land', zorder=3)
    ax_mesh.scatter(lon[is_port], lat[is_port], s=18, c='red', marker='^', label='port', zorder=4)
    ax_mesh.set_title(f'Mesh: {nf.shape[0]} nodes, {n_edges} edges')
    ax_mesh.legend(loc='upper right', markerscale=2)

    for ax in (ax_map, ax_mesh):
        ax.set_xlim(min_lon, max_lon); ax.set_ylim(min_lat, max_lat)
        ax.set_aspect('equal'); ax.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------
# failure analysis
# --------------------------------------------------------------------

def turn_angle_deg(context_xy):
    """
    Turn across the last two context steps, from positions rather than
    the COG field (COG is noisy/stale for slow vessels).
    """
    if len(context_xy) < 3:
        return 0.0
    v1, v2 = context_xy[-2] - context_xy[-3], context_xy[-1] - context_xy[-2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))


def speed_variability(context_xy):
    """Std of per-step displacement: high means accelerating/decelerating."""
    if len(context_xy) < 3:
        return 0.0
    steps = [np.linalg.norm(context_xy[i+1] - context_xy[i]) for i in range(len(context_xy)-1)]
    return float(np.std(steps))


def analyze_failures(model, dataset, norm_scale, ctx_obj, n_samples=32, min_move_km=1.0):
    """Per-window metrics plus candidate explanatory factors."""
    port_arr = np.array(ctx_obj.ports)
    records = []
    for i in range(len(dataset)):
        r = sample_positions(model, dataset, i, norm_scale, n_samples)
        if r['moved_km'] < min_move_km:
            continue
        a = r['anchor']
        d_port = float(np.min(np.sqrt((port_arr[:, 0]-a[0])**2 + (port_arr[:, 1]-a[1])**2)) * 111.0)
        ctx, ego_rows, _ = dataset[i]
        records.append({
            'idx': i, 'moved_km': r['moved_km'], 'err_km': r['err_km'],
            'spread_km': r['spread_km'], 'beats_baseline': r['err_km'] < r['moved_km'],
            'turn_deg': turn_angle_deg(r['context_xy']),
            'speed_var': speed_variability(r['context_xy']),
            'dist_to_port_km': d_port,
            'n_vessels_in_snapshot': int(ctx[-1]['vessel'].x.shape[0]),
            'sog_knots': float(ctx[-1]['vessel'].x[ego_rows[-1], 2].cpu()),
        })
    return records


def summarize_failures(records, top_frac=0.2, verbose=True):
    """
    Compare worst-predicted windows against best, factor by factor.
    A ratio near 1.0 means the factor doesn't distinguish failures;
    well above 1.0 means failures have much more of it.
    """
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
        print(f"mean error {df['err_km'].mean():.2f}km vs mean movement {df['moved_km'].mean():.2f}km\n")
        print(f"{'factor':<26}{'worst 20%':>12}{'best 20%':>12}{'ratio':>9}")
        print('-' * 59)
        for col in ['turn_deg', 'speed_var', 'dist_to_port_km',
                     'n_vessels_in_snapshot', 'sog_knots', 'moved_km', 'spread_km']:
            w, b = worst[col].mean(), best[col].mean()
            print(f"{col:<26}{w:>12.2f}{b:>12.2f}{(w/b if abs(b) > 1e-9 else float('nan')):>9.2f}")
    return df