"""
AIS parsing, resampling, vessel selection, and shared world-snapshot
construction.
"""
import pandas as pd
import numpy as np

from graph_data import build_world_snapshot


VESSEL_TYPE_VOCAB = {
    'cargo': 0, 'tanker': 1, 'fishing': 2, 'passenger': 3,
    'military': 4, 'other': 5, 'unknown': 6,
}
N_VESSEL_TYPES = len(VESSEL_TYPE_VOCAB)
VESSEL_FEATURE_DIM = 5 + N_VESSEL_TYPES   # lon, lat, sog, cog_sin, cog_cos, onehot

DMA_COLUMN_MAP = {
    'MMSI': 'mmsi', 'Timestamp': 'timestamp', 'Longitude': 'lon',
    'Latitude': 'lat', 'SOG': 'sog', 'COG': 'cog', 'Ship type': 'vessel_type',
}


def load_dma_ais_csv(path, bounds, start_time, end_time):
    """
    Loads a raw DMA daily AIS CSV. Handles the '#'-prefixed header some
    export batches carry, drops Base Station rows (fixed shore
    transmitters, not vessels), and parses day-first timestamps.
    """
    df = pd.read_csv(path)
    df.columns = [c.lstrip('#').strip() for c in df.columns]
    df = df[list(DMA_COLUMN_MAP.keys()) + ['Type of mobile']]
    df = df[df['Type of mobile'] != 'Base Station'].drop(columns=['Type of mobile'])
    df = df.rename(columns=DMA_COLUMN_MAP)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

    min_lon, min_lat, max_lon, max_lat = bounds
    df = df[df['lon'].between(min_lon, max_lon) &
            df['lat'].between(min_lat, max_lat) &
            df['timestamp'].between(start_time, end_time)].copy()

    df['vessel_type'] = df['vessel_type'].fillna('unknown').str.lower()
    df['vessel_type_id'] = df['vessel_type'].map(
        lambda t: VESSEL_TYPE_VOCAB.get(t, VESSEL_TYPE_VOCAB['unknown']))
    return df.sort_values('timestamp')


def resample_to_snapshots(df, interval_minutes=15):
    """
    Bins pings into fixed intervals, interpolating across single missed
    intervals. A timestep is dropped if ANY of lon/lat/sog/cog is still
    missing -- real AIS legitimately has blank SOG/COG on some pings, and
    checking only position let NaNs reach the loss.
    """
    df = df.copy()
    df['bin'] = df['timestamp'].dt.floor(f'{interval_minutes}min')
    snapshots = {}
    for mmsi, g in df.groupby('mmsi'):
        g = g.set_index('bin')[['lon', 'lat', 'sog', 'cog', 'vessel_type_id']]
        g = g[~g.index.duplicated(keep='last')]
        full = pd.date_range(g.index.min(), g.index.max(), freq=f'{interval_minutes}min')
        g = g.reindex(full)
        g[['lon', 'lat', 'sog', 'cog']] = g[['lon', 'lat', 'sog', 'cog']].interpolate(limit=1)
        g['vessel_type_id'] = g['vessel_type_id'].ffill().bfill()
        for ts, row in g.iterrows():
            if row[['lon', 'lat', 'sog', 'cog']].isna().any():
                continue
            snapshots.setdefault(ts, {})[mmsi] = row.values.astype(float)
    return snapshots


def one_hot_vessel_type(type_ids, n_types=N_VESSEL_TYPES):
    ids = np.asarray(type_ids).astype(int)
    oh = np.zeros((len(ids), n_types))
    oh[np.arange(len(ids)), np.clip(ids, 0, n_types - 1)] = 1.0
    return oh


def select_ego_vessels_stratified(ais_df, min_pings=40, max_median_ping_interval_sec=1.0,
                                    sog_threshold_knots=2.0, n_underway=20,
                                    n_stationary=20, rng_seed=0):
    """
    Balanced underway/stationary ego selection. Stationary vessels are
    sampled rather than excluded: a deployed predictor must handle "stays
    put" as a valid outcome, and the original failure mode was an
    unbalanced distribution, not the presence of stationary vessels.
    """
    stats = ais_df.groupby('mmsi').agg(
        n_pings=('timestamp', 'count'), median_sog=('sog', 'median'),
        t_min=('timestamp', 'min'), t_max=('timestamp', 'max'))
    stats['span_sec'] = (stats['t_max'] - stats['t_min']).dt.total_seconds()
    stats['median_ping_interval_sec'] = stats['span_sec'] / stats['n_pings'].clip(lower=1)

    ok = stats[(stats['n_pings'] >= min_pings) &
               (stats['median_ping_interval_sec'] >= max_median_ping_interval_sec)]
    under = ok[ok['median_sog'] >= sog_threshold_knots]
    stat = ok[ok['median_sog'] < sog_threshold_knots]
    up = under.sample(n=min(n_underway, len(under)), random_state=rng_seed) if len(under) else under
    sp = stat.sample(n=min(n_stationary, len(stat)), random_state=rng_seed) if len(stat) else stat
    picked = pd.concat([up, sp])
    picked['regime'] = ['underway'] * len(up) + ['stationary'] * len(sp)
    return picked


def build_world_snapshots(ais_snapshots, mesh_node_features, mesh_edge_index,
                            mesh_x_tensor=None, mesh_edge_tensor=None, mesh_tree=None,
                            progress_every=None):
    """
    Builds ONE graph per timestamp, shared by every ego vessel.

    Returns (world_snapshots, timestamps, mmsi_row_maps) where
    mmsi_row_maps[i] maps mmsi -> row index within world_snapshots[i].

    This replaces the previous per-(ego, timestamp) construction. The
    world at a timestamp does not depend on which vessel you're
    forecasting, so building it once per timestamp instead of once per
    (vessel, timestamp) removes a 150-600x duplication that made
    multi-day, multi-hundred-vessel runs exhaust GPU memory.

    Vessel features: [lon, lat, sog, cog_sin, cog_cos, *type_onehot(7)].
    COG is sin/cos encoded (a raw 0-359 scalar would imply 359 deg and
    1 deg are nearly opposite); type is one-hot (a raw id would imply a
    false ordering between categories).
    """
    timestamps = sorted(ais_snapshots.keys())
    world, row_maps = [], []
    for i, ts in enumerate(timestamps):
        vessels = ais_snapshots[ts]
        mmsi_list = list(vessels.keys())
        raw = np.stack([vessels[m] for m in mmsi_list])       # lon,lat,sog,cog,type

        cog_rad = np.radians(raw[:, 3])
        feats = np.concatenate([
            raw[:, 0:3],
            np.sin(cog_rad)[:, None], np.cos(cog_rad)[:, None],
            one_hot_vessel_type(raw[:, 4]),
        ], axis=1)

        world.append(build_world_snapshot(
            mesh_node_features, mesh_edge_index, feats,
            mesh_x_tensor=mesh_x_tensor, mesh_edge_tensor=mesh_edge_tensor,
            mesh_tree=mesh_tree))
        row_maps.append({m: r for r, m in enumerate(mmsi_list)})

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    built {i+1}/{len(timestamps)} world snapshots", flush=True)

    return world, timestamps, row_maps


def vessel_presence(row_maps, ego_mmsi):
    """
    (world_idx, ego_row) for each timestamp where this vessel appears --
    the index pair a per-vessel dataset needs to address shared snapshots.
    """
    widx, erow = [], []
    for i, rm in enumerate(row_maps):
        r = rm.get(ego_mmsi)
        if r is not None:
            widx.append(i)
            erow.append(r)
    return widx, erow