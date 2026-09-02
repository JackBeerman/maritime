"""
Ego-anchored irregular AIS ingestion.

The fixed-interval branch resamples every vessel onto shared 15-minute
bins and interpolates across gaps. That is convenient offline but bakes
in an assumption that is false in live operation: that observations
arrive on a uniform grid. Measured on one real DMA day, per-vessel
median ping intervals span 10s (p10) to 284s (p90) -- a 28x range -- so
"the next 4 pings" means anywhere from 40 seconds to ~19 minutes of
lookahead depending on the vessel.

This module keeps raw reported positions and represents time explicitly
instead:
  * A "snapshot" is anchored to a single ego-vessel ping. The ego row is
    that vessel's exact reported state at that instant.
  * Every other vessel is represented by its most recent report BEFORE
    that instant, carrying a staleness feature (how old the report is).
    That mirrors what a live system actually knows: you never have a
    neighbor's current position, only its last transmission.
  * No interpolation is performed anywhere. Targets are real observed
    pings, each paired with its true elapsed time.
"""
import numpy as np
import pandas as pd
import torch

from graph_data import build_hetero_snapshot


VESSEL_TYPE_VOCAB = {
    'cargo': 0, 'tanker': 1, 'fishing': 2, 'passenger': 3,
    'military': 4, 'other': 5, 'unknown': 6,
}
N_VESSEL_TYPES = len(VESSEL_TYPE_VOCAB)

DMA_COLUMN_MAP = {
    'MMSI': 'mmsi',
    'Timestamp': 'timestamp',
    'Longitude': 'lon',
    'Latitude': 'lat',
    'SOG': 'sog',
    'COG': 'cog',
    'Ship type': 'vessel_type',
}

# Feature layout (14 dims):
#   [lon, lat, sog, cog_sin, cog_cos, dt_norm, staleness_norm, *type_onehot(7)]
VESSEL_FEATURE_DIM = 7 + N_VESSEL_TYPES

DT_REFERENCE_SEC = 60.0      # log-scale reference for own-report spacing
STALENESS_REFERENCE_SEC = 60.0


def load_dma_ais_csv(path, bounds, start_time=None, end_time=None):
    """
    Loads a raw DMA daily AIS CSV. Same cleaning as the fixed-interval
    branch (header '#' prefix, Base Station rows, day-first timestamps),
    but no resampling downstream.
    """
    df = pd.read_csv(path)
    df.columns = [c.lstrip('#').strip() for c in df.columns]
    df = df[list(DMA_COLUMN_MAP.keys()) + ['Type of mobile']]
    df = df[df['Type of mobile'] != 'Base Station'].drop(columns=['Type of mobile'])
    df = df.rename(columns=DMA_COLUMN_MAP)

    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

    min_lon, min_lat, max_lon, max_lat = bounds
    mask = df['lon'].between(min_lon, max_lon) & df['lat'].between(min_lat, max_lat)
    if start_time is not None:
        mask &= df['timestamp'] >= start_time
    if end_time is not None:
        mask &= df['timestamp'] <= end_time
    df = df[mask].copy()

    df['vessel_type'] = df['vessel_type'].fillna('unknown').str.lower()
    df['vessel_type_id'] = df['vessel_type'].map(
        lambda t: VESSEL_TYPE_VOCAB.get(t, VESSEL_TYPE_VOCAB['unknown'])
    )
    # A ping missing position, speed or course can't be used as an ego
    # target or a neighbor state; drop rather than impute.
    df = df.dropna(subset=['lon', 'lat', 'sog', 'cog'])
    return df.sort_values('timestamp').reset_index(drop=True)


def one_hot_vessel_type(type_ids, n_types=N_VESSEL_TYPES):
    type_ids = np.asarray(type_ids).astype(int)
    onehot = np.zeros((len(type_ids), n_types))
    onehot[np.arange(len(type_ids)), np.clip(type_ids, 0, n_types - 1)] = 1.0
    return onehot


def thin_pings(times_sec, min_gap_sec):
    """
    Indices of pings kept when enforcing a minimum spacing.

    Some vessels report every 2-3 seconds. Consecutive reports that close
    together carry almost no new information but would dominate the
    window count and make "next N pings" a sub-minute horizon, so it is
    usually worth thinning. Returns all indices when min_gap_sec <= 0.
    """
    if min_gap_sec is None or min_gap_sec <= 0:
        return np.arange(len(times_sec))
    keep, last = [], None
    for i, t in enumerate(times_sec):
        if last is None or (t - last) >= min_gap_sec:
            keep.append(i)
            last = t
    return np.asarray(keep, dtype=int)


def select_ego_vessels_stratified(ais_df, min_pings=60, sog_threshold_knots=2.0,
                                    max_median_ping_interval_sec=1.0,
                                    n_underway=100, n_stationary=100, rng_seed=0):
    """
    Balanced underway/stationary ego-vessel selection, same reasoning as
    the fixed-interval branch: a deployed predictor must handle "stays
    put" as a valid outcome, so stationary vessels are sampled rather
    than excluded. Quality filters (enough pings, not an anomalous
    sub-second transmitter) apply to both regimes.
    """
    stats = ais_df.groupby('mmsi').agg(
        n_pings=('timestamp', 'count'),
        median_sog=('sog', 'median'),
        t_min=('timestamp', 'min'),
        t_max=('timestamp', 'max'),
    )
    stats['span_sec'] = (stats['t_max'] - stats['t_min']).dt.total_seconds()
    stats['median_ping_interval_sec'] = stats['span_sec'] / stats['n_pings'].clip(lower=1)

    ok = stats[(stats['n_pings'] >= min_pings) &
               (stats['median_ping_interval_sec'] >= max_median_ping_interval_sec)]
    underway = ok[ok['median_sog'] >= sog_threshold_knots]
    stationary = ok[ok['median_sog'] < sog_threshold_knots]

    up = underway.sample(n=min(n_underway, len(underway)), random_state=rng_seed) if len(underway) else underway
    sp = stationary.sample(n=min(n_stationary, len(stationary)), random_state=rng_seed) if len(stationary) else stationary
    picked = pd.concat([up, sp])
    picked['regime'] = ['underway'] * len(up) + ['stationary'] * len(sp)
    return picked


def build_ego_anchored_snapshots(ais_df, ego_mmsi, mesh_node_features, mesh_edge_index,
                                   staleness_cutoff_sec=1800, neighbor_radius_deg=0.5,
                                   max_neighbors=150, min_ping_gap_sec=60,
                                   mesh_x_tensor=None, mesh_edge_tensor=None, mesh_tree=None):
    """
    Builds one HeteroData snapshot per (thinned) ego ping.

    Returns (snapshots, ego_idx_per_step, ping_times_sec, positions), where
    ping_times_sec are seconds since the ego vessel's first kept ping and
    positions are its exact reported lon/lat -- both used by
    IrregularVesselDataset to construct time-aware windows.

    Neighbours are resolved by a single ordered pass over nearby traffic,
    carrying a running "last known state per vessel" table. This is O(n)
    in the number of nearby pings, versus a per-neighbour asof-join which
    would be intractable at ~29M pings/day.
    """
    ego = ais_df[ais_df['mmsi'] == ego_mmsi].sort_values('timestamp')
    if len(ego) < 2:
        return [], [], np.array([]), np.zeros((0, 2))

    t_origin = ego['timestamp'].iloc[0]
    ego_times_all = (ego['timestamp'] - t_origin).dt.total_seconds().to_numpy()
    keep = thin_pings(ego_times_all, min_ping_gap_sec)
    ego = ego.iloc[keep]
    ego_times = ego_times_all[keep]
    if len(ego) < 2:
        return [], [], np.array([]), np.zeros((0, 2))

    ego_lon = ego['lon'].to_numpy()
    ego_lat = ego['lat'].to_numpy()
    ego_sog = ego['sog'].to_numpy()
    ego_cog = ego['cog'].to_numpy()
    ego_type = ego['vessel_type_id'].to_numpy()

    # Spatial + temporal prefilter: only traffic that could plausibly be a
    # neighbour anywhere along this vessel's track.
    lo_lon, hi_lon = ego_lon.min() - neighbor_radius_deg, ego_lon.max() + neighbor_radius_deg
    lo_lat, hi_lat = ego_lat.min() - neighbor_radius_deg, ego_lat.max() + neighbor_radius_deg
    t_lo = t_origin + pd.Timedelta(seconds=float(ego_times[0] - staleness_cutoff_sec))
    t_hi = t_origin + pd.Timedelta(seconds=float(ego_times[-1]))
    nearby = ais_df[
        (ais_df['mmsi'] != ego_mmsi) &
        ais_df['lon'].between(lo_lon, hi_lon) &
        ais_df['lat'].between(lo_lat, hi_lat) &
        ais_df['timestamp'].between(t_lo, t_hi)
    ].sort_values('timestamp')

    nb_t = (nearby['timestamp'] - t_origin).dt.total_seconds().to_numpy()
    nb_mmsi = nearby['mmsi'].to_numpy()
    nb_vals = nearby[['lon', 'lat', 'sog', 'cog', 'vessel_type_id']].to_numpy()

    state_t = {}
    state_v = {}
    ptr = 0

    snapshots, ego_idx_per_step = [], []
    prev_ego_t = None

    for i, et in enumerate(ego_times):
        while ptr < len(nb_t) and nb_t[ptr] <= et:
            m = nb_mmsi[ptr]
            state_t[m] = nb_t[ptr]
            state_v[m] = nb_vals[ptr]
            ptr += 1

        own_dt = 0.0 if prev_ego_t is None else (et - prev_ego_t)
        prev_ego_t = et

        rows = [np.array([ego_lon[i], ego_lat[i], ego_sog[i], ego_cog[i],
                           ego_type[i], own_dt, 0.0])]

        if state_t:
            m_arr = np.fromiter(state_t.keys(), dtype=np.int64, count=len(state_t))
            t_arr = np.fromiter((state_t[m] for m in m_arr), dtype=np.float64, count=len(m_arr))
            stale = et - t_arr
            fresh = stale <= staleness_cutoff_sec
            m_arr, t_arr, stale = m_arr[fresh], t_arr[fresh], stale[fresh]

            if len(m_arr):
                vals = np.stack([state_v[m] for m in m_arr])
                d = np.hypot(vals[:, 0] - ego_lon[i], vals[:, 1] - ego_lat[i])
                within = d <= neighbor_radius_deg
                m_arr, stale, vals, d = m_arr[within], stale[within], vals[within], d[within]

                if len(m_arr) > max_neighbors:
                    sel = np.argsort(d)[:max_neighbors]
                    stale, vals = stale[sel], vals[sel]

                for j in range(len(vals)):
                    # neighbours have no meaningful own-report spacing from
                    # the ego vessel's perspective; staleness carries the
                    # timing information instead
                    rows.append(np.array([vals[j, 0], vals[j, 1], vals[j, 2],
                                           vals[j, 3], vals[j, 4], 0.0, stale[j]]))

        raw = np.stack(rows)
        cog_rad = np.radians(raw[:, 3])
        feats = np.concatenate([
            raw[:, 0:3],
            np.sin(cog_rad)[:, None],
            np.cos(cog_rad)[:, None],
            np.log1p(np.clip(raw[:, 5], 0, None))[:, None] / np.log1p(DT_REFERENCE_SEC),
            np.log1p(np.clip(raw[:, 6], 0, None))[:, None] / np.log1p(STALENESS_REFERENCE_SEC),
            one_hot_vessel_type(raw[:, 4]),
        ], axis=1)

        snapshots.append(build_hetero_snapshot(
            mesh_node_features, mesh_edge_index, feats, ego_idx=0,
            mesh_x_tensor=mesh_x_tensor, mesh_edge_tensor=mesh_edge_tensor,
            mesh_tree=mesh_tree,
        ))
        ego_idx_per_step.append(0)

    positions = np.stack([ego_lon, ego_lat], axis=1)
    return snapshots, ego_idx_per_step, ego_times, positions
