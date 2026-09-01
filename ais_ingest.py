"""
Loads raw AIS records, resamples to fixed-interval snapshots, and builds
the HeteroData sequence needed by VesselSequenceDataset.
"""
import pandas as pd
import numpy as np
from mesh import sample_domain_points, build_mesh
from graph_data import build_hetero_snapshot


VESSEL_TYPE_VOCAB = {
    'cargo': 0, 'tanker': 1, 'fishing': 2, 'passenger': 3,
    'military': 4, 'other': 5, 'unknown': 6,
}


def load_ais_csv(path, bounds, start_time, end_time):
    """Loads AIS data already in the internal schema:
    mmsi, timestamp, lon, lat, sog, cog, vessel_type"""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    min_lon, min_lat, max_lon, max_lat = bounds
    df = df[
        (df['lon'].between(min_lon, max_lon)) &
        (df['lat'].between(min_lat, max_lat)) &
        (df['timestamp'].between(start_time, end_time))
    ].copy()
    df['vessel_type'] = df['vessel_type'].fillna('unknown').str.lower()
    df['vessel_type_id'] = df['vessel_type'].map(
        lambda t: VESSEL_TYPE_VOCAB.get(t, VESSEL_TYPE_VOCAB['unknown'])
    )
    return df.sort_values('timestamp')


DMA_COLUMN_MAP = {
    'MMSI': 'mmsi',
    'Timestamp': 'timestamp',
    'Longitude': 'lon',
    'Latitude': 'lat',
    'SOG': 'sog',
    'COG': 'cog',
    'Ship type': 'vessel_type',
}


def load_dma_ais_csv(path, bounds, start_time, end_time):
    """
    Loads a raw Danish Maritime Authority daily AIS CSV/zip (as downloaded
    from aisdata.ais.dk) and converts it to the internal schema used
    downstream: mmsi, timestamp, lon, lat, sog, cog, vessel_type,
    vessel_type_id.

    DMA files include both vessels ('Class A'/'Class B' AIS units) and
    fixed shore-based 'Base Station' transmitters -- base stations are
    dropped here since they aren't vessels and would otherwise show up as
    stationary "vessels" with no MMSI-linked ship type.

    Some DMA export batches prefix the first header column with '# '
    (e.g. '# Timestamp' instead of 'Timestamp') -- stripping that here
    before selecting columns, rather than relying on usecols at read time,
    means this works whether or not a given file has the prefix.
    """
    df = pd.read_csv(path)
    df.columns = [c.lstrip('#').strip() for c in df.columns]
    df = df[list(DMA_COLUMN_MAP.keys()) + ['Type of mobile']]

    df = df[df['Type of mobile'] != 'Base Station'].drop(columns=['Type of mobile'])
    df = df.rename(columns=DMA_COLUMN_MAP)

    # DMA timestamps are day-first: "15/02/2023 00:00:00"
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

    min_lon, min_lat, max_lon, max_lat = bounds
    df = df[
        df['lon'].between(min_lon, max_lon) &
        df['lat'].between(min_lat, max_lat) &
        df['timestamp'].between(start_time, end_time)
    ].copy()

    df['vessel_type'] = df['vessel_type'].fillna('unknown').str.lower()
    df['vessel_type_id'] = df['vessel_type'].map(
        lambda t: VESSEL_TYPE_VOCAB.get(t, VESSEL_TYPE_VOCAB['unknown'])
    )
    return df.sort_values('timestamp')


def resample_to_snapshots(df, interval_minutes=15):
    """
    Bins AIS pings into fixed-interval snapshots per vessel, linearly
    interpolating position/speed/course to fill gaps up to one missed
    interval (longer gaps are left as genuine absences -- a vessel going
    dark is itself a meaningful signal to preserve rather than paper over).
    """
    df = df.copy()
    df['bin'] = df['timestamp'].dt.floor(f'{interval_minutes}min')
    snapshots = {}
    for mmsi, g in df.groupby('mmsi'):
        g = g.set_index('bin')[['lon', 'lat', 'sog', 'cog', 'vessel_type_id']]
        g = g[~g.index.duplicated(keep='last')]
        full_range = pd.date_range(g.index.min(), g.index.max(), freq=f'{interval_minutes}min')
        g = g.reindex(full_range)
        g[['lon', 'lat', 'sog', 'cog']] = g[['lon', 'lat', 'sog', 'cog']].interpolate(limit=1)
        g['vessel_type_id'] = g['vessel_type_id'].ffill().bfill()
        for ts, row in g.iterrows():
            if row[['lon', 'lat', 'sog', 'cog']].isna().any():
                continue
            snapshots.setdefault(ts, {})[mmsi] = row.values.astype(float)
    return snapshots


def one_hot_vessel_type(type_ids, n_types=None):
    """One-hot encodes vessel type IDs -- fed as a raw integer, the network
    would otherwise implicitly treat 'fishing' (2) as numerically between
    'tanker' (1) and 'passenger' (3), a false ordinal relationship for a
    categorical variable."""
    if n_types is None:
        n_types = len(VESSEL_TYPE_VOCAB)
    type_ids = type_ids.astype(int)
    onehot = np.zeros((len(type_ids), n_types))
    onehot[np.arange(len(type_ids)), type_ids] = 1.0
    return onehot


def build_snapshot_sequence(snapshots, mesh_node_features, mesh_edge_index,
                             ego_mmsi, min_vessels_per_snapshot=1,
                             mesh_x_tensor=None, mesh_edge_tensor=None, mesh_tree=None):
    """
    Converts the {timestamp: {mmsi: state}} dict into an ordered list of
    HeteroData graphs plus the ego vessel's row-index at each timestep,
    for timestamps where the ego vessel is actually present.

    mesh_x_tensor/mesh_edge_tensor/mesh_tree: pass precomputed versions
    (built once outside this function) to avoid rebuilding them on every
    single timestep -- see build_hetero_snapshot's docstring for why
    this matters a lot at real scale.

    Vessel feature layout: [lon, lat, sog, cog_sin, cog_cos, *type_onehot]
    -- lon/lat kept as raw real-world degrees (k-NN graph construction,
    target/displacement computation, and plotting all depend on reading
    columns 0-1 as real coordinates elsewhere in the pipeline); sog stays
    raw too (normalized only inside the model, at the point of
    consumption). COG is encoded as sin/cos rather than a raw 0-359
    scalar, since a raw scalar would tell the network that 359 degrees
    and 1 degree are nearly maximally different, when they're actually
    2 degrees apart. Vessel type is one-hot encoded (see
    one_hot_vessel_type) rather than left as a raw ordinal-looking integer.
    """
    timestamps = sorted(snapshots.keys())
    seq, ego_idx_per_step = [], []
    for ts in timestamps:
        vessels = snapshots[ts]
        if ego_mmsi not in vessels or len(vessels) < min_vessels_per_snapshot:
            continue
        mmsi_list = list(vessels.keys())
        ego_idx = mmsi_list.index(ego_mmsi)
        raw = np.stack([vessels[m] for m in mmsi_list])  # (V, 5): lon, lat, sog, cog, type_id

        cog_rad = np.radians(raw[:, 3])
        cog_sin = np.sin(cog_rad)[:, None]
        cog_cos = np.cos(cog_rad)[:, None]
        type_onehot = one_hot_vessel_type(raw[:, 4])

        vessel_states = np.concatenate(
            [raw[:, 0:3], cog_sin, cog_cos, type_onehot], axis=1
        )  # (V, 3 + 2 + n_types)

        data = build_hetero_snapshot(
            mesh_node_features, mesh_edge_index, vessel_states, ego_idx,
            mesh_x_tensor=mesh_x_tensor, mesh_edge_tensor=mesh_edge_tensor, mesh_tree=mesh_tree,
        )
        seq.append(data)
        ego_idx_per_step.append(ego_idx)
    return seq, ego_idx_per_step


def select_ego_vessels_stratified(ais_df, min_pings=40, max_median_ping_interval_sec=1.0,
                                    sog_threshold_knots=2.0, n_underway=20, n_stationary=20,
                                    rng_seed=0):
    """
    Selects a balanced mix of underway and stationary/anchored vessels as
    ego training targets, rather than excluding stationary vessels
    entirely.

    Why balance instead of exclude: a real deployed predictor has to
    handle both regimes -- an anchored vessel staying put is a perfectly
    valid target, and if the model never sees that during training it's
    likely to do something worse (spurious predicted movement) when asked
    to forecast a genuinely stationary vessel at inference. The failure
    mode from the original GC-VTP wasn't "stationary vessels in the
    training set" -- it was an UNBALANCED distribution (mostly stationary)
    combined with a decoder that couldn't express "it depends", which
    together taught the model to always predict near-zero movement even
    for vessels that were actually moving. A roughly even mix, with SOG
    already present in the vessel feature vector as a conditioning signal,
    lets the model learn to distinguish the two regimes instead of
    averaging over them.

    Quality filters (enough pings, not an anomalous high-frequency
    transmitter) still apply to both regimes equally.
    """
    stats = ais_df.groupby('mmsi').agg(
        n_pings=('timestamp', 'count'),
        median_sog=('sog', 'median'),
        t_min=('timestamp', 'min'),
        t_max=('timestamp', 'max'),
    )
    stats['span_sec'] = (stats['t_max'] - stats['t_min']).dt.total_seconds()
    stats['median_ping_interval_sec'] = stats['span_sec'] / stats['n_pings'].clip(lower=1)

    quality_ok = stats[
        (stats['n_pings'] >= min_pings) &
        (stats['median_ping_interval_sec'] >= max_median_ping_interval_sec)
    ]

    underway = quality_ok[quality_ok['median_sog'] >= sog_threshold_knots]
    stationary = quality_ok[quality_ok['median_sog'] < sog_threshold_knots]

    underway_pick = underway.sample(n=min(n_underway, len(underway)), random_state=rng_seed) if len(underway) else underway
    stationary_pick = stationary.sample(n=min(n_stationary, len(stationary)), random_state=rng_seed) if len(stationary) else stationary

    picked = pd.concat([underway_pick, stationary_pick])
    picked['regime'] = ['underway'] * len(underway_pick) + ['stationary'] * len(stationary_pick)
    return picked