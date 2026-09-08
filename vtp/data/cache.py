"""
Disk cache for the load + resample stage.

Loading 12 daily CSVs (~236M records) and resampling them takes on the
order of 20-30 minutes, and it is repeated identically on every job --
every smoke test, every resumed run, every hyperparameter change. None
of that work depends on the model, so it only needs doing once.

The cache key covers the input files (path, size, mtime) and the
resampling interval, so editing the data or changing `--interval-minutes`
invalidates it automatically rather than silently serving stale results.

Cached artifacts:
  * `ais_snapshots` -- {timestamp: {mmsi: [lon, lat, sog, cog, type_id]}}
  * `vessel_stats`  -- the small per-vessel table used for ego selection,
    so the full DataFrame never has to be reconstructed
"""
import hashlib
import os
import pickle
import time

import numpy as np
import pandas as pd


def _fingerprint(paths, interval_minutes, bounds):
    """Identity of the inputs -- changing any of them invalidates the cache."""
    h = hashlib.sha256()
    for p in sorted(paths):
        st = os.stat(p)
        h.update(p.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    h.update(f"{interval_minutes}".encode())
    h.update(",".join(f"{b:.4f}" for b in bounds).encode())
    return h.hexdigest()[:16]


def vessel_stats_from_df(ais_df):
    """
    Condense the full DataFrame to the per-vessel table ego selection
    needs. Keeping this instead of the DataFrame is what lets the cache
    stay small -- ~15k rows rather than ~236M.
    """
    stats = ais_df.groupby('mmsi').agg(
        n_pings=('timestamp', 'count'),
        median_sog=('sog', 'median'),
        t_min=('timestamp', 'min'),
        t_max=('timestamp', 'max'),
    )
    stats['span_sec'] = (stats['t_max'] - stats['t_min']).dt.total_seconds()
    stats['median_ping_interval_sec'] = stats['span_sec'] / stats['n_pings'].clip(lower=1)
    return stats


def select_ego_vessels_from_stats(stats, min_pings=40, max_median_ping_interval_sec=1.0,
                                    sog_threshold_knots=2.0, n_underway=20,
                                    n_stationary=20, rng_seed=0):
    """
    Same stratified selection as `ais_ingest.select_ego_vessels_stratified`,
    but reading the cached stats table instead of a full DataFrame. Keep
    the two in sync -- they must pick identical vessels for a cached run
    to match an uncached one.
    """
    ok = stats[(stats['n_pings'] >= min_pings) &
               (stats['median_ping_interval_sec'] >= max_median_ping_interval_sec)]
    under = ok[ok['median_sog'] >= sog_threshold_knots]
    stat = ok[ok['median_sog'] < sog_threshold_knots]
    up = under.sample(n=min(n_underway, len(under)), random_state=rng_seed) if len(under) else under
    sp = stat.sample(n=min(n_stationary, len(stat)), random_state=rng_seed) if len(stat) else stat
    picked = pd.concat([up, sp])
    picked['regime'] = ['underway'] * len(up) + ['stationary'] * len(sp)
    return picked


def load_and_resample_cached(paths, bounds, interval_minutes, loader_fn, resampler_fn,
                              cache_dir='cache', verbose=True, force_rebuild=False):
    """
    Returns (ais_snapshots, vessel_stats), from cache when possible.

    loader_fn(path, bounds) -> DataFrame and resampler_fn(df, interval)
    are injected rather than imported so this works unchanged on both
    branches, whose ingest modules differ.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _fingerprint(paths, interval_minutes, bounds)
    cache_path = os.path.join(cache_dir, f"ais_{key}.pkl")

    if os.path.exists(cache_path) and not force_rebuild:
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            blob = pickle.load(f)
        if verbose:
            print(f"  loaded cache {cache_path} in {time.time()-t0:.1f}s "
                  f"({len(blob['snapshots'])} timestamps, "
                  f"{len(blob['stats'])} vessels)", flush=True)
        return blob['snapshots'], blob['stats']

    if verbose:
        print(f"  no cache for key {key}; building (this is the slow path)", flush=True)
    t0 = time.time()
    dfs = []
    for p in paths:
        d = loader_fn(p, bounds)
        if verbose:
            print(f"    {p}: {len(d)} records", flush=True)
        dfs.append(d)
    ais_df = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
    del dfs
    if verbose:
        print(f"    total {len(ais_df)} records, {ais_df['mmsi'].nunique()} vessels "
              f"({time.time()-t0:.1f}s)", flush=True)

    stats = vessel_stats_from_df(ais_df)
    t1 = time.time()
    snapshots = resampler_fn(ais_df, interval_minutes)
    if verbose:
        print(f"    resampled to {len(snapshots)} timestamps ({time.time()-t1:.1f}s)", flush=True)
    del ais_df

    t2 = time.time()
    with open(cache_path, 'wb') as f:
        pickle.dump({'snapshots': snapshots, 'stats': stats}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        size_mb = os.path.getsize(cache_path) / 1024**2
        print(f"  wrote cache {cache_path} ({size_mb:.0f} MB, {time.time()-t2:.1f}s)", flush=True)

    return snapshots, stats