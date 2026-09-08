"""
Build a benchmark set from real DMA AIS data.

The resulting .npz is the shareable artefact: it fixes the vessels, the
windows, the horizons and the ground truth, so any model evaluated
against it is solving exactly the same problem. Publish it alongside a
paper and the baseline table becomes reproducible without this codebase.

Windows are built from RAW pings, never resampled or interpolated, so
the benchmark does not privilege either branch's assumptions. A
fixed-interval model is free to resample internally -- that information
loss is part of what is being compared.
"""
import argparse
import json

import numpy as np
import pandas as pd

from .protocol import Window, BenchmarkSet


def build_windows_for_vessel(ais_df, mmsi, seq_len, future_len,
                              min_ping_gap_sec=900.0, stride=1,
                              neighbor_radius_deg=0.5, staleness_cutoff_sec=1800.0,
                              max_neighbors=150, max_windows=None, rng=None):
    """
    Extract benchmark windows for one ego vessel.

    `min_ping_gap_sec` thins dense reporting. It is the main control on
    forecast horizon: with `future_len` targets, the horizon is roughly
    `future_len * min_ping_gap_sec`. Set it to match the horizon you want
    to benchmark at -- comparing methods at different horizons is
    meaningless.
    """
    ego = ais_df[ais_df['mmsi'] == mmsi].sort_values('timestamp')
    if len(ego) < seq_len + future_len:
        return []

    t0 = ego['timestamp'].iloc[0]
    t_all = (ego['timestamp'] - t0).dt.total_seconds().to_numpy()

    keep, last = [], None
    for i, t in enumerate(t_all):
        if last is None or (t - last) >= min_ping_gap_sec:
            keep.append(i); last = t
    keep = np.asarray(keep)
    if len(keep) < seq_len + future_len:
        return []

    e = ego.iloc[keep]
    t = t_all[keep]
    lon, lat = e['lon'].to_numpy(), e['lat'].to_numpy()
    sog, cog = e['sog'].to_numpy(), e['cog'].to_numpy()
    vtype = str(e['vessel_type'].iloc[0]) if 'vessel_type' in e else ''

    # spatial + temporal prefilter for neighbour lookup
    pad = neighbor_radius_deg
    nearby = ais_df[(ais_df['mmsi'] != mmsi) &
                    ais_df['lon'].between(lon.min()-pad, lon.max()+pad) &
                    ais_df['lat'].between(lat.min()-pad, lat.max()+pad)]
    nb_t = (nearby['timestamp'] - t0).dt.total_seconds().to_numpy()
    order = np.argsort(nb_t)
    nb_t = nb_t[order]
    nb_vals = nearby[['lon', 'lat', 'sog', 'cog']].to_numpy()[order]
    nb_mmsi = nearby['mmsi'].to_numpy()[order]

    windows = []
    starts = list(range(0, len(t) - seq_len - future_len + 1, max(1, stride)))
    if max_windows and len(starts) > max_windows:
        rng = rng or np.random.default_rng(0)
        starts = sorted(rng.choice(starts, size=max_windows, replace=False))

    for s in starts:
        a = s + seq_len - 1                       # anchor index
        anchor_t = t[a]

        # neighbours: last report before the anchor, within radius
        cut = np.searchsorted(nb_t, anchor_t, side='right')
        state = {}
        lo = max(0, cut - 200000)                 # bound the scan
        for k in range(lo, cut):
            state[nb_mmsi[k]] = (nb_t[k], nb_vals[k])
        npos, nsog, ncog, nstale = [], [], [], []
        for m, (tt, v) in state.items():
            age = anchor_t - tt
            if age > staleness_cutoff_sec:
                continue
            if abs(v[0] - lon[a]) > pad or abs(v[1] - lat[a]) > pad:
                continue
            npos.append([v[0], v[1]]); nsog.append(v[2]); ncog.append(v[3]); nstale.append(age)
        if npos and len(npos) > max_neighbors:
            d = np.hypot(np.array(npos)[:, 0] - lon[a], np.array(npos)[:, 1] - lat[a])
            sel = np.argsort(d)[:max_neighbors]
            npos = [npos[i] for i in sel]; nsog = [nsog[i] for i in sel]
            ncog = [ncog[i] for i in sel]; nstale = [nstale[i] for i in sel]

        ctx = slice(s, s + seq_len)
        fut = slice(a + 1, a + 1 + future_len)
        moved = np.hypot(lon[fut][-1] - lon[a], lat[fut][-1] - lat[a]) * 111.0

        windows.append(Window(
            context_positions=np.stack([lon[ctx], lat[ctx]], axis=1),
            context_times=t[ctx] - anchor_t,
            context_sog=sog[ctx], context_cog=cog[ctx],
            target_times=t[fut] - anchor_t,
            target_positions=np.stack([lon[fut], lat[fut]], axis=1),
            neighbor_positions=np.asarray(npos) if npos else None,
            neighbor_sog=np.asarray(nsog) if npos else None,
            neighbor_cog=np.asarray(ncog) if npos else None,
            neighbor_staleness=np.asarray(nstale) if npos else None,
            mmsi=int(mmsi), vessel_type=vtype,
            regime='underway' if moved > 1.0 else 'stationary',
        ))
    return windows


def build_benchmark(ais_df, vessel_ids, seq_len=12, future_len=4,
                     min_ping_gap_sec=900.0, stride=4, max_windows_per_vessel=20,
                     neighbor_radius_deg=0.5, staleness_cutoff_sec=1800.0,
                     max_neighbors=150, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    all_w = []
    for i, m in enumerate(vessel_ids):
        ws = build_windows_for_vessel(
            ais_df, m, seq_len, future_len, min_ping_gap_sec, stride,
            neighbor_radius_deg, staleness_cutoff_sec, max_neighbors,
            max_windows=max_windows_per_vessel, rng=rng)
        all_w.extend(ws)
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(vessel_ids)} vessels -> {len(all_w)} windows", flush=True)

    meta = {'seq_len': seq_len, 'future_len': future_len,
            'min_ping_gap_sec': min_ping_gap_sec, 'stride': stride,
            'max_windows_per_vessel': max_windows_per_vessel,
            'neighbor_radius_deg': neighbor_radius_deg,
            'staleness_cutoff_sec': staleness_cutoff_sec,
            'max_neighbors': max_neighbors, 'seed': seed,
            'source': 'DMA AIS (aisdata.ais.dk)'}
    return BenchmarkSet(all_w, meta)


def main():
    p = argparse.ArgumentParser(description='Build a shareable benchmark set from AIS.')
    p.add_argument('--ais-path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seq-len', type=int, default=12)
    p.add_argument('--future-len', type=int, default=4)
    p.add_argument('--min-ping-gap-sec', type=float, default=900.0)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--max-windows-per-vessel', type=int, default=20)
    p.add_argument('--n-underway', type=int, default=100)
    p.add_argument('--n-stationary', type=int, default=100)
    p.add_argument('--min-pings', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    from vtp.data.ingest import load_dma_ais_csv, select_ego_vessels_stratified
    from vtp.data.coastline import DMA_BOUNDS

    print(f"loading {args.ais_path} ...", flush=True)
    df = load_dma_ais_csv(args.ais_path, DMA_BOUNDS)
    print(f"  {len(df)} records, {df['mmsi'].nunique()} vessels", flush=True)

    good = select_ego_vessels_stratified(
        df, min_pings=args.min_pings, n_underway=args.n_underway,
        n_stationary=args.n_stationary, rng_seed=args.seed)
    print(f"  selected {len(good)} vessels: {good['regime'].value_counts().to_dict()}", flush=True)

    bench = build_benchmark(
        df, list(good.index), seq_len=args.seq_len, future_len=args.future_len,
        min_ping_gap_sec=args.min_ping_gap_sec, stride=args.stride,
        max_windows_per_vessel=args.max_windows_per_vessel, seed=args.seed)
    bench.save(args.out)

    print(f"\nwrote {args.out}")
    for k, v in bench.summary().items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
