"""
Benchmark protocol: a portable, model-agnostic definition of the
prediction task.

The point is that a baseline table in a paper should not require anyone
to adopt this codebase. A benchmark set here is a plain array file
containing observed track segments and their true futures; any model --
a Kalman filter, an LSTM, a transformer from another repo, a commercial
system -- can be evaluated against it by implementing one method.

A `Window` deliberately carries only what is observable at prediction
time:

  * the ego vessel's own recent reports, with their REAL elapsed times
  * nearby traffic as of the anchor instant, each with a staleness
  * the times at which a prediction is requested

and, separately, the ground truth used for scoring. Nothing about the
mesh, the graph, or any model's internals appears, so the format cannot
smuggle in an advantage for the model that produced it.

Times are seconds relative to the anchor (the most recent observation),
so `context_times` are <= 0 and `target_times` are > 0. This is what
lets irregularly sampled and fixed-interval models be scored on exactly
the same task.
"""
from dataclasses import dataclass, asdict
from typing import Optional, List
import json

import numpy as np


@dataclass
class Window:
    """One prediction problem: what was observed, and what is asked."""

    # --- ego vessel history (oldest -> newest; last row is the anchor) ---
    context_positions: np.ndarray   # (T, 2) lon, lat
    context_times: np.ndarray       # (T,)   seconds relative to anchor, <= 0
    context_sog: np.ndarray         # (T,)   knots
    context_cog: np.ndarray         # (T,)   degrees

    # --- what is being asked ---
    target_times: np.ndarray        # (F,)   seconds ahead of anchor, > 0

    # --- ground truth, for scoring only ---
    target_positions: np.ndarray    # (F, 2) lon, lat

    # --- surrounding traffic as of the anchor instant ---
    # Positions are each neighbour's most recent report BEFORE the anchor,
    # with staleness saying how old it is. This mirrors what a live system
    # knows; a neighbour's current position is never available.
    neighbor_positions: Optional[np.ndarray] = None   # (N, 2)
    neighbor_sog: Optional[np.ndarray] = None         # (N,)
    neighbor_cog: Optional[np.ndarray] = None         # (N,)
    neighbor_staleness: Optional[np.ndarray] = None   # (N,) seconds

    # --- identification / stratification, not model input ---
    mmsi: Optional[int] = None
    vessel_type: Optional[str] = None
    regime: Optional[str] = None      # 'underway' | 'stationary'

    @property
    def anchor(self) -> np.ndarray:
        return self.context_positions[-1]

    @property
    def horizon_sec(self) -> float:
        return float(self.target_times[-1])

    @property
    def n_context(self) -> int:
        return len(self.context_times)

    @property
    def n_targets(self) -> int:
        return len(self.target_times)


class BenchmarkSet:
    """
    A collection of windows plus the metadata needed to reproduce and
    cite it. Saved as a single .npz so it can be shared without this
    package.
    """

    def __init__(self, windows: List[Window], meta: Optional[dict] = None):
        self.windows = windows
        self.meta = meta or {}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i]

    def save(self, path):
        """
        Write to .npz. Windows may differ in context and target length
        (that is the nature of irregular sampling), so arrays are
        concatenated and split by offset rather than stacked.
        """
        ctx_pos, ctx_t, ctx_sog, ctx_cog = [], [], [], []
        tgt_t, tgt_pos = [], []
        nb_pos, nb_sog, nb_cog, nb_stale = [], [], [], []
        ctx_off, tgt_off, nb_off = [0], [0], [0]
        mmsi, vtype, regime = [], [], []

        for w in self.windows:
            ctx_pos.append(w.context_positions); ctx_t.append(w.context_times)
            ctx_sog.append(w.context_sog); ctx_cog.append(w.context_cog)
            ctx_off.append(ctx_off[-1] + w.n_context)

            tgt_t.append(w.target_times); tgt_pos.append(w.target_positions)
            tgt_off.append(tgt_off[-1] + w.n_targets)

            n_nb = 0 if w.neighbor_positions is None else len(w.neighbor_positions)
            if n_nb:
                nb_pos.append(w.neighbor_positions); nb_sog.append(w.neighbor_sog)
                nb_cog.append(w.neighbor_cog); nb_stale.append(w.neighbor_staleness)
            nb_off.append(nb_off[-1] + n_nb)

            mmsi.append(-1 if w.mmsi is None else int(w.mmsi))
            vtype.append(w.vessel_type or '')
            regime.append(w.regime or '')

        def cat(xs, dim):
            if not xs:
                return np.zeros((0, dim) if dim else (0,), dtype=np.float64)
            return np.concatenate(xs, axis=0)

        np.savez_compressed(
            path,
            context_positions=cat(ctx_pos, 2), context_times=cat(ctx_t, 0),
            context_sog=cat(ctx_sog, 0), context_cog=cat(ctx_cog, 0),
            context_offsets=np.array(ctx_off),
            target_times=cat(tgt_t, 0), target_positions=cat(tgt_pos, 2),
            target_offsets=np.array(tgt_off),
            neighbor_positions=cat(nb_pos, 2), neighbor_sog=cat(nb_sog, 0),
            neighbor_cog=cat(nb_cog, 0), neighbor_staleness=cat(nb_stale, 0),
            neighbor_offsets=np.array(nb_off),
            mmsi=np.array(mmsi), vessel_type=np.array(vtype),
            regime=np.array(regime), meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        co, to, no = z['context_offsets'], z['target_offsets'], z['neighbor_offsets']
        windows = []
        for i in range(len(co) - 1):
            a, b = co[i], co[i+1]
            c, d = to[i], to[i+1]
            e, f = no[i], no[i+1]
            windows.append(Window(
                context_positions=z['context_positions'][a:b],
                context_times=z['context_times'][a:b],
                context_sog=z['context_sog'][a:b],
                context_cog=z['context_cog'][a:b],
                target_times=z['target_times'][c:d],
                target_positions=z['target_positions'][c:d],
                neighbor_positions=z['neighbor_positions'][e:f] if f > e else None,
                neighbor_sog=z['neighbor_sog'][e:f] if f > e else None,
                neighbor_cog=z['neighbor_cog'][e:f] if f > e else None,
                neighbor_staleness=z['neighbor_staleness'][e:f] if f > e else None,
                mmsi=int(z['mmsi'][i]) if z['mmsi'][i] >= 0 else None,
                vessel_type=str(z['vessel_type'][i]) or None,
                regime=str(z['regime'][i]) or None,
            ))
        return cls(windows, json.loads(str(z['meta'])))

    def summary(self):
        hor = np.array([w.horizon_sec for w in self.windows]) / 60.0
        moved = np.array([
            haversine_km(w.anchor[0], w.anchor[1],
                          w.target_positions[-1, 0], w.target_positions[-1, 1])
            for w in self.windows])
        regimes = {}
        for w in self.windows:
            regimes[w.regime or 'unknown'] = regimes.get(w.regime or 'unknown', 0) + 1
        return {
            'n_windows': len(self.windows),
            'n_vessels': len({w.mmsi for w in self.windows if w.mmsi is not None}),
            'horizon_min_median': float(np.median(hor)),
            'horizon_min_p10': float(np.percentile(hor, 10)),
            'horizon_min_p90': float(np.percentile(hor, 90)),
            'displacement_km_median': float(np.median(moved)),
            'regimes': regimes,
            **self.meta,
        }


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))
