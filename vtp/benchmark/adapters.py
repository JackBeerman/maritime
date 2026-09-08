"""
Adapters that let this project's trained models be scored on the
benchmark alongside external ones.

The benchmark deliberately knows nothing about meshes or graphs, so each
adapter is responsible for rebuilding whatever internal representation
its model needs from the plain `Window`. That indirection is the point:
it keeps the benchmark honest, and it means a model from another
codebase competes on exactly the same inputs.
"""
import numpy as np
import torch

from .predictors import Predictor


class IrregularVTPPredictor(Predictor):
    """
    Wraps the irregular-sampling model.

    Rebuilds ego-anchored snapshots from the window's raw observations,
    runs the Δt-conditioned head at the window's own target times, and
    converts the normalized velocity residual back to absolute lon/lat:
    `output * vel_scale * dt + anchor`.

    Because the head is conditioned on Δt, this model can answer the
    benchmark's arbitrary target times directly -- no interpolation or
    rounding to a grid.
    """
    name = "irregular-vtp"
    probabilistic = True

    def __init__(self, model, vel_scale, mesh_node_features, mesh_edge_index,
                 mesh_tree=None, device=None, name=None,
                 staleness_cutoff_sec=1800.0, max_neighbors=150):
        from scipy.spatial import cKDTree
        self.model = model.eval()
        self.vel_scale = vel_scale
        self.mnf, self.mei = mesh_node_features, mesh_edge_index
        self.mesh_tree = mesh_tree or cKDTree(mesh_node_features[:, :2])
        self.mesh_x = torch.as_tensor(mesh_node_features, dtype=torch.float)
        self.mesh_e = torch.as_tensor(mesh_edge_index, dtype=torch.long)
        self.device = device or next(model.parameters()).device
        self.staleness_cutoff_sec = staleness_cutoff_sec
        self.max_neighbors = max_neighbors
        if name:
            self.name = name

    def _snapshot(self, ego_row, neighbors, staleness, dt_own):
        """Build one ego-anchored HeteroData from plain arrays."""
        from vtp.data.graphs import build_hetero_snapshot
        from vtp.data.ingest import one_hot_vessel_type, DT_REFERENCE_SEC, STALENESS_REFERENCE_SEC

        rows = [ego_row]
        if neighbors is not None and len(neighbors):
            rows.extend(neighbors)
        raw = np.stack(rows)                       # lon, lat, sog, cog, type
        n = len(raw)

        dts = np.zeros(n); dts[0] = dt_own
        stale = np.zeros(n)
        if neighbors is not None and len(neighbors):
            stale[1:] = staleness

        cog_rad = np.radians(raw[:, 3])
        feats = np.concatenate([
            raw[:, 0:3],
            np.sin(cog_rad)[:, None], np.cos(cog_rad)[:, None],
            (np.log1p(np.clip(dts, 0, None)) / np.log1p(DT_REFERENCE_SEC))[:, None],
            (np.log1p(np.clip(stale, 0, None)) / np.log1p(STALENESS_REFERENCE_SEC))[:, None],
            one_hot_vessel_type(raw[:, 4]),
        ], axis=1)
        return build_hetero_snapshot(self.mnf, self.mei, feats, ego_idx=0,
                                      mesh_x_tensor=self.mesh_x,
                                      mesh_edge_tensor=self.mesh_e,
                                      mesh_tree=self.mesh_tree)

    @torch.no_grad()
    def predict(self, window, n_samples=32):
        T = window.n_context
        # neighbours are only known as of the anchor instant, so earlier
        # context steps carry ego state alone -- the same information a
        # live system would have had
        snaps = []
        for t in range(T):
            ego_row = np.array([window.context_positions[t, 0],
                                 window.context_positions[t, 1],
                                 window.context_sog[t], window.context_cog[t], 0.0])
            dt_own = 0.0 if t == 0 else float(window.context_times[t] - window.context_times[t-1])
            if t == T - 1 and window.neighbor_positions is not None and len(window.neighbor_positions):
                keep = window.neighbor_staleness <= self.staleness_cutoff_sec
                idx = np.where(keep)[0][: self.max_neighbors]
                nb = np.stack([window.neighbor_positions[idx, 0],
                                window.neighbor_positions[idx, 1],
                                window.neighbor_sog[idx], window.neighbor_cog[idx],
                                np.zeros(len(idx))], axis=1)
                snaps.append(self._snapshot(ego_row, nb, window.neighbor_staleness[idx], dt_own))
            else:
                snaps.append(self._snapshot(ego_row, None, None, dt_own))

        snaps = [d.to(self.device) for d in snaps]
        ego_rows = [0] * T
        ctx_times = torch.as_tensor(window.context_times, dtype=torch.float, device=self.device)
        dts = torch.as_tensor(window.target_times, dtype=torch.float, device=self.device)

        out = self.model(snaps, ego_rows, ctx_times, dts, n_samples=n_samples)
        dt_col = dts.clamp(min=1e-6).view(1, 1, -1, 1)
        anchor = torch.as_tensor(window.anchor, dtype=torch.float, device=self.device)
        real = out * self.vel_scale * dt_col + anchor
        return real[0].cpu().numpy()


class FixedIntervalVTPPredictor(Predictor):
    """
    Wraps the fixed-interval model.

    This model only knows how to answer a fixed grid (+15/30/45/60 min),
    so scoring it on arbitrary benchmark target times requires
    interpolating its gridded output to those times. That is a real
    limitation of the approach, not an artefact of the benchmark, and it
    is why this adapter is more involved than the irregular one -- worth
    stating plainly when the two appear in the same table.
    """
    name = "fixed-interval-vtp"
    probabilistic = True

    def __init__(self, model, norm_scale, mesh_node_features, mesh_edge_index,
                 mesh_tree=None, device=None, interval_minutes=15, name=None):
        from scipy.spatial import cKDTree
        self.model = model.eval()
        self.norm_scale = norm_scale
        self.mnf, self.mei = mesh_node_features, mesh_edge_index
        self.mesh_tree = mesh_tree or cKDTree(mesh_node_features[:, :2])
        self.mesh_x = torch.as_tensor(mesh_node_features, dtype=torch.float)
        self.mesh_e = torch.as_tensor(mesh_edge_index, dtype=torch.long)
        self.device = device or next(model.parameters()).device
        self.interval_sec = interval_minutes * 60
        if name:
            self.name = name

    @torch.no_grad()
    def predict(self, window, n_samples=32):
        from vtp.data.graphs import build_world_snapshot
        from ais_ingest import one_hot_vessel_type

        # resample the window's irregular context onto this model's grid,
        # which is exactly the information loss the fixed-interval
        # approach imposes
        seq_len = getattr(self.model, 'expected_seq_len', 12)
        grid_t = np.array([-(seq_len - 1 - i) * self.interval_sec for i in range(seq_len)])
        ct, cp = window.context_times, window.context_positions
        lon = np.interp(grid_t, ct, cp[:, 0])
        lat = np.interp(grid_t, ct, cp[:, 1])
        sog = np.interp(grid_t, ct, window.context_sog)
        cog = np.interp(grid_t, ct, window.context_cog)

        snaps, ego_rows = [], []
        for t in range(seq_len):
            raw = np.array([[lon[t], lat[t], sog[t], cog[t], 0.0]])
            cog_rad = np.radians(raw[:, 3])
            feats = np.concatenate([raw[:, 0:3], np.sin(cog_rad)[:, None],
                                     np.cos(cog_rad)[:, None],
                                     one_hot_vessel_type(raw[:, 4])], axis=1)
            snaps.append(build_world_snapshot(self.mnf, self.mei, feats,
                                               mesh_x_tensor=self.mesh_x,
                                               mesh_edge_tensor=self.mesh_e,
                                               mesh_tree=self.mesh_tree).to(self.device))
            ego_rows.append(0)

        out = self.model(snaps, ego_rows, n_samples=n_samples, training=False)
        anchor = torch.as_tensor(window.anchor, dtype=torch.float, device=self.device)
        grid_pred = (out * self.norm_scale + anchor)[0].cpu().numpy()   # (S, F_grid, 2)

        # interpolate the gridded prediction to the benchmark's target times
        F_grid = grid_pred.shape[1]
        grid_times = np.array([(k + 1) * self.interval_sec for k in range(F_grid)])
        S = grid_pred.shape[0]
        out_pred = np.empty((S, len(window.target_times), 2))
        for s in range(S):
            out_pred[s, :, 0] = np.interp(window.target_times, grid_times, grid_pred[s, :, 0])
            out_pred[s, :, 1] = np.interp(window.target_times, grid_times, grid_pred[s, :, 1])
        return out_pred
