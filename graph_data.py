import torch
from torch_geometric.data import HeteroData
import numpy as np
from scipy.spatial import cKDTree


def assign_vessels_to_mesh(vessel_xy, mesh_xy, k=3, mesh_tree=None):
    """
    Each vessel connects to its k nearest mesh nodes. KD-tree based --
    see the fixed-interval branch for the measured justification and the
    (benign) near-tie caveat vs brute-force distance.
    """
    if mesh_tree is None:
        mesh_tree = cKDTree(mesh_xy)
    vessel_xy_np = vessel_xy if isinstance(vessel_xy, np.ndarray) else np.asarray(vessel_xy)
    k = min(k, mesh_tree.n)
    _, idx = mesh_tree.query(vessel_xy_np, k=k)
    if k == 1:
        idx = idx[:, None]
    idx = torch.as_tensor(idx, dtype=torch.long)
    src = idx.flatten()
    dst = torch.arange(len(vessel_xy_np)).repeat_interleave(k)
    return torch.stack([dst, src], dim=0)


def knn_graph_manual(xy, k):
    """k-NN edge list (src -> dst) via KD-tree, no pyg-lib dependency."""
    xy_np = xy if isinstance(xy, np.ndarray) else np.asarray(xy)
    n = xy_np.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return torch.empty((2, 0), dtype=torch.long)
    tree = cKDTree(xy_np)
    _, idx = tree.query(xy_np, k=k + 1)  # +1: a point always finds itself
    if idx.ndim == 1:
        idx = idx[None, :]
    idx = idx[:, 1:k + 1]
    idx = torch.as_tensor(idx, dtype=torch.long)
    dst = torch.arange(n).repeat_interleave(k)
    src = idx.flatten()
    return torch.stack([src, dst], dim=0)


def build_hetero_snapshot(mesh_node_features, mesh_edge_index, vessel_states,
                           ego_idx, k_vessel_mesh=3, k_vessel_vessel=6,
                           mesh_x_tensor=None, mesh_edge_tensor=None, mesh_tree=None):
    """
    Builds one ego-anchored HeteroData snapshot, always on CPU.

    Unlike the fixed-interval branch, a "snapshot" here is anchored to a
    single ego-vessel ping rather than a shared time bin: the ego row
    holds its exact reported position at that instant, and every other
    row holds that vessel's most recent report BEFORE that instant, with
    a staleness feature recording how old it is (see irregular_ingest).
    """
    data = HeteroData()
    if mesh_x_tensor is None:
        mesh_x_tensor = torch.as_tensor(mesh_node_features, dtype=torch.float, device='cpu')
    if mesh_edge_tensor is None:
        mesh_edge_tensor = torch.as_tensor(mesh_edge_index, dtype=torch.long, device='cpu')
    data['mesh'].x = mesh_x_tensor
    data['mesh', 'to', 'mesh'].edge_index = mesh_edge_tensor

    data['vessel'].x = torch.as_tensor(vessel_states, dtype=torch.float, device='cpu')
    data['vessel'].ego_mask = torch.zeros(len(vessel_states), dtype=torch.bool, device='cpu')
    data['vessel'].ego_mask[ego_idx] = True

    vessel_xy = vessel_states[:, :2]
    mesh_xy = mesh_node_features[:, :2]
    v2m = assign_vessels_to_mesh(vessel_xy, mesh_xy, k=k_vessel_mesh, mesh_tree=mesh_tree)
    data['vessel', 'near', 'mesh'].edge_index = v2m
    data['mesh', 'rev_near', 'vessel'].edge_index = v2m.flip(0)

    if len(vessel_states) > 1:
        data['vessel', 'near', 'vessel'].edge_index = knn_graph_manual(vessel_xy, k=k_vessel_vessel)
    else:
        data['vessel', 'near', 'vessel'].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data


class IrregularVesselDataset(torch.utils.data.Dataset):
    """
    Sliding windows over a single ego vessel's irregular ping sequence.

    Unlike the fixed-interval version, each item carries explicit timing:
      ctx              -- seq_len ego-anchored HeteroData snapshots
      ctx_times        -- (seq_len,) seconds, relative to the window's last
                          context ping (so the last entry is always 0.0)
      target_positions -- (future_len, 2) actual observed lon/lat, never
                          interpolated
      target_dts       -- (future_len,) seconds from the last context ping
                          to each target ping

    Windows are contiguous in PING INDEX, not in wall-clock time -- there
    is no notion of a "missing" step here, since gaps are represented
    directly by the dt values rather than by absent bins.
    """
    def __init__(self, snapshots, ego_idx_per_step, ping_times_sec, positions,
                 seq_len=12, future_len=4, max_window_span_sec=None):
        self.snapshots = snapshots
        self.ego_idx_per_step = ego_idx_per_step
        self.ping_times_sec = np.asarray(ping_times_sec, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.seq_len = seq_len
        self.future_len = future_len

        starts = []
        for s in range(len(snapshots) - seq_len - future_len + 1):
            if max_window_span_sec is not None:
                span = self.ping_times_sec[s + seq_len + future_len - 1] - self.ping_times_sec[s]
                if span > max_window_span_sec:
                    continue
            starts.append(s)
        self.valid_starts = starts

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        ctx = self.snapshots[s:s + self.seq_len]
        anchor_i = s + self.seq_len - 1
        t0 = self.ping_times_sec[anchor_i]

        ctx_times = torch.as_tensor(
            self.ping_times_sec[s:s + self.seq_len] - t0, dtype=torch.float)

        fut = slice(anchor_i + 1, anchor_i + 1 + self.future_len)
        target_positions = torch.as_tensor(self.positions[fut], dtype=torch.float)
        target_dts = torch.as_tensor(self.ping_times_sec[fut] - t0, dtype=torch.float)

        return ctx, ctx_times, target_positions, target_dts
