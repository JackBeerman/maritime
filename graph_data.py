import torch
from torch_geometric.data import HeteroData
import numpy as np
from scipy.spatial import cKDTree


def assign_vessels_to_mesh(vessel_xy, mesh_xy, k=3, mesh_tree=None):
    """
    Each vessel connects to its k nearest mesh nodes, via KD-tree.
    A KD-tree can pick a different k-th neighbour than brute-force cdist
    when two candidates are within floating-point noise of tied (~5% of
    vessels at real scale, always sub-1%-relative-distance ties) --
    verified benign, not a correctness bug.
    """
    if mesh_tree is None:
        mesh_tree = cKDTree(mesh_xy)
    xy = vessel_xy if isinstance(vessel_xy, np.ndarray) else np.asarray(vessel_xy)
    k = min(k, mesh_tree.n)
    _, idx = mesh_tree.query(xy, k=k)
    if k == 1:
        idx = idx[:, None]
    idx = torch.as_tensor(idx, dtype=torch.long)
    return torch.stack([torch.arange(len(xy)).repeat_interleave(k), idx.flatten()], dim=0)


def knn_graph_manual(xy, k):
    """k-NN edge list (src -> dst) via KD-tree; avoids the pyg-lib dependency."""
    xy_np = xy if isinstance(xy, np.ndarray) else np.asarray(xy)
    n = xy_np.shape[0]
    k = min(k, n - 1)
    if k < 1:
        return torch.empty((2, 0), dtype=torch.long)
    tree = cKDTree(xy_np)
    _, idx = tree.query(xy_np, k=k + 1)          # +1: a point finds itself first
    if idx.ndim == 1:
        idx = idx[None, :]
    idx = torch.as_tensor(idx[:, 1:k + 1], dtype=torch.long)
    return torch.stack([idx.flatten(), torch.arange(n).repeat_interleave(k)], dim=0)


def build_world_snapshot(mesh_node_features, mesh_edge_index, vessel_states,
                          k_vessel_mesh=3, k_vessel_vessel=6,
                          mesh_x_tensor=None, mesh_edge_tensor=None, mesh_tree=None):
    """
    Builds ONE graph for a timestamp, holding every vessel present.

    Deliberately carries NO ego_mask. The world at a given timestamp is
    identical no matter which vessel you are forecasting -- only the
    choice of ego row differs -- so baking ego_mask in would force a
    separate copy of the same graph per ego vessel. At real scale that
    was the difference between ~1.2 GB and several hundred GB (the OOM
    that motivated this refactor). The ego row index is instead carried
    alongside, by VesselSequenceDataset.
    """
    data = HeteroData()
    if mesh_x_tensor is None:
        mesh_x_tensor = torch.as_tensor(mesh_node_features, dtype=torch.float, device='cpu')
    if mesh_edge_tensor is None:
        mesh_edge_tensor = torch.as_tensor(mesh_edge_index, dtype=torch.long, device='cpu')
    data['mesh'].x = mesh_x_tensor
    data['mesh', 'to', 'mesh'].edge_index = mesh_edge_tensor

    data['vessel'].x = torch.as_tensor(vessel_states, dtype=torch.float, device='cpu')

    vessel_xy = vessel_states[:, :2]
    v2m = assign_vessels_to_mesh(vessel_xy, mesh_node_features[:, :2],
                                  k=k_vessel_mesh, mesh_tree=mesh_tree)
    data['vessel', 'near', 'mesh'].edge_index = v2m
    data['mesh', 'rev_near', 'vessel'].edge_index = v2m.flip(0)

    if len(vessel_states) > 1:
        data['vessel', 'near', 'vessel'].edge_index = knn_graph_manual(vessel_xy, k=k_vessel_vessel)
    else:
        data['vessel', 'near', 'vessel'].edge_index = torch.empty((2, 0), dtype=torch.long)
    return data


def share_mesh_on_device(world_snapshots, device):
    """
    Moves world snapshots to `device` while keeping ONE shared copy of the
    mesh tensors.

    HeteroData.to() would otherwise allocate a private mesh copy per
    snapshot (~348 KB each), which across thousands of snapshots is tens
    of GB of pure duplication. Here the mesh is moved once and the same
    device tensor is assigned to every snapshot.
    """
    if not world_snapshots:
        return world_snapshots
    mesh_x = world_snapshots[0]['mesh'].x.to(device)
    mesh_ei = world_snapshots[0]['mesh', 'to', 'mesh'].edge_index.to(device)
    for d in world_snapshots:
        d['vessel'].x = d['vessel'].x.to(device)
        for et in [('vessel', 'near', 'mesh'), ('mesh', 'rev_near', 'vessel'),
                    ('vessel', 'near', 'vessel')]:
            d[et].edge_index = d[et].edge_index.to(device)
        d['mesh'].x = mesh_x
        d['mesh', 'to', 'mesh'].edge_index = mesh_ei
    return world_snapshots


class VesselSequenceDataset(torch.utils.data.Dataset):
    """
    Sliding windows for one ego vessel over SHARED world snapshots.

    Instead of owning its own copies of the graphs, this holds indices
    into a single shared list of per-timestamp world snapshots, plus the
    ego vessel's row index within each. Windows are consecutive in the
    vessel's own presence list (matching the previous behaviour), so a
    window can still span a timestamp where the vessel was absent.

    __getitem__ returns (ctx_snapshots, ego_rows, target_positions).
    ego_rows is required because the shared snapshots carry no ego_mask.
    """
    def __init__(self, world_snapshots, present_world_idx, present_ego_row,
                 seq_len=4, future_len=4):
        self.world = world_snapshots
        self.widx = np.asarray(present_world_idx, dtype=np.int64)
        self.erow = np.asarray(present_ego_row, dtype=np.int64)
        self.seq_len = seq_len
        self.future_len = future_len
        n = len(self.widx)
        self.valid_starts = list(range(n - seq_len - future_len + 1))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        s = self.valid_starts[idx]
        ctx_slice = slice(s, s + self.seq_len)
        ctx = [self.world[i] for i in self.widx[ctx_slice]]
        ego_rows = [int(r) for r in self.erow[ctx_slice]]

        fut = slice(s + self.seq_len, s + self.seq_len + self.future_len)
        target = torch.stack([
            self.world[wi]['vessel'].x[int(er), :2]
            for wi, er in zip(self.widx[fut], self.erow[fut])
        ], dim=0)
        return ctx, ego_rows, target