import torch
from torch_geometric.data import HeteroData
import numpy as np


def assign_vessels_to_mesh(vessel_xy, mesh_xy, k=3):
    v = torch.as_tensor(vessel_xy, dtype=torch.float, device='cpu')
    m = torch.as_tensor(mesh_xy, dtype=torch.float, device='cpu')
    dists = torch.cdist(v, m)
    knn_idx = torch.topk(dists, k=k, largest=False).indices
    src = knn_idx.flatten()
    dst = torch.arange(len(vessel_xy)).repeat_interleave(k)
    return torch.stack([dst, src], dim=0)


def knn_graph_manual(xy, k):
    """
    Undirected-style k-NN edge list (src -> dst, each node connected to its
    k nearest neighbors), built with plain torch.cdist instead of the
    optional pyg-lib backend that knn_graph() requires — pyg-lib wheels are
    version-pinned to specific torch/CUDA builds and aren't guaranteed to be
    available on Rivanna's stack, so this avoids depending on it.
    """
    x = torch.as_tensor(xy, dtype=torch.float, device='cpu')
    n = x.shape[0]
    dists = torch.cdist(x, x)
    dists.fill_diagonal_(float('inf'))
    k = min(k, n - 1)
    idx = torch.topk(dists, k=k, largest=False).indices  # (n, k)
    dst = torch.arange(n).repeat_interleave(k)
    src = idx.flatten()
    return torch.stack([src, dst], dim=0)


def build_hetero_snapshot(mesh_node_features, mesh_edge_index,
                           vessel_states, ego_idx, k_vessel_mesh=3,
                           k_vessel_vessel=6):
    """
    Builds a per-timestep HeteroData graph, always on CPU regardless of
    what device mesh_node_features/mesh_edge_index happen to already be
    on (e.g. if a prior cell reassigned them to a CUDA tensor) -- data
    should be built on CPU and moved to GPU only right before the model
    needs it, so a stray device on an upstream variable can't silently
    produce a graph with mixed-device tensors.
    """
    data = HeteroData()
    data['mesh'].x = torch.as_tensor(mesh_node_features, dtype=torch.float, device='cpu')
    data['mesh', 'to', 'mesh'].edge_index = torch.as_tensor(mesh_edge_index, dtype=torch.long, device='cpu')

    data['vessel'].x = torch.as_tensor(vessel_states, dtype=torch.float, device='cpu')
    data['vessel'].ego_mask = torch.zeros(len(vessel_states), dtype=torch.bool, device='cpu')
    data['vessel'].ego_mask[ego_idx] = True

    vessel_xy = vessel_states[:, :2]
    mesh_xy = mesh_node_features[:, :2]
    v2m = assign_vessels_to_mesh(vessel_xy, mesh_xy, k=k_vessel_mesh)
    data['vessel', 'near', 'mesh'].edge_index = v2m
    data['mesh', 'rev_near', 'vessel'].edge_index = v2m.flip(0)

    if len(vessel_states) > 1:
        v2v = knn_graph_manual(vessel_xy, k=k_vessel_vessel)
        data['vessel', 'near', 'vessel'].edge_index = v2v
    else:
        data['vessel', 'near', 'vessel'].edge_index = torch.empty((2, 0), dtype=torch.long, device='cpu')

    return data


class VesselSequenceDataset(torch.utils.data.Dataset):
    """
    Wraps a list of per-timestep HeteroData snapshots into (seq_len ->
    future_len) training windows for a single ego vessel track.
    """
    def __init__(self, snapshots, ego_idx_per_step, seq_len=4, future_len=4):
        self.snapshots = snapshots
        self.ego_idx_per_step = ego_idx_per_step
        self.seq_len = seq_len
        self.future_len = future_len
        self.valid_starts = list(range(len(snapshots) - seq_len - future_len + 1))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        ctx = self.snapshots[start:start + self.seq_len]
        fut_start = start + self.seq_len
        future_positions = []
        for t in range(fut_start, fut_start + self.future_len):
            ego_i = self.ego_idx_per_step[t]
            future_positions.append(self.snapshots[t]['vessel'].x[ego_i, :2])
        target = torch.stack(future_positions, dim=0)
        return ctx, target
