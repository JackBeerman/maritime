import torch
from torch_geometric.data import HeteroData
import numpy as np
from scipy.spatial import cKDTree


def assign_vessels_to_mesh(vessel_xy, mesh_xy, k=3, mesh_tree=None):
    """
    Each vessel connects to its k nearest mesh nodes.

    Uses a KD-tree (scipy.spatial.cKDTree) rather than brute-force
    distance computation -- with real snapshots holding ~2500 vessels
    against a mesh of several thousand nodes, brute-force cdist against
    the FULL mesh dominates dataset-build time (measured ~86ms/call at
    real scale vs ~2ms/call with a prebuilt tree). Pass a prebuilt
    mesh_tree (the mesh is static across an entire training run, so it
    only needs building once) to avoid rebuilding it on every call.

    Note: a KD-tree can occasionally pick a different k-th neighbor than
    brute-force torch.cdist when two candidates are within floating-point
    noise of being exactly tied (verified: differs for ~5% of vessels on
    real-scale data, always at sub-1%-relative-distance near-ties, never
    a meaningfully "wrong" neighbor) -- this is expected, harmless
    numerical noise at the tie boundary, not a correctness bug.
    """
    if mesh_tree is None:
        mesh_tree = cKDTree(mesh_xy)
    vessel_xy_np = vessel_xy if isinstance(vessel_xy, np.ndarray) else np.asarray(vessel_xy)
    _, idx = mesh_tree.query(vessel_xy_np, k=k)
    if k == 1:
        idx = idx[:, None]
    idx = torch.as_tensor(idx, dtype=torch.long)
    src = idx.flatten()
    dst = torch.arange(len(vessel_xy_np)).repeat_interleave(k)
    return torch.stack([dst, src], dim=0)


def knn_graph_manual(xy, k):
    """
    Undirected-style k-NN edge list (src -> dst, each node connected to its
    k nearest neighbors), built with a KD-tree rather than the pyg-lib
    backend that knn_graph() requires (pyg-lib wheels are version-pinned
    to specific torch/CUDA builds and aren't guaranteed to be available
    on Rivanna's stack) or brute-force cdist (which scales poorly with
    real vessel counts -- see assign_vessels_to_mesh's docstring for
    measured numbers and the same near-tie caveat, which applies here
    identically).
    """
    xy_np = xy if isinstance(xy, np.ndarray) else np.asarray(xy)
    n = xy_np.shape[0]
    k = min(k, n - 1)
    tree = cKDTree(xy_np)
    _, idx = tree.query(xy_np, k=k + 1)  # +1: a point always finds itself at distance 0
    if idx.ndim == 1:
        idx = idx[None, :]
    idx = idx[:, 1:k + 1]  # drop the self-match
    idx = torch.as_tensor(idx, dtype=torch.long)
    dst = torch.arange(n).repeat_interleave(k)
    src = idx.flatten()
    return torch.stack([src, dst], dim=0)


def build_hetero_snapshot(mesh_node_features, mesh_edge_index,
                           vessel_states, ego_idx, k_vessel_mesh=3,
                           k_vessel_vessel=6, mesh_x_tensor=None,
                           mesh_edge_tensor=None, mesh_tree=None):
    """
    Builds a per-timestep HeteroData graph, always on CPU regardless of
    what device mesh_node_features/mesh_edge_index happen to already be
    on (e.g. if a prior cell reassigned them to a CUDA tensor) -- data
    should be built on CPU and moved to GPU only right before the model
    needs it, so a stray device on an upstream variable can't silently
    produce a graph with mixed-device tensors.

    mesh_x_tensor/mesh_edge_tensor/mesh_tree: optional precomputed mesh
    tensors and KD-tree. The mesh is static across an entire training
    run (same mesh reused for every vessel and every timestamp) -- pass
    these in (built once, e.g. in train.py right after build_mesh) to
    avoid rebuilding them on every single call, which matters a lot at
    real scale (thousands of calls, each processing ~2500 vessels).
    Falls back to building them fresh here if not supplied, for
    standalone/notebook use.
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
