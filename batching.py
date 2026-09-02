"""
Batched window encoding for the irregular-sampling branch.

Same motivation as the fixed-interval branch: at batch_size=1 the GPU is
mostly idle on Python and kernel-launch overhead, which made 100k-window
epochs unusable.

One important difference: **snapshot deduplication does not apply here.**
Snapshots are anchored to each ego vessel's own ping times, with
neighbours resolved at those exact instants, so two windows essentially
never reference the same graph object. The saving is purely from graph
batching (amortized launch overhead), not from reuse -- expect a smaller
speedup than the fixed-interval branch gets.

Windows here also carry per-window timing (`ctx_times`, `target_dts`),
which must be stacked into batch tensors so the time-aware transformer
and the dt-conditioned head see the right values per row.
"""
import torch
from torch_geometric.data import Batch


def collate_windows(batch):
    """Keep windows as a plain list; graph batching happens in the encoder."""
    return batch


def encode_windows_batched(model, windows, gnn_chunk_size=16):
    """
    windows: list of (ctx, ctx_times, target_positions, target_dts).
    Returns (B, hidden) context embeddings, matching what the unbatched
    model.encode_context() produces per window.
    """
    B = len(windows)
    T = len(windows[0][0])

    flat, ego_rows = [], []
    for ctx, _, _, _ in windows:
        for d in ctx:
            flat.append(d)
            ego_rows.append(int(d['vessel'].ego_mask.nonzero()[0].item()))

    embeds = []
    for i in range(0, len(flat), gnn_chunk_size):
        chunk = flat[i:i + gnn_chunk_size]
        if len(chunk) == 1:
            embeds.append(model.gnn(chunk[0]))
            continue
        b = Batch.from_data_list(chunk)
        out = model.gnn(b)
        ptr = b['vessel'].ptr
        for j in range(len(chunk)):
            embeds.append(out[ptr[j]:ptr[j + 1]])

    ego_embeds = torch.stack([embeds[k][ego_rows[k]] for k in range(len(flat))])
    seq = ego_embeds.view(B, T, -1)

    device = seq.device
    times = torch.stack([w[1].to(device) for w in windows])          # (B, T)
    return model.temporal(seq, times)[:, -1, :]


def batch_timing_and_targets(windows, device):
    """
    anchors (B, 2), target positions (B, F, 2), target dts (B, F).
    Anchors come from each window's last context snapshot, read at that
    window's own ego row.
    """
    anchors, targets, dts = [], [], []
    for ctx, _, tpos, tdts in windows:
        ego = int(ctx[-1]['vessel'].ego_mask.nonzero()[0].item())
        anchors.append(ctx[-1]['vessel'].x[ego, :2])
        targets.append(tpos.to(device))
        dts.append(tdts.to(device))
    return torch.stack(anchors), torch.stack(targets), torch.stack(dts)