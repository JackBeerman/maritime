"""
Batched window encoding.

At batch_size=1 each training step is one forward/backward over a full
GNN (~5,600 mesh nodes plus thousands of vessels) for a single window,
so the GPU spends most of its time idle on Python and kernel-launch
overhead rather than arithmetic. With 100k+ windows per epoch that put
both branches past any usable epoch time.

Two savings are available here, and this module takes both:

1. **Graph batching.** Several graphs are combined with
   Batch.from_data_list and pushed through the GNN in one call,
   amortizing launch overhead across them.

2. **Snapshot deduplication (fixed-interval branch only).** Windows
   share world snapshots -- a batch of 32 windows x 12 timesteps holds
   384 snapshot references drawn from only ~1,152 distinct world states,
   so many are literally the same object. The GNN output for a snapshot
   does not depend on which vessel is "ego" (ego is just a row index),
   so each distinct snapshot is encoded ONCE per step and the ego rows
   are gathered afterwards. The larger the batch, the better this ratio
   gets.

Memory note: batching replicates the mesh once per graph in the chunk
(~348 KB each, plus activations), so `gnn_chunk_size` bounds transient
memory. It is deliberately separate from `batch_size`: the batch decides
how many windows share a gradient step, the chunk decides how many
graphs go through the GNN at once.
"""
import torch
from torch_geometric.data import Batch


def collate_windows(batch):
    """Keep windows as a plain list; graph batching happens in the encoder."""
    return batch


def encode_windows_batched(model, windows, gnn_chunk_size=32):
    """
    windows: list of (ctx, ego_rows, target) from VesselSequenceDataset.
    Returns (B, hidden) context embeddings -- the same value the
    unbatched model.encode_context() produces per window.
    """
    B = len(windows)
    T = len(windows[0][0])

    # Deduplicate by object identity. Windows from the same vessel (and
    # across vessels) reference the same shared world snapshots.
    uniq_list, uniq_index = [], {}
    ref, ego_rows_flat = [], []
    for ctx, ego_rows, _ in windows:
        for t, d in enumerate(ctx):
            key = id(d)
            if key not in uniq_index:
                uniq_index[key] = len(uniq_list)
                uniq_list.append(d)
            ref.append(uniq_index[key])
            ego_rows_flat.append(int(ego_rows[t]))

    # One GNN pass per distinct snapshot, chunked to bound memory.
    embeds = []
    for i in range(0, len(uniq_list), gnn_chunk_size):
        chunk = uniq_list[i:i + gnn_chunk_size]
        if len(chunk) == 1:
            embeds.append(model.gnn(chunk[0]))
            continue
        b = Batch.from_data_list(chunk)
        out = model.gnn(b)
        ptr = b['vessel'].ptr
        for j in range(len(chunk)):
            embeds.append(out[ptr[j]:ptr[j + 1]])

    ego_embeds = torch.stack([embeds[ref[k]][ego_rows_flat[k]]
                               for k in range(len(ref))])
    seq = ego_embeds.view(B, T, -1)
    return model.temporal(seq)[:, -1, :]


def batch_anchors_and_targets(windows):
    """
    Anchor positions (B, 2) and target positions (B, future_len, 2),
    stacked from a list of windows.
    """
    anchors = torch.stack([w[0][-1]['vessel'].x[int(w[1][-1]), :2] for w in windows])
    targets = torch.stack([w[2] for w in windows])
    return anchors, targets


def dedup_ratio(windows):
    """Distinct snapshots / total references -- useful for logging how
    much work deduplication is actually saving at a given batch size."""
    total = sum(len(w[0]) for w in windows)
    uniq = len({id(d) for w in windows for d in w[0]})
    return uniq, total