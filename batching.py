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

Snapshots are kept on CPU and only the current chunk is moved to the
GPU. That matters for two reasons:

  * **Memory.** Every snapshot carries its own mesh copy (~348 KB).
    Holding a whole dataset on the GPU meant tens of GB of duplicated
    mesh -- 50 GB on the irregular branch, which simply OOM'd. With
    per-chunk transfer the GPU holds a few MB of mesh at a time, so
    dataset size no longer bounds what fits.
  * **Safety.** `Batch.from_data_list(...)` returns a NEW object, so
    moving it leaves the CPU originals untouched. `HeteroData.to()`
    mutates in place, and mutating shared snapshots mid-epoch was the
    source of an earlier class of corruption.

Transfer cost is small next to the GNN forward/backward (a 32-graph
chunk is ~30 MB, a few milliseconds), so `gnn_chunk_size` now bounds GPU
memory almost entirely. Lower it first if you hit OOM.
"""
import torch
from torch_geometric.data import Batch


def collate_windows(batch):
    """Keep windows as a plain list; graph batching happens in the encoder."""
    return batch


def _model_device(model):
    return next(model.parameters()).device


def encode_windows_batched(model, windows, gnn_chunk_size=32, device=None):
    """
    windows: list of (ctx, ego_rows, target) from VesselSequenceDataset,
    holding CPU snapshots. Returns (B, hidden) context embeddings on the
    model's device -- the same value the unbatched encode_context()
    produces per window.
    """
    device = device or _model_device(model)
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
        # collate on CPU, then move the (new) batch object to the GPU --
        # the source snapshots stay on CPU and are never mutated
        b = Batch.from_data_list(chunk).to(device)
        out = model.gnn(b)
        ptr = b['vessel'].ptr.cpu()
        for j in range(len(chunk)):
            embeds.append(out[int(ptr[j]):int(ptr[j + 1])])

    ego_embeds = torch.stack([embeds[ref[k]][ego_rows_flat[k]]
                               for k in range(len(ref))])
    seq = ego_embeds.view(B, T, -1)
    return model.temporal(seq)[:, -1, :]


def batch_anchors_and_targets(windows, device=None):
    """
    Anchor positions (B, 2) and target positions (B, future_len, 2).
    Read from CPU snapshots and moved once, rather than per window.
    """
    anchors = torch.stack([w[0][-1]['vessel'].x[int(w[1][-1]), :2] for w in windows])
    targets = torch.stack([w[2] for w in windows])
    if device is not None:
        anchors, targets = anchors.to(device), targets.to(device)
    return anchors, targets


def dedup_ratio(windows):
    """Distinct snapshots / total references -- useful for logging how
    much work deduplication is actually saving at a given batch size."""
    total = sum(len(w[0]) for w in windows)
    uniq = len({id(d) for w in windows for d in w[0]})
    return uniq, total