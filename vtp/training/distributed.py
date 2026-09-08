"""
Multi-GPU training via DistributedDataParallel.

Each rank owns one GPU, holds its own copy of the model, and processes a
disjoint slice of the training windows; gradients are all-reduced so
every rank stays in step. Throughput scales close to linearly with GPU
count for this workload, because the bottleneck is per-window GNN
compute rather than communication.

This is only practical because snapshots now live on CPU (see
batching.py). Each rank moves its own chunks to its own GPU, so N ranks
do NOT mean N copies of the dataset in GPU memory. It also means the
world snapshots are built once per rank on CPU -- redundant work, but
CPU RAM is plentiful here and it avoids the complexity of shared memory.

Launch with torchrun:

    torchrun --nproc_per_node=4 train.py --distributed [other args]

Notes on correctness with DDP:

* Only rank 0 prints, writes checkpoints, and runs validation. Otherwise
  four ranks race to write the same file.
* Checkpoints store the UNWRAPPED state dict (`model.module`), so they
  load fine in single-GPU inference and in viz.py.
* The DistributedSampler must be re-seeded each epoch (`set_epoch`) or
  every rank sees the same window order every epoch, silently reducing
  the effective shuffle to nothing.
* Effective batch size becomes `--batch-size x nproc_per_node`. Larger
  batches usually want a higher learning rate; the launcher below scales
  it by sqrt(world_size) unless you override.
"""
import os
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader


def is_distributed():
    return int(os.environ.get('WORLD_SIZE', 1)) > 1


def get_rank():
    return int(os.environ.get('RANK', 0))


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def is_main_process():
    return get_rank() == 0


def setup_distributed(backend='nccl'):
    """
    Initialize the process group and bind this rank to its GPU.
    Returns (device, rank, world_size). Safe to call when not launched
    under torchrun -- it falls back to single-process.
    """
    if not is_distributed():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device, 0, 1

    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        # gloo/CPU fallback so the distributed code path can be exercised
        # without GPUs (useful for testing the plumbing)
        backend = 'gloo'
        device = 'cpu'
    dist.init_process_group(backend=backend)
    return device, get_rank(), get_world_size()


def cleanup_distributed():
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model, device):
    """DDP-wrap when distributed, otherwise return the model unchanged."""
    if not is_distributed():
        return model
    if torch.cuda.is_available():
        return DDP(model, device_ids=[get_local_rank()],
                   output_device=get_local_rank(), find_unused_parameters=False)
    return DDP(model, find_unused_parameters=False)


def unwrap(model):
    """The underlying module, for checkpointing and for reaching .gnn/.head."""
    return model.module if hasattr(model, 'module') else model


def make_loader(dataset, batch_size, collate_fn, shuffle=True, seed=0):
    """
    DataLoader that shards across ranks when distributed.
    Returns (loader, sampler); the sampler is None in single-process, and
    must have set_epoch() called each epoch when it is not.
    """
    if not is_distributed():
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_fn), None
    sampler = DistributedSampler(dataset, num_replicas=get_world_size(),
                                  rank=get_rank(), shuffle=shuffle, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        collate_fn=collate_fn)
    return loader, sampler


def scaled_lr(base_lr, world_size, rule='sqrt'):
    """
    Adjust learning rate for the larger effective batch.

    Effective batch is base_batch x world_size. Linear scaling is the
    common recipe for image classification; sqrt is gentler and tends to
    be safer for small batches and noisy objectives, which is the case
    here (energy score over sampled trajectories). Pass rule='none' to
    keep the base rate.
    """
    if world_size <= 1 or rule == 'none':
        return base_lr
    if rule == 'linear':
        return base_lr * world_size
    return base_lr * math.sqrt(world_size)


def all_reduce_mean(value, device):
    """Average a Python float across ranks (for logging a true epoch loss)."""
    if not is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / get_world_size()).item()


def barrier():
    if is_distributed() and dist.is_initialized():
        dist.barrier()


def rank0_print(*args, **kwargs):
    """Print from rank 0 only -- otherwise every line appears N times."""
    if is_main_process():
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)