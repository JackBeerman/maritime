"""
Multi-step inference: warms up the KV cache over an observed context window,
then rolls forward using the FGN head's samples to condition each subsequent
step. Since we don't have real future AIS pings during rollout, each
sampled future position is fed back in as if it were the next observation,
independently per sample thread.
"""
import numpy as np
import torch
from cached_attention import KVCache
from graph_data import build_hetero_snapshot


@torch.no_grad()
def warm_up_cache(model, context_snapshots, ego_idx_per_step):
    cache = KVCache(model.n_layers)
    context_embed = None
    for t, snapshot in enumerate(context_snapshots):
        context_embed = model.rollout_step(snapshot, cache, position=t)
    return cache, context_embed, len(context_snapshots)


@torch.no_grad()
def forecast_multi_step(model, context_snapshots, ego_idx_per_step,
                         mesh_node_features, mesh_edge_index,
                         other_vessel_states_future, goal_xy,
                         n_future_steps=4, n_samples=16):
    """
    other_vessel_states_future: list (length n_future_steps - 1) of (V-1, F)
        arrays -- other vessels' states at each future step, coming from
        their own separate tracks/forecasts, since GC-VTP predicts one ego
        vessel at a time conditioned on everyone else's positions.
    goal_xy: (1, 2) tensor, or a NaN row if unknown.

    Returns: (n_samples, n_future_steps, 2) -- each sample is an internally
    consistent rollout thread (step t and t+1 come from the same thread,
    not independently resampled each step).
    """
    cache, context_embed, next_pos = warm_up_cache(model, context_snapshots, ego_idx_per_step)
    goal_embed = model.goal_encoder(goal_xy, training=False)

    first_step_samples = model.head.sample(context_embed, goal_embed, n_samples=n_samples)
    ego_next_per_sample = first_step_samples[0, :, 0, :]  # (n_samples, 2)
    all_steps = [ego_next_per_sample]

    # Each sample thread's hypothesized ego position diverges from the
    # others, so each needs its own cache going forward.
    per_sample_caches = [_clone_cache(cache) for _ in range(n_samples)]
    last_ego_xy = ego_next_per_sample

    for step in range(1, n_future_steps):
        other_states = other_vessel_states_future[step - 1]
        next_positions = []
        for s in range(n_samples):
            vessel_states = _assemble_vessel_states(last_ego_xy[s], other_states)
            snapshot = build_hetero_snapshot(
                mesh_node_features, mesh_edge_index, vessel_states, ego_idx=0
            )
            ctx_embed = model.rollout_step(snapshot, per_sample_caches[s], position=next_pos + step - 1)
            sample_out = model.head.sample(ctx_embed, goal_embed, n_samples=1)
            next_positions.append(sample_out[0, 0, 0, :])
        last_ego_xy = torch.stack(next_positions, dim=0)
        all_steps.append(last_ego_xy)

    return torch.stack(all_steps, dim=1)  # (n_samples, n_future_steps, 2)


def _assemble_vessel_states(ego_xy, other_states):
    ego_row = np.zeros((1, other_states.shape[1]))
    ego_row[0, 0] = ego_xy[0].item()
    ego_row[0, 1] = ego_xy[1].item()
    return np.concatenate([ego_row, other_states], axis=0)


def _clone_cache(cache):
    new_cache = KVCache(len(cache.k))
    new_cache.k = [k.clone() if k is not None else None for k in cache.k]
    new_cache.v = [v.clone() if v is not None else None for v in cache.v]
    return new_cache
