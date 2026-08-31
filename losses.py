"""
Energy score loss: trains the FGN-style sampler to match the true future
trajectory distribution without collapsing to a single point (the failure
mode the direct decoder hit before).
"""
import torch


def energy_score_loss(samples, target, beta=1.0):
    """
    samples: (B, S, future_len, 2)
    target:  (B, future_len, 2)
    """
    B, S, T, _ = samples.shape
    target_exp = target.unsqueeze(1)  # (B, 1, T, 2)

    # term 1: E||X - y||  (accuracy)
    diff_target = torch.norm(samples - target_exp, dim=-1) ** beta  # (B, S, T)
    term1 = diff_target.mean(dim=1).sum(dim=-1)  # (B,)

    # term 2: 0.5 * E||X - X'||  (spread / diversity penalty)
    s1 = samples.unsqueeze(2)  # (B, S, 1, T, 2)
    s2 = samples.unsqueeze(1)  # (B, 1, S, T, 2)
    pairwise = torch.norm(s1 - s2, dim=-1) ** beta  # (B, S, S, T)
    term2 = 0.5 * pairwise.mean(dim=(1, 2)).sum(dim=-1)  # (B,)

    return (term1 - term2).mean()
