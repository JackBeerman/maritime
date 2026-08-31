"""
GC-VTP rebuild:
  1. Heterogeneous GNN pass per timestep (vessel <-> mesh, vessel <-> vessel)
  2. Cached causal transformer over the resulting per-timestep ego embedding
     sequence — full-window pass for training, incremental .step() for
     autoregressive rollout / live AIS ingestion.
  3. FGN-style decoder: samples multiple plausible future trajectories from
     a shared conditioned generator instead of one point estimate.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from cached_attention import CachedTemporalTransformer, KVCache


class MeshVesselGNN(nn.Module):
    def __init__(self, mesh_in=4, vessel_in=8, hidden=128):
        super().__init__()
        self.mesh_proj = nn.Linear(mesh_in, hidden)
        self.vessel_proj = nn.Linear(vessel_in, hidden)
        spec = lambda: {
            ('vessel', 'near', 'mesh'): SAGEConv((hidden, hidden), hidden),
            ('mesh', 'rev_near', 'vessel'): SAGEConv((hidden, hidden), hidden),
            ('mesh', 'to', 'mesh'): SAGEConv(hidden, hidden),
            ('vessel', 'near', 'vessel'): SAGEConv(hidden, hidden),
        }
        self.conv1 = HeteroConv(spec(), aggr='mean')
        self.conv2 = HeteroConv(spec(), aggr='mean')

    def forward(self, data):
        x_dict = {
            'mesh': torch.relu(self.mesh_proj(data['mesh'].x)),
            'vessel': torch.relu(self.vessel_proj(data['vessel'].x)),
        }
        edge_index_dict = data.edge_index_dict
        x_dict = {k: torch.relu(v) for k, v in self.conv1(x_dict, edge_index_dict).items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict['vessel']


class GoalEncoder(nn.Module):
    """Goal token with dropout so naval/no-filed-destination traffic is representable."""
    def __init__(self, hidden=128, dropout_p=0.3):
        super().__init__()
        self.proj = nn.Linear(2, hidden)
        self.null_token = nn.Parameter(torch.randn(hidden) * 0.02)
        self.dropout_p = dropout_p

    def forward(self, goal_xy, training=True):
        has_goal = ~torch.isnan(goal_xy[:, 0])
        goal_xy_safe = torch.nan_to_num(goal_xy, nan=0.0)
        embed = self.proj(goal_xy_safe)
        embed = torch.where(has_goal.unsqueeze(-1), embed, self.null_token.expand_as(embed))
        if training:
            drop_mask = (torch.rand(embed.shape[0], device=embed.device) < self.dropout_p)
            embed = torch.where(drop_mask.unsqueeze(-1), self.null_token.expand_as(embed), embed)
        return embed


class FGNDecoderHead(nn.Module):
    """
    Functional-generative-style head: conditions a small generator on
    (context embedding, goal embedding, noise) so many plausible trajectory
    samples can be drawn cheaply per forward pass. Trained with the energy
    score loss (losses.py) rather than single-point regression, which is
    what the prior direct decoder got stuck on.
    """
    def __init__(self, hidden=128, future_len=4, noise_dim=32):
        super().__init__()
        self.future_len = future_len
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(hidden * 2 + noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, future_len * 2),
        )

    def sample(self, context_embed, goal_embed, n_samples=32):
        B = context_embed.shape[0]
        device = context_embed.device
        cond = torch.cat([context_embed, goal_embed], dim=-1)
        cond = cond.unsqueeze(1).expand(-1, n_samples, -1)
        noise = torch.randn(B, n_samples, self.noise_dim, device=device)
        inp = torch.cat([cond, noise], dim=-1)
        out = self.net(inp)
        return out.view(B, n_samples, self.future_len, 2)


class GCVTP(nn.Module):
    def __init__(self, mesh_in=4, vessel_in=8, hidden=128, future_len=4,
                 n_heads=4, n_layers=3, max_cache_len=256):
        super().__init__()
        self.gnn = MeshVesselGNN(mesh_in, vessel_in, hidden)
        self.temporal = CachedTemporalTransformer(hidden, n_heads, n_layers, max_cache_len)
        self.goal_encoder = GoalEncoder(hidden)
        self.head = FGNDecoderHead(hidden, future_len)
        self.hidden = hidden
        self.n_layers = n_layers

    def encode_context(self, snapshot_seq, ego_idx_per_step):
        """Full-window pass (training): returns (1, hidden) summary embedding."""
        ego_embeds = []
        for t, data in enumerate(snapshot_seq):
            v_embed = self.gnn(data)
            ego_embeds.append(v_embed[ego_idx_per_step[t]])
        ego_seq = torch.stack(ego_embeds, dim=0).unsqueeze(0)  # (1, T, hidden)
        out_seq = self.temporal(ego_seq)
        return out_seq[:, -1, :]

    def forward(self, snapshot_seq, ego_idx_per_step, goal_xy, n_samples=32, training=True):
        context = self.encode_context(snapshot_seq, ego_idx_per_step)
        goal_embed = self.goal_encoder(goal_xy, training=training)
        return self.head.sample(context, goal_embed, n_samples=n_samples)

    def rollout_step(self, snapshot, cache: KVCache, position: int):
        """
        Incremental inference: one new timestep's snapshot -> GNN -> cached
        transformer step. O(1) attention cost against the growing cache
        instead of recomputing over the full history — use this for
        autoregressive multi-step forecasting or live AIS ingestion.
        """
        v_embed = self.gnn(snapshot)
        ego_i = snapshot['vessel'].ego_mask.nonzero()[0].item()
        new_embed = v_embed[ego_i].view(1, 1, self.hidden)
        out = self.temporal.step(new_embed, cache, position=position)
        return out[:, -1, :]
