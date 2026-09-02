import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv
from cached_attention import CachedTemporalTransformer, KVCache
from coastline import DMA_BOUNDS


class MeshVesselGNN(nn.Module):
    """
    Normalizes lon/lat (centered and scaled to the domain bounds) and SOG
    (scaled by a typical max speed) internally, right before the linear
    projection into the network -- this only affects what the network
    sees, not the stored data['mesh'].x / data['vessel'].x tensors
    themselves, since k-NN graph construction, target/displacement
    computation, and plotting elsewhere in the pipeline all depend on
    those columns holding raw real-world coordinates.
    """
    def __init__(self, mesh_in=4, vessel_in=12, hidden=128, bounds=DMA_BOUNDS, max_sog_knots=30.0):
        super().__init__()
        self.mesh_proj = nn.Linear(mesh_in, hidden)
        self.vessel_proj = nn.Linear(vessel_in, hidden)

        min_lon, min_lat, max_lon, max_lat = bounds
        self.register_buffer('lon_center', torch.tensor((min_lon + max_lon) / 2, dtype=torch.float))
        self.register_buffer('lon_scale', torch.tensor((max_lon - min_lon) / 2, dtype=torch.float))
        self.register_buffer('lat_center', torch.tensor((min_lat + max_lat) / 2, dtype=torch.float))
        self.register_buffer('lat_scale', torch.tensor((max_lat - min_lat) / 2, dtype=torch.float))
        self.max_sog_knots = max_sog_knots

        conv_spec = {
            ('vessel', 'near', 'mesh'): SAGEConv((hidden, hidden), hidden),
            ('mesh', 'rev_near', 'vessel'): SAGEConv((hidden, hidden), hidden),
            ('mesh', 'to', 'mesh'): SAGEConv(hidden, hidden),
            ('vessel', 'near', 'vessel'): SAGEConv(hidden, hidden),
        }
        self.conv1 = HeteroConv(conv_spec, aggr='mean')
        conv_spec2 = {
            ('vessel', 'near', 'mesh'): SAGEConv((hidden, hidden), hidden),
            ('mesh', 'rev_near', 'vessel'): SAGEConv((hidden, hidden), hidden),
            ('mesh', 'to', 'mesh'): SAGEConv(hidden, hidden),
            ('vessel', 'near', 'vessel'): SAGEConv(hidden, hidden),
        }
        self.conv2 = HeteroConv(conv_spec2, aggr='mean')

    def _normalize_lonlat(self, lon, lat):
        return (lon - self.lon_center) / self.lon_scale, (lat - self.lat_center) / self.lat_scale

    def _normalize_mesh_features(self, x):
        lon_n, lat_n = self._normalize_lonlat(x[:, 0:1], x[:, 1:2])
        return torch.cat([lon_n, lat_n, x[:, 2:]], dim=-1)

    def _normalize_vessel_features(self, x):
        lon_n, lat_n = self._normalize_lonlat(x[:, 0:1], x[:, 1:2])
        sog_n = x[:, 2:3] / self.max_sog_knots
        return torch.cat([lon_n, lat_n, sog_n, x[:, 3:]], dim=-1)

    def forward(self, data):
        mesh_x = self._normalize_mesh_features(data['mesh'].x)
        vessel_x = self._normalize_vessel_features(data['vessel'].x)
        x_dict = {
            'mesh': torch.relu(self.mesh_proj(mesh_x)),
            'vessel': torch.relu(self.vessel_proj(vessel_x)),
        }
        edge_index_dict = data.edge_index_dict
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict['vessel']


class FGNDecoderHead(nn.Module):
    """
    Functional-generative-style head: conditions a small generator on
    (context embedding, noise) so many plausible trajectory samples can
    be drawn cheaply per forward pass. Trained with the energy score loss
    (losses.py) rather than single-point regression.
    """
    def __init__(self, hidden=128, future_len=4, noise_dim=32):
        super().__init__()
        self.future_len = future_len
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(hidden + noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, future_len * 2),
        )

    def sample(self, context_embed, n_samples=32):
        B = context_embed.shape[0]
        device = context_embed.device
        cond = context_embed.unsqueeze(1).expand(-1, n_samples, -1)
        noise = torch.randn(B, n_samples, self.noise_dim, device=device)
        inp = torch.cat([cond, noise], dim=-1)
        out = self.net(inp)
        return out.view(B, n_samples, self.future_len, 2)


class GCVTP(nn.Module):
    def __init__(self, mesh_in=4, vessel_in=12, hidden=128, future_len=4,
                 n_heads=4, n_layers=3, max_cache_len=256, bounds=DMA_BOUNDS,
                 max_sog_knots=30.0):
        super().__init__()
        self.gnn = MeshVesselGNN(mesh_in, vessel_in, hidden, bounds=bounds, max_sog_knots=max_sog_knots)
        self.temporal = CachedTemporalTransformer(hidden, n_heads, n_layers, max_cache_len)
        self.head = FGNDecoderHead(hidden, future_len)
        self.hidden = hidden
        self.n_layers = n_layers

    def encode_context(self, snapshot_seq, ego_idx_per_step):
        """Full-window pass (training): returns (B=1, hidden) summary embedding."""
        ego_embeds = []
        for t, data in enumerate(snapshot_seq):
            v_embed = self.gnn(data)
            ego_embeds.append(v_embed[ego_idx_per_step[t]])
        ego_seq = torch.stack(ego_embeds, dim=0).unsqueeze(0)  # (1, T, hidden)
        out_seq = self.temporal(ego_seq)  # (1, T, hidden)
        return out_seq[:, -1, :]  # last-step summary

    def forward(self, snapshot_seq, ego_idx_per_step, n_samples=32, training=True):
        context = self.encode_context(snapshot_seq, ego_idx_per_step)
        samples = self.head.sample(context, n_samples=n_samples)
        return samples

    def rollout_step(self, snapshot, cache: KVCache, position: int, ego_idx=None):
        """
        Incremental inference: one new timestep's snapshot -> GNN -> cached
        transformer step.

        ego_idx must be supplied because shared world snapshots carry no
        ego_mask -- the same graph serves every ego vessel, so which row
        is 'us' is caller state, not a property of the graph. Falls back
        to ego_mask if present, for compatibility with older snapshots.
        """
        v_embed = self.gnn(snapshot)
        if ego_idx is None:
            ego_idx = snapshot['vessel'].ego_mask.nonzero()[0].item()
        new_embed = v_embed[ego_idx].view(1, 1, self.hidden)
        out = self.temporal.step(new_embed, cache, position=position)
        return out[:, -1, :]