import math
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

from cached_attention import TimeAwareTemporalTransformer, KVCache
from coastline import DMA_BOUNDS
from irregular_ingest import VESSEL_FEATURE_DIM


class MeshVesselGNN(nn.Module):
    """
    Same heterogeneous GNN as the fixed-interval branch, with lon/lat and
    SOG normalized internally (stored tensors keep raw real-world values,
    since k-NN construction, target computation and plotting all read
    columns 0-1 as real coordinates).
    """
    def __init__(self, mesh_in=4, vessel_in=VESSEL_FEATURE_DIM, hidden=128,
                 bounds=DMA_BOUNDS, max_sog_knots=30.0):
        super().__init__()
        self.mesh_proj = nn.Linear(mesh_in, hidden)
        self.vessel_proj = nn.Linear(vessel_in, hidden)

        min_lon, min_lat, max_lon, max_lat = bounds
        self.register_buffer('lon_center', torch.tensor((min_lon + max_lon) / 2, dtype=torch.float))
        self.register_buffer('lon_scale', torch.tensor((max_lon - min_lon) / 2, dtype=torch.float))
        self.register_buffer('lat_center', torch.tensor((min_lat + max_lat) / 2, dtype=torch.float))
        self.register_buffer('lat_scale', torch.tensor((max_lat - min_lat) / 2, dtype=torch.float))
        self.max_sog_knots = max_sog_knots

        spec = lambda: {
            ('vessel', 'near', 'mesh'): SAGEConv((hidden, hidden), hidden),
            ('mesh', 'rev_near', 'vessel'): SAGEConv((hidden, hidden), hidden),
            ('mesh', 'to', 'mesh'): SAGEConv(hidden, hidden),
            ('vessel', 'near', 'vessel'): SAGEConv(hidden, hidden),
        }
        self.conv1 = HeteroConv(spec(), aggr='mean')
        self.conv2 = HeteroConv(spec(), aggr='mean')

    def _norm_lonlat(self, lon, lat):
        return (lon - self.lon_center) / self.lon_scale, (lat - self.lat_center) / self.lat_scale

    def forward(self, data):
        mx = data['mesh'].x
        lon_n, lat_n = self._norm_lonlat(mx[:, 0:1], mx[:, 1:2])
        mesh_x = torch.cat([lon_n, lat_n, mx[:, 2:]], dim=-1)

        vx = data['vessel'].x
        vlon_n, vlat_n = self._norm_lonlat(vx[:, 0:1], vx[:, 1:2])
        vessel_x = torch.cat([vlon_n, vlat_n, vx[:, 2:3] / self.max_sog_knots, vx[:, 3:]], dim=-1)

        x_dict = {'mesh': torch.relu(self.mesh_proj(mesh_x)),
                  'vessel': torch.relu(self.vessel_proj(vessel_x))}
        ei = data.edge_index_dict
        x_dict = {k: torch.relu(v) for k, v in self.conv1(x_dict, ei).items()}
        x_dict = self.conv2(x_dict, ei)
        return x_dict['vessel']


class DtConditionedFGNHead(nn.Module):
    """
    FGN sampling head conditioned on each target's elapsed time.

    With irregular pings the same "next 4 observations" can span 40
    seconds or 19 minutes, so the horizon has to be an explicit input
    rather than something the model infers. Each target dt is encoded
    (log-scaled, then sinusoidally embedded) and concatenated with the
    context embedding and per-sample noise, so one forward pass produces
    a coherent trajectory across heterogeneous horizons -- and at
    inference the model can be queried at arbitrary future times, not
    just the ones that happened to appear in training.

    Output is a normalized VELOCITY residual (see train_irregular.py):
    displacement divided by dt, then by a global velocity scale. That
    makes targets dimensionless and comparable across horizons; the head
    can still widen its spread for large dt because dt is an input.
    """
    def __init__(self, hidden=128, noise_dim=32, dt_embed_dim=32,
                 dt_reference_sec=900.0):
        super().__init__()
        self.noise_dim = noise_dim
        self.dt_embed_dim = dt_embed_dim
        self.dt_reference_sec = dt_reference_sec
        self.dt_proj = nn.Sequential(
            nn.Linear(dt_embed_dim, dt_embed_dim), nn.ReLU(),
            nn.Linear(dt_embed_dim, dt_embed_dim))
        self.net = nn.Sequential(
            nn.Linear(hidden + noise_dim + dt_embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))

    def _dt_features(self, dts):
        """dts: (B, F) seconds -> (B, F, dt_embed_dim)."""
        x = torch.log1p(dts.clamp(min=0.0)) / math.log1p(self.dt_reference_sec)
        half = self.dt_embed_dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(100.0), half,
                                          device=dts.device, dtype=torch.float))
        args = x.unsqueeze(-1) * freqs
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if enc.shape[-1] < self.dt_embed_dim:
            enc = torch.cat([enc, torch.zeros(*enc.shape[:-1],
                                               self.dt_embed_dim - enc.shape[-1],
                                               device=enc.device)], dim=-1)
        return self.dt_proj(enc)

    def sample(self, context_embed, target_dts, n_samples=32):
        """
        context_embed: (B, hidden); target_dts: (B, F) seconds ahead.
        Returns (B, n_samples, F, 2) normalized velocity residuals.

        One noise draw per sample is shared across all horizons of that
        sample, so a given sample is a coherent trajectory rather than
        independent guesses per step.
        """
        B, F = target_dts.shape
        dt_feat = self._dt_features(target_dts)                       # (B, F, D)
        ctx = context_embed.unsqueeze(1).unsqueeze(1).expand(B, n_samples, F, -1)
        dtf = dt_feat.unsqueeze(1).expand(B, n_samples, F, -1)
        noise = torch.randn(B, n_samples, 1, self.noise_dim,
                             device=context_embed.device).expand(B, n_samples, F, -1)
        return self.net(torch.cat([ctx, noise, dtf], dim=-1))


class IrregularVTP(nn.Module):
    def __init__(self, mesh_in=4, vessel_in=VESSEL_FEATURE_DIM, hidden=128,
                 n_heads=4, n_layers=3, max_cache_len=256, bounds=DMA_BOUNDS,
                 max_sog_knots=30.0, noise_dim=32):
        super().__init__()
        self.gnn = MeshVesselGNN(mesh_in, vessel_in, hidden, bounds, max_sog_knots)
        self.temporal = TimeAwareTemporalTransformer(hidden, n_heads, n_layers, max_cache_len)
        self.head = DtConditionedFGNHead(hidden, noise_dim=noise_dim)
        self.hidden = hidden
        self.n_layers = n_layers

    def encode_context(self, snapshot_seq, ego_idx_per_step, ctx_times):
        """ctx_times: (T,) elapsed seconds, last entry 0.0 (the anchor)."""
        embeds = [self.gnn(d)[ego_idx_per_step[t]] for t, d in enumerate(snapshot_seq)]
        seq = torch.stack(embeds, dim=0).unsqueeze(0)             # (1, T, hidden)
        times = ctx_times.to(seq.device).unsqueeze(0)              # (1, T)
        return self.temporal(seq, times)[:, -1, :]

    def forward(self, snapshot_seq, ego_idx_per_step, ctx_times, target_dts,
                n_samples=32):
        context = self.encode_context(snapshot_seq, ego_idx_per_step, ctx_times)
        return self.head.sample(context, target_dts.to(context.device).unsqueeze(0)
                                 if target_dts.dim() == 1 else target_dts.to(context.device),
                                 n_samples=n_samples)

    def rollout_step(self, snapshot, time_sec, cache: KVCache):
        """Streaming: one newly-arrived ego ping at elapsed time_sec."""
        v = self.gnn(snapshot)
        ego_i = snapshot['vessel'].ego_mask.nonzero()[0].item()
        embed = v[ego_i].view(1, 1, self.hidden)
        t = torch.as_tensor([[float(time_sec)]], device=embed.device)
        return self.temporal.step(embed, t, cache)[:, -1, :]
