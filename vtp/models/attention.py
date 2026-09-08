import math
import torch
import torch.nn as nn


def continuous_time_encoding(t_sec, hidden, min_period=1.0, max_period=86400.0):
    """
    Sinusoidal encoding of REAL elapsed time rather than integer step
    index.

    The fixed-interval branch used a learned embedding looked up by step
    number, which silently asserts that every step is the same duration
    apart. With irregular pings that is false -- consecutive observations
    can be 10 seconds or 20 minutes apart -- so position is encoded as a
    continuous function of seconds, letting attention distinguish a
    tightly-sampled burst from a sparse one.

    t_sec: (B, T) elapsed seconds (typically <= 0, relative to the
           window's anchor ping). Returns (B, T, hidden).
    """
    half = hidden // 2
    freqs = torch.exp(torch.linspace(
        math.log(1.0 / min_period), math.log(1.0 / max_period), half,
        device=t_sec.device, dtype=torch.float))
    args = t_sec.unsqueeze(-1).float() * freqs * (2 * math.pi)
    enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if enc.shape[-1] < hidden:  # odd hidden size
        enc = torch.cat([enc, torch.zeros(*enc.shape[:-1], hidden - enc.shape[-1],
                                           device=enc.device)], dim=-1)
    return enc


class KVCache:
    """Growing per-layer K/V cache for incremental rollout."""
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def update(self, layer_idx, k_new, v_new):
        # concat along the SEQUENCE dim (2), not heads (1) -- getting this
        # wrong silently corrupts attention rather than erroring
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k_new, v_new
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)
        return self.k[layer_idx], self.v[layer_idx]

    def trim(self, max_len):
        for i in range(len(self.k)):
            if self.k[i] is not None and self.k[i].shape[2] > max_len:
                self.k[i] = self.k[i][:, :, -max_len:]
                self.v[i] = self.v[i][:, :, -max_len:]


class CachedCausalAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        assert hidden % n_heads == 0
        self.h = n_heads
        self.d = hidden // n_heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, cache: KVCache = None, layer_idx: int = None):
        B, T_new, H = x.shape
        qkv = self.qkv(x).view(B, T_new, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]

        if cache is not None:
            k, v = cache.update(layer_idx, k_new, v_new)
        else:
            k, v = k_new, v_new

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        if cache is None:
            T_total = k.shape[2]
            mask = torch.triu(torch.ones(T_new, T_total, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T_new, H)
        return self.out(out)


class CachedTransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.attn = CachedCausalAttention(hidden, n_heads)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.ReLU(), nn.Linear(hidden * 4, hidden))

    def forward(self, x, cache=None, layer_idx=None):
        x = x + self.attn(self.norm1(x), cache=cache, layer_idx=layer_idx)
        x = x + self.ff(self.norm2(x))
        return x


class TimeAwareTemporalTransformer(nn.Module):
    """
    Causal transformer over irregularly-spaced timesteps.

    forward(seq, times_sec) -- full-window pass for training
    step(embed, time_sec, cache) -- incremental pass for streaming rollout

    Both take explicit elapsed times instead of assuming uniform spacing.
    """
    def __init__(self, hidden=128, n_heads=4, n_layers=3, max_cache_len=256,
                 min_period=1.0, max_period=86400.0):
        super().__init__()
        self.layers = nn.ModuleList([CachedTransformerBlock(hidden, n_heads)
                                       for _ in range(n_layers)])
        self.hidden = hidden
        self.max_cache_len = max_cache_len
        self.min_period = min_period
        self.max_period = max_period
        self.time_proj = nn.Linear(hidden, hidden)

    def _time_embed(self, t_sec):
        enc = continuous_time_encoding(t_sec, self.hidden,
                                        self.min_period, self.max_period)
        return self.time_proj(enc)

    def forward(self, seq, times_sec):
        """seq: (B, T, hidden); times_sec: (B, T) elapsed seconds."""
        x = seq + self._time_embed(times_sec)
        for layer in self.layers:
            x = layer(x, cache=None)
        return x

    def step(self, new_embed, time_sec, cache: KVCache):
        """
        new_embed: (B, 1, hidden); time_sec: (B, 1) elapsed seconds for
        this observation. No integer position needed -- timing comes from
        the value itself, which is what makes streaming irregular pings
        work.
        """
        x = new_embed + self._time_embed(time_sec)
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache, layer_idx=i)
        cache.trim(self.max_cache_len)
        return x
