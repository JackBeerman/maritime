import torch
import torch.nn as nn
import math


class KVCache:
    """Holds growing K/V tensors per layer across a rollout."""
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def update(self, layer_idx, k_new, v_new):
        # k_new/v_new: (B, heads, T_new, d) — concat along the SEQUENCE dim (2),
        # not the heads dim. Concatenating on dim=1 silently corrupts attention
        # instead of erroring, so this is worth getting right.
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k_new, v_new
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)
        return self.k[layer_idx], self.v[layer_idx]

    def trim(self, max_len):
        """Delta/sliding-window variant: drop oldest entries beyond max_len
        so cache size stays bounded during long live-tracking sessions."""
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
        """
        x: (B, T_new, hidden) — only the NEW timestep(s) when a cache is
           supplied. Without a cache, behaves like full causal self-attention
           over x (used during training on fixed windows).
        """
        B, T_new, H = x.shape
        qkv = self.qkv(x).view(B, T_new, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]  # (B, heads, T_new, d)

        if cache is not None:
            k, v = cache.update(layer_idx, k_new, v_new)
        else:
            k, v = k_new, v_new

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        if cache is None:
            T_total = k.shape[2]
            mask = torch.triu(torch.ones(T_new, T_total, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        # with a cache, q only covers new steps attending to all past+new
        # keys — no mask needed since nothing "future" exists yet

        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T_new, H)
        return self.out(out)


class CachedTransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.attn = CachedCausalAttention(hidden, n_heads)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.ReLU(), nn.Linear(hidden * 4, hidden)
        )

    def forward(self, x, cache=None, layer_idx=None):
        x = x + self.attn(self.norm1(x), cache=cache, layer_idx=layer_idx)
        x = x + self.ff(self.norm2(x))
        return x


class CachedTemporalTransformer(nn.Module):
    """
    forward(): full-window pass for training, standard causal masking.
    step(): incremental decoding — O(1) new-token attention against the
            growing cache instead of O(T) recompute, for rollout/live AIS.
    """
    def __init__(self, hidden=128, n_heads=4, n_layers=3, max_cache_len=256):
        super().__init__()
        self.layers = nn.ModuleList(
            [CachedTransformerBlock(hidden, n_heads) for _ in range(n_layers)]
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_cache_len, hidden) * 0.02)
        self.max_cache_len = max_cache_len

    def forward(self, ego_seq):
        T = ego_seq.shape[1]
        x = ego_seq + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer(x, cache=None)
        return x  # (B, T, hidden) — caller takes [:, -1, :] for a summary

    def step(self, new_embed, cache: KVCache, position: int):
        x = new_embed + self.pos_embed[:, position:position + 1, :]
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache, layer_idx=i)
        cache.trim(self.max_cache_len)
        return x
