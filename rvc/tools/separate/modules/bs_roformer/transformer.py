import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .attend import Attend


def qkv_to_bnhd(qkv, heads):
    return qkv.reshape(qkv.shape[0], qkv.shape[1], 3, heads, qkv.shape[-1] // (3 * heads)).unbind(dim=2)


class RMSNorm(Module):
    def __init__(self, dim, scale=True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else 1

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = RMSNorm(dim)
        self.linear1 = nn.Linear(dim, inner_dim, bias=False)
        self.linear2 = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def apply_rotary_emb_fast(cos, sin, t):
    if t.is_cuda and t.dtype == torch.float16:
        rot = torch.complex(cos[..., ::2], sin[..., ::2])
        rotated = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2)) * rot
        return torch.view_as_real(rotated).reshape_as(t)

    cos, sin, t_even, t_odd = cos[..., ::2], sin[..., ::2], t[..., ::2], t[..., 1::2]
    out = torch.empty_like(t)
    out[..., ::2] = t_even * cos - t_odd * sin
    out[..., 1::2] = t_odd * cos + t_even * sin
    return out


def cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype):
    cache = getattr(rotary_embed, "_pymss_cos_sin_cache", None)
    if cache is None:
        rotary_embed._pymss_cos_sin_cache = cache = {}

    key = (seq_len, device.type, device.index, dtype)
    cached = cache.get(key)
    if cached is not None:
        return cached

    freqs = rotary_embed.forward(
        lambda: rotary_embed.get_seq_pos(seq_len, device=device, dtype=dtype, offset=0), cache_key=f"freqs:{seq_len}|offset:0"
    )[None, :, None, :].to(device=device, dtype=dtype)
    cached = (freqs.cos(), freqs.sin())
    cache[key] = cached
    return cached


def rotate_qk_fast_bnhd(rotary_embed, q, k):
    cos, sin = cached_rotary_cos_sin(rotary_embed, q.shape[1], q.device, q.dtype)
    return apply_rotary_emb_fast(cos, sin, q), apply_rotary_emb_fast(cos, sin, k)


class Attention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.0,
        flash=False,
        rotary_embed=None,
        shared_qkv_bias=None,
        shared_out_bias=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.flash = flash
        self.rotary_embed = rotary_embed

        if shared_qkv_bias is not None:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            self.to_qkv.bias = shared_qkv_bias
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if shared_out_bias is not None:
            self.to_out = nn.Linear(inner_dim, dim, bias=False)
            self.to_out.bias = shared_out_bias
        else:
            self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.to_gates = nn.Linear(dim, heads, bias=False)
        self.norm = RMSNorm(dim)

    def _attention(self, q, k, v):
        dropout_p = self.dropout if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    def forward(self, x):
        x = self.norm(x)
        q, k, v = qkv_to_bnhd(self.to_qkv(x), self.heads)

        if self.rotary_embed is not None:
            q, k = rotate_qk_fast_bnhd(self.rotary_embed, q, k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = self._attention(q, k, v)
        return self.to_out((out.transpose(1, 2) * self.to_gates(x).unsqueeze(-1).sigmoid()).flatten(start_dim=-2))


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        shared_qkv_bias=None,
        shared_out_bias=None,
    ):
        super().__init__()
        self.layers = ModuleList(
            [
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            shared_qkv_bias=shared_qkv_bias,
                            shared_out_bias=shared_out_bias,
                            rotary_embed=rotary_embed,
                            flash=flash_attn,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
