"""The three pieces every Transformer layer is made of."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def attention(q, k, v):
    """Causal attention written out by hand. Only used as the reference the tests compare against;
    the model itself calls F.scaled_dot_product_attention, which is much faster."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    n = q.shape[-2]
    future = torch.triu(torch.ones(n, k.shape[-2], dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class RoPE(nn.Module):
    """Rotary position embedding: rotates each pair (x_2k, x_2k+1) at position i
    by the angle i / theta ** (2k / d_head)."""

    def __init__(self, theta, d_head, max_len):
        super().__init__()
        freqs = theta ** (-torch.arange(0, d_head, 2).float() / d_head)
        angles = torch.arange(max_len).float()[:, None] * freqs[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        n = x.shape[-2]
        cos, sin = self.cos[:n], self.sin[:n]
        even, odd = x[..., 0::2], x[..., 1::2]
        out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention. One RoPE module is shared by every layer, so it is
    passed in rather than built here."""

    def __init__(self, d_model, num_heads, rope):
        super().__init__()
        self.num_heads = num_heads
        self.rope = rope
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        split = lambda t: rearrange(t, "... n (h d) -> ... h n d", h=self.num_heads)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        q, k = self.rope(q), self.rope(k)  # positions go into the queries and keys only
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(rearrange(y, "... h n d -> ... n (h d)"))


class FeedForward(nn.Module):
    """SwiGLU: W2 (SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
