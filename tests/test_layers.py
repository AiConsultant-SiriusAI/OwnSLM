import math

import torch
import torch.nn.functional as F

from minilm.model.layers import CausalSelfAttention, FeedForward, RoPE, attention

CFG = dict(d_model=64, num_heads=4)


def test_hand_written_attention_matches_pytorch_3d():
    q, k, v = (torch.randn(3, 7, 16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(attention(q, k, v), ref, atol=1e-5)


def test_hand_written_attention_matches_pytorch_4d():
    q, k, v = (torch.randn(2, 4, 9, 16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(attention(q, k, v), ref, atol=1e-5)


def slow_rope(x, theta, d_head):
    """Builds the 2x2 rotation for every pair explicitly, the obvious slow way."""
    out = torch.empty_like(x)
    for i in range(x.shape[-2]):
        for k in range(d_head // 2):
            a = i / theta ** (2 * k / d_head)
            c, s = math.cos(a), math.sin(a)
            x1, x2 = x[..., i, 2 * k], x[..., i, 2 * k + 1]
            out[..., i, 2 * k] = c * x1 - s * x2
            out[..., i, 2 * k + 1] = s * x1 + c * x2
    return out


def test_rope_matches_the_slow_version():
    x = torch.randn(2, 3, 11, 16)
    rope = RoPE(10000.0, 16, 32)
    assert torch.allclose(rope(x), slow_rope(x, 10000.0, 16), atol=1e-5)


def test_rope_preserves_the_norm_of_each_pair():
    x = torch.randn(1, 5, 8)
    y = RoPE(10000.0, 8, 16)(x)
    assert torch.allclose(x.pow(2).sum(-1), y.pow(2).sum(-1), atol=1e-5)


def test_shapes_are_preserved_with_extra_batch_dims():
    rope = RoPE(10000.0, CFG["d_model"] // CFG["num_heads"], 32)
    attn = CausalSelfAttention(CFG["d_model"], CFG["num_heads"], rope)
    ff = FeedForward(CFG["d_model"], 128)
    for shape in [(5, 64), (3, 5, 64), (2, 3, 5, 64)]:
        x = torch.randn(*shape)
        assert attn(x).shape == x.shape
        assert ff(x).shape == x.shape


def test_attention_output_does_not_depend_on_future_tokens():
    torch.manual_seed(0)
    rope = RoPE(10000.0, 16, 32)
    attn = CausalSelfAttention(64, 4, rope)
    x = torch.randn(1, 10, 64)
    y = x.clone()
    y[:, 6:] = torch.randn(1, 4, 64)
    assert torch.allclose(attn(x)[:, :6], attn(y)[:, :6], atol=1e-6)
