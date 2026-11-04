"""The Transformer block and the full language model."""

from torch import nn

from minilm.model.layers import CausalSelfAttention, FeedForward, RoPE


class TransformerBlock(nn.Module):
    """Pre-norm block: z = x + Attn(norm(x)), y = z + FF(norm(z))."""

    def __init__(self, d_model, num_heads, d_ff, rope):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.attn = CausalSelfAttention(d_model, num_heads, rope)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        z = x + self.attn(self.norm1(x))
        return z + self.ff(self.norm2(z))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.context_length = context_length
        self.embed = nn.Embedding(vocab_size, d_model)
        rope = RoPE(rope_theta, d_model // num_heads, context_length)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))
