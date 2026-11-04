"""Numbers for Questions 5-8: parameters, FLOPs and memory. All worked out from the shapes."""

from minilm.model.transformer import TransformerLM

OURS = dict(name="ours", vocab=10000, ctx=256, d=512, layers=4, heads=16, d_ff=1344)
GPT2 = dict(name="gpt-2 small", vocab=50257, ctx=1024, d=768, layers=12, heads=12, d_ff=3072)


def params(c):
    """Parameter count per part of the model."""
    attn = 4 * c["d"] * c["d"]
    ff = 3 * c["d"] * c["d_ff"]
    norms = 2 * c["d"]
    return {
        "embedding": c["vocab"] * c["d"],
        "attention (all layers)": c["layers"] * attn,
        "feed-forward (all layers)": c["layers"] * ff,
        "rmsnorm (all layers + final)": c["layers"] * norms + c["d"],
        "output projection": c["d"] * c["vocab"],
    }


def flops(c):
    """Every matmul in one forward pass over ctx tokens. An m x n times n x p matmul is 2mnp."""
    n, d, L, f, V = c["ctx"], c["d"], c["layers"], c["d_ff"], c["vocab"]
    per_layer = {
        "Q projection": 2 * n * d * d,
        "K projection": 2 * n * d * d,
        "V projection": 2 * n * d * d,
        "scores QK^T (all heads)": 2 * n * n * d,
        "weights x V (all heads)": 2 * n * n * d,
        "output projection": 2 * n * d * d,
        "W1 (gate)": 2 * n * d * f,
        "W3 (up)": 2 * n * d * f,
        "W2 (down)": 2 * n * f * d,
    }
    return per_layer, {k: v * L for k, v in per_layer.items()} | {"LM head": 2 * n * d * V}


def show(c):
    print(f"\n=== {c['name']}: vocab {c['vocab']}, ctx {c['ctx']}, d_model {c['d']}, "
          f"{c['layers']} layers, {c['heads']} heads, d_ff {c['d_ff']} ===")

    p = params(c)
    total = sum(p.values())
    print("\nQ5  parameters")
    for k, v in p.items():
        print(f"  {k:32s} {v:>12,}  {100 * v / total:5.1f}%")
    print(f"  {'TOTAL':32s} {total:>12,}")
    embed_share = (p["embedding"] + p["output projection"]) / total
    print(f"  embedding + output layers: {100 * embed_share:.1f}% of the total")

    per_layer, whole = flops(c)
    print(f"\nQ6/Q7  forward-pass FLOPs over {c['ctx']} tokens")
    print("  per layer:")
    for k, v in per_layer.items():
        print(f"    {k:28s} {v:>16,}")
    print(f"    {'one layer total':28s} {sum(per_layer.values()):>16,}")
    t = sum(whole.values())
    print("  whole model:")
    for k, v in whole.items():
        print(f"    {k:28s} {v:>16,}  {100 * v / t:5.1f}%")
    print(f"    {'TOTAL':28s} {t:>16,}  ({t / 1e9:.2f} GFLOPs)")

    ff_share = sum(v for k, v in whole.items() if k.startswith("W"))
    attn_share = t - ff_share - whole["LM head"]
    print(f"  attention {100 * attn_share / t:.1f}%  "
          f"feed-forward {100 * ff_share / t:.1f}%  LM head {100 * whole['LM head'] / t:.1f}%")
    return total


def memory(total):
    """Q8: fp32 memory for parameters, gradients and the AdamW state."""
    mb = total * 4 / 2**20
    print("\nQ8  fp32 memory")
    print(f"  parameters              {mb:7.1f} MB")
    print(f"  gradients               {mb:7.1f} MB")
    print(f"  AdamW exp_avg           {mb:7.1f} MB")
    print(f"  AdamW exp_avg_sq        {mb:7.1f} MB")
    print(f"  TOTAL                   {4 * mb:7.1f} MB")
    print("  AdamW keeps two running averages per parameter (the gradient mean and the")
    print("  squared-gradient mean), so its state is twice the size of the weights.")


if __name__ == "__main__":
    total = show(OURS)
    show(GPT2)
    memory(total)
    model = TransformerLM(10000, 256, 512, 4, 16, 1344, 10000.0)
    print(f"\ncheck: sum(p.numel()) = {sum(p.numel() for p in model.parameters()):,}")
