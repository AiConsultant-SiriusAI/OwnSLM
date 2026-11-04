"""The warm-up experiments: UTF-8 (Part 1), sqrt(d_k) (Q3), RoPE distances (Q4), SGD (Part 5)."""

import sys

import torch

from minilm.model.layers import RoPE

sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252


def utf8():
    print("=== Part 1 warm-ups ===")
    print(f"chr(0) printed: {chr(0)!s:<4} repr: {chr(0)!r}")
    for s in ["hi", "नमस्ते", "🙂"]:
        print(f"  {s:8} utf-8 {len(s.encode('utf-8')):2}B  "
              f"utf-16 {len(s.encode('utf-16')):2}B  utf-32 {len(s.encode('utf-32')):2}B")
    try:
        [bytes([b]).decode("utf-8") for b in "é".encode("utf-8")]
    except UnicodeDecodeError as e:
        print(f"  byte-at-a-time on 'é' (2 bytes) fails: {e}")
    try:
        b"\xc3\x28".decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"  b'\\xc3\\x28' is not valid utf-8: {e}")


def scaling():
    """Q3: why attention divides by sqrt(d_k)."""
    print("\n=== Q3: dividing by sqrt(d_k), d_k = 32 ===")
    torch.manual_seed(0)
    d_k = 32
    q, k = torch.randn(8, d_k), torch.randn(8, d_k)
    scores = q @ k.T
    for label, s in [("unscaled", scores), ("scaled", scores / d_k**0.5)]:
        p = torch.softmax(s, -1)
        print(f"  {label:9} score std {s.std():5.2f}  max prob {p.max():.3f}  "
              f"entropy {-(p * p.log()).sum(-1).mean():.3f} nats")


def rope_distance():
    """Q4: after RoPE, q_i . k_j depends only on i - j."""
    print("\n=== Q4: RoPE dot products depend only on i - j ===")
    torch.manual_seed(0)
    d_head = 16
    rope = RoPE(10000.0, d_head, 64)
    q = torch.randn(1, d_head).expand(64, d_head).clone()  # the same vector at every position
    k = torch.randn(1, d_head).expand(64, d_head).clone()
    qr, kr = rope(q), rope(k)
    for gap in [0, 1, 5]:
        dots = [float(qr[i] @ kr[i - gap]) for i in (10, 20, 30, 40)]
        print(f"  i - j = {gap}:  " + "  ".join(f"{d:+.5f}" for d in dots))
    print("  each row is constant, so only the gap matters, not the absolute positions")


def sgd():
    """Part 5 warm-up: what the learning rate does."""
    print("\n=== Part 5 warm-up: SGD on mean(x^2), 10 steps ===")
    for lr in [1, 10, 100, 1000]:
        torch.manual_seed(0)
        x = torch.nn.Parameter(5 * torch.randn(10, 10))
        opt = torch.optim.SGD([x], lr=lr)
        first = (x**2).mean().item()
        for _ in range(10):
            loss = (x**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  lr {lr:5}: {first:10.3f} -> {(x**2).mean().item():12.4g}")
    print("  the gradient is 2x/100, so a step multiplies x by (1 - lr/50):")
    print("  lr 1 and 10 shrink x, lr 100 flips its sign but keeps the size, lr 1000 explodes")


if __name__ == "__main__":
    utf8()
    scaling()
    rope_distance()
    sgd()
