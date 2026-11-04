"""Generate a story from a released model. Imports nothing from train.py."""

import argparse
import json
import time
from pathlib import Path

import torch

from minilm.generate import generate
from minilm.model.transformer import TransformerLM
from minilm.tokenizer.tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="release")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    d = Path(args.model_dir)
    config = json.loads((d / "config.json").read_text())
    model = TransformerLM(**config).to(args.device)
    model.load_state_dict(torch.load(d / "model.pt", map_location=args.device))
    tok = Tokenizer.from_files(d / "vocab.json", d / "merges.txt", ["<|endoftext|>"])

    t0 = time.time()
    text, n = generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_p)
    elapsed = time.time() - t0

    print(args.prompt + text)
    print(f"\n{n} tokens in {elapsed:.1f}s ({n / elapsed:.1f} tokens/s)")


if __name__ == "__main__":
    main()
