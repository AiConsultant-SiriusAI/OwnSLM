"""Part 1: train the BPE tokenizer on TinyStories and save it."""

import argparse
import time
from pathlib import Path

from minilm.tokenizer.bpe import train_bpe
from minilm.tokenizer.tokenizer import save_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--input", default="../data/TinyStoriesV2-GPT4-train.txt")
p.add_argument("--out-dir", default="../data")
p.add_argument("--vocab-size", type=int, default=10000)
p.add_argument("--max-bytes", type=int, default=None, help="train on only the first N bytes")
args = p.parse_args()

t0 = time.time()
vocab, merges = train_bpe(args.input, args.vocab_size, ["<|endoftext|>"], args.max_bytes)
out = Path(args.out_dir)
save_tokenizer(vocab, merges, out / "vocab.json", out / "merges.txt")

longest = max(vocab.values(), key=len)
print(f"trained {len(vocab)} tokens in {time.time() - t0:.1f}s")
print(f"longest token: {longest!r} ({len(longest)} bytes)")
