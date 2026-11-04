"""Part 2: encode both TinyStories files to uint16 .npy arrays."""

import argparse
import time
from pathlib import Path

import numpy as np

from minilm.tokenizer.tokenizer import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="../data")
p.add_argument("--splits", nargs="+", default=["valid", "train"])
args = p.parse_args()

d = Path(args.data_dir)
tok = Tokenizer.from_files(d / "vocab.json", d / "merges.txt", ["<|endoftext|>"])

for split in args.splits:
    src = d / f"TinyStoriesV2-GPT4-{split}.txt"
    t0 = time.time()
    with open(src, encoding="utf-8") as f:
        ids = np.fromiter(tok.encode_iterable(f), dtype=np.uint16)
    np.save(d / f"{split}.npy", ids)
    secs = time.time() - t0
    nbytes = src.stat().st_size
    print(f"{split}: {len(ids):,} tokens in {secs:.1f}s "
          f"({nbytes / secs / 1e6:.2f} MB/s, {nbytes / len(ids):.3f} bytes/token)")
