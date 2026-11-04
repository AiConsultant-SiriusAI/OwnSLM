"""Part 8: strip a checkpoint down to the four files inference.py needs."""

import argparse
import json
import shutil
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="checkpoints/run.pt")
p.add_argument("--data-dir", default="../data")
p.add_argument("--out", default="release")
p.add_argument("--vocab-size", type=int, default=10000)
p.add_argument("--context-length", type=int, default=256)
p.add_argument("--d-model", type=int, default=512)
p.add_argument("--num-layers", type=int, default=4)
p.add_argument("--num-heads", type=int, default=16)
p.add_argument("--d-ff", type=int, default=1344)
p.add_argument("--rope-theta", type=float, default=10000.0)
args = p.parse_args()

out = Path(args.out)
out.mkdir(exist_ok=True)
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
# torch.compile prefixes every key with _orig_mod.
state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
torch.save(state, out / "model.pt")

config = {k: getattr(args, k) for k in
          ["vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta"]}
(out / "config.json").write_text(json.dumps(config, indent=2))
for name in ["vocab.json", "merges.txt"]:
    shutil.copy(Path(args.data_dir) / name, out / name)
print(f"wrote {out}/ ({(out / 'model.pt').stat().st_size / 1e6:.1f} MB)")
