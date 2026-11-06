"""Part 6: short pilot runs to pick a learning rate."""

import argparse
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
p.add_argument("--steps", type=int, default=300)
p.add_argument("--out-dir", default="checkpoints")
p.add_argument("rest", nargs=argparse.REMAINDER, help="extra args passed through to train.py")
args = p.parse_args()
rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest  # argparse keeps the "--"

for lr in args.lrs:
    tag = f"{args.out_dir}/pilot_{lr:g}"
    print(f"\n### lr {lr:g}")
    subprocess.run(
        [sys.executable, "train.py", "--lr", str(lr), "--steps", str(args.steps),
         "--eval-every", "50", "--ckpt-every", "0", "--out", tag] + rest,
        check=True,
    )
