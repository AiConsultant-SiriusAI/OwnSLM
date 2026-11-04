"""Plot loss curves from the CSV logs train.py writes."""

import argparse
import csv

import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("csvs", nargs="+")
p.add_argument("--out", default="loss.png")
p.add_argument("--x", default="step", choices=["step", "seconds"])
args = p.parse_args()

fig, ax = plt.subplots(figsize=(7, 4))
for path in args.csvs:
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["step"]]
    x = [float(r[args.x]) for r in rows]
    ax.plot(x, [float(r["train_loss"]) for r in rows], lw=0.8, label=f"{path} train")
    val = [(float(r[args.x]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    if val:
        ax.plot(*zip(*val), marker="o", ms=3, label=f"{path} val")

ax.set_xlabel(args.x)
ax.set_ylabel("cross-entropy loss")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"wrote {args.out}")
