"""Train the language model. Logs to a CSV; resume with --resume."""

import argparse
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

from minilm.model.transformer import TransformerLM
from minilm.train import checkpoint
from minilm.train.data import get_batch
from minilm.train.optim import make_optimizer, make_scheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="../data/train.npy")
    p.add_argument("--val-data", default="../data/valid.npy")
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--out", default="checkpoints/run")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="fp16 autocast + GradScaler")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--single-batch", action="store_true", help="overfit one batch, as a sanity check")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fixed_val_batches(data, n, batch_size, context_length, device):
    """The same held-out batches every time, so validation numbers are comparable."""
    state = np.random.get_state()
    np.random.seed(1234)
    batches = [get_batch(data, batch_size, context_length, device) for _ in range(n)]
    np.random.set_state(state)
    return batches


def evaluate(model, batches, vocab_size):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in batches:
            total += F.cross_entropy(model(x).reshape(-1, vocab_size), y.reshape(-1)).item()
    model.train()
    return total / len(batches)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    model = TransformerLM(
        args.vocab_size, args.context_length, args.d_model,
        args.num_layers, args.num_heads, args.d_ff, args.rope_theta,
    ).to(args.device)
    if args.compile:
        model = torch.compile(model)

    opt = make_optimizer(model, args.lr, args.weight_decay)
    warmup = max(1, int(args.warmup_frac * args.steps))
    sched = make_scheduler(opt, args.lr, args.min_lr, warmup, args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start = 0
    if args.resume:
        start = checkpoint.load(args.resume, model, opt, sched, scaler, args.device)

    val_batches = fixed_val_batches(
        val_data, args.eval_batches, args.batch_size, args.context_length, args.device
    )
    held = get_batch(train_data, args.batch_size, args.context_length, args.device)

    log = open(f"{args.out}.csv", "a", newline="")
    writer = csv.writer(log)
    if start == 0:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "grad_norm", "seconds"])

    t0 = time.time()
    model.train()
    for step in range(start, args.steps):
        x, y = held if args.single_batch else get_batch(
            train_data, args.batch_size, args.context_length, args.device
        )
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            loss = F.cross_entropy(model(x).reshape(-1, args.vocab_size), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # must come before clipping, or we clip scaled gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        sched.step()

        val = ""
        if step % args.eval_every == 0 or step == args.steps - 1:
            val = evaluate(model, val_batches, args.vocab_size)
            print(f"step {step}  train {loss.item():.4f}  val {val:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  gnorm {norm:.2f}")
        writer.writerow([step, f"{loss.item():.5f}", val, f"{sched.get_last_lr()[0]:.3e}",
                         f"{float(norm):.4f}", f"{time.time() - t0:.1f}"])
        log.flush()

        if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            checkpoint.save(f"{args.out}.pt", step + 1, model, opt, sched, scaler)

    checkpoint.save(f"{args.out}.pt", args.steps, model, opt, sched, scaler)
    final = evaluate(model, val_batches, args.vocab_size)
    print(f"final val loss {final:.4f}  perplexity {np.exp(final):.2f}")
    log.close()


if __name__ == "__main__":
    main()
