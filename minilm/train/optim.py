import math

import torch


def make_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    """Weight matrices get weight decay, RMSNorm weights (1-D tensors) do not."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def lr_at(t, max_lr, min_lr, warmup_steps, cosine_steps):
    """Linear warm-up, then cosine decay down to min_lr, then flat."""
    if t < warmup_steps:
        return t / warmup_steps * max_lr
    if t <= cosine_steps:
        progress = (t - warmup_steps) / (cosine_steps - warmup_steps)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    return min_lr


def make_scheduler(optimizer, max_lr, min_lr, warmup_steps, cosine_steps):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda t: lr_at(t, max_lr, min_lr, warmup_steps, cosine_steps) / max_lr
    )
