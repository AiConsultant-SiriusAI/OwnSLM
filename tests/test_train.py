import math

import numpy as np
import torch
import torch.nn.functional as F

from minilm.model.transformer import TransformerLM
from minilm.train import checkpoint
from minilm.train.data import get_batch
from minilm.train.optim import lr_at, make_optimizer, make_scheduler


def small():
    return TransformerLM(vocab_size=50, context_length=16, d_model=32,
                         num_layers=2, num_heads=4, d_ff=64, rope_theta=10000.0)


def test_cross_entropy_matches_hand_computed():
    logits = torch.randn(3, 5, 7)
    targets = torch.randint(0, 7, (3, 5))
    loss = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    by_hand = -torch.log(torch.softmax(logits, -1))
    expected = by_hand.gather(-1, targets.unsqueeze(-1)).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_param_groups_cover_every_parameter_once():
    model = small()
    opt = make_optimizer(model, 1e-3)
    seen = [p for g in opt.param_groups for p in g["params"]]
    assert len(seen) == len(list(model.parameters()))
    assert {id(p) for p in seen} == {id(p) for p in model.parameters()}
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_lr_schedule_matches_hand_computed_values():
    f = lambda t: lr_at(t, max_lr=1e-3, min_lr=1e-4, warmup_steps=10, cosine_steps=110)
    assert f(0) == 0.0
    assert math.isclose(f(5), 5e-4)            # halfway through warm-up
    assert math.isclose(f(10), 1e-3)           # peak
    assert math.isclose(f(60), 1e-4 + 0.5 * 9e-4)   # cos(pi/2) = 0, so halfway down
    assert math.isclose(f(110), 1e-4, abs_tol=1e-12)  # bottom, exactly at the last step
    assert f(500) == 1e-4


def test_gradient_clipping_caps_the_norm():
    model = small()
    ids = torch.randint(0, 50, (2, 16))
    F.cross_entropy(model(ids).reshape(-1, 50), ids.reshape(-1)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    after = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    assert after <= 1.0 + 1e-5


def test_get_batch_shifts_targets_and_stays_in_bounds():
    data = np.arange(200, dtype=np.uint16)
    x, y = get_batch(data, batch_size=8, context_length=16, device="cpu")
    assert x.shape == y.shape == (8, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert y.max().item() <= 199  # data is arange, so the values are the indices read


def test_save_and_resume_reproduces_the_same_losses(tmp_path):
    def build(seed):
        torch.manual_seed(seed)
        model = small()
        opt = make_optimizer(model, 1e-3)
        sched = make_scheduler(opt, 1e-3, 1e-4, 2, 20)
        return model, opt, sched, torch.amp.GradScaler("cpu", enabled=False)

    def steps(state, start, n):
        model, opt, sched, _ = state
        losses = []
        for t in range(start, start + n):
            torch.manual_seed(100 + t)  # the same batch at the same step either way
            ids = torch.randint(0, 50, (2, 16))
            loss = F.cross_entropy(model(ids).reshape(-1, 50), ids.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        return losses

    state = build(0)
    steps(state, 0, 3)  # a few steps first, so the AdamW moments are not all zero
    checkpoint.save(tmp_path / "ck.pt", 3, *state)
    straight = steps(state, 3, 5)

    other = build(1)  # a different init, so only a real reload can reproduce the losses
    assert checkpoint.load(tmp_path / "ck.pt", *other) == 3
    resumed = steps(other, 3, 5)

    for a, b in zip(straight, resumed):
        assert abs(a - b) < 1e-3
