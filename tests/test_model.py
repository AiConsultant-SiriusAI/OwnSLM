import math

import torch
import torch.nn.functional as F

from minilm.model.transformer import TransformerLM

CFG = dict(
    vocab_size=10000, context_length=256, d_model=512,
    num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0,
)


def small():
    return TransformerLM(vocab_size=100, context_length=32, d_model=32,
                         num_layers=2, num_heads=4, d_ff=64, rope_theta=10000.0)


def test_logits_shape():
    model = TransformerLM(**CFG)
    ids = torch.randint(0, 10000, (2, 17))
    assert model(ids).shape == (2, 17, 10000)


def test_model_cannot_see_the_future():
    torch.manual_seed(0)
    model = small().eval()
    a = torch.randint(0, 100, (1, 16))
    b = a.clone()
    b[0, 9:] = torch.randint(0, 100, (7,))  # change everything after position 8
    with torch.no_grad():
        assert torch.equal(model(a)[:, :9], model(b)[:, :9])


def test_initial_loss_is_about_ln_vocab():
    # an untrained model has no reason to prefer any token, so softmax is roughly uniform
    # over the vocabulary and the cross entropy is about -log(1/V) = ln(V).
    torch.manual_seed(0)
    model = TransformerLM(**CFG)
    ids = torch.randint(0, 10000, (2, 64))
    with torch.no_grad():
        loss = F.cross_entropy(model(ids).reshape(-1, 10000), ids.reshape(-1))
    assert abs(loss.item() - math.log(10000)) < 0.3


def test_parameter_count():
    model = TransformerLM(**CFG)
    v, d, L, f = CFG["vocab_size"], CFG["d_model"], CFG["num_layers"], CFG["d_ff"]
    expected = v * d + L * (4 * d * d + 3 * d * f + 2 * d) + d + d * v
    assert sum(p.numel() for p in model.parameters()) == expected == 22_696_448
