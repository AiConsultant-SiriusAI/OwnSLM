import torch

from minilm.generate import _sample_top_p, generate
from minilm.model.transformer import TransformerLM
from minilm.tokenizer.bpe import train_bpe
from minilm.tokenizer.tokenizer import Tokenizer

TEXT = "Once upon a time there was a small cat who liked fish.\n" * 20


def setup(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text(TEXT, encoding="utf-8")
    vocab, merges = train_bpe(str(p), 400, ["<|endoftext|>"])
    tok = Tokenizer(vocab, merges, ["<|endoftext|>"])
    torch.manual_seed(0)
    model = TransformerLM(len(tok.vocab), 32, 32, 2, 4, 64, 10000.0)
    return model, tok


def test_greedy_is_deterministic(tmp_path):
    model, tok = setup(tmp_path)
    a, _ = generate(model, tok, "Once upon", 20, temperature=0)
    b, _ = generate(model, tok, "Once upon", 20, temperature=0)
    assert a == b


def test_stops_at_max_new_tokens(tmp_path):
    model, tok = setup(tmp_path)
    _, n = generate(model, tok, "Once", 12, temperature=0.8, top_p=0.9)
    assert n <= 12


def test_context_is_cropped_to_the_window(tmp_path):
    # a prompt longer than context_length must not crash: only the last 32 tokens go in
    model, tok = setup(tmp_path)
    long_prompt = "Once upon a time there was a small cat who liked fish. " * 10
    assert len(tok.encode(long_prompt)) > model.context_length
    _, n = generate(model, tok, long_prompt, 5, temperature=0)
    assert n <= 5


def test_top_p_only_samples_from_the_kept_group():
    probs = torch.tensor([0.6, 0.3, 0.07, 0.03])
    # 0.6 alone is under 0.9, 0.6 + 0.3 reaches it, so only ids 0 and 1 are allowed
    picks = {_sample_top_p(probs, 0.9) for _ in range(200)}
    assert picks <= {0, 1}
    assert {_sample_top_p(probs, 0.01) for _ in range(20)} == {0}
