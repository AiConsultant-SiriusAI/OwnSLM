import random

from minilm.tokenizer.bpe import learn_merges, train_bpe


def test_hand_worked_merges():
    # the worked example from the brief: split on spaces only, so no PAT here
    counts = {"low": 5, "lower": 2, "widest": 3, "newest": 6}
    _, merges = learn_merges(counts, 262, [])
    assert merges == [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]


def test_no_merge_crosses_a_special_token(tmp_path):
    p = tmp_path / "two.txt"
    p.write_text("aaaa aaaa aaaa<|endoftext|>bbbb bbbb bbbb", encoding="utf-8")
    vocab, _ = train_bpe(str(p), 300, ["<|endoftext|>"])
    for t in vocab.values():
        if t != b"<|endoftext|>":
            assert not (b"a" in t and b"b" in t)


def test_vocab_size_and_unique_ids(tmp_path):
    rng = random.Random(0)
    words = ["".join(rng.choices("abcdefghij", k=rng.randint(3, 9))) for _ in range(400)]
    p = tmp_path / "s.txt"
    p.write_text(" ".join(words), encoding="utf-8")
    vocab, merges = train_bpe(str(p), 300, ["<|endoftext|>"])
    assert len(vocab) == 300
    assert len(set(vocab.keys())) == 300
    assert len(set(vocab.values())) == 300
    assert len(merges) == 300 - 256 - 1


def test_specials_are_in_the_vocab(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("hello world", encoding="utf-8")
    vocab, _ = train_bpe(str(p), 280, ["<|endoftext|>"])
    assert b"<|endoftext|>" in vocab.values()
