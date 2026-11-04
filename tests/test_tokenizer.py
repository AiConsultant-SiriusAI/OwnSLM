from minilm.tokenizer.bpe import train_bpe
from minilm.tokenizer.tokenizer import Tokenizer, load_tokenizer, save_tokenizer

TEXT = "Once upon a time there was a small cat. The cat liked fish.\n" * 20


def tiny():
    vocab = {
        0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t",
        6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at",
    }
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    return Tokenizer(vocab, merges)


def test_hand_worked_encoding():
    assert tiny().encode("the cat ate") == [9, 7, 1, 5, 10, 3]


def test_roundtrip():
    tok = trained()
    for s in ["Once upon a time", "नमस्ते दुनिया", "a 🙂 b 🎉", "", "   \n\t  "]:
        assert tok.decode(tok.encode(s)) == s


def test_special_token_is_one_id_and_blocks_merging():
    tok = trained()
    ids = tok.encode("cat<|endoftext|>cat")
    eot = tok.ids[b"<|endoftext|>"]
    assert ids.count(eot) == 1
    i = ids.index(eot)
    # the halves either side encode exactly as they would on their own
    assert ids[:i] == tok.encode("cat") and ids[i + 1:] == tok.encode("cat")


def test_overlapping_specials_match_the_longer_one():
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [], ["<|end|>", "<|endoftext|>"])
    assert tok.encode("<|endoftext|>") == [tok.ids[b"<|endoftext|>"]]


def test_encode_iterable_matches_encode(tmp_path):
    tok = trained()
    p = tmp_path / "f.txt"
    p.write_text(TEXT, encoding="utf-8")
    with open(p, encoding="utf-8") as f:
        streamed = list(tok.encode_iterable(f))
    assert streamed == tok.encode(TEXT)


def test_save_and_load_roundtrip(tmp_path):
    src = tmp_path / "c.txt"
    src.write_text(TEXT, encoding="utf-8")
    vocab, merges = train_bpe(str(src), 400, ["<|endoftext|>"])
    save_tokenizer(vocab, merges, tmp_path / "v.json", tmp_path / "m.txt")
    v2, m2 = load_tokenizer(tmp_path / "v.json", tmp_path / "m.txt")
    assert v2 == vocab and m2 == merges
    tok = Tokenizer.from_files(tmp_path / "v.json", tmp_path / "m.txt", ["<|endoftext|>"])
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_decode_replaces_broken_bytes():
    vocab = {i: bytes([i]) for i in range(256)}
    assert Tokenizer(vocab, []).decode([0xFF]) == "�"


_trained = None


def trained():
    """One tokenizer trained on TEXT, reused across tests."""
    global _trained
    if _trained is None:
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.write(fd, TEXT.encode("utf-8"))
        os.close(fd)
        vocab, merges = train_bpe(path, 400, ["<|endoftext|>"])
        os.unlink(path)
        _trained = Tokenizer(vocab, merges, ["<|endoftext|>"])
    return _trained
