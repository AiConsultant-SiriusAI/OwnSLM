"""Encoding and decoding with a trained BPE vocabulary."""

import json

import regex

from minilm.tokenizer.bpe import PAT


def save_tokenizer(vocab, merges, vocab_path, merges_path):
    """vocab.json maps the token (as latin-1 text, which round-trips every byte) to its id.
    merges.txt holds one JSON pair per line, so tokens containing spaces stay unambiguous."""
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({v.decode("latin-1"): k for k, v in vocab.items()}, f)
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(json.dumps([a.decode("latin-1"), b.decode("latin-1")]) + "\n")


def load_tokenizer(vocab_path, merges_path):
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {i: t.encode("latin-1") for t, i in json.load(f).items()}
    with open(merges_path, encoding="utf-8") as f:
        merges = [tuple(s.encode("latin-1") for s in json.loads(line)) for line in f if line.strip()]
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.specials = list(special_tokens or [])
        for t in self.specials:
            if t.encode("utf-8") not in self.vocab.values():
                self.vocab[len(self.vocab)] = t.encode("utf-8")
        self.ids = {v: k for k, v in self.vocab.items()}
        self.ranks = {p: i for i, p in enumerate(merges)}
        # longest first, so an overlapping special token matches the longer one
        self.special_re = (
            regex.compile("(" + "|".join(regex.escape(t) for t in sorted(self.specials, key=len, reverse=True)) + ")")
            if self.specials
            else None
        )

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        vocab, merges = load_tokenizer(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)

    def _merge(self, token):
        """Apply the learned merges to one pre-token, lowest rank first."""
        parts = [bytes([b]) for b in token.encode("utf-8")]
        while len(parts) > 1:
            pair = min(zip(parts, parts[1:]), key=lambda p: self.ranks.get(p, len(self.ranks)))
            if pair not in self.ranks:
                break
            merged, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return [self.ids[p] for p in parts]

    def encode(self, text):
        ids = []
        pieces = self.special_re.split(text) if self.special_re else [text]
        for piece in pieces:
            if piece in self.specials:
                ids.append(self.ids[piece.encode("utf-8")])
            else:
                for m in regex.finditer(PAT, piece):
                    ids.extend(self._merge(m.group()))
        return ids

    def encode_iterable(self, iterable):
        """Encode a stream (e.g. an open file) without holding it all in memory."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
