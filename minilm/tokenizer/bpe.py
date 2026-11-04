"""Byte-pair encoding training."""

import heapq
from collections import Counter, defaultdict

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class _Desc:
    """Wraps a pair so the min-heap pops the tuple-largest pair on a count tie."""

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def pretoken_counts(text, special_tokens):
    """Count pre-tokens, never letting one span a special token."""
    counts = Counter()
    pieces = regex.split("|".join(regex.escape(t) for t in special_tokens), text)
    for piece in pieces:
        for m in regex.finditer(PAT, piece):
            counts[m.group()] += 1
    return counts


def learn_merges(counts, vocab_size, special_tokens):
    """Merge the most frequent pair until the vocabulary is full. counts maps pre-token -> freq."""
    vocab = {i: bytes([i]) for i in range(256)}
    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")

    # each pre-token is a list of byte-strings; freqs[i] is how often it occurs
    words = [[bytes([b]) for b in w.encode("utf-8")] for w in counts]
    freqs = list(counts.values())

    pair_counts = Counter()
    where = defaultdict(set)  # pair -> indices of words containing it
    for i, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_counts[p] += freqs[i]
            where[p].add(i)

    heap = [(-c, _Desc(p), p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    while len(vocab) < vocab_size and heap:
        c, _, best = heapq.heappop(heap)
        if pair_counts.get(best, 0) != -c:
            continue  # stale entry, the real count changed after it was pushed
        merges.append(best)
        new = best[0] + best[1]
        vocab[len(vocab)] = new

        touched = Counter()  # pairs whose count changed, so we re-push them once
        for i in list(where[best]):
            w, f = words[i], freqs[i]
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                where[p].discard(i)
                touched[p] = pair_counts[p]
            nw, j = [], 0
            while j < len(w):
                if j + 1 < len(w) and (w[j], w[j + 1]) == best:
                    nw.append(new)
                    j += 2
                else:
                    nw.append(w[j])
                    j += 1
            words[i] = nw
            for p in zip(nw, nw[1:]):
                pair_counts[p] += f
                where[p].add(i)
                touched[p] = pair_counts[p]
        for p, n in touched.items():
            if n > 0:
                heapq.heappush(heap, (-n, _Desc(p), p))
            else:
                pair_counts.pop(p, None)

    return vocab, merges


def train_bpe(input_path, vocab_size, special_tokens, max_bytes=None):
    """Learn a BPE vocabulary from a file. Returns (vocab: id -> bytes, merges: list of pairs)."""
    with open(input_path, "rb") as f:
        raw = f.read(max_bytes) if max_bytes else f.read()
    if max_bytes:
        # cut back to the last story boundary so we never train on half a story
        cut = raw.rfind(special_tokens[0].encode("utf-8"))
        if cut > 0:
            raw = raw[:cut]
    text = raw.decode("utf-8", errors="ignore")
    return learn_merges(pretoken_counts(text, special_tokens), vocab_size, special_tokens)
