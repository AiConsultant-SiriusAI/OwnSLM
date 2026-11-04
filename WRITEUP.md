# Writeup

Every number below marked **[TODO]** needs the T4 run or the full 2.2 GB file; the command that
produces it is given with it. Everything else is real output from `python -m scripts.analysis`
and `python -m scripts.warmups` on this repo.

---

## Part 1 warm-ups

**`chr(0)`** is the NUL character, a real one-character string. `print` sends the raw character to
the terminal, which shows nothing; `repr` shows the escaped source form `'\x00'`, so you can see
what is actually there.

**Why UTF-8 bytes.** UTF-8 is the only one of the three with no wasted padding on ASCII, which is
most of TinyStories, and it has no byte-order mark or endianness to worry about:

| string | UTF-8 | UTF-16 | UTF-32 |
| --- | --- | --- | --- |
| `hi` | 2 B | 6 B | 12 B |
| `नमस्ते` | 18 B | 14 B | 28 B |
| `🙂` | 4 B | 6 B | 8 B |

A byte alphabet also has exactly 256 symbols, so the base vocabulary is small and nothing is ever
out of vocabulary. UTF-16 would need 65,536 base symbols and still use surrogate pairs.

**Decoding one byte at a time is wrong** because UTF-8 is a multi-byte encoding: a character
outside ASCII is spread over 2–4 bytes that only mean something together. `"é"` is `b'\xc3\xa9'`,
and `bytes([0xc3]).decode("utf-8")` raises
`'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data`.

**An invalid two-byte sequence:** `b'\xc3\x28'`. `0xc3` announces a two-byte character, so the next
byte must be a continuation byte of the form `10xxxxxx` (0x80–0xBF). `0x28` is `(`, which is not,
so Python raises `invalid continuation byte`.

---

## Part 1 — tokenizer training

- Training time: **[TODO]** — `python -m scripts.train_tokenizer` prints it.
- Peak memory: **[TODO]** — the text is held as one Python `str`, so expect roughly the file size
  plus the pre-token table.
- Longest token: **[TODO]** — the script prints it. Expect a long common word or a word with its
  leading space; if it is something like a run of spaces or newlines, that is a sign the
  pre-tokenizer let whitespace runs through.

**Question 1 — where does the time go?** **[TODO]**
```bash
python -c "import cProfile,pstats; from minilm.tokenizer.bpe import train_bpe; \
cProfile.run('train_bpe(\"../data/TinyStoriesV2-GPT4-valid.txt\", 10000, [\"<|endoftext|>\"])', \
'prof'); pstats.Stats('prof').sort_stats('cumtime').print_stats(15)"
```
Expectation to check against: pre-tokenization is one `regex.finditer` pass over the whole file,
while merging touches only the pre-tokens that contain the merged pair. On a 2 GB file the single
regex pass usually dominates; on a small file the 9,743 merge iterations do.

---

## Part 2 — encoding

- Compression ratio: **[TODO]** bytes/token on 10 random stories. On a 3 MB slice of the validation
  file with a 2,000-token vocabulary this implementation gets **3.68 bytes/token**; a
  10,000-token vocabulary should do better.
- Speed: **[TODO]**. Measured here at **1.04 MB/s** on one CPU core, which puts the 2.23 GB
  training file at roughly 36 minutes.
- Encoding time for the training file: **[TODO]** — `python -m scripts.encode_data` prints it.

**Question 2 — why `uint16`?** A `uint16` holds 0–65,535 and the vocabulary is 10,000, so every ID
fits with room to spare, at 2 bytes instead of the 8 a Python int or an `int64` array would use.
That matters because the array is memory-mapped during training: halving the width halves the disk
reads per batch. It stops being good enough the moment the vocabulary exceeds 65,536 tokens —
GPT-2's 50,257 still fits, but a 100k-token vocabulary (Llama 3, GPT-4) would silently wrap around
and would need `uint32`.

---

## Part 3 — the pieces

All tests pass: `pytest tests/test_layers.py` — hand-written attention matches
`F.scaled_dot_product_attention` to 1e-5 for 3-D and 4-D inputs, RoPE matches an explicit
rotation-matrix version, and every piece preserves its input shape with extra batch dimensions.

**Question 3 — why divide by sqrt(d_k)?** With `d_k = 32` and standard normal q and k, each score
is a sum of 32 products, so its standard deviation grows like sqrt(d_k):

| | score std | max softmax prob | entropy |
| --- | --- | --- | --- |
| unscaled | 5.92 | 0.998 | 0.435 nats |
| scaled by sqrt(32) | 1.05 | 0.593 | 1.674 nats |

Unscaled, the softmax is already almost one-hot at initialisation: it puts 99.8% on a single token
chosen by random initialisation noise. That is bad in itself, and worse for training, because the
gradient of a saturated softmax is nearly zero, so attention can barely learn. Dividing by
sqrt(d_k) holds the scores at unit scale whatever the head dimension, which keeps the distribution
soft and the gradients alive.

**Question 4 — RoPE depends only on i − j.** Taking one fixed query vector and one fixed key vector
and placing copies of them at every position, the dot product after RoPE is:

| gap i − j | at i=10 | i=20 | i=30 | i=40 |
| --- | --- | --- | --- | --- |
| 0 | +2.98395 | +2.98395 | +2.98395 | +2.98395 |
| 1 | +1.51567 | +1.51567 | +1.51567 | +1.51567 |
| 5 | −0.19526 | −0.19526 | −0.19526 | −0.19526 |

Each row is constant: the absolute positions do not matter, only the gap. The reason is that RoPE
rotates each 2-D pair by an angle proportional to the position, and a rotation by `a_i` followed by
the inverse rotation by `a_j` composes into one rotation by `a_i − a_j`, which depends only on
`i − j`. Reproduce with `python -m scripts.warmups`.

---

## Part 4 — the full model

Shape test `(2, 17)` → `(2, 17, 10000)` passes; the causality test (perturb every token after
position 8, outputs at positions 0–8 unchanged **bit for bit**) passes; the initial loss on random
tokens is within 0.3 of `ln(10000) = 9.2103`.

**Question 5 — parameter count.**

| part | parameters | formula | share |
| --- | --- | --- | --- |
| embedding | 5,120,000 | V·d | 22.6% |
| attention, 4 layers | 4,194,304 | L·4d² | 18.5% |
| feed-forward, 4 layers | 8,257,536 | L·3·d·d_ff | 36.4% |
| RMSNorm, 4 layers + final | 4,608 | L·2d + d | 0.0% |
| output projection | 5,120,000 | d·V | 22.6% |
| **total** | **22,696,448** | | |

`sum(p.numel() for p in model.parameters())` = **22,696,448**, matching exactly.
Embedding + output = 10,240,000 = **45.1%** of the model. Nearly half the parameters do nothing
but map between token IDs and vectors — that is what a small `d_model` with a 10k vocabulary buys.

**Question 6 — FLOPs for one forward pass over 256 tokens** (an m×n by n×p matmul is 2mnp):

per layer:

| matmul | shape | FLOPs |
| --- | --- | --- |
| Q projection | 256×512 · 512×512 | 134,217,728 |
| K projection | 256×512 · 512×512 | 134,217,728 |
| V projection | 256×512 · 512×512 | 134,217,728 |
| scores QKᵀ, 16 heads | 16 × (256×32 · 32×256) | 67,108,864 |
| weights × V, 16 heads | 16 × (256×256 · 256×32) | 67,108,864 |
| output projection | 256×512 · 512×512 | 134,217,728 |
| W1 (gate) | 256×512 · 512×1344 | 352,321,536 |
| W3 (up) | 256×512 · 512×1344 | 352,321,536 |
| W2 (down) | 256×1344 · 1344×512 | 352,321,536 |
| **one layer** | | **1,728,053,248** |

whole model: 4 × 1,728,053,248 + LM head (256×512 · 512×10000 = 2,621,440,000)
= **9,533,652,992 FLOPs (9.53 GFLOPs)**.

Split: feed-forward **44.3%**, attention **28.2%**, LM head **27.5%**. The feed-forward layers do
the most work — more than attention, even though attention is the interesting part. The single
biggest matmul in the model is the LM head, at 27.5% of the whole forward pass on its own.

**Question 7 — the same for GPT-2 small** (12 layers, d_model 768, 12 heads, vocab 50,257,
context 1,024): one layer is 22,548,578,304 FLOPs, the LM head is 79,047,426,048, and the total is
**349,630,365,696 (349.6 GFLOPs)**, 37× ours.

| | ours (256 ctx) | GPT-2 small (1024 ctx) |
| --- | --- | --- |
| attention | 28.2% | 27.6% |
| feed-forward | 44.3% | 49.8% |
| LM head | 27.5% | 22.6% |

As the model gets bigger the LM head shrinks as a share of the work even though the vocabulary is
5× larger, because the layers grow faster than the head does. Within attention, the quadratic part
(QKᵀ and weights×V) grows from 5.6% of the total to 11.1%: it scales with context length squared,
so at 1,024 tokens it is already comparable to the projections, and at 8k tokens it would dominate.

---

## Part 5 — training setup

**Warm-up: what the learning rate does.** Minimising `mean(x²)` for a 10×10 parameter, 10 steps:

| lr | start | after 10 steps |
| --- | --- | --- |
| 1 | 26.271 | 17.54 |
| 10 | 26.271 | 0.3029 |
| 100 | 26.271 | 26.27 |
| 1000 | 26.271 | 9.875e+26 |

The gradient of `mean(x²)` is `2x/100`, so one step maps `x → x(1 − lr/50)`. At lr 1 the factor is
0.98 — correct but slow. At lr 10 it is 0.8, and the loss falls by two orders of magnitude. At
lr 100 the factor is exactly −1: every step flips the sign of x and leaves its magnitude untouched,
so the loss never moves. Above that the factor exceeds 1 in magnitude and it diverges: at lr 1000
the factor is −19, and 19¹⁰ ≈ 6e12 squared gives the 1e26 we see. The edge of stability is at
lr 100, and the best rate is just below it — exactly the pattern Question 9 asks about.

All training tests pass: cross entropy matches `-log softmax` by hand; the two AdamW groups contain
every parameter exactly once with weight decay 0.1 and 0.0; the schedule matches hand-computed
values at steps 0, 5, 10, 60, 110 and 500; post-clip gradient norm ≤ 1.0; targets are inputs
shifted one step with no slice past the end; and save → 5 steps → reload → the same 5 steps gives
losses within 1e-3 (the reload starts from a *differently seeded* model, so the test fails if the
checkpoint does not actually restore the weights, the optimizer moments and the scheduler).

**`torch.compile` speedup: [TODO]** — run `train.py --steps 300` with and without `--compile`
and compare the `seconds` column of the two CSVs.

**Question 8 — fp32 memory.**

| | |
| --- | --- |
| parameters | 86.6 MB |
| gradients | 86.6 MB |
| AdamW `exp_avg` | 86.6 MB |
| AdamW `exp_avg_sq` | 86.6 MB |
| **total** | **346.3 MB** |

AdamW needs roughly twice the weights again because it keeps **two** running averages per
parameter: `exp_avg`, the decayed mean of the gradient (the momentum term, β₁), and `exp_avg_sq`,
the decayed mean of the squared gradient (the per-parameter scale, β₂). Both are the same shape as
the weights, so the optimizer state alone is 2× the model. Activations are on top of all this and
are what actually decides the batch size you can fit.

---

## Part 6 — the training script

- **Single-batch check:** verified on CPU with a 2,000-token vocabulary and a 2-layer model: train
  loss **7.8101 → 0.0232** over 300 steps while validation loss *rises* from 7.77 to 7.56, which is
  the memorisation we want to see. Plot: **[TODO, `checkpoints/overfit.png`]**.
- **Resume check:** stopped at step 60, resumed from the checkpoint, and the loss continued at
  6.75 → 6.47 with no jump. Plot: **[TODO, `checkpoints/resume.png`]**.
  One real gotcha found here: resuming with a different `--steps` rebuilds the cosine schedule for
  a different horizon and the learning rate jumps. Resume with the same `--steps`.
- **Gradient norms over the first 100 steps:** healthy and flat in the CPU runs (0.5–0.8, no NaN,
  no growth). **[TODO on the real model]** — the `grad_norm` column of the CSV.

**Learning-rate pilots: [TODO]** — `python -m scripts.pilot_lr --steps 300`, then
`python -m scripts.plots checkpoints/pilot_*.csv --out pilots.png`.
Rates chosen: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 — one per half-decade across the range the brief gives,
so the grid is even on a log scale (which is how learning rates behave), with 1e-2 included
specifically because it should be high enough to diverge and show where the ceiling is.

**Question 9 — how does the blow-up rate compare with the best one? [TODO]**
Answer from the pilot chart: give the ratio between the lowest diverging rate and the best one, and
say whether the best really sat just below the edge. The SGD warm-up above is the clean version of
this: there, the best stable rate and the edge of stability were within a factor of 10.

---

## Part 7 — the main run

- Final validation loss: **[TODO]**, perplexity `e^loss` = **[TODO]**.
- Loss curve against steps and against time: **[TODO]** — `scripts/plots.py --x step` and `--x seconds`.
- Three stories of 256+ tokens from "Once upon a time" at two temperature/top-p settings:
  **[TODO]** — e.g. `--temperature 0.8 --top-p 0.9` and `--temperature 1.0 --top-p 0.95`.

**Question 10 — how fluent are the stories? [TODO]**
Two things to look for, and to back with your own samples: (a) **temperature and top-p** — low
temperature gives grammatical but repetitive text that loops on one phrase, high temperature gives
variety but loses the thread and invents words; (b) **the 256-token context** — the model cannot
see further back than 256 tokens, so a character introduced early is forgotten by the end, and
long stories drift off topic rather than resolve. Say which failures are *bugs you found and fixed*
and which are *limits of a 22.7M model trained on 41M tokens*; the brief grades that distinction.

---

## Part 8 — packaging

The acceptance test passes on the smoke-test release (fresh `release/` built by
`scripts/make_release.py`, loaded with `map_location`, run on CPU): the same `--seed 42` gave a
byte-identical story on two consecutive runs, and `inference.py` imports only `minilm.generate`,
`minilm.model` and `minilm.tokenizer` — nothing from `train.py`.

**Question 11 — CPU generation speed and the KV cache. [TODO for the real model]**

Measured on the smoke-test model: 462–550 tokens/s on this laptop (Windows 11, CPU-only PyTorch
2.15). Report your own number for the real 22.7M model generating 256 tokens.

The repeated work: with no cache, generating token `n` runs the whole model over all `n` tokens,
but only the last position's output is used. Over 256 tokens that is
`1 + 2 + ... + 256 = 32,896` token-forward-passes, where a KV cache would need only 256 — every
earlier position's keys and values are recomputed identically every step. So about
**99.2% of the work is repeated** (`1 - 256/32896`).

A KV cache would not make it 128× faster in wall-clock terms, though. The uncached version runs one
big batched matmul per step, which uses the CPU well; the cached version runs a sequence of
matrix-vector products, which are memory-bound and far less efficient per FLOP. Expect a large
speedup that grows with sequence length, but measure it rather than quoting the FLOP ratio.

---

## Honest summary

**[TODO after the main run]** — separate (a) bugs in my code that I found and fixed from (b) limits
of a small model trained briefly. Bugs found so far during development: the single-batch check and
the differently-seeded resume test both caught real wiring problems, and the resume-with-different-
`--steps` learning-rate jump described in Part 6 is a genuine gotcha in this script.
