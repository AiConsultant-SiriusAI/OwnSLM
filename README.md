# OwnSLM

A 22.7M-parameter GPT-style language model that writes short children's
stories.

## Generate a story

```bash
pip install -r requirements.txt
python inference.py --model-dir release --prompt "Once upon a time" \
    --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
```

Runs on an ordinary laptop with no GPU. The same `--seed` always gives the same story.

`release/` is produced by `scripts/make_release.py` after training (see below) and holds just four
files: `model.pt`, `config.json`, `vocab.json`, `merges.txt`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Model

| | |
| --- | --- |
| vocabulary | 10,000 BPE tokens trained on TinyStories |
| context length | 256 |
| layers / d_model / heads / d_ff | 4 / 512 / 16 / 1,344 |
| parameters | 22,696,448 (12,455,936 in the blocks, 45.1% in embedding + output) |
| position encoding | RoPE, theta 10,000 |
| normalisation | pre-norm RMSNorm |
| feed-forward | SwiGLU |
| trained on | 40,960,000 tokens (5,000 steps x batch 32 x context 256) |

## Layout

```
minilm/tokenizer/   train_bpe() and the Tokenizer class
minilm/model/       RoPE, attention, SwiGLU, the Transformer
minilm/train/       batching, optimizer, schedule, checkpoints
minilm/generate.py  temperature + top-p sampling
train.py            the training script
inference.py        standalone generation from release/
scripts/            tokenizer training, data encoding, analysis, plots, packaging
tests/              pytest
notebooks/colab.ipynb
```

## Reproducing the training

Scripts under `scripts/` are run as modules from the repo root, so `minilm` is importable:

```bash
# 1. BPE on TinyStories (slow on 2 cores; --max-bytes trains on a prefix instead)
python -m scripts.train_tokenizer --input ../data/TinyStoriesV2-GPT4-train.txt \
    --out-dir ../data --vocab-size 10000

# 2. encode both splits to uint16 .npy
python -m scripts.encode_data --data-dir ../data

# 3. sanity check: one batch over and over, the loss must fall to about zero
python train.py --steps 300 --single-batch --lr 1e-3 --out checkpoints/overfit

# 4. learning-rate pilots
python -m scripts.pilot_lr --steps 300

# 5. the main run (on a T4, add --amp for fp16)
python train.py --steps 5000 --lr <best> --amp --out checkpoints/main

# 6. package it
python -m scripts.make_release --checkpoint checkpoints/main.pt --data-dir ../data
```

Resuming: pass `--resume checkpoints/main.pt` **with the same `--steps`**, or the cosine schedule
is rebuilt for a different horizon and the learning rate jumps.

`python -m scripts.analysis` prints the parameter, FLOP and memory tables;
`python -m scripts.warmups` prints the UTF-8, sqrt(d_k), RoPE and SGD experiments.

See [WRITEUP.md](WRITEUP.md) for the answers to Questions 1-11.
