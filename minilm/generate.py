import torch


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature=1.0, top_p=1.0):
    """Sample a continuation. temperature=0 means greedy, which is deterministic."""
    model.eval()
    device = next(model.parameters()).device
    eot = tokenizer.ids[b"<|endoftext|>"]
    ids = tokenizer.encode(prompt)
    new = []
    for _ in range(max_new_tokens):
        window = ids[-model.context_length :]
        logits = model(torch.tensor([window], device=device))[0, -1]
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = _sample_top_p(probs, top_p)
        if nxt == eot:
            break
        ids.append(nxt)
        new.append(nxt)
    return tokenizer.decode(new), len(new)


def _sample_top_p(probs, top_p):
    """Keep the smallest group of most likely tokens that adds up to at least p."""
    sorted_probs, idx = torch.sort(probs, descending=True)
    keep = (torch.cumsum(sorted_probs, 0) - sorted_probs) < top_p
    sorted_probs = sorted_probs[keep]
    return int(idx[keep][torch.multinomial(sorted_probs / sorted_probs.sum(), 1)])
