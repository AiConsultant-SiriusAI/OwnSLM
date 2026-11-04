import numpy as np
import torch


def get_batch(data, batch_size, context_length, device):
    """Pick random windows. Targets are the inputs shifted one step right."""
    starts = np.random.randint(0, len(data) - context_length, size=batch_size)
    x = np.stack([data[i : i + context_length] for i in starts])
    y = np.stack([data[i + 1 : i + 1 + context_length] for i in starts])
    return (
        torch.from_numpy(x).long().to(device),
        torch.from_numpy(y).long().to(device),
    )
