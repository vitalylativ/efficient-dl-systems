from enum import Enum
import time
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer import TransformerModel, generate_square_subsequent_mask
from dataset import (
    BrainDataset,
    BigBrainDataset,
    collate_fn,
    MAX_LENGTH,
)


DATA_DIR = "wikitext-103-raw-v1"
BATCH_SIZE = 32
VOCAB_SIZE = 30522


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_gpt2_model(vocab_size: int = VOCAB_SIZE) -> TransformerModel:
    return TransformerModel(ntoken=vocab_size, d_model=1024, nhead=8, d_hid=1024 * 4, nlayers=1)


def run_epoch(
    data_mode: DataMode,
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
) -> dict[str, float]:
    # Build dataloader based on mode
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        raise NotImplementedError(f"{data_mode} not implemented yet")

    print(f"Running epoch with {data_mode.name} on {device}")

    # Model, loss, optimizer
    model = get_gpt2_model().to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warmup: a few iterations to warm up the GPU
    warmup_iter = iter(dataloader)
    for _ in range(5):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            break
        if isinstance(batch, (list, tuple)):
            x, y = batch
        x, y = x.to(device), y.to(device)
        # TransformerModel expects (seq_len, batch)
        mask = generate_square_subsequent_mask(x.size(1)).to(device)
        logits = model(x.T, mask)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed epoch
    batch_times = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, y = batch
        x, y = x.to(device), y.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        mask = generate_square_subsequent_mask(x.size(1)).to(device)
        logits = model(x.T, mask)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
        batch_times.append(time.time() - start)

    stats = {
        "min": min(batch_times),
        "max": max(batch_times),
        "mean": statistics.mean(batch_times),
        "median": statistics.median(batch_times),
    }
    print(f"[{data_mode.name}] {len(batch_times)} batches")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}s")
    return stats

if __name__ == "__main__":
    for mode in DataMode:
        try:
            run_epoch(mode)
        except NotImplementedError as e:
            print(e)
