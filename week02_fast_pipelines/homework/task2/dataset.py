from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from glob import glob


MAX_LENGTH = 640


class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.pad_id = self.tokenizer.pad_token_id

        # Read all non-empty lines first
        lines = []
        for file_path in sorted(glob(f"{data_path}/train-*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)

        encoded = self.tokenizer(lines, add_special_tokens=False)
        self.samples = encoded["input_ids"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        seq_len = self.max_length + 1

        ids = ids[:seq_len]

        if len(ids) < seq_len:
            ids = ids + [self.pad_id] * (seq_len - len(ids))

        ids = torch.tensor(ids, dtype=torch.long)
        return ids[:-1], ids[1:]

class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.pad_id = self.tokenizer.pad_token_id

        lines = []
        for file_path in sorted(glob(f"{data_path}/train-*.txt")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)

        encoded = self.tokenizer(lines, add_special_tokens=False)
        self.samples = encoded["input_ids"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        ids = ids[:self.max_length + 1]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch: list[torch.Tensor], max_length: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence to the max length *within this batch*.
    Each item in batch is a 1D tensor of token IDs (variable length).
    """
    # Find the longest sequence in the batch (capped at max_length+1 if provided)
    batch_max = max(len(seq) for seq in batch)
    if max_length is not None:
        batch_max = min(batch_max, max_length + 1)

    pad_id = 0  # BERT pad token id
    inputs = []
    targets = []
    for seq in batch:
        seq = seq[:batch_max]
        # Pad to batch_max
        if len(seq) < batch_max:
            seq = torch.cat([seq, torch.full((batch_max - len(seq),), pad_id, dtype=torch.long)])
        inputs.append(seq[:-1])
        targets.append(seq[1:])

    return torch.stack(inputs), torch.stack(targets)


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        pass

    def __getitem__(self, idx: int):
        pass


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __getitem__(self, idx: int):
        pass

class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass
