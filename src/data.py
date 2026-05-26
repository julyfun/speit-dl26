"""Dataset and dataloader helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.features import (
    METADATA_NAMES,
    build_text,
    compute_metadata_stats,
    extract_metadata,
    normalize_metadata,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_fold_indices(
    labels: np.ndarray,
    fold: int,
    num_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, val_idx = splits[fold]
    return train_idx, val_idx


class TweetDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        metadata_stats: dict[str, np.ndarray] | None,
        max_length: int,
        use_metadata: bool,
        labels: np.ndarray | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.metadata_stats = metadata_stats
        self.max_length = max_length
        self.use_metadata = use_metadata
        self.labels = labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        text = build_text(record)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if self.use_metadata:
            metadata = extract_metadata(record)
            if self.metadata_stats is not None:
                metadata = normalize_metadata(metadata, self.metadata_stats)
            item["metadata"] = torch.tensor(metadata, dtype=torch.float32)

        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        item["challenge_id"] = torch.tensor(int(record["challenge_id"]), dtype=torch.long)
        return item


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def build_metadata_stats(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    matrix = np.stack([extract_metadata(r) for r in records], axis=0)
    return compute_metadata_stats(matrix)


def create_dataloader(
    dataset: TweetDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=shuffle,
    )
