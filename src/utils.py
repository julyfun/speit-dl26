"""Checkpoint and trainer state helpers."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


def ensure_hub_cached(model_name: str, accelerator) -> Path:
    """Download full repo (incl. safetensors) on main process, then all ranks read local cache."""
    from huggingface_hub import snapshot_download

    if accelerator.is_main_process:
        print(f"Downloading {model_name} (waiting for full weights)...")
        snapshot_download(repo_id=model_name)
    accelerator.wait_for_everyone()
    cache_dir = Path(snapshot_download(repo_id=model_name, local_files_only=True))
    if accelerator.is_main_process:
        print(f"Model ready: {cache_dir}")
    return cache_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_trainer_state(
    save_dir: Path,
    state: dict[str, Any],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_trainer_state(load_dir: Path) -> dict[str, Any]:
    with open(load_dir / "trainer_state.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _write_checkpoint_files(
    save_dir: Path,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    metadata_stats: dict[str, np.ndarray] | None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.encoder.save_pretrained(save_dir / "encoder")
    torch.save(unwrapped.state_dict(), save_dir / "model.pt")
    tokenizer.save_pretrained(save_dir / "tokenizer")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")

    if metadata_stats is not None:
        np.savez(
            save_dir / "metadata_stats.npz",
            mean=metadata_stats["mean"],
            std=metadata_stats["std"],
        )


def save_checkpoint(
    save_dir: Path,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    args: dict[str, Any],
    metadata_stats: dict[str, np.ndarray] | None,
    completed_train_iteration: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
) -> None:
    _write_checkpoint_files(
        save_dir, model, tokenizer, optimizer, scheduler, metadata_stats
    )

    state = {
        "completed_train_iteration": completed_train_iteration,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_name": args["model_name"],
        "args": args,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_trainer_state(save_dir, state)


def save_best_checkpoint(
    save_dir: Path,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    metadata_stats: dict[str, np.ndarray] | None,
    trainer_state: dict[str, Any],
) -> None:
    best_dir = save_dir / "best"
    _write_checkpoint_files(best_dir, model, tokenizer, None, None, metadata_stats)
    save_trainer_state(best_dir, trainer_state)


def load_metadata_stats(load_dir: Path) -> dict[str, np.ndarray] | None:
    stats_path = load_dir / "metadata_stats.npz"
    if not stats_path.exists():
        return None
    data = np.load(stats_path)
    return {"mean": data["mean"], "std": data["std"]}


def load_model_weights(model: torch.nn.Module, load_dir: Path) -> None:
    state_dict = torch.load(load_dir / "model.pt", map_location="cpu", weights_only=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.load_state_dict(state_dict, strict=False)
