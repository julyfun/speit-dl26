#!/usr/bin/env python3
"""HF fine-tuning entry point for Influencer vs Observer classification."""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import (
    TweetDataset,
    build_metadata_stats,
    create_dataloader,
    get_fold_indices,
    load_jsonl,
)
from src.features import METADATA_NAMES
from src.model import FusionClassifier
from src.utils import (
    ensure_hub_cached,
    load_metadata_stats,
    load_model_weights,
    load_trainer_state,
    save_best_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF tweet classification fine-tuning")
    parser.add_argument("--load", type=str, default=None, help="Load checkpoint path")
    parser.add_argument("--save", type=str, default=None, help="Save checkpoint path")
    parser.add_argument(
        "--target-train-iteration",
        type=int,
        default=5,
        help="Total number of training epochs to reach",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate on test set")
    parser.add_argument("--train-path", type=str, default="Kaggle2025/train.jsonl")
    parser.add_argument("--test-path", type=str, default="Kaggle2025/kaggle_test.jsonl")
    parser.add_argument(
        "--model-name",
        type=str,
        default="cardiffnlp/twitter-xlm-roberta-large-2022",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-probs", type=str, default=None)
    parser.add_argument("--use-metadata", action="store_true")
    parser.add_argument(
        "--use-best",
        action="store_true",
        help="Load best/ checkpoint for eval (default when best/ exists)",
    )
    return parser.parse_args()


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {k: v for k, v in vars(args).items() if k != "eval"}


def class_weights_from_labels(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def format_cls_metrics(labels: np.ndarray, preds: np.ndarray) -> str:
    parts = [f"pred1={preds.mean():.3f}"]
    for cls in (0, 1):
        mask = labels == cls
        if mask.any():
            parts.append(f"rec{cls}={(preds[mask] == cls).mean():.3f}")
    return " ".join(parts)


def resolve_pretrained_paths(
    model_name: str,
    accelerator: Accelerator,
    checkpoint_dir: Path | None,
) -> tuple[str, str]:
    """Return (tokenizer_path, encoder_path), always fully local after this call."""
    if checkpoint_dir and (checkpoint_dir / "tokenizer").exists():
        if (checkpoint_dir / "encoder").exists():
            return str(checkpoint_dir / "tokenizer"), str(checkpoint_dir / "encoder")
        cache_dir = ensure_hub_cached(model_name, accelerator)
        return str(checkpoint_dir / "tokenizer"), str(cache_dir)

    cache_dir = ensure_hub_cached(model_name, accelerator)
    return str(cache_dir), str(cache_dir)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    accelerator: Accelerator,
    loss_fn: nn.Module,
    train: bool,
) -> tuple[float, float, str]:
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    progress = tqdm(dataloader, disable=not accelerator.is_local_main_process, leave=False)

    for batch in progress:
        labels = batch.pop("labels")
        batch.pop("challenge_id", None)
        batch_size = labels.size(0)

        step_ctx = accelerator.accumulate(model) if train else contextlib.nullcontext()
        with torch.set_grad_enabled(train):
            with step_ctx:
                with accelerator.autocast():
                    if "metadata" in batch:
                        logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            metadata=batch["metadata"],
                        )
                    else:
                        logits = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    loss = loss_fn(logits, labels)

                if train:
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach().float().item() * batch_size
        total_samples += batch_size
        preds = logits.detach().argmax(dim=-1)
        gathered_preds, gathered_labels = accelerator.gather_for_metrics((preds, labels))
        all_preds.extend(gathered_preds.cpu().tolist())
        all_labels.extend(gathered_labels.cpu().tolist())

        if accelerator.is_local_main_process:
            progress.set_postfix(loss=f"{loss.detach().float().item():.4f}")

    avg_loss = 0.0
    if total_samples > 0:
        loss_sum = torch.tensor(total_loss, device=accelerator.device, dtype=torch.float64)
        sample_count = torch.tensor(float(total_samples), device=accelerator.device, dtype=torch.float64)
        loss_sum = accelerator.reduce(loss_sum, reduction="sum")
        sample_count = accelerator.reduce(sample_count, reduction="sum")
        avg_loss = (loss_sum / sample_count).item()

    label_arr = np.asarray(all_labels)
    pred_arr = np.asarray(all_preds)
    accuracy = accuracy_score(label_arr, pred_arr)
    summary = "" if train else format_cls_metrics(label_arr, pred_arr)
    return avg_loss, accuracy, summary


@torch.no_grad()
def predict_test(
    model: nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_ids: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        challenge_ids = batch.pop("challenge_id")
        batch.pop("labels", None)

        with accelerator.autocast():
            if "metadata" in batch:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    metadata=batch["metadata"],
                )
            else:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        gathered_ids, gathered_preds, gathered_probs = accelerator.gather_for_metrics(
            (challenge_ids, preds, probs)
        )

        all_ids.extend(gathered_ids.cpu().tolist())
        all_preds.extend(gathered_preds.cpu().tolist())
        all_probs.extend(gathered_probs.cpu().tolist())

    return (
        np.asarray(all_ids, dtype=np.int64),
        np.asarray(all_preds, dtype=np.int64),
        np.asarray(all_probs, dtype=np.float32),
    )


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用，训练会落在 CPU 上（极慢）。\n"
            f"  torch={torch.__version__}, cuda={torch.version.cuda}\n"
            "  请执行: uv sync --reinstall-package torch\n"
            "  并确认: uv run python -c \"import torch; print(torch.cuda.is_available())\" 输出 True\n"
            "  需使用 cu124 版 PyTorch（驱动 CUDA 12.4），不要用 cu130。"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_cuda_available()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )

    if args.eval:
        if not args.load:
            raise ValueError("--eval requires --load")
        run_eval(args, accelerator)
        return

    run_train(args, accelerator)


def run_train(args: argparse.Namespace, accelerator: Accelerator) -> None:
    if not args.save:
        raise ValueError("Training requires --save")

    save_dir = Path(args.save)
    train_records = load_jsonl(args.train_path)
    labels = np.asarray([int(r["label"]) for r in train_records], dtype=np.int64)
    train_idx, val_idx = get_fold_indices(labels, args.fold, args.num_folds, args.seed)

    train_subset = [train_records[i] for i in train_idx]
    val_subset = [train_records[i] for i in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    completed_epochs = 0
    global_step = 0
    best_metric = -1.0
    best_epoch = 0
    metadata_stats = None
    model_name = args.model_name
    use_metadata = args.use_metadata

    if args.load and Path(args.load).exists():
        state = load_trainer_state(Path(args.load))
        completed_epochs = int(state["completed_train_iteration"])
        global_step = int(state.get("global_step", 0))
        best_metric = float(state.get("best_metric", -1.0))
        best_epoch = int(state.get("best_epoch", 0))
        model_name = state.get("model_name", args.model_name)
        use_metadata = state.get("args", {}).get("use_metadata", use_metadata)
        metadata_stats = load_metadata_stats(Path(args.load))
        if accelerator.is_main_process:
            print(f"Resuming from epoch {completed_epochs}, best acc={best_metric:.4f}")

    if metadata_stats is None and use_metadata:
        metadata_stats = build_metadata_stats(train_subset)

    if completed_epochs >= args.target_train_iteration:
        if accelerator.is_main_process:
            print(
                f"Already reached target ({completed_epochs} >= {args.target_train_iteration}). "
                "Increase --target-train-iteration to continue."
            )
        return

    checkpoint_dir = Path(args.load) if args.load and Path(args.load).exists() else None
    tokenizer_path, encoder_path = resolve_pretrained_paths(model_name, accelerator, checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    train_dataset = TweetDataset(
        train_subset,
        tokenizer,
        metadata_stats,
        args.max_length,
        use_metadata,
        train_labels,
    )
    val_dataset = TweetDataset(
        val_subset,
        tokenizer,
        metadata_stats,
        args.max_length,
        use_metadata,
        val_labels,
    )

    per_device_batch = args.batch_size
    train_loader = create_dataloader(
        train_dataset, per_device_batch, shuffle=True, num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        val_dataset, per_device_batch, shuffle=False, num_workers=args.num_workers
    )

    model = FusionClassifier(
        model_name=encoder_path,
        num_metadata=len(METADATA_NAMES),
        use_metadata=use_metadata,
        dropout=args.dropout,
        local_files_only=True,
    )
    if args.load and (Path(args.load) / "model.pt").exists():
        load_model_weights(model, Path(args.load))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.load and (Path(args.load) / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(Path(args.load) / "optimizer.pt", map_location="cpu", weights_only=True)
        )

    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_weights_from_labels(train_labels).to(accelerator.device)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    if accelerator.is_main_process:
        print(
            f"Train class counts: 0={class_counts[0]}, 1={class_counts[1]}, "
            f"weights={[round(w, 3) for w in class_weights.tolist()]}"
        )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    remaining_epochs = max(args.target_train_iteration - completed_epochs, 0)
    total_updates = updates_per_epoch * remaining_epochs
    warmup_steps = int(total_updates * args.warmup_ratio) if total_updates > 0 else 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    if args.load and (Path(args.load) / "scheduler.pt").exists():
        scheduler.load_state_dict(
            torch.load(Path(args.load) / "scheduler.pt", map_location="cpu", weights_only=True)
        )
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        print(
            f"Train samples={len(train_subset)}, val samples={len(val_subset)}, "
            f"per_device_batch={per_device_batch}, gpus={accelerator.num_processes}, "
            f"steps_per_epoch={len(train_loader)}, "
            f"epochs {completed_epochs}->{args.target_train_iteration}"
        )

    patience_counter = 0
    for epoch in range(completed_epochs, args.target_train_iteration):
        if accelerator.is_main_process:
            print(f"\n=== Epoch {epoch + 1}/{args.target_train_iteration} ===")

        train_loss, train_acc, _ = run_epoch(
            model, train_loader, optimizer, scheduler, accelerator, loss_fn, True
        )
        val_loss, val_acc, val_diag = run_epoch(
            model, val_loader, optimizer, None, accelerator, loss_fn, False
        )
        global_step += updates_per_epoch
        completed_epochs = epoch + 1

        if accelerator.is_main_process:
            print(
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f} | {val_diag}"
            )

        improved = val_acc > best_metric
        if improved:
            best_metric = val_acc
            best_epoch = completed_epochs
            patience_counter = 0
        else:
            patience_counter += 1

        if accelerator.is_main_process:
            saved_args = args_to_dict(args)
            saved_args["use_metadata"] = use_metadata
            trainer_state = {
                "completed_train_iteration": completed_epochs,
                "global_step": global_step,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "model_name": model_name,
                "args": saved_args,
            }
            unwrapped = accelerator.unwrap_model(model)
            save_checkpoint(
                save_dir,
                unwrapped,
                tokenizer,
                optimizer,
                scheduler,
                saved_args,
                metadata_stats,
                completed_epochs,
                global_step,
                best_metric,
                best_epoch,
            )
            if improved:
                save_best_checkpoint(
                    save_dir,
                    unwrapped,
                    tokenizer,
                    metadata_stats,
                    trainer_state,
                )
            tag = " (best)" if improved else ""
            print(f"Saved checkpoint to {save_dir}{tag}")

        accelerator.wait_for_everyone()

        if patience_counter >= args.early_stopping_patience:
            if accelerator.is_main_process:
                print(f"Early stopping after {args.early_stopping_patience} epochs without improvement.")
            break

    if accelerator.is_main_process:
        print(
            f"Training finished. Best validation accuracy: {best_metric:.4f} "
            f"(epoch {best_epoch}). Best checkpoint: {save_dir / 'best'}"
        )


def resolve_eval_checkpoint(load_path: str, use_best: bool) -> Path:
    load_dir = Path(load_path)
    best_dir = load_dir / "best"
    if use_best or (best_dir / "model.pt").exists():
        if (best_dir / "model.pt").exists():
            return best_dir
    if not (load_dir / "model.pt").exists():
        raise FileNotFoundError(f"No checkpoint found under {load_dir}")
    return load_dir


def run_eval(args: argparse.Namespace, accelerator: Accelerator) -> None:
    load_dir = resolve_eval_checkpoint(args.load, args.use_best)
    if accelerator.is_main_process and load_dir.name == "best":
        print(f"Using best checkpoint: {load_dir}")
    state = load_trainer_state(load_dir)
    model_name = state.get("model_name", args.model_name)
    use_metadata = state["args"].get("use_metadata", args.use_metadata)
    max_length = state["args"].get("max_length", args.max_length)
    metadata_stats = load_metadata_stats(load_dir)

    encoder_path = str(load_dir / "encoder") if (load_dir / "encoder").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(load_dir / "tokenizer")
    test_records = load_jsonl(args.test_path)

    test_dataset = TweetDataset(
        test_records,
        tokenizer,
        metadata_stats,
        max_length,
        use_metadata,
        labels=None,
    )
    test_loader = create_dataloader(
        test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = FusionClassifier(
        model_name=encoder_path,
        num_metadata=len(METADATA_NAMES),
        use_metadata=use_metadata,
        dropout=state["args"].get("dropout", args.dropout),
        local_files_only=(load_dir / "encoder").exists(),
    )
    load_model_weights(model, load_dir)
    model, test_loader = accelerator.prepare(model, test_loader)

    ids, preds, probs = predict_test(model, test_loader, accelerator)

    if accelerator.is_main_process:
        unique_ids, unique_idx = np.unique(ids, return_index=True)
        unique_preds = preds[unique_idx]
        unique_probs = probs[unique_idx]

        output_csv = args.output_csv or str(load_dir / "submission.csv")
        df = pd.DataFrame({"ID": unique_ids, "Prediction": unique_preds})
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved submission to {output_csv}")

        probs_path = args.output_probs or str(load_dir / "test_probs.npy")
        np.save(probs_path, unique_probs)
        print(f"Saved probabilities to {probs_path}")


if __name__ == "__main__":
    main()
