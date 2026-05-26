#!/usr/bin/env bash
# Fold 0 训练脚本 — 双卡 RTX 4090 (48GB)
# 若 OOM，将 BATCH_SIZE 降到 48 或 32，并相应增大 GRAD_ACCUM

set -euo pipefail
cd "$(dirname "$0")/.."

BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

uv run accelerate launch --num_processes "${NUM_PROCESSES}" main.py \
  --model-name cardiffnlp/twitter-xlm-roberta-base \
  --train-path Kaggle2025/train.jsonl \
  --test-path Kaggle2025/kaggle_test.jsonl \
  --save outputs/twitter-xlm-r-base-fold0 \
  --target-train-iteration 5 \
  --fold 0 \
  --num-folds 5 \
  --seed 42 \
  --max-length 256 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --use-metadata
