#!/usr/bin/env bash
# 训练指定 fold — twitter-xlm-roberta-large-2022，双卡 RTX 4090 (48GB)
#
# 用法:
#   bash scripts/train_fold.sh 0
#   bash scripts/train_fold.sh 3
#   FOLD=1 BATCH_SIZE=64 bash scripts/train_fold.sh
#
# 若 OOM: BATCH_SIZE=64 GRAD_ACCUM=2 bash scripts/train_fold.sh 0
# 续训:  LOAD=outputs/twitter-xlm-r-large-fold0 bash scripts/train_fold.sh 0

set -euo pipefail
cd "$(dirname "$0")/.."

FOLD="${FOLD:-${1:-0}}"
NUM_FOLDS="${NUM_FOLDS:-5}"
SAVE_DIR="${SAVE_DIR:-outputs/twitter-xlm-r-large-fold${FOLD}}"
LOAD="${LOAD:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

ARGS=(
  --model-name cardiffnlp/twitter-xlm-roberta-large-2022
  --train-path Kaggle2025/train.jsonl
  --test-path Kaggle2025/kaggle_test.jsonl
  --save "${SAVE_DIR}"
  --target-train-iteration 5
  --fold "${FOLD}"
  --num-folds "${NUM_FOLDS}"
  --seed 42
  --max-length 256
  --batch-size "${BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}"
  --lr 2e-5
  --weight-decay 0.01
  --dropout 0.15
  --label-smoothing 0.02
  --use-metadata
)

[[ -n "${LOAD}" ]] && ARGS+=(--load "${LOAD}")

echo "Training fold ${FOLD}/${NUM_FOLDS} -> ${SAVE_DIR}"

uv run accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  main.py "${ARGS[@]}"
