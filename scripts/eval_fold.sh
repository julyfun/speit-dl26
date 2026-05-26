#!/usr/bin/env bash
# 对指定 fold 的 checkpoint 生成提交与概率文件
#
# 用法:
#   bash scripts/eval_fold.sh 0
#   bash scripts/eval_fold.sh 2 outputs/submission_fold2.csv

set -euo pipefail
cd "$(dirname "$0")/.."

FOLD="${FOLD:-${1:-0}}"
CHECKPOINT="${CHECKPOINT:-${2:-outputs/twitter-xlm-r-large-fold${FOLD}}}"
OUTPUT_CSV="${OUTPUT_CSV:-${3:-outputs/submission_fold${FOLD}.csv}}"
OUTPUT_PROBS="${OUTPUT_PROBS:-${CHECKPOINT}/best/test_probs.npy}"

echo "Eval fold ${FOLD}: ${CHECKPOINT} -> ${OUTPUT_CSV}"

uv run accelerate launch --num_processes 2 --mixed_precision bf16 main.py \
  --load "${CHECKPOINT}" \
  --test-path Kaggle2025/kaggle_test.jsonl \
  --eval \
  --use-best \
  --batch-size 128 \
  --output-csv "${OUTPUT_CSV}" \
  --output-probs "${OUTPUT_PROBS}"
