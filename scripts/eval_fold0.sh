#!/usr/bin/env bash
# 生成 fold0 提交文件（默认使用 best checkpoint）

set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-outputs/twitter-xlm-r-base-fold0}"
OUTPUT_CSV="${2:-outputs/submission_fold0.csv}"

uv run accelerate launch --num_processes 2 main.py \
  --load "${CHECKPOINT}" \
  --test-path Kaggle2025/kaggle_test.jsonl \
  --eval \
  --use-best \
  --batch-size 128 \
  --output-csv "${OUTPUT_CSV}"
