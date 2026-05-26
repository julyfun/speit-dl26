# SPEIT DL26 — Influencer vs Observer 分类

基于 Hugging Face 的三路特征融合微调：`cardiffnlp/twitter-xlm-roberta-base` + 结构化 metadata MLP。

## 环境

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 应看到 True 和 device_count=2
```

**注意**：本机驱动为 CUDA 12.4，必须使用 `cu124` 版 PyTorch，不能用 `cu130`（会 silently 回退 CPU）。

## 训练（双卡 4090，fold 0）

默认 `batch_size=64` / 卡，`max_length=256`，bf16，约可吃满 48GB 显存。若 OOM：

```bash
BATCH_SIZE=48 GRAD_ACCUM=2 bash scripts/train_fold0.sh
# 或
BATCH_SIZE=32 GRAD_ACCUM=2 bash scripts/train_fold0.sh
```

或直接：

```bash
uv run accelerate launch --num_processes 2 main.py \
  --model-name cardiffnlp/twitter-xlm-roberta-base \
  --train-path Kaggle2025/train.jsonl \
  --test-path Kaggle2025/kaggle_test.jsonl \
  --save outputs/twitter-xlm-r-base-fold0 \
  --target-train-iteration 5 \
  --fold 0 \
  --num-folds 5 \
  --seed 42 \
  --max-length 256 \
  --batch-size 64 \
  --use-metadata
```

## 断点续训

已训 2 轮、目标 5 轮 → 只会再训 3 轮：

```bash
uv run accelerate launch --num_processes 2 main.py \
  --load outputs/twitter-xlm-r-base-fold0 \
  --save outputs/twitter-xlm-r-base-fold0 \
  --target-train-iteration 5
```

## 生成提交

```bash
bash scripts/eval_fold0.sh
# 或
uv run accelerate launch main.py \
  --load outputs/twitter-xlm-r-base-fold0 \
  --eval \
  --use-best \
  --output-csv outputs/submission_fold0.csv
```

输出：
- `outputs/submission_fold0.csv` — Kaggle 格式 `ID,Prediction`
- `outputs/twitter-xlm-r-base-fold0/best/test_probs.npy` — 概率，供后续 ensemble

## 目录结构

```
main.py              # 训练 / 评估入口
src/
  features.py        # 文本与 metadata 特征
  data.py            # Dataset / DataLoader / 5-fold 划分
  model.py           # Transformer + Metadata MLP 融合
  utils.py           # checkpoint / seed
outputs/             # checkpoint 与提交（gitignore）
scripts/
  train_fold0.sh
  eval_fold0.sh
```

## Checkpoint 说明

每次 epoch 保存到 `--save` 目录；验证集 accuracy 提升时额外保存到 `--save/best/`。

`trainer_state.json` 字段：`completed_train_iteration`、`global_step`、`best_metric`、`best_epoch`、`model_name`、`args`。
