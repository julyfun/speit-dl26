# SPEIT DL26 — Influencer vs Observer 分类

基于 Hugging Face 的三路特征融合微调：`cardiffnlp/twitter-xlm-roberta-large-2022` + 结构化 metadata MLP。

## 环境

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 应看到 True 和 device_count=2
```

**注意**：本机驱动为 CUDA 12.4，必须使用 `cu124` 版 PyTorch，不能用 `cu130`（会 silently 回退 CPU）。

## 训练（双卡 4090，任意 fold）

```bash
bash scripts/train_fold.sh 0    # fold 0
bash scripts/train_fold.sh 1    # fold 1
# ... 到 fold 4
```

默认 `batch_size=128` / 卡，checkpoint 目录 `outputs/twitter-xlm-r-large-fold{N}`。

若 OOM：

```bash
BATCH_SIZE=64 GRAD_ACCUM=2 bash scripts/train_fold.sh 0
```

`train_fold0.sh` 仍可用，等价于 `train_fold.sh 0`。

或直接：

```bash
uv run accelerate launch --num_processes 2 main.py \
  --model-name cardiffnlp/twitter-xlm-roberta-large-2022 \
  --train-path Kaggle2025/train.jsonl \
  --test-path Kaggle2025/kaggle_test.jsonl \
  --save outputs/twitter-xlm-r-large-fold0 \
  --target-train-iteration 5 \
  --fold 0 \
  --num-folds 5 \
  --seed 42 \
  --max-length 256 \
  --batch-size 32 \
  --use-metadata
```

## 断点续训

已训 2 轮、目标 5 轮 → 只会再训 3 轮：

```bash
uv run accelerate launch --num_processes 2 main.py \
  --load outputs/twitter-xlm-r-large-fold0 \
  --save outputs/twitter-xlm-r-large-fold0 \
  --target-train-iteration 5
```

## 生成提交

```bash
bash scripts/eval_fold.sh 0
bash scripts/eval_fold.sh 1
# ...
```

或：

```bash
bash scripts/eval_fold0.sh
```

输出：
- `outputs/submission_fold0.csv` — Kaggle 格式 `ID,Prediction`
- `outputs/twitter-xlm-r-large-fold0/best/test_probs.npy` — 概率，供后续 ensemble

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
  train_fold.sh      # bash scripts/train_fold.sh {0..4}
  eval_fold.sh
  train_fold0.sh     # -> train_fold.sh 0
  eval_fold0.sh
```

## Checkpoint 说明

每次 epoch 保存到 `--save` 目录；验证集 accuracy 提升时额外保存到 `--save/best/`。

`trainer_state.json` 字段：`completed_train_iteration`、`global_step`、`best_metric`、`best_epoch`、`model_name`、`args`。
