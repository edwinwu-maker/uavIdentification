# CPP 归一化与 Segment 合并验证方案

## 目标

只验证两个变量对 CPP/FAM 分类效果的影响：

1. CPP 归一化方式。
2. segment 合并方式。

本轮不验证 `fam_nfft/fam_hop`、模型结构、RF segmentation 或其他新特征，避免变量过多导致无法归因。

## 固定条件

除当前实验变量外，所有实验固定以下配置：

- 原始数据：`/mnt/data/wurixin/DroneRFa`
- 小样本抽取：`--files-per-class 3`
- FAM 参数：`--fam-nfft 256 --fam-hop 256`
- 模型：`resnet18-small-stem`
- 训练 batch size：`64`
- 学习率：`0.001`
- train/val/test split seed：`42`
- early stopping patience：`10`
- 不启用 `--use-rf-segmentation`

实验顺序必须先做归一化验证，再在最佳归一化方式上做 segment 合并验证。

## Experiment 1: CPP 归一化验证

### 目的

验证当前 `max` 归一化是否是 CPP 收敛慢、低 SNR 表现差的主要原因。

### 固定参数

- `--fam-merge mean`
- `--fam-nfft 256`
- `--fam-hop 256`
- `--model resnet18-small-stem`

### 对比组

| 实验名 | 归一化方式 | 合并方式 | FAM 参数 |
| --- | --- | --- | --- |
| `cpp_max_mean_256` | `max` | `mean` | `256/256` |
| `cpp_logz_mean_256` | `log-zscore-sample` | `mean` | `256/256` |

### 预计算命令

```bash
python scripts/precompute_cpp_h5.py \
  --data-dir /mnt/data/wurixin/DroneRFa \
  --output-dir /mnt/ssd/wurixin/small_cpp_norm_max_mean_h5 \
  --files-per-class 3 \
  --cpp-normalization max \
  --fam-merge mean \
  --fam-nfft 256 \
  --fam-hop 256
```

```bash
python scripts/precompute_cpp_h5.py \
  --data-dir /mnt/data/wurixin/DroneRFa \
  --output-dir /mnt/ssd/wurixin/small_cpp_norm_logz_mean_h5 \
  --files-per-class 3 \
  --cpp-normalization log-zscore-sample \
  --fam-merge mean \
  --fam-nfft 256 \
  --fam-hop 256
```

### 训练命令

```bash
python scripts/train.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/small_cpp_norm_max_mean_h5 \
  --model resnet18-small-stem \
  --checkpoint-path small_cpp_norm_max_mean.pth \
  --batch-size 64 \
  --device cuda:3
```

```bash
python scripts/train.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/small_cpp_norm_logz_mean_h5 \
  --model resnet18-small-stem \
  --checkpoint-path small_cpp_norm_logz_mean.pth \
  --batch-size 64 \
  --device cuda:3
```

### 测试命令

```bash
python scripts/evaluate.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/small_cpp_norm_max_mean_h5 \
  --model resnet18-small-stem \
  --model-path outputs/checkpoints/small_cpp_norm_max_mean.pth \
  --batch-size 64 \
  --device cuda:3
```

```bash
python scripts/evaluate.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/small_cpp_norm_logz_mean_h5 \
  --model resnet18-small-stem \
  --model-path outputs/checkpoints/small_cpp_norm_logz_mean.pth \
  --batch-size 64 \
  --device cuda:3
```

### 判定标准

- 优先比较 test accuracy 和 macro F1。
- 同时观察前 10 个 epoch 的收敛速度。
- 如果 `log-zscore-sample` 收敛更快，且 best val/test 指标更高，后续实验全部使用 `log-zscore-sample`。
- 如果两者接近，保留 `max` 作为 baseline，但仍继续做 segment 合并验证。

## Experiment 2: Segment 合并方式验证

### 目的

验证 `mean` 合并是否会稀释短时有效 CPP 峰值，以及 `max` 合并是否更适合当前任务。

### 固定参数

- 使用 Experiment 1 胜出的归一化方式。
- `--fam-nfft 256`
- `--fam-hop 256`
- `--model resnet18-small-stem`

### 对比组

以下命令假设 Experiment 1 中 `log-zscore-sample` 胜出。如果 `max` 胜出，将命令中的 `--cpp-normalization log-zscore-sample` 和路径名替换为 `max`。

| 实验名 | 归一化方式 | 合并方式 | FAM 参数 |
| --- | --- | --- | --- |
| `cpp_logz_merge_mean_256` | `log-zscore-sample` | `mean` | `256/256` |
| `cpp_logz_merge_max_256` | `log-zscore-sample` | `max` | `256/256` |

### 预计算命令

`mean` 组可复用 Experiment 1 的 `small_cpp_norm_logz_mean_h5`。如果需要独立命名，也可重新生成：

```bash
python scripts/precompute_cpp_h5.py \
  --data-dir /mnt/data/wurixin/DroneRFa \
  --output-dir /mnt/ssd/wurixin/small_cpp_logz_merge_mean_h5 \
  --files-per-class 3 \
  --cpp-normalization log-zscore-sample \
  --fam-merge mean \
  --fam-nfft 256 \
  --fam-hop 256
```

```bash
python scripts/precompute_cpp_h5.py \
  --data-dir /mnt/data/wurixin/DroneRFa \
  --output-dir /mnt/ssd/wurixin/small_cpp_logz_merge_max_h5 \
  --files-per-class 3 \
  --cpp-normalization log-zscore-sample \
  --fam-merge max \
  --fam-nfft 256 \
  --fam-hop 256
```

### 训练命令

```bash
python scripts/train.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/small_cpp_logz_merge_mean_h5 \
  --model resnet18-small-stem \
  --checkpoint-path small_cpp_logz_merge_mean.pth \
  --batch-size 64 \
  --device cuda:3
```

```bash
python scripts/train.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/small_cpp_logz_merge_max_h5 \
  --model resnet18-small-stem \
  --checkpoint-path small_cpp_logz_merge_max.pth \
  --batch-size 64 \
  --device cuda:3
```

### 测试命令

```bash
python scripts/evaluate.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/small_cpp_logz_merge_mean_h5 \
  --model resnet18-small-stem \
  --model-path outputs/checkpoints/small_cpp_logz_merge_mean.pth \
  --batch-size 64 \
  --device cuda:3
```

```bash
python scripts/evaluate.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/small_cpp_logz_merge_max_h5 \
  --model resnet18-small-stem \
  --model-path outputs/checkpoints/small_cpp_logz_merge_max.pth \
  --batch-size 64 \
  --device cuda:3
```

### 判定标准

- 如果 `max` 明显优于 `mean`，说明关键 CPP 峰值可能被均值合并削弱。
- 如果 `mean` 更好或更稳定，说明当前 CPP 更依赖全段稳定循环平稳统计。
- 如果两种合并方式都明显低于 STFT，下一轮再验证 `fam_nfft/fam_hop`，本轮不引入新变量。

## 记录指标

每组实验至少记录以下字段：

```text
exp_name | normalization | fam_merge | best_val | stopped_epoch | test_acc | macro_f1 | train_time
```

建议同时保存：

- 训练日志。
- test confusion matrix。
- `outputs/metrics/*.npy` 指标文件。
- `outputs/figures/*.png` 混淆矩阵图。
- 预计算输出目录名和 checkpoint 名称。

## 最终结论模板

实验完成后只输出三条结论：

```text
1. CPP 归一化：max vs log-zscore-sample，哪个更好，差距是多少。
2. Segment 合并：mean vs max，哪个更好，差距是多少。
3. 最佳 CPP 配置相比当前 STFT baseline，差距是否缩小。
```

如果最佳 CPP 仍明显低于 STFT，再进入下一轮 FAM 参数验证；本轮不修改 `fam_nfft/fam_hop`。
