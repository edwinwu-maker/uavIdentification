# CPP Full 极低 SNR 低于 STFT 的下一步排查方向

## 最新结论

截至 `2026-07-08 22:28:16` 的 full SNR 评估日志，CPP 在 `-15 dB` 和 `-10 dB` 明显低于 STFT，`-7.5 dB` 基本接近 STFT；从 `-5 dB` 和 `-2.5 dB` 开始，CPP 已经略高于 STFT。

因此，当前问题不应表述为“CPP full 数据整体不如 STFT”，而应表述为“在 STFT 和 CPP 都使用 `[-5, 15] dB` random-SNR 训练时，CPP 在训练分布外的极低 SNR 区间外推鲁棒性弱于 STFT”。下一步优先确认 CPP/STFT 训练 H5 的 random-SNR 范围确实一致，再验证 CPP 是否能通过扩展训练 SNR 范围补回 `-15 dB` 和 `-10 dB` 的差距。

## 1. 当前现象

当前 full 数据集评估日志显示：

- 原始数据规模：`389` 个 `.mat` 文件。
- 总样本数：`58980`。
- 划分方式：`train=35388`、`val=11796`、`test=11796`。
- CPP checkpoint：`outputs/checkpoints/cpp_norm_logz_mean.pth`。
- CPP 评估参数已显式使用：`--cpp-normalization log-zscore-sample`。

目前已经观察到的 CPP full SNR accuracy：

| SNR | CPP accuracy | STFT accuracy | CPP - STFT |
| --- | ---: | ---: | ---: |
| `-15 dB` | `0.3187` | `0.4627` | `-0.1440` |
| `-10 dB` | `0.6761` | `0.7375` | `-0.0614` |
| `-7.5 dB` | `0.8466` | `0.8528` | `-0.0062` |
| `-5 dB` | `0.9324` | `0.9257` | `+0.0067` |
| `-2.5 dB` | `0.9670` | `0.9618` | `+0.0052` |

从当前已完成点看，CPP 不是在所有 SNR 区间都明显弱于 STFT，而是主要输在 `-15 dB` 和 `-10 dB` 这类极低 SNR 区间。到 `-7.5 dB` 时，CPP 与 STFT 已经非常接近；从 `-5 dB` 开始，CPP 已经略高于 STFT。

因此，后续排查不应笼统判断为“CPP full 整体不如 STFT”，而应优先定位“CPP 极低 SNR 鲁棒性不足”的原因。

## 2. 已确认与暂时排除的因素

服务器上运行的 CPP SNR 评估命令为：

```bash
nohup python scripts/eval_snr_accuracy_cpp.py \
  --model-path outputs/checkpoints/cpp_norm_logz_mean.pth \
  --device cuda:1 \
  --cpp-normalization log-zscore-sample \
  > eval_cpp_norm_logz_mean_accuracy.log 2>&1 &
```

日志确认模型加载和数据规模如下：

```text
Loaded 58980 samples from 389 files; split train=35388 val=11796 test=11796
Loading model from outputs/checkpoints/cpp_norm_logz_mean.pth
```

因此，“CPP SNR 评估脚本默认使用 `max` normalization，导致评估端与训练端不一致”不再是首要嫌疑。当前评估端已经显式使用 `log-zscore-sample`。

后续仍需确认的是：`cpp_norm_logz_mean.pth` 对应的训练 H5 是否也确实由 `log-zscore-sample + fam-merge mean + fam-nfft 256 + fam-hop 256` 生成。如果训练 H5 和评估参数不匹配，仍可能导致性能偏差。

## 3. 优先排查方向

### 3.1 等 CPP 全部 SNR 点跑完并保存结果

当前已经看到 `-15 dB` 到 `-2.5 dB` 的结果。需要等 CPP 全部 SNR 点跑完，再和 STFT 曲线完整比较。

重点确认：

- `0 dB` 及以上是否继续追平或超过 STFT。
- 高 SNR 区间是否存在明显下降。
- CPP 与 STFT 的差距是否只集中在 `-15 dB` 和 `-10 dB`。

建议不要覆盖已有 CSV 和日志。保存方式示例：

```bash
cp outputs/metrics/cpp_snr_accuracy.csv outputs/metrics/cpp_norm_logz_mean_full_snr_accuracy.csv
cp eval_cpp_norm_logz_mean_accuracy.log outputs/logs/eval_cpp_norm_logz_mean_accuracy.log
```

如果当前脚本默认输出路径会覆盖旧结果，后续评估命令应显式指定输出文件：

```bash
python scripts/eval_snr_accuracy_cpp.py \
  --model-path outputs/checkpoints/cpp_norm_logz_mean.pth \
  --device cuda:1 \
  --cpp-normalization log-zscore-sample \
  --output-csv outputs/metrics/cpp_norm_logz_mean_full_snr_accuracy.csv \
  --output-png outputs/figures/cpp_norm_logz_mean_full_snr_accuracy.png
```

### 3.2 确认 CPP/STFT 训练集的随机 SNR 范围是否一致

重点回查 `cpp_norm_logz_mean.pth` 和 STFT baseline checkpoint 的训练来源：

- 训练命令使用的 `--data-dir` 是哪个 H5 目录。
- 对应 H5 目录是否分别由 `scripts/precompute_cpp_h5.py` 和 `scripts/precompute_stft_h5.py` 生成。
- 预计算时是否传入了 `--clean`。
- 预计算时是否显式传入了 `--snr-min` 和 `--snr-max`。

当前 `scripts/precompute_cpp_h5.py` 的默认行为是：

- 不传 `--clean` 时，会在预计算阶段添加 random-SNR AWGN。
- 默认 `--snr-min -5`。
- 默认 `--snr-max 15`。

`scripts/precompute_stft_h5.py` 也使用相同的默认 random-SNR 范围。因此，如果 CPP 和 STFT 训练 H5 都使用默认 random-SNR 设置，那么二者训练数据都只覆盖 `[-5, 15] dB`。此时评估中的 `-15 dB` 和 `-10 dB` 对两者都属于训练分布之外。

这意味着：训练 SNR 范围不足不是 CPP 相比 STFT 落后的唯一解释。更准确的判断是，在相同 `[-5, 15] dB` 训练范围下，STFT 对 `-15 dB` 和 `-10 dB` 的分布外泛化更强，而 CPP 在极低 SNR 下退化更明显。

### 3.3 检查 small 与 full 实验是否只改变了数据规模

small 数据集下 CPP SNR 曲线优于 STFT，但 full 数据集下 CPP 在极低 SNR 落后。需要确认 small/full 对比时是否只改变了数据规模。

需要核对：

- small 和 full 是否使用相同的 CPP 参数：
  - `--cpp-normalization log-zscore-sample`
  - `--fam-merge mean`
  - `--fam-nfft 256`
  - `--fam-hop 256`
- small 和 full 是否使用相同模型：
  - `resnet18-small-stem`
- small 和 full 是否使用相同训练策略：
  - batch size
  - learning rate
  - early stopping patience
  - split seed
- small 和 full 是否使用相同 random-SNR 范围。

如果 small 使用 clean 数据或不同 random-SNR 范围，而 full 使用 `[-5, 15]` random-SNR，则不能直接把差异归因于数据规模。

## 4. 单变量对照实验：只改训练 SNR 范围

如果确认当前 CPP 和 STFT full checkpoint 的训练 random-SNR 范围都是默认 `[-5, 15]`，下一步最优先的单变量实验是：只把 CPP 训练集 random-SNR 改为 `[-15, 15]`，其他参数全部保持不变。

这个实验的目的不是证明 CPP/STFT 使用了不同训练范围，而是验证：CPP 的极低 SNR 短板能否通过把 `-15 dB` 和 `-10 dB` 纳入训练分布来补回。

固定参数：

- CPP normalization：`log-zscore-sample`
- FAM merge：`mean`
- FAM 参数：`--fam-nfft 256 --fam-hop 256`
- 模型：`resnet18-small-stem`
- 不启用 RF segmentation
- 训练、验证、测试 split seed 保持默认 `42`

### 4.1 生成 `[-15, 15]` CPP 训练 H5

注意：输出路径使用新目录，避免覆盖现有 full CPP H5。

```bash
python scripts/precompute_cpp_h5.py \
  --data-dir /mnt/data/wurixin/DroneRFa \
  --output-dir /mnt/ssd/wurixin/DroneRFa_cpp_logz_mean_awgn_m15_15_h5 \
  --cpp-normalization log-zscore-sample \
  --fam-merge mean \
  --fam-nfft 256 \
  --fam-hop 256 \
  --snr-min -15 \
  --snr-max 15 \
  --noise-seed 42 \
  --device cuda:1
```

### 4.2 训练新的 CPP checkpoint

```bash
python scripts/train.py \
  --feature cpp \
  --data-dir /mnt/ssd/wurixin/DroneRFa_cpp_logz_mean_awgn_m15_15_h5 \
  --model resnet18-small-stem \
  --checkpoint-path cpp_norm_logz_mean_awgn_m15_15.pth \
  --batch-size 64 \
  --device cuda:1
```

### 4.3 用同一 SNR 评估脚本测试

```bash
python scripts/eval_snr_accuracy_cpp.py \
  --model-path outputs/checkpoints/cpp_norm_logz_mean_awgn_m15_15.pth \
  --device cuda:1 \
  --cpp-normalization log-zscore-sample \
  --output-csv outputs/metrics/cpp_norm_logz_mean_awgn_m15_15_snr_accuracy.csv \
  --output-png outputs/figures/cpp_norm_logz_mean_awgn_m15_15_snr_accuracy.png
```

### 4.4 对比对象

至少比较三条曲线：

1. 当前 CPP：`[-5, 15]` random-SNR 训练。
2. 新 CPP：`[-15, 15]` random-SNR 训练。
3. STFT baseline。

重点比较：

- `-15 dB`
- `-10 dB`
- `-7.5 dB`
- `-5 dB`
- `0 dB` 及以上

如果新 CPP 在 `-15 dB` 和 `-10 dB` 明显提升，同时 `0 dB` 以上没有明显下降，则可以认为 CPP 的极低 SNR 短板主要可以通过训练 SNR 覆盖补偿。

如果极低 SNR 提升，但中高 SNR 明显下降，则说明简单扩大到 `[-15, 15]` 会引入训练目标冲突，后续可以考虑非均匀 SNR 采样、分段采样或混合 clean/noisy 训练集。

如果极低 SNR 没有明显提升，则应转向 CPP 特征参数和模型结构排查。

## 5. Per-class 低 SNR 诊断

只看整体 accuracy 不足以判断 CPP 的问题来源。下一步应在低 SNR 点输出每类准确率和混淆矩阵。

建议优先分析：

- `-15 dB`
- `-10 dB`
- `-7.5 dB`

诊断目标：

- 判断 CPP 是否只有少数类别严重掉点。
- 判断 CPP 是否在所有类别上普遍下降。
- 判断低 SNR 下是否集中混淆到某几个类别。

结果解释：

- 如果只有少数类别拖累整体 accuracy，优先查看这些类别的 CPP 图、STFT 图和混淆矩阵。
- 如果所有类别都普遍下降，说明问题更可能来自 CPP 特征在极低 SNR 下整体信号结构被噪声破坏，或者当前训练策略没有学到足够鲁棒的 CPP 表征。
- 如果 CPP 在低 SNR 下集中预测为某几个类别，需要重点检查类别不均衡、训练 split、以及这些类别的随机噪声增强后特征是否过于相似。

## 6. 后续才考虑的方向

在完成训练 SNR 范围对照之前，不建议直接扩大 CPP 参数搜索。否则变量过多，难以判断改进来自哪里。

如果 `[-15, 15]` random-SNR 训练后，CPP 在 `-15 dB` 和 `-10 dB` 仍明显低于 STFT，再进入以下排查。

### 6.1 比较 segment 合并方式

比较：

- `fam-merge mean`
- `fam-merge max`

可能解释：

- `mean` 合并可能稀释短时有效循环平稳峰值。
- `max` 合并可能更保留局部显著峰值，但也可能更容易放大噪声峰值。

### 6.2 比较 FAM 参数

比较：

- `fam_nfft/fam_hop = 256/256`
- `fam_nfft/fam_hop = 64/64`

可能解释：

- `256/256` 频率分辨率更高，但低 SNR 下估计可能更稀疏、更不稳定。
- `64/64` 频率分辨率更低，但统计上可能更平滑，低 SNR 下可能更稳。

### 6.3 检查模型结构

当前 CPP 使用 `resnet18-small-stem`，已经比标准 ResNet-18 更适合小尺寸 CPP 图。

如果特征参数和训练 SNR 范围都无法改善，再考虑：

- 更轻量的 CNN。
- 对 CPP 峰值更友好的 attention 模块。
- 多尺度卷积结构。

不建议过早引入复杂模型。应先确认 CPP 特征和训练数据分布没有明显问题。

## 7. 判定标准

### 7.1 支持“CPP 可通过扩展训练 SNR 覆盖补偿”的证据

满足以下条件时，可以认为 CPP 的极低 SNR 短板主要来自训练分布覆盖不足，且可以通过扩展训练 SNR 范围补偿：

- `[-15, 15]` 训练后，`-15 dB` 和 `-10 dB` 明显提升。
- `-7.5 dB` 到 `10 dB` 不明显低于当前 CPP。
- full SNR 曲线整体更接近 STFT。
- per-class 结果显示大多数类别低 SNR 都有改善。

### 7.2 支持“CPP 特征参数不足”的证据

满足以下条件时，应转向 FAM 参数和合并方式：

- `[-15, 15]` 训练后，`-15 dB` 和 `-10 dB` 提升很小。
- CPP 在低 SNR 下仍明显低于 STFT。
- per-class 结果显示大多数类别普遍下降，而不是个别类别问题。

### 7.3 支持“类别或数据分布问题”的证据

满足以下条件时，应优先做类别级分析：

- 只有少数类别在低 SNR 下 accuracy 接近 0。
- 混淆矩阵显示错误集中在固定类别对之间。
- small 数据集表现好，但 full 数据集中新增文件导致某些类别显著退化。

## 8. 当前最可能方向

当前最可能的方向是：

CPP full 不是整体不如 STFT，而是在相同 `[-5, 15] dB` random-SNR 训练条件下，CPP 对 `-15 dB` 和 `-10 dB` 的分布外泛化弱于 STFT。

如果当前 CPP 和 STFT 训练集都只覆盖默认 `[-5, 15] dB`，那么下一步应先做 CPP 的 `[-15, 15] dB` random-SNR 训练对照，而不是立刻修改 FAM 参数或模型结构。当前 `-5 dB` 和 `-2.5 dB` 已经超过 STFT，这进一步说明排查重点应放在训练范围外的 `-15 dB` 和 `-10 dB`，并验证 CPP 是否能通过训练覆盖补回这两个点。
