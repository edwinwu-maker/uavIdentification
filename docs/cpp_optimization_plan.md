# CPP 快速排查与效果优化方案

## Summary

当前 DroneRFa 数据集包含 14 类无人机/背景类别，原始数据规模约 380G。全量预计算和训练成本过高，不适合直接用于排查 CPP 归一化、FAM 参数和模型结构问题。

优化路线是先建立小规模、类别均衡、可快速复现的实验集，用低成本实验筛掉无效方案；再逐级放大到中等规模和全量数据验证。整体顺序为：

1. 小规模均衡数据集快速筛选方案。
2. 优化 CPP 归一化和 FAM 参数。
3. 适配 CPP 的模型结构。
4. 最后进行全量验证。

## Key Changes

### 建立分层数据集

建立三档实验数据集：

- `smoke`：每类 1 个 `.mat` 文件，每文件 1-3 个样本，用于验证预计算、训练和评估流程是否能跑通。
- `debug`：每类 3-5 个 `.mat` 文件，每文件 5-20 个样本，用于比较 CPP 归一化、FAM 参数和模型结构。
- `full`：全量数据，只用于验证已经在 `debug` 阶段稳定有效的方案。

预计算脚本需要支持按类别和样本数筛选，避免只使用现有 `--max-files` 造成类别不均衡。建议增加：

- `--classes T0000 T0010 ...`
- `--files-per-class N`
- `--max-samples-per-file N`

STFT 和 CPP 对比时必须使用同一批原始 `.mat` 文件和样本索引，保证结果可比。

### 优先优化 CPP 输入

暂时不使用当前 `--use-rf-segmentation`。当前 RF segmentation 默认只保留较短片段，并且两个 RF 通道独立选帧，可能破坏双通道时间对应关系，也会降低 FAM 估计稳定性。

优先比较以下 CPP 归一化方案：

- 当前 `max normalization`
- `log1p + per-channel per-sample z-score`
- `log1p + train-set per-channel mean/std`

优先比较以下 FAM 参数：

- 当前 `fam_nfft=256, fam_hop=256`
- `fam_nfft=64, fam_hop=64`
- `fam_nfft=64, fam_hop=32`

### 适配 CPP 模型结构

保留当前 ResNet-18 作为基线，同时增加 CPP-ResNet-small：

- `conv1` 从 `7x7 stride=2` 改为 `3x3 stride=1`
- 去掉或延后早期 maxpool

目标是避免 CPP 稀疏峰值在前几层被过早下采样抹掉。

Transformer 不作为第一优先级。如果 CPP-ResNet-small 仍然不足，再尝试 CNN + Attention。纯 Transformer/ViT 只作为最后的对照实验。

## Experiment Flow

### 1. 构建 debug 数据集

每类均衡抽取少量 `.mat` 文件和样本，生成对应的 STFT H5 和 CPP H5。

要求：

- 每类文件数一致或尽量接近。
- 每个 `.mat` 文件抽取的样本数一致。
- STFT 和 CPP 使用同一批原始样本。
- 保存实验数据清单，记录类别、文件名和样本索引。

### 2. 验证 CPP 归一化

固定模型和 FAM 参数，只改变归一化方式：

- A：当前 CPP，`max normalization`
- B：`log1p + per-channel per-sample z-score`
- C：`log1p + train-set per-channel mean/std`

如果 B/C 相比 A 没有稳定提升，先不进入模型替换实验。

### 3. 验证 FAM 参数

在最佳归一化方案上比较：

- `fam_nfft=256, fam_hop=256`
- `fam_nfft=64, fam_hop=64`
- `fam_nfft=64, fam_hop=32`

优先保留训练曲线更稳定、macro F1 更高、混淆矩阵改善更明确的配置。

### 4. 验证模型结构

使用最佳 CPP H5，对比：

- 当前 ResNet-18
- CPP-ResNet-small

如果 CPP-ResNet-small 明显优于原始 ResNet-18，再进入 CNN + Attention 实验。

### 5. 逐级放大

debug 阶段有效后，扩展到更多文件和样本。中等规模仍然有效后，再跑 14 类全量训练。

小规模实验只用于筛选方向，最终结论必须以中等规模和全量验证为准。

## Test Plan

每组实验固定：

- 相同类别。
- 相同 `.mat` 文件。
- 相同样本索引。
- 相同 train/val/test seed。
- 相同 epoch、patience、batch size、learning rate。

每组实验记录：

- 数据规模：类别数、文件数、每文件样本数、总样本数。
- CPP 参数：归一化方式、`fam_nfft`、`fam_hop`。
- 模型参数：原始 ResNet-18、CPP-ResNet-small 或后续 attention 模型。
- best val accuracy。
- test accuracy。
- macro F1。
- confusion matrix。
- CPP PNG 可视化。
- 训练耗时。

debug 阶段每个候选方案至少跑 2-3 个 seed，避免小样本偶然结论。

只有满足以下条件的方案才进入 full：

- debug 集指标优于当前 CPP 基线。
- 中等规模数据集仍然有效。
- 训练曲线没有明显过拟合或剧烈震荡。
- CPP PNG 可视化没有明显退化。
- 多 seed 排名基本一致。

## Assumptions

- 当前目标是快速定位 CPP 效果差的原因，而不是直接跑全量最优实验。
- 小规模数据集只用于筛选方向，不作为最终结论。
- 暂不优先使用当前 RF segmentation。
- 暂不优先直接替换为纯 Transformer。
- 后续代码实现应保持改动可控，每次实验只改变一个关键变量。
