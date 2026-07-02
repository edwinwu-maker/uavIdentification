# 随机 SNR 加噪训练集成方案

## 方案概述

采用训练数据预计算阶段 IQ 层随机加噪方案：每个原始 IQ 样本只随机分配一个 SNR，加入 AWGN 后再计算 STFT/CPP 特征并保存到新的 H5 目录。

默认 SNR 范围使用 `[-5, 15]` dB，数据量保持约 `1x`，不会像多 SNR 副本方案一样膨胀到数倍。训练脚本继续读取 H5，不改主训练流程。

## 关键改动

- 在 STFT/CPP 预计算脚本中新增随机加噪参数：
  - `--random-snr`
  - `--snr-min -5`
  - `--snr-max 15`
  - `--noise-seed 42`
- 加噪位置固定在原始 IQ 层：
  - 读取 `.mat` 中的 IQ 样本。
  - 为每个样本随机采样一个 `snr_db`。
  - 调用现有 AWGN 逻辑加噪。
  - 再计算 STFT 或 CPP 特征。
- H5 结构保持训练兼容：
  - STFT：`/stft`、`/labels`
  - CPP：`/cpp`、`/labels`
  - 不额外写入 `/snr_db`，避免改变下游 H5 数据结构预期。
- 输出目录建议区分 clean 与 noisy：
  - `DroneRFa_stft_awgn_random_h5`
  - `DroneRFa_cpp_awgn_random_h5`

## 实现细节

- 复用 `src.evaluation.snr_accuracy.add_awgn_for_snr`，避免重复实现 AWGN。
- STFT 路线：
  - 修改 `scripts/precompute_stft_h5.py`。
  - 在 `read_iq_batch(...)` 后，对 batch 内每条 IQ 分配随机 SNR 并加噪。
  - 将加噪后的 batch 传入 `compute_stft(...)`。
- CPP 路线：
  - 修改 `scripts/precompute_cpp_h5.py`。
  - 在 `read_iq_batch(...)` 后，对当前 IQ 样本分配随机 SNR 并加噪。
  - 加噪后再执行可选 RF segmentation 和 `compute_cpp(...)`。
- 随机性要求：
  - 同一 `--noise-seed`、同一输入数据、同一参数下，生成的 SNR 和特征应可复现。
  - 每个样本使用独立 RNG，避免 batch size 或处理顺序改变导致 SNR 分配变化。
- 训练方式保持不变：

```bash
python scripts/train.py --feature stft --data-dir <DroneRFa_stft_awgn_random_h5>
python scripts/train.py --feature cpp --data-dir <DroneRFa_cpp_awgn_random_h5>
```

## 存储与精度权衡

- 数据量约等于 clean H5：每个原始样本只生成一个含噪特征样本。
- 默认 `[-5, 15]` dB 的原因：
  - 避免大量样本落入 `-15` 到 `-5` dB 的极低 SNR 区间，减少训练标签信息被噪声淹没。
  - 对主流可识别 SNR 区间更友好，训练更稳定。
- 仍建议保留参数支持 `[-15, 15]`：
  - 用于和 TASE-Net 论文设置对齐。
  - 用于测试极低 SNR 鲁棒性是否提升。
- 推荐实验组合：
  - 主模型：`--snr-min -5 --snr-max 15`
  - 对比模型：`--snr-min -15 --snr-max 15`
  - 统一用现有 SNR accuracy 脚本在 `[-15, 15]` 全范围评估。

## 测试计划

小规模生成验证：

```bash
python scripts/precompute_stft_h5.py \
  --data-dir <raw_drone_rfa_dir> \
  --output-dir <tmp_awgn_stft_h5> \
  --max-files 1 \
  --max-samples-per-file 2 \
  --random-snr \
  --snr-min -5 \
  --snr-max 15 \
  --noise-seed 42
```

检查 H5：

- `/stft` 或 `/cpp` 样本数不膨胀。
- H5 仅包含训练需要的数据集，不新增 `/snr_db`。

复现性验证：

- 相同参数和 `noise_seed` 生成两次，小样本特征数据完全一致。

训练验证：

- 使用小 H5 跑 1 个 epoch，确认训练入口无需修改。

效果验证：

- 用 clean 训练模型、`[-5, 15]` noisy 训练模型、`[-15, 15]` noisy 训练模型分别跑现有 SNR accuracy 评估脚本。
- 比较低 SNR、中高 SNR 和整体曲线差异。

## 默认假设

- 加噪发生在原始复数 IQ 信号层面，不做特征级加噪。
- 默认不保留 clean 副本，避免存储增长。
- 默认随机 SNR 范围为 `[-5, 15]` dB。
- 当前训练、评估脚本继续基于 H5 特征数据工作。
