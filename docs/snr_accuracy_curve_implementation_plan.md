# SNR-Accuracy Curve Implementation Plan

## 背景与结论

当前项目使用 DroneRFa 实采无人机射频信号数据集。该数据集样本文件名和现有 `.h5` 缓存中没有真实信噪比标签，现有 `SpectrogramDataset` 和 `CppDataset` 也只返回特征张量与类别标签。因此，训练完成后不能直接绘制严格意义上的“真实采集 SNR-准确率曲线”。

可行方案是绘制“人工加噪 SNR-准确率曲线”：保持训练好的模型不变，在测试阶段从原始 IQ 信号出发，对测试样本叠加不同强度的复高斯白噪声，再重新计算 STFT 或 CPP/FAM 特征，最后评估每个 SNR 档位下的分类准确率。

这条曲线的含义是模型在不同人工噪声强度下的鲁棒性，而不是数据集中真实环境 SNR 的统计表现。

## 当前项目相关结构

项目当前流程如下：

```text
原始 .mat IQ 文件
  -> scripts/precompute_stft_h5.py / scripts/precompute_cpp_h5.py
  -> train_stft.py / train_cpp.py
  -> test_stft.py / test_cpp.py
  -> checkpoints/best_stft_model.pth / checkpoints/best_cpp_model.pth
```

现有测试脚本 `test_stft.py` 和 `test_cpp.py` 会在预计算 `.h5` 测试集上输出整体 accuracy、precision、recall、F1 和混淆矩阵，但不会按 SNR 分组，因为 `.h5` 文件中目前只有：

```text
/stft 或 /cpp
/labels
```

没有：

```text
/snr
```

也没有每条样本对应的 SNR 元数据。

## 目标

新增一个独立评估流程，在不改动训练逻辑、不重新训练模型的前提下，生成：

```text
checkpoints/stft_snr_accuracy.csv
checkpoints/stft_snr_accuracy.png
```

或：

```text
checkpoints/cpp_snr_accuracy.csv
checkpoints/cpp_snr_accuracy.png
```

CSV 记录每个 SNR 档位的样本数、正确数和准确率；PNG 绘制 SNR-Accuracy 曲线。

## 推荐实现方案

推荐新增独立脚本，而不是修改 `test_stft.py` / `test_cpp.py`：

```text
scripts/eval_snr_accuracy_stft.py
scripts/eval_snr_accuracy_cpp.py
```

原因：

1. 现有测试脚本基于预计算 `.h5`，无法在不同 SNR 下动态加噪。
2. SNR 曲线评估必须从原始 IQ 信号开始，否则无法控制输入信号的目标 SNR。
3. 独立脚本不会影响已有训练、测试和混淆矩阵逻辑，改动最小。

## SNR 加噪方法

对每个复 IQ 样本 `x`，目标信噪比为 `snr_db` 时：

```python
signal_power = np.mean(np.abs(x) ** 2)
noise_power = signal_power / (10.0 ** (snr_db / 10.0))
noise = np.sqrt(noise_power / 2.0) * (
    rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
)
y = x + noise
```

其中：

- `x` 是复数 IQ 信号。
- `signal_power` 是样本平均功率。
- `noise_power` 根据目标 SNR 反推。
- 除以 `2.0` 是因为复噪声的实部和虚部分别承担一半噪声功率。

项目中已有类似函数：

```text
src/utils/synthetic_signal.py
```

其中的 `add_awgn_for_snr` 可复用，或在新评估脚本中实现一个本地版本以减少跨用途耦合。

## 测试集划分一致性

为了与现有训练/测试逻辑一致，SNR 曲线评估应复用当前固定随机种子划分：

```python
random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)
```

但注意：现有 `.h5` Dataset 只索引预计算特征，不适合动态加噪。新增脚本可以使用相同的文件排序、样本切片规则和 `manual_seed(42)` 思路，从原始 `.mat` 样本构造测试索引。

建议保持以下成功标准：

```text
同一批原始测试样本
同一组 SNR 档位
同一个训练完成的 checkpoint
每个 SNR 档位独立加噪并完整评估一次
```

## STFT 曲线实现流程

STFT 路线最直接，因为 `scripts/precompute_stft_h5.py` 已经提供了可复用函数：

```text
count_iq_samples
_read_iq_batch
compute_stft
_parse_label
```

建议流程：

1. 扫描原始 `.mat` 文件。
2. 按文件名解析类别标签。
3. 按 `sample_length=1_000_000` 计算每个文件可切出的样本数。
4. 构建所有样本索引：

```text
(mat_path, sample_idx, label)
```

5. 使用与训练一致的 6:2:2 划分，取测试部分。
6. 对每个 `snr_db`：
   - 读取测试 IQ batch。
   - 对 batch 中每个样本叠加 AWGN。
   - 调用 `compute_stft` 得到 `(B, 2, 1024, 1024)`。
   - 输入 `DroneRFaResNet18`。
   - 累计正确数和总数。
7. 保存 CSV 和 PNG。

推荐默认 SNR 档位：

```python
[-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
```

## CPP/FAM 曲线实现流程

CPP 路线也可实现，但计算成本明显高于 STFT。原因是每个 SNR 档位都要重新计算 FAM/CPP，且 FAM 计算比 STFT 更重。

可复用函数来自：

```text
scripts/precompute_cpp_h5.py
```

关键函数：

```text
count_iq_samples
iter_iq_pairs
compute_cpp_pair
_parse_label
```

建议流程与 STFT 类似，但每个样本需要：

1. 读取 RF0 和 RF1 两路复 IQ。
2. 对两路 IQ 分别按相同目标 SNR 加 AWGN。
3. 调用 `compute_cpp_pair` 得到 `(2, alpha_bins, f_bins)`。
4. 送入训练好的 CPP 模型。

考虑计算成本，CPP 曲线脚本建议支持：

```text
--max-samples
--max-files
--snrs
--device
--pair-chunk-size
```

用于先做小规模验证，再跑完整评估。

## 推荐命令

STFT 评估命令建议设计为：

```powershell
conda activate droneRFa
python scripts/eval_snr_accuracy_stft.py `
  --data-dir E:/dataSet/DroneRFa `
  --model-path checkpoints/best_stft_model.pth `
  --device cuda:0 `
  --batch-size 8 `
  --snrs -20 -15 -10 -5 0 5 10 15 20 25 30
```

CPP 评估命令建议设计为：

```powershell
conda activate droneRFa
python scripts/eval_snr_accuracy_cpp.py `
  --data-dir E:/dataSet/DroneRFa `
  --model-path checkpoints/best_cpp_model.pth `
  --device cuda:0 `
  --snrs -20 -15 -10 -5 0 5 10 15 20 25 30
```

## 输出 CSV 格式

建议 CSV 字段：

```csv
snr_db,num_samples,num_correct,accuracy
-20,500,37,0.074000
-15,500,58,0.116000
-10,500,102,0.204000
-5,500,196,0.392000
0,500,305,0.610000
5,500,411,0.822000
10,500,465,0.930000
15,500,482,0.964000
20,500,489,0.978000
25,500,491,0.982000
30,500,492,0.984000
```

## 绘图样式

建议绘图代码：

```python
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(snrs, accuracies, marker="o", linewidth=2)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Accuracy")
ax.set_title("SNR-Accuracy Curve")
ax.set_ylim(0.0, 1.0)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(output_png, dpi=300)
```

如果同时比较 STFT 和 CPP，可以使用同一张图：

```text
横轴：SNR (dB)
纵轴：Accuracy
曲线 1：STFT
曲线 2：CPP/FAM
```

这种对比能直接反映不同特征在低信噪比下的鲁棒性差异。

## 结果分析方法

曲线通常应满足以下趋势：

1. 低 SNR 区间准确率低。
   - 例如 `-20 dB` 到 `-10 dB`。
   - 噪声功率远高于信号功率，设备和调制相关特征被严重破坏。

2. 中 SNR 区间准确率快速上升。
   - 例如 `-5 dB` 到 `10 dB`。
   - 模型开始恢复对无人机射频特征的辨别能力。

3. 高 SNR 区间准确率趋于平台。
   - 例如 `15 dB` 到 `30 dB`。
   - 曲线应接近原始测试集准确率。

如果出现异常，需要按以下方向分析：

```text
高 SNR 准确率仍低：
  模型训练效果、checkpoint、类别映射或测试集划分可能有问题。

曲线剧烈抖动：
  测试样本数太少，或每个 SNR 使用了不同测试样本。

低 SNR 准确率明显高于预期：
  可能存在数据泄漏、划分不一致，或噪声添加方式没有真正改变输入。

STFT 高于 CPP：
  当前 CPP 参数或 FAM 聚合方式可能没有充分保留可分特征。

CPP 低 SNR 高于 STFT：
  说明循环平稳特征对随机噪声更稳健，这是合理且值得重点分析的结果。
```

## 论文或报告中的表述建议

建议避免写成：

```text
在真实 SNR 为 x dB 时，模型准确率为 y。
```

更严谨的说法是：

```text
由于 DroneRFa 数据集未提供逐样本 SNR 标注，本文在实采 IQ 信号基础上叠加不同功率的复高斯白噪声，构造目标 SNR 测试条件，并评估模型在不同人工噪声强度下的分类准确率。该曲线用于衡量模型抗噪声鲁棒性。
```

## 最小改动清单

推荐优先实现 STFT 版本：

```text
Create: scripts/eval_snr_accuracy_stft.py
Output: checkpoints/stft_snr_accuracy.csv
Output: checkpoints/stft_snr_accuracy.png
```

验证通过后，再实现 CPP 版本：

```text
Create: scripts/eval_snr_accuracy_cpp.py
Output: checkpoints/cpp_snr_accuracy.csv
Output: checkpoints/cpp_snr_accuracy.png
```

不建议修改：

```text
train_stft.py
train_cpp.py
test_stft.py
test_cpp.py
src/data/stft_dataset.py
src/data/cpp_dataset.py
```

除非后续决定把 SNR 曲线评估正式纳入统一测试接口。

## 实施优先级

1. 先实现 `eval_snr_accuracy_stft.py`。
2. 使用少量样本和 3 个 SNR 档位快速验证：

```text
--snrs -10 0 10
--max-samples 20
```

3. 确认 CSV 和 PNG 正常生成。
4. 跑完整 SNR 档位。
5. 如需比较特征鲁棒性，再实现 CPP/FAM 版本。

## 风险与注意事项

1. 计算成本较高。
   - 每个 SNR 档位都需要重新生成特征。
   - CPP/FAM 尤其耗时。

2. 随机噪声需要固定种子。
   - 建议提供 `--seed` 参数。
   - 这样曲线可复现。

3. 不要把加噪测试样本混入训练。
   - 当前目标是评估鲁棒性，不是噪声增强训练。

4. 需要保证类别映射完全一致。
   - 复用现有 `_parse_label` 和 `LABEL_MAPPING`。

5. 需要保证模型输入归一化一致。
   - STFT 必须复用 `compute_stft`。
   - CPP 必须复用 `compute_cpp_pair`。

## 后续扩展

如果 SNR 曲线显示低信噪比性能不足，可以进一步做噪声增强训练：

```text
训练阶段随机采样 SNR
对 IQ 加 AWGN
再生成 STFT/CPP
训练模型
```

但这会显著增加训练成本。更实际的折中方案是预生成多组加噪 `.h5` 缓存，例如：

```text
stft_h5_snr_-10
stft_h5_snr_0
stft_h5_snr_10
```

用于训练增强或离线评估。
