# RF Signal Acquisition and Segmentation 集成方案

## 结论

推荐按“独立 RF segmentation 模块 + CPP 前置可选开关”集成。第一版不要直接实现完整 order-statistics 自适应 `φ` 作为默认训练路径，因为它会导致每个样本输出长度不稳定，影响现有 H5、DataLoader、ResNet 训练链路。先用论文一致但工程稳定的 `target_len >= 100_000` 推导 `top_k`，跑通并验证收益。

## 核心集成位置

### 1. 新增 RF segmentation 模块

新增文件：

```text
src/preprocess/rf_segmentation.py
```

推荐接口：

```python
segment_predominant_rf(
    x,
    *,
    frame_len=10_000,
    target_len=100_000,
    top_k=None,
    sort_by_time=True,
    eps=1e-12,
    device="cpu",
)
```

输出：

```python
selected_signal, selected_indices, eser_values
```

算法流程：

```text
IQ 序列
-> 非重叠分帧
-> FFT
-> power spectrum
-> spectral probability
-> spectral entropy
-> ST-ESER = energy / entropy
-> 选 top-k 帧
-> 按原始时间顺序拼接
-> 输入 CPP/FAM
```

### 2. 接入 CPP 预计算

修改文件：

```text
scripts/precompute_cpp_h5.py
```

新增 CLI 参数：

```bash
--use-rf-segmentation
--rf-frame-len 10000
--rf-target-len 100000
--rf-top-k
```

处理顺序：

```text
read_iq_batch
-> RF0/RF1 分别做 ST-ESER segmentation
-> compute_cpp
-> 写入 H5
```

默认关闭 RF segmentation，保证现有 CPP 预计算行为完全不变。

### 3. 接入 SNR CPP 评估

修改文件：

```text
scripts/eval_snr_accuracy_cpp.py
```

处理顺序应为：

```text
读取 clean IQ
-> 按目标 SNR 加 AWGN
-> RF segmentation
-> CPP
-> 模型预测
```

原因是论文评估的是不同 SNR 下的接收信号识别，ST-ESER 也应作用在加噪后的观测信号上。

## 参数建议

论文里容易混淆两个 `N`：

- RF segmentation 分帧：`M=1e8`，`I=1e4`，所以帧长应是 `frame_len=10_000`。
- CPP/FAM 参数：论文使用 `N'=256`，sample size `N=16384`，这是 FAM/SCS 阶段参数，不是 ST-ESER 分帧长度。

第一版默认：

```text
rf_frame_len = 10_000
rf_target_len = 100_000
rf_top_k = ceil(100_000 / 10_000) = 10
sort_by_time = True
```

如果本项目当前样本长度是 `sample_length=1_000_000`，那么每个样本有 100 个 ST-ESER 帧，保留 10 帧，正好得到 100k IQ 点。

## 方案对比

### 方案 A：`target_len/top-k` 工程版

优点：

- 输出长度稳定，容易接入 H5、CPP、训练链路。
- 实现和测试成本低。
- 与论文“至少提取 `10^5` 个样本形成新序列 `x(n)`”的实验设置一致。

缺点：

- 不是最严格复现论文公式 (7)-(9) 的 order-statistics `φ`。

结论：推荐作为第一版默认方案。

### 方案 B：完整 order-statistics 版

优点：

- 最贴近论文公式 (7)-(9)。
- 可以复现 `ρ=0.95`、`e_bar=mode(ST-ESER)` 的自适应帧数选择。

缺点：

- 样本间 `φ` 可能不同，需要 padding、截断或变长处理。
- 会增加 H5 预计算、批处理训练和结果对比的复杂度。

结论：第二版作为 `selection_mode="order_statistic"` 分析选项，不默认用于训练。

### 方案 C：直接嵌进 `compute_cpp`

优点：

- 调用端改动少。

缺点：

- `compute_cpp` 职责变重。
- 后续不方便单测、复用和对比开启/关闭 segmentation 的结果。

结论：不推荐。

## 实施阶段

### 阶段 1：纯模块与单测

新增：

```text
src/preprocess/rf_segmentation.py
tests/test_rf_segmentation.py
```

验证内容：

- 高能低熵合成片段能被 top-k 选中。
- `target_len` 正确推导 `top_k`。
- complex IQ 输入保持 complex 输出。
- 零信号不产生 NaN/Inf。
- `sort_by_time=True` 时输出按原时间顺序拼接。

### 阶段 2：接入 CPP 离线预计算

修改：

```text
scripts/precompute_cpp_h5.py
```

示例命令：

```bash
~/Desktop/venv/bin/python scripts/precompute_cpp_h5.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --output-dir ~/Desktop/dataset/DroneRFa_cpp_rfseg_h5 \
  --use-rf-segmentation \
  --rf-frame-len 10000 \
  --rf-target-len 100000 \
  --device mps
```

### 阶段 3：接入 SNR CPP 评估

修改：

```text
scripts/eval_snr_accuracy_cpp.py
```

要求：

- 与预计算脚本保持同名参数。
- 确保顺序是 `加噪 -> segmentation -> CPP`。

### 阶段 4：文档和实验记录

更新：

```text
readme.md
docs/RF_Signal_Acquisition_and_Segmentation_实现流程构思_公式修正版.md
```

明确记录：

- RF segmentation 帧长 `10_000`。
- CPP/FAM `N=16384` 是另一阶段参数。
- 第一版默认 `target_len=100_000`。
- order-statistics 暂不作为默认训练路径。

## 验收标准

最小测试：

```bash
~/Desktop/venv/bin/python -m pytest \
  tests/test_rf_segmentation.py \
  tests/test_preprocess_compute_modules.py \
  tests/test_precompute_scripts.py \
  tests/test_eval_snr_accuracy_cpp.py
```

功能验证：

```bash
~/Desktop/venv/bin/python scripts/precompute_cpp_h5.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --max-files 1 \
  --max-samples-per-file 1 \
  --use-rf-segmentation \
  --rf-frame-len 10000 \
  --rf-target-len 100000
```

成功标准：

- H5 仍写出固定形状 `cpp: (N, 2, alpha_bins, f_bins)`。
- 无 NaN/Inf。
- 默认不开启 segmentation 时，现有行为不变。
- 开启后 RF0/RF1 先被压缩到约 `100_000` 点，再进入 `compute_cpp`。

