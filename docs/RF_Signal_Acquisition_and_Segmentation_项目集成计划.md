# RFSignalAcquisitionSegmentation 类封装与评估链路接入计划

## Summary

在 `src/preprocess/rf_segmentation.py` 中新增 `RFSignalAcquisitionSegmentation(nn.Module)`，保留现有 `segment_predominant_rf()` 函数作为兼容入口，并让 CPP 预计算与 SNR 评估脚本改用类实例。新增 `--rf-selection-mode`，支持当前稳定的 `target_len/top_k` 策略，以及用于分析的 `order_statistic` 策略；order-statistic 输出在脚本链路中截断或补零到 `target_len`，保证 CPP/H5 形状稳定。

## Key Changes

- `src/preprocess/rf_segmentation.py`
  - 新增 `RFSignalAcquisitionSegmentation(nn.Module)`。
  - 构造参数：`frame_len=10000`、`target_len=100000`、`top_k=None`、`selection_mode="target_len"`、`sort_by_time=True`、`eps=1e-12`、`device=None`、`confidence=0.95`、`mode_bins=64`。
  - `forward(x)` 返回 `(selected, indices, eser)`，其中 `x` 支持 `[M]` 和 `[B, M]`，保持一维输入返回一维 `selected/indices/eser` 的现有习惯。
  - 保留 `segment_predominant_rf(...)`，内部实例化 `RFSignalAcquisitionSegmentation` 后调用，避免破坏现有导入。
- 选择策略
  - `selection_mode="target_len"`：按 `ceil(target_len / frame_len)` 推导保留帧数；若传入 `top_k`，仍优先使用 `top_k`，兼容当前行为。
  - `selection_mode="top_k"`：必须提供 `top_k`，直接选择 ESER 最大的 `top_k` 帧。
  - `selection_mode="order_statistic"`：每个样本单独计算 ESER 分布，使用 histogram 估计众数 `e_bar`，基于 `confidence=0.95` 搜索满足条件的 `iota`，得到 `phi = I - iota + 1` 后选 top-`phi`。
  - order-statistic 输出拼接后统一处理到 `target_len`：超过则截断，不足则右侧补零；`indices` 保留真实选中帧索引，不为 padding 伪造索引。
- 脚本接入
  - `scripts/precompute_cpp_h5.py` 和 `scripts/eval_snr_accuracy_cpp.py` 改为创建一个 `RFSignalAcquisitionSegmentation` 实例，然后对 RF0/RF1 分别调用。
  - 两个脚本新增同名 CLI 参数：`--rf-selection-mode {target_len,top_k,order_statistic}`、`--rf-confidence`、`--rf-mode-bins`。
  - 默认仍为 `target_len`，不开启 `--use-rf-segmentation` 时行为完全不变。
  - SNR 评估顺序保持：读取 clean IQ -> 加 AWGN -> RF segmentation -> CPP -> 模型预测。

## Test Plan

- 新增 `tests/test_rf_segmentation.py`。
  - 验证 `RFSignalAcquisitionSegmentation` 与 `segment_predominant_rf()` 在默认 `target_len/top_k` 路径输出一致。
  - 验证高能低熵合成 frame 会被选中。
  - 验证 `[M]` 和 `[B, M]` 输入的输出维度。
  - 验证 complex IQ 输入保持 complex dtype，零信号不会产生 NaN/Inf。
  - 验证 `order_statistic` 模式输出长度等于 `target_len`，且 `indices` 只包含真实 frame 下标。
  - 验证 `sort_by_time=True` 时 selected 按原始 frame 顺序拼接。
- 轻量脚本验证
  - 运行 `python -m pytest tests/test_rf_segmentation.py`。
  - 用 `--max-files 1 --max-samples-per-file 1 --use-rf-segmentation --rf-selection-mode target_len` 验证 CPP 预计算仍能写出固定 H5。
  - 用 `--max-samples 1 --snrs 0 --use-rf-segmentation --rf-selection-mode order_statistic` 验证 SNR 评估链路可运行。

## Assumptions

- 类封装是主接口，函数保留为兼容层，不删除现有函数名。
- order-statistic 是分析选项，不作为默认训练/预计算策略。
- `target_len=100000` 是脚本链路中的固定输出长度目标；order-statistic 变长结果必须截断/补零到该长度。
- 当前仓库没有现成 `tests/` 目录，实施时会新增该目录和 focused 单测。
- 不改 CPP/FAM 计算逻辑，不新增训练 feature 类型。
