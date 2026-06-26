# 集成 RF Signal Acquisition and Segmentation 到 CPP 预处理链路

## Summary

将论文 Section II-A 的 ST-ESER predominant segment 选择实现为独立预处理模块，并接入现有 `scripts/precompute_cpp_h5.py` 与 SNR 评估链路。第一版默认使用固定输出长度 `target_len=100_000`，保证 H5 中 CPP 样本形状稳定；order-statistics 自适应 φ 先不作为训练默认路径。

## Key Changes

- 新增 `src/preprocess/rf_segmentation.py`：
  - 输入单通道或批量复数 IQ：`(..., L)`。
  - 非重叠分帧：`frame_len`，不足一帧尾部截断。
  - 计算 FFT power、谱概率、谱熵、ST-ESER。
  - 按 ST-ESER 选择 `top_k` 或由 `target_len` 推导 `ceil(target_len / frame_len)` 个帧。
  - 默认按原始时间顺序拼接，返回 `selected_signal, selected_indices, eser`。
- 接入 CPP 预计算：
  - 在 `scripts/precompute_cpp_h5.py` 增加参数：`--use-rf-segmentation`、`--rf-frame-len`、`--rf-target-len`、`--rf-top-k`。
  - 开启后，对 RF0/RF1 两个通道分别分段选择，再把选出的信号传入现有 `compute_cpp`。
  - 默认关闭，保持当前 CPP 预计算行为完全不变。
- 接入 SNR CPP 评估：
  - `scripts/eval_snr_accuracy_cpp.py` 增加同名参数。
  - 顺序为：读取 IQ -> 加噪声 -> RF segmentation -> CPP -> 模型预测。
- 文档同步：
  - 在 `readme.md` 增加一个“带 RF segmentation 的 CPP 预计算”命令示例。
  - 在现有构思文档中补充本项目落点：模块文件、CLI 参数、默认策略。

## Interfaces

推荐函数接口：

```python
segment_predominant_rf(
    x: np.ndarray | torch.Tensor,
    *,
    frame_len: int,
    target_len: int | None = 100_000,
    top_k: int | None = None,
    sort_by_time: bool = True,
    eps: float = 1e-12,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

- `top_k` 优先级高于 `target_len`；两者都为空时报错。
- 输出信号长度固定为 `selected_frames * frame_len`，不额外 padding；后续 FAM 的 padding 继续由现有 `compute_fam_grid_segmented` 负责。

## Test Plan

- 新增 `tests/test_rf_segmentation.py`：
  - 合成高能片段信号，验证 top-k 能选中对应帧。
  - 验证 `target_len` 正确推导帧数。
  - 验证复数 IQ 输入输出长度、dtype 和 shape。
  - 验证零信号不会产生 NaN/Inf。
- 轻量集成验证：
  - 用小数组直接调用 `compute_cpp` 前的 segmentation 路径，确认 RF0/RF1 输出可进入现有 CPP 函数。
  - 运行 `pytest tests/test_rf_segmentation.py tests/test_generate_stft_png.py`。
- 不在单测中跑完整 `.mat -> .h5`，避免依赖本地大数据集和长时间 FAM 计算。

## Assumptions

- 当前项目最自然的落点是 CPP/FAM 前置预处理，不新增 `--feature rf_segmentation` 训练类型。
- 第一版不实现论文 order-statistics φ，因为现有 H5/DataLoader/ResNet 流程需要固定特征形状；该模式可后续作为分析工具加入。
- PDF 当前环境缺少可用文本抽取工具；本计划依据仓库中的公式修正版构思文档和现有代码结构制定。实施时若能安装/使用 PDF 文本抽取工具，应再核对论文参数描述。
- 工作区已有大量未提交改动，实施时只触碰上述相关文件，不整理无关 diff。
