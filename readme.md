# 无人机识别

本项目基于 DroneRFa 无人机射频信号数据集，提供 STFT 频谱图和 CPP/FAM 特征的预计算、可视化、训练与测试脚本。

## 数据集

- DroneRFa 下载链接：[DroneRFa](https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d&traceId=e1937db6-783a-47dd-9df2-16c869a5dc33)
- 数据集说明文档：`docs/DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf`

默认数据路径：

- Windows: `E:/dataSet/DroneRFa`
- macOS: `~/Desktop/dataset/droneRFa`
- Linux: `/mnt/data/wurixin/DroneRFa`

## 运行环境

macOS 上优先使用本地虚拟环境：

```bash
~/Desktop/venv/bin/python <script-or-command>
```

Windows 上使用 `droneRFa` conda 环境：

```powershell
conda activate droneRFa
python <script-or-command>
```

## 项目目录结构

```text
.
├── AGENTS.md                         # 代码代理协作规则和项目运行约定
├── readme.md                         # 项目说明文档
├── requirements.txt                  # 项目依赖列表
├── scripts/
│   ├── train.py                      # 统一训练入口，支持 --feature stft|cpp
│   ├── evaluate.py                   # 统一评估入口，支持 --feature stft|cpp
│   ├── eval_snr_accuracy_stft.py     # 评估 STFT 模型在不同 SNR 下的准确率
│   ├── eval_snr_accuracy_cpp.py      # 评估 CPP/FAM 模型在不同 SNR 下的准确率
│   ├── precompute_stft_h5.py         # 将原始 .mat IQ 数据预计算为 STFT .h5
│   ├── precompute_cpp_h5.py          # 将原始 .mat IQ 数据预计算为 CPP/FAM .h5
│   ├── generate_stft_png.py          # 从 STFT .h5 生成双通道频谱图 PNG
│   └── generate_cpp_png.py           # 从 CPP/FAM .h5 生成双通道 CPP/FAM PNG
├── src/
│   ├── data/
│   │   ├── drone_rfa_io.py           # DroneRFa 原始 .mat 数据读取工具
│   │   ├── h5_dataset.py             # STFT/CPP .h5 数据集通用加载基类
│   │   ├── splits.py                 # 数据集划分工具
│   │   ├── stft_dataset.py           # 按样本索引懒加载 STFT .h5 数据
│   │   └── cpp_dataset.py            # 按样本索引懒加载 CPP/FAM .h5 数据
│   ├── evaluation/
│   │   └── snr_accuracy.py           # 不同 SNR 条件下的准确率评估工具
│   ├── models/
│   │   └── resnet.py                 # 适配双通道特征输入的 ResNet-18
│   ├── preprocess/
│   │   ├── h5_precompute.py          # STFT/CPP .h5 预计算通用流程
│   │   ├── random_snr_awgn.py        # 随机 SNR 加性高斯白噪声增强
│   │   ├── rf_segmentation.py        # RF 片段选择与截取工具
│   │   ├── stft.py                   # STFT 频谱图特征生成
│   │   ├── fam_defaults.py           # FAM 默认坐标范围
│   │   ├── cpp.py                    # CPP/FAM 网格生成与双通道 CPP 特征组装
│   │   └── fam_torch.py              # PyTorch FAM/SCF 计算工具，支持 CPU/CUDA/MPS
│   ├── training/
│   │   ├── checkpoint.py             # 模型 checkpoint 保存与加载
│   │   ├── evaluator.py              # 验证/测试循环
│   │   ├── metrics.py                # 训练评估指标计算
│   │   └── trainer.py                # 模型训练循环
│   ├── visualization/
│   │   ├── h5_png_export.py          # 从 .h5 样本批量导出 PNG 的通用工具
│   │   └── plot_utils.py             # 通用时域、频域和 STFT 可视化工具
│   └── utils/
│       ├── logger.py                 # 全局日志工具
│       ├── feature_specs.py          # STFT/CPP 特征配置定义
│       ├── device.py                 # 设备选择辅助函数
│       └── paths.py                  # 输出目录路径约定
├── docs/
│   ├── TODO.md
│   ├── random_snr_awgn_training_plan.md
│   ├── Yan 等 - 2025 - TASE-Net A Novel Robust Deep-Learning Network for Open-Set Few-Shot UAV Recognition.pdf
│   └── DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf
└── outputs/
    ├── README.md                     # 统一实验输出目录规划
    ├── checkpoints/                  # 模型 checkpoint 输出目录
    ├── figures/                      # 图片和曲线输出目录
    ├── logs/                         # 训练与评估日志输出目录
    └── metrics/                      # 指标、曲线数据和混淆矩阵输出目录
```

`outputs/` 是统一实验输出目录。日志默认写入 `outputs/logs/`，模型 checkpoint 默认写入 `outputs/checkpoints/`，图片默认写入 `outputs/figures/`，指标和混淆矩阵数组默认写入 `outputs/metrics/`。

仓库内代码默认使用 `src.*` 导入路径，不使用 `drone_rfa.*` 别名。FAM 相关实现位于 `src.preprocess`，绘图工具位于 `src.visualization`。

## 常用命令

以下命令默认从仓库根目录运行。macOS 可将 `python` 替换为 `~/Desktop/venv/bin/python`。

### 预计算 STFT 特征

```bash
python scripts/precompute_stft_h5.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --output-dir ~/Desktop/dataset/droneRFa/stft_h5 \
  --device mps
```

### 预计算 CPP/FAM 特征

```bash
python scripts/precompute_cpp_h5.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --output-dir ~/Desktop/dataset/droneRFa/cpp_h5 \
  --device mps
```

开启 ST-ESER predominant segment 选择后再预计算 CPP/FAM：

```bash
python scripts/precompute_cpp_h5.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --output-dir ~/Desktop/dataset/droneRFa/cpp_rfseg_h5 \
  --use-rf-segmentation \
  --rf-frame-len 10000 \
  --rf-target-len 100000 \
  --device mps
```

### 生成特征图片

```bash
python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5
python scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5
```

STFT 图片默认保存到 `outputs/figures/stft_png/`，CPP/FAM 图片默认保存到 `outputs/figures/cpp_png/`。

### 训练模型

```bash
python scripts/train.py --feature stft --data-dir ~/Desktop/dataset/droneRFa/stft_h5 --batch-size 64
python scripts/train.py --feature cpp --data-dir ~/Desktop/dataset/droneRFa/cpp_h5 --batch-size 64
```

可通过 `--device` 指定设备，例如 `cuda:0`、`mps` 或 `cpu`。

### 测试模型

```bash
python scripts/evaluate.py \
  --feature stft \
  --data-dir ~/Desktop/dataset/droneRFa/stft_h5 \
  --model-path outputs/checkpoints/best_stft_model.pth

python scripts/evaluate.py \
  --feature cpp \
  --data-dir ~/Desktop/dataset/droneRFa/cpp_h5 \
  --model-path outputs/checkpoints/best_cpp_model.pth
```

测试脚本会输出 accuracy、precision、recall、F1-score 和 loss，并将混淆矩阵数组保存到 `outputs/metrics/`，混淆矩阵图片保存到 `outputs/figures/`。
