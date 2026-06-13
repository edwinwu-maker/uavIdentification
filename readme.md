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
│   ├── precompute_stft_h5.py         # 将原始 .mat IQ 数据预计算为 STFT .h5
│   ├── precompute_cpp_h5.py          # 将原始 .mat IQ 数据预计算为 CPP/FAM .h5
│   ├── generate_stft_png.py          # 从 STFT .h5 生成双通道频谱图 PNG
│   ├── generate_cpp_png.py           # 从 CPP/FAM .h5 生成双通道 CPP/FAM PNG
│   └── plot_fam_demo.py              # 生成合成信号的时域、频域和 FAM 可视化示例
├── src/
│   ├── data/
│   │   ├── stft_dataset.py           # 按样本索引懒加载 STFT .h5 数据
│   │   └── cpp_dataset.py            # 按样本索引懒加载 CPP/FAM .h5 数据
│   ├── models/
│   │   └── resnet.py                 # 适配双通道特征输入的 ResNet-18
│   ├── preprocess/
│   │   ├── fam.py                    # CPU/NumPy 版本 FAM/SCF 计算工具
│   │   ├── fam_constants.py          # FAM 相关常量与边界定义
│   │   ├── fam_grid.py               # FAM 稀疏点到 CPP/FAM 网格的聚合工具
│   │   └── fam_torch.py              # PyTorch 版本 FAM/SCF 计算工具
│   ├── signal/
│   │   └── synthetic_signal.py       # FAM 示例使用的合成信号生成器
│   ├── visualization/
│   │   ├── fam_plot.py               # FAM/CPP 热力图、三维曲面和时频图绘制工具
│   │   └── plot_utils.py             # 通用时域、频域和 STFT 可视化工具
│   └── utils/
│       ├── logger.py                 # 全局日志工具
│       ├── feature_specs.py          # STFT/CPP 特征配置定义
│       ├── config.py                 # 轻量 YAML 配置解析
│       ├── device.py                 # 设备选择辅助函数
│       └── paths.py                  # 输出目录路径约定
├── docs/
│   ├── project_structure_optimization.md
│   ├── snr_accuracy_curve_implementation_plan.md
│   └── DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf
├── outputs/
│   └── README.md                       # 统一实验输出目录规划
├── checkpoints/                      # 旧 checkpoint 输出目录，已由 .gitignore 忽略
└── logs/                             # 旧日志目录，已由 .gitignore 忽略
```

`outputs/` 是统一实验输出目录。日志默认写入 `outputs/logs/`，模型 checkpoint 默认写入 `outputs/checkpoints/`，图片默认写入 `outputs/figures/`，指标和混淆矩阵数组默认写入 `outputs/metrics/`。

仓库内代码默认使用 `src.*` 导入路径，不使用 `drone_rfa.*` 别名。合成信号和 FAM 相关实现分别位于 `src.signal` 和 `src.preprocess`，绘图工具位于 `src.visualization`。

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
