# 无人机识别

本项目基于 DroneRFa 无人机射频信号数据集，提供 STFT 频谱图和 CPP/FAM 特征的预计算、可视化、训练与测试脚本。

无人机信号研究对象为 `T0010-T10000` 共 15 类，保留 `T0000` 背景类，训练和评估共 16 类。`T0001` 的时频图经人工鉴定为基于 WiFi 的射频信号，易与 WiFi 混淆，不作为研究对象；`T10001-T11000` 飞控器也不纳入研究。

原始标签仍为 T 编码的二进制整数值，即 `0、2-16`，H5 和 split manifest 保留原始标签；模型内部连续映射为 `0-15`，混淆矩阵显示原始标签。原始文件选择、H5 加载、可视化、DEC 清洗和 SNR 评估均排除 `T0001`，无需额外指定 `--exclude-labels 1`。

`T0000` 的 `RF0` 和 `RF1` 分别生成独立单通道样本；其他文件按 `S` 编码选择通道：`S0000-S0111` 使用 `RF0`，`S1000-S1111` 使用 `RF1`。STFT 和 CPP/FAM 特征均为 `(N, 1, H, W)`，H5 通过逐样本 `/rf_channel` 数据集记录实际通道。

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
│   ├── generate_raw_stft_png.py      # 从原始 .mat IQ 按指定 SNR 生成单通道 STFT PNG
│   ├── generate_raw_cpp_png.py       # 从原始 .mat IQ 按指定 SNR 生成单通道 CPP/FAM PNG
│   ├── generate_stft_png.py          # 从 STFT .h5 生成单通道频谱图 PNG
│   ├── generate_cpp_png.py           # 从 CPP/FAM .h5 生成单通道 CPP/FAM PNG
│   ├── train_stft_deep_cluster.py    # 训练 STFT DCEC 并生成盲审样本
│   └── export_clean_stft_h5.py       # 根据人工审核结果剔除纯背景并导出 H5
├── src/
│   ├── data/
│   │   ├── drone_rfa_io.py           # DroneRFa 原始 .mat 数据读取工具
│   │   ├── h5_dataset.py             # STFT/CPP .h5 数据集通用加载基类
│   │   ├── splits.py                 # 数据集划分工具
│   │   ├── stft_dataset.py           # 按样本索引懒加载 STFT .h5 数据
│   │   ├── stft_clustering.py        # clean STFT 聚类索引与模型输入预处理
│   │   ├── stft_cleaning_export.py   # 人工标定、阈值校准与 H5 导出
│   │   └── cpp_dataset.py            # 按样本索引懒加载 CPP/FAM .h5 数据
│   ├── evaluation/
│   │   └── snr_accuracy.py           # 不同 SNR 条件下的准确率评估工具
│   ├── models/
│   │   ├── resnet.py                 # 适配单通道特征输入的 ResNet-18
│   │   ├── autoencoder.py            # STFT 卷积自编码器
│   │   ├── clustering.py             # DEC/DCEC 聚类层与目标分布计算
│   │   └── dcec.py                   # STFT DCEC 模型
│   ├── preprocess/
│   │   ├── h5_precompute.py          # STFT/CPP .h5 预计算通用流程
│   │   ├── random_snr_awgn.py        # 随机 SNR 加性高斯白噪声增强
│   │   ├── rf_segmentation.py        # RF 片段选择与截取工具
│   │   ├── stft.py                   # STFT 频谱图特征生成
│   │   ├── cpp.py                    # 单通道 CPP/FAM 网格生成
│   │   └── fam_torch.py              # PyTorch FAM/SCF 计算与默认坐标范围，支持 CPU/CUDA/MPS
│   ├── training/
│   │   ├── model_io.py               # 模型权重保存与加载
│   │   ├── evaluator.py              # 验证/测试循环
│   │   ├── metrics.py                # 训练评估指标计算
│   │   ├── stft_deep_cluster.py      # 自编码器预训练、K 选择和 DCEC 训练
│   │   └── trainer.py                # 模型训练循环
│   └── utils/
│       ├── logger.py                 # 全局日志工具
│       ├── feature_config.py         # STFT/CPP 特征配置定义
│       ├── device.py                 # 设备选择辅助函数
│       ├── h5topng.py                # 从 .h5 样本批量导出 PNG 的通用工具
│       ├── plot.py                   # 通用时域、频域和 STFT 可视化工具
│       └── paths.py                  # 输出目录路径约定
├── docs/
│   ├── TODO.md
│   ├── random_snr_awgn_training_plan.md
│   ├── Yan 等 - 2025 - TASE-Net A Novel Robust Deep-Learning Network for Open-Set Few-Shot UAV Recognition.pdf
│   └── DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf
└── outputs/
    ├── README.md                     # 统一实验输出目录规划
    ├── models/                       # 模型权重输出目录
    ├── figures/                      # 图片和曲线输出目录
    ├── logs/                         # 训练与评估日志输出目录
    └── metrics/                      # 指标、曲线数据和混淆矩阵输出目录
```

`outputs/` 是统一实验输出目录。日志默认写入 `outputs/logs/`，模型权重默认写入 `outputs/models/`，图片默认写入 `outputs/figures/`，指标和混淆矩阵数组默认写入 `outputs/metrics/`。

仓库内代码默认使用 `src.*` 导入路径，不使用 `drone_rfa.*` 别名。FAM 相关实现位于 `src.preprocess`，绘图和 H5 导图工具位于 `src.utils`。

## 常用命令

以下命令默认从仓库根目录运行。macOS 可将 `python` 替换为 `~/Desktop/venv/bin/python`。

### 预计算 STFT 特征

```bash
python scripts/precompute_stft_h5.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --output-dir ~/Desktop/dataset/DroneRFa_stft_awgn_random_16class_h5 \
  --device mps
```

### 预计算 CPP/FAM 特征

```bash
python scripts/precompute_cpp_h5.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --output-dir ~/Desktop/dataset/DroneRFa_cpp_awgn_random_16class_h5 \
  --device mps
```

### 生成特征图片

```bash
# 原始 IQ 直接生成多个指定 SNR 的单通道 STFT 图片
python scripts/generate_raw_stft_png.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --snrs -10 0 10 \
  --device mps

# 原始 IQ 直接生成多个指定 SNR 的单通道 CPP/FAM 图片
python scripts/generate_raw_cpp_png.py \
  --data-dir ~/Desktop/dataset/droneRFa \
  --snrs -10 0 10 \
  --device mps

# 不添加 AWGN，直接由原始 IQ 生成图片（STFT/CPP 均支持 --clean）
python scripts/generate_raw_stft_png.py --data-dir ~/Desktop/dataset/droneRFa --clean --device mps
python scripts/generate_raw_cpp_png.py --data-dir ~/Desktop/dataset/droneRFa --clean --device mps

# 从预计算 H5 生成图片
python scripts/generate_stft_png.py --h5-dir ~/Desktop/dataset/droneRFa/stft_h5
python scripts/generate_cpp_png.py --h5-dir ~/Desktop/dataset/droneRFa/cpp_h5
```

原始 IQ 直出的 STFT 图片默认保存到数据目录旁的 `DroneRFa_stft_snr_png/`，并按
`SNR/类别/原始文件名` 三级目录保存；
`T0000` 同时输出 RF0/RF1，`S0000-S0111` 选择 RF0，`S1000-S1111` 选择 RF1。图片文件名包含通道编号。图片与模型输入一致，显示逐样本 z-score
归一化功率，适合比较信号结构，但色条不表示可跨图片比较的绝对功率。

原始 IQ 直出的 CPP/FAM 图片采用相同的通道选择和三级目录结构，默认保存到数据目录旁的
`DroneRFa_cpp_snr_png/`。图片经过与 CPP 模型输入一致的 log-zscore-sample 归一化。
`--snrs` 与 `--clean` 必须二选一；clean 图片保存在 `clean/类别/原始文件名` 目录中。

H5 生成的 STFT 图片默认保存到 H5 目录旁的 `stft_png/`，CPP/FAM 图片默认保存到 H5
目录旁的 `cpp_png/`。

新实验使用带 `16class` 的独立目录和文件名，不覆盖历史产物。原始标签及单通道格式正确的旧 17 类 H5 可显式指定为输入，加载时自动跳过 `T0001`；应重新生成 split manifest，旧 17 类 manifest 和模型权重不用于当前实验。旧 14 类 H5 的压缩标签及文件级 `rf_channel` 属性与当前格式不兼容。DEC 也应使用新的空 `work-dir` 重新生成审核、训练和导出产物。

### DEC 单阶段 STFT 清洗

项目仅使用 DEC（DCEC 实现）从 T0010-T10000 中剔除高置信纯背景，保留其他可见信号。
输入 H5 须由 `precompute_stft_h5.py --clean` 生成；`--clean` 仅表示不额外添加 AWGN，
不表示已完成数据清洗。训练会跳过 `T0000`，并生成 250 条盲审图片及 `review.csv`，
其中 150 条用于簇语义映射、50 条用于阈值校准、50 条用于独立审核：

```bash
python scripts/train_stft_deep_cluster.py \
  --data-dir ~/Desktop/dataset/DroneRFa_stft_16class_h5 \
  --work-dir outputs/stft_dec_background \
  --device mps
```

查看 `outputs/stft_dec_background/review_images/`，在 `review.csv` 的 `manual_label` 列填写
`background`、`signal` 或 `uncertain`。`signal` 表示任何可见信号，不要求必须是图传。
完成审核后导出 DEC 背景过滤结果：

```bash
python scripts/export_clean_stft_h5.py \
  --data-dir ~/Desktop/dataset/DroneRFa_stft_16class_h5 \
  --work-dir outputs/stft_dec_background \
  --output-dir ~/Desktop/dataset/DroneRFa_stft_dec_filtered_h5
```

DEC 仅删除归属于背景簇且背景分数达到校准阈值的样本；信号簇、混合簇和未达到阈值的
样本全部保留，`T0000` 的全部样本原样保留。清洗不会按图传与非图传进一步筛选信号。
正式导出要求独立审核样本上的信号保留率为 100%、背景召回率不低于 90%；这些门槛只代表
审核样本上的结果。全量决定和独立审核指标分别保存在 `cleaning_manifest.csv` 与
`cleaning_report.json`，结果写入新 H5 目录，源 H5 不会被修改。
旧 DEC `video/non_video` 复核不兼容，必须使用新的空 `work-dir` 重新生成。

### 训练模型

STFT 与 CPP 统一使用单通道输入的 `DroneRFaResNet18`，保持对比实验的网络结构一致。

```bash
python scripts/train.py --feature stft --data-dir ~/Desktop/dataset/DroneRFa_stft_awgn_random_16class_h5 \
  --split-manifest outputs/splits/full_16class_seed42.csv --batch-size 64
python scripts/train.py --feature cpp --data-dir ~/Desktop/dataset/DroneRFa_cpp_awgn_random_16class_h5 \
  --split-manifest outputs/splits/full_16class_seed42.csv --batch-size 64
```

可通过 `--device` 指定设备，例如 `cuda:0`、`mps` 或 `cpu`。

### 测试模型

```bash
python scripts/evaluate.py \
  --feature stft \
  --data-dir ~/Desktop/dataset/DroneRFa_stft_awgn_random_16class_h5 \
  --split-manifest outputs/splits/full_16class_seed42.csv \
  --model-path outputs/models/best_stft_16class_model.pth

python scripts/evaluate.py \
  --feature cpp \
  --data-dir ~/Desktop/dataset/DroneRFa_cpp_awgn_random_16class_h5 \
  --split-manifest outputs/splits/full_16class_seed42.csv \
  --model-path outputs/models/best_cpp_16class_model.pth
```

测试脚本会输出 accuracy、precision、recall、F1-score 和 loss，并将混淆矩阵数组保存到 `outputs/metrics/`，混淆矩阵图片保存到 `outputs/figures/`。

## 可选类别屏蔽实验

在固定排除 `T0001` 的基础上，可通过 `--exclude-labels` 额外排除原始标签。标签现在与 T 编码的二进制值一致，例如 `T1101` 和 `T1110` 对应 `--exclude-labels 13 14`。程序会在模型内部重映射为连续标签，混淆矩阵坐标仍显示原始标签。

```bash
# STFT：训练与常规评估
python scripts/train.py --feature stft --data-dir "$STFT_H5" \
  --split-manifest "$MANIFEST" --exclude-labels 13 14 --device cuda:0
python scripts/evaluate.py --feature stft --data-dir "$STFT_H5" \
  --split-manifest "$MANIFEST" --exclude-labels 13 14 --device cuda:0

# CPP：训练与常规评估
python scripts/train.py --feature cpp --data-dir "$CPP_H5" \
  --split-manifest "$MANIFEST" --exclude-labels 13 14 --device cuda:0
python scripts/evaluate.py --feature cpp --data-dir "$CPP_H5" \
  --split-manifest "$MANIFEST" --exclude-labels 13 14 --device cuda:0
```

未显式指定模型路径或混淆矩阵路径时，屏蔽实验自动在文件名中附加排除标签。固定 SNR 评估同样传入
`--exclude-labels 13 14`，并应显式指定对应的 CSV、PNG、混淆矩阵前缀和诊断 CSV；
其余 STFT/CPP 参数必须与预计算配置保持一致。
