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
│   ├── train_stft_deepcluster.py     # DeepCluster 训练、伪标签反馈与人工审核
│   └── export_deepcluster_stft_h5.py # 按人工确认决定导出 keep/quarantine H5
├── src/
│   ├── data/
│   │   ├── drone_rfa_io.py           # DroneRFa 原始 .mat 数据读取工具
│   │   ├── h5_dataset.py             # STFT/CPP .h5 数据集通用加载基类
│   │   ├── splits.py                 # 数据集划分工具
│   │   ├── stft_dataset.py           # 按样本索引懒加载 STFT .h5 数据
│   │   ├── stft_clustering.py        # clean STFT 聚类索引与模型输入预处理
│   │   ├── stft_cleaning_export.py   # H5 行复制与 CSV/JSON 写出工具
│   │   ├── deepcluster_export.py     # DeepCluster 人工审核、盲审评估与导出
│   │   ├── deepcluster_review.py     # 样本指纹与复核材料
│   │   ├── deepcluster_visualization.py # DeepCluster t-SNE 与处置图
│   │   └── cpp_dataset.py            # 按样本索引懒加载 CPP/FAM .h5 数据
│   ├── evaluation/
│   │   └── snr_accuracy.py           # 不同 SNR 条件下的准确率评估工具
│   ├── models/
│   │   ├── resnet.py                 # 适配单通道特征输入的 ResNet-18
│   │   ├── stft_cnn.py               # 轻量 Conv-BN-ReLU 聚类特征提取器
│   │   └── stft_resnet.py            # 改进 ResNet-18 聚类特征提取器
│   ├── preprocess/
│   │   ├── h5_precompute.py          # STFT/CPP .h5 预计算通用流程
│   │   ├── random_snr_awgn.py        # 随机 SNR 加性高斯白噪声增强
│   │   ├── stft.py                   # STFT 频谱图特征生成
│   │   ├── cpp.py                    # 单通道 CPP/FAM 网格生成
│   │   └── fam_torch.py              # PyTorch FAM/SCF 计算与默认坐标范围，支持 CPU/CUDA/MPS
│   ├── training/
│   │   ├── model_io.py               # 模型权重保存与加载
│   │   ├── evaluator.py              # 验证/测试循环
│   │   ├── metrics.py                # 训练评估指标计算
│   │   ├── stft_deepcluster.py       # 聚类伪标签迭代训练与可选 NMI 早停
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

新实验使用带 `16class` 的独立目录和文件名，不覆盖历史产物。原始标签及单通道格式正确的旧 17 类 H5 可显式指定为输入，加载时自动跳过 `T0001`；应重新生成 split manifest，旧 17 类 manifest 和模型权重不用于当前实验。旧 14 类 H5 的压缩标签及文件级 `rf_channel` 属性与当前格式不兼容。DeepCluster 实验使用独立 `work-dir` 保存训练与审核产物。

### DeepCluster 独立图传清洗

从随机初始化的 CNN 开始，交替执行全量编码、PCA 白化与 L2 归一化、KMeans 伪标签和分类训练。默认处理低质量的 10 类 `4,6,8,10,11,12,13,14,15,16`，不使用机型标签训练。`clean` 只表示未额外添加 AWGN，并非有效图传标注。

```bash
~/Desktop/venv/bin/python scripts/train_stft_deepcluster.py \
  --data-dir <原始clean-H5目录> --work-dir outputs/stft_deepcluster --device mps
```

默认 K=32、PCA 最多64个有效维度、20轮聚类反馈，每轮伪标签训练一个 epoch；可分别用 `--n-clusters 16`、`--n-clusters 64` 做对照。每轮重置分类头、按簇均衡抽样，最终更新后重新聚类。CNN 不包含 decoder，也不使用重建误差阈值。

可通过 `--early-stop-patience 5` 开启聚类稳定性早停；默认值0关闭，`--epochs` 仍默认20，作为最大训练轮数。`--early-stop-nmi-threshold` 默认0.99，`--early-stop-min-epochs` 默认10：从第10轮开始，轮间 NMI 连续5轮达到阈值则停止，最早完成第14轮后退出；未达标清零，首轮 NMI 为空不计数。预算不足时正常跑满。阈值是初始实验配置，需要真实数据校准。

```bash
~/Desktop/venv/bin/python scripts/train_stft_deepcluster.py \
  --data-dir <原始clean-H5目录> --epochs 50 --early-stop-patience 5 \
  --work-dir outputs/stft_deepcluster_early_stop --device mps
```

判据使用每轮训练前的聚类，完成本轮训练后检查，不增加每轮编码或聚类次数；停止后保留最后权重并重新编码、聚类，最终聚类种子按实际完成轮数计算。history 的 `stable_rounds` 和日志记录连续达标轮数；checkpoint 与训练报告的 `training_summary` 同时记录 `epochs_requested`、`epochs_completed`、`stopped_early`、`stop_reason`（`max_epochs` 或 `assignment_stable`）。达到最大轮数统一视为正常完成。稳定不代表语义正确；每轮分类头重置，loss 不直接用于早停，人工审核及盲审标签不参与停止决策。

输入形式由 `--input-mode grayscale|rgb` 选择，默认 `grayscale` 保持原单通道浮点矩阵输入。两种模式均从 H5 读取 STFT，截断至 `[-5, 5]`、除以 5 并平均池化至 256×256，保持原坐标方向；`rgb` 再以固定 256 色 `jet` 查找表映射为三通道像素矩阵，RGB 分量统一缩放至 `[-1, 1]`。输入不包含标题、坐标轴、色条或留白，也不需要保存中间 PNG。该映射使用的强度范围与审核图的 `[-3, 3]` 显示范围不同。

```bash
~/Desktop/venv/bin/python scripts/train_stft_deepcluster.py \
  --data-dir <原始clean-H5目录> --input-mode rgb \
  --work-dir outputs/stft_deepcluster_rgb --device mps
```

CNN 首层随模式使用 1 或 3 个输入通道。`deepcluster.pt` 记录 `input_mode`、`in_channels` 和 `colormap`，`training_report.json` 的 `parameters.input_mode` 记录本次选择；重建模型时按 checkpoint 的 `in_channels` 初始化。两种模式的权重不能直接互换。对照实验使用独立工作目录，并可通过 `--reuse-review-dir` 固定审核样本。

骨干由 `--backbone cnn|resnet18` 选择，默认 `cnn` 为轻量五层 CNN：通道依次为 `32,64,128,256,256`，每层采用 `3×3 / stride=2 / padding=1` 无偏置卷积、BatchNorm2d 和 ReLU；空间尺寸依次为 `128,64,32,16,8`，末端经全局平均池化和 `256→128` 线性层输出特征。卷积采用 Kaiming normal（fan_in、ReLU）初始化，避免低幅值输入的首轮随机特征逐层衰减。BN 训练时更新运行统计，编码时使用 eval 模式及运行统计，不额外预热或校准。改进 ResNet-18 从随机权重开始，保留 `[2,2,2,2]` BasicBlock；入口为 `3×3 / stride=2` 卷积，移除 MaxPool，所有归一化均为 `GroupNorm(8, channels)`。256×256 输入经过总计16倍下采样得到 `512×16×16`，再通过 `2×2` 自适应平均池化保留粗略时频位置，展平后由 `2048→128` 线性层输出特征。特征末端没有额外激活或归一化，仍使用相同的 PCA、L2、KMeans 和逐轮重置分类头流程。

```bash
~/Desktop/venv/bin/python scripts/train_stft_deepcluster.py \
  --data-dir <原始clean-H5目录> --backbone resnet18 --batch-size 8 \
  --work-dir outputs/stft_deepcluster_resnet18 --device mps
```

两种骨干均支持 `--input-mode grayscale|rgb`。checkpoint 和报告的 `model_type` 分别为 `stft_light_cnn`、`stft_resnet18`，`parameters.backbone` 记录选择。重建时根据 `model_type` 选择 `src.models.stft_cnn.StftCNN` 或 `src.models.stft_resnet.StftResNet18`，传入保存的 `latent_dim`、`in_channels` 后严格加载 `model_state_dict`，提取特征前调用 `eval()`；BN 运行均值、运行方差及批次计数随 state_dict 保存。旧 `model_type=stft_cnn` 权重与新的轻量 CNN 不兼容，不自动转换；请保留历史产物并使用独立工作目录训练新模型。

单通道改进 ResNet-18 约1143万参数、8.87 GMAC/样本，轻量 CNN 约101万参数、0.269 GMAC（256×256 输入，仅统计卷积和全连接）。实际耗时和显存需实测，可先使用较小批次。此改造无法恢复1024→256输入池化已经丢失的信息，也不预设聚类质量一定提升。对照时固定数据、输入模式、批次大小、seed、K、PCA维度、轮数和学习率，使用独立工作目录，并通过 `--reuse-review-dir` 复用相同盲审样本及逐样本审核；比较有效图传误删率、删除集合无效样本比例和隔离率。

审核工作目录中的 `cluster_review.csv` 和 `sample_review.csv`：`decision` 为 `keep/drop/quarantine`，明确填写 `confirmed=true` 才生效。逐样本决定优先；混合簇设 `mixed=true` 后整簇默认隔离，只允许逐样本覆盖。未确认样本默认隔离。仅保留可确认的可用图传；纯背景、仅控制跳频、明确非目标通信可剔除；弱信号、混合信号或无法确认的通信来源应隔离。簇代表图仅辅助审核，距离近或抽样一致不证明整个簇有效。

`audit.csv` 是与开发审核分离的固定盲审样本，`manual_label` 使用 `video/background/control_only/interference/weak_or_mixed/uncertain`；应在处置决定冻结后填写，仅用于评估，不参与清洗决定。原始高分辨率审核图用于检查池化可能掩盖的细窄信号。重跑保留同身份、同输入指纹的样本审核；簇编号不具有跨运行语义，簇级处置需要重新确认。

不同方法使用独立 `work-dir`，添加 `--reuse-review-dir <参考实验目录>` 复用同一盲审样本和已有逐样本审核；要求输入指纹一致。这样可固定审核集，而非仅依赖相同随机种子。各方法的簇处置仍需重新确认。

```bash
~/Desktop/venv/bin/python scripts/export_deepcluster_stft_h5.py \
  --data-dir <原始clean-H5目录> --work-dir outputs/stft_deepcluster \
  --output-dir <新的清洗输出目录>
```

输出目录的 `keep/` 与 `quarantine/` 保存原始 H5 行及元数据，剔除样本仅记录清单；源文件不修改，目标不覆盖。T0000 透传保留，T0001 排除；未纳入本轮类别的文件不导出，另列报告。训练及导出通过源 H5 全文件 SHA256 核验内容，数据量大时需要额外顺序读取。审核采用人工语义，表征训练无监督。该实验允许全量无标签聚类，属于现有数据整理，不能据此宣称未见文件泛化能力。

训练结束后自动生成 `deepcluster_tsne_clusters.png`（簇编号）和 `deepcluster_tsne_original_labels.png`（原始 T 类别）。投影使用最终参与 KMeans 的 PCA 白化＋L2 特征，单通道与 RGB 模式采用相同流程。`deepcluster_tsne.csv` 保存坐标与完整样本身份，`training_report.json` 的 `tsne` 记录参数及抽样数量。

审核导出时复用同一坐标，在输出目录生成 `deepcluster_tsne_decisions.png`，按 `keep/drop/quarantine` 着色；T0000 不参与投影。旧目录缺少坐标时仍可导出 H5，报告注明未生成处置图。坐标与当前分配表身份不匹配时拒绝导出。

最多展示10,000条样本，超出时按实验 seed 均匀无放回抽样，小簇不保证覆盖。投影使用二维、`perplexity=min(30, 展示数量-1)`、自动学习率和1,000次迭代；PCA 初始化，单维特征改用随机初始化，常量特征使用零坐标并注明退化。同次实验的各图共享坐标范围；不同实验分别拟合，位置不可直接比较。t-SNE 只用于观察邻域与审核结果，不参与聚类或清洗判定。

对照实验应使用相同数据和人工审核预算，比较单通道 CNN+DeepCluster+审核、RGB CNN+DeepCluster+审核及不同骨干。关注明确图传错误剔除率、隔离率、删除集合无效样本比例及逐类/逐文件覆盖；训练损失、NMI 和 t-SNE 不代替清洗验收。下游识别须固定原始文件划分与同一个测试集，不能清洗后重新随机划分。

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
