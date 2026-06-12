# 项目结构规划优化方向

本文基于当前 `droneRFa` 项目的实际目录与代码组织情况，给出后续结构优化建议。

## 当前结构观察

项目当前已经具备基本分层：

```text
droneRFa/
  readme.md
  requirements.txt
  train_stft.py
  train_cpp.py
  test_stft.py
  test_cpp.py
  scripts/
  src/
    data/
    models/
    utils/
  docs/
  logs/
  checkpoints/
```

其中：

- `src/data` 负责 STFT、CPP/FAM 的 H5 数据集读取。
- `src/models` 目前主要包含 ResNet 模型定义。
- `src/utils` 包含 FAM、绘图、日志、合成信号等工具函数。
- `scripts` 存放预处理、图片生成和演示脚本。
- 根目录包含 STFT/CPP 两套训练与测试入口。

整体结构适合早期实验，但如果继续扩展模型、特征、实验配置和评估流程，当前结构会逐渐出现重复代码多、入口分散、配置难复现的问题。

## 主要问题

### 1. 根目录入口过多

当前根目录包含：

```text
train_stft.py
train_cpp.py
test_stft.py
test_cpp.py
```

这些脚本承担了训练、评估、参数解析、设备选择、数据切分、checkpoint 管理等多种职责。STFT 和 CPP 两套流程高度相似，后续新增特征或模型时容易继续复制脚本。

### 2. `src` 包命名不够清晰

当前代码使用：

```python
from src.data.stft_dataset import SpectrogramDataset
from src.models.resnet import DroneRFaResNet18
```

这在实验阶段可以接受，但长期项目中建议将 `src` 作为源码根目录，将真实包名改为业务语义更明确的 `drone_rfa`。

### 3. 训练、评估、指标逻辑没有公共模块

当前多个脚本中重复存在：

- 默认设备选择。
- 默认数据目录选择。
- 数据集切分。
- DataLoader 参数构造。
- 训练循环。
- 评估循环。
- 指标计算。
- 混淆矩阵保存。
- checkpoint 路径管理。

这些逻辑应沉淀到 `training`、`evaluation`、`metrics` 等公共模块中。

### 4. 配置硬编码较多

如下参数散落在脚本中：

```text
NUM_CLASSES
BATCH_SIZE
LEARNING_RATE
TRAIN_RATIO
VAL_RATIO
TEST_RATIO
PATIENCE
CHECKPOINT_NAME
data_dir
device
num_workers
```

这会影响实验复现，也不利于批量运行不同配置。

### 5. 文档与实际结构不一致

当前 `readme.md` 存在中文编码错乱问题，并且目录说明中包含已经不存在或已经改名的文件，例如：

- `droneRFa_dataset.py`
- `spectrogram_dataset.py`
- `transforms.py`

README 应优先修复为当前真实结构，避免交接和复现实验时误导。

### 6. 运行产物和源码边界不够清晰

项目中存在：

- `logs/`
- `checkpoints/`
- `.pytest_cache/`
- 多处 `__pycache__/`

虽然 `.gitignore` 已经忽略部分产物，但建议进一步统一输出目录，明确源码、数据、缓存、模型权重和实验结果的边界。

## 推荐目标结构

建议中期目标结构如下：

```text
droneRFa/
  README.md
  requirements.txt
  pyproject.toml
  AGENTS.md

  configs/
    default.yaml
    stft.yaml
    cpp.yaml

  src/
    drone_rfa/
      __init__.py

      data/
        __init__.py
        h5_dataset.py
        stft_dataset.py
        cpp_dataset.py
        splits.py

      models/
        __init__.py
        resnet.py
        factory.py

      training/
        __init__.py
        trainer.py
        evaluator.py
        metrics.py
        checkpoint.py

      preprocessing/
        __init__.py
        stft.py
        cpp.py
        fam.py
        fam_torch.py
        fam_grid.py

      visualization/
        __init__.py
        plot_utils.py
        fam_plot.py
        confusion_matrix.py

      utils/
        __init__.py
        logger.py
        device.py
        paths.py

  scripts/
    train.py
    evaluate.py
    precompute_stft_h5.py
    precompute_cpp_h5.py
    generate_stft_png.py
    generate_cpp_png.py
    plot_fam_demo.py

  tests/
    test_data.py
    test_metrics.py
    test_shapes.py

  docs/
    dataset.md
    structure.md
    project_structure_optimization.md
    snr_accuracy_curve_implementation_plan.md

  outputs/
    checkpoints/
    logs/
    figures/
    metrics/
```

其中 `outputs/` 应加入 `.gitignore`，作为统一实验输出目录。

## 模块职责建议

### `configs/`

保存实验配置，避免在脚本中硬编码训练参数。

示例：

```yaml
feature: stft
num_classes: 25
batch_size: 64
learning_rate: 0.001
epochs: 200
patience: 10
split:
  train: 0.6
  val: 0.2
  test: 0.2
data:
  dir: E:/dataSet/DroneRFa/stft_h5
output:
  checkpoint: outputs/checkpoints/best_stft_model.pth
```

### `data/`

负责数据读取、H5 索引、数据切分。

建议抽象一个通用 H5 数据集：

```python
H5FeatureDataset(cache_dir, feature_key)
```

然后 STFT 和 CPP/FAM 数据集只保留差异化封装：

```python
SpectrogramDataset(cache_dir) -> feature_key="stft"
CppDataset(cache_dir) -> feature_key="cpp"
```

### `training/`

负责训练、评估、指标和 checkpoint 管理。

建议拆分为：

- `trainer.py`：训练循环、early stopping。
- `evaluator.py`：验证集和测试集推理。
- `metrics.py`：accuracy、precision、recall、f1、confusion matrix。
- `checkpoint.py`：模型保存、加载、路径管理。

### `preprocessing/`

保存信号处理和特征计算逻辑，例如：

- STFT 计算。
- CPP/FAM 计算。
- FAM grid 聚合。
- Torch/NumPy 两种实现。

这样可以将算法计算逻辑从通用 `utils` 中移出，职责更明确。

### `visualization/`

负责所有可视化输出，例如：

- STFT 图。
- CPP/FAM 图。
- 混淆矩阵图。
- 时域、频域和 FAM 演示图。

### `scripts/`

只保留命令行入口，不承载复杂业务逻辑。

推荐入口形式：

```powershell
python scripts/train.py --feature stft --config configs/stft.yaml
python scripts/train.py --feature cpp --config configs/cpp.yaml
python scripts/evaluate.py --feature stft --checkpoint outputs/checkpoints/best_stft_model.pth
python scripts/evaluate.py --feature cpp --checkpoint outputs/checkpoints/best_cpp_model.pth
```

## 优先级建议

### 第一阶段：低风险整理

优先处理不改变核心行为的结构问题：

1. 修复 `readme.md` 编码和过期目录说明。
2. 新增 `configs/`，迁移训练参数。
3. 新增 `outputs/` 目录规划，并加入 `.gitignore`。
4. 抽取设备选择、路径选择、数据切分公共逻辑。
5. 清理不应进入项目结构认知的 `__pycache__/`、`.pytest_cache/` 等缓存目录。

### 第二阶段：训练与评估管线收敛

将 STFT/CPP 的重复训练和测试流程合并：

1. 抽取 `training/trainer.py`。
2. 抽取 `training/evaluator.py`。
3. 抽取 `training/metrics.py`。
4. 抽取 `training/checkpoint.py`。
5. 将根目录的 `train_stft.py`、`train_cpp.py`、`test_stft.py`、`test_cpp.py` 收敛为 `scripts/train.py` 和 `scripts/evaluate.py`。

### 第三阶段：包名和目录语义升级

将 `src` 从包名改为源码根目录：

```text
src/
  drone_rfa/
```

同步将导入路径从：

```python
from src.data.stft_dataset import SpectrogramDataset
```

迁移为：

```python
from drone_rfa.data.stft_dataset import SpectrogramDataset
```

这个阶段改动范围较大，建议在前两阶段稳定后执行。

### 第四阶段：测试与复现能力建设

补充最小测试集：

- H5 Dataset 能正确读取样本。
- STFT/CPP 样本 shape 符合模型输入。
- split 结果在固定 seed 下可复现。
- metrics 计算结果正确。
- ResNet 前向输出维度为 `(batch_size, num_classes)`。

## 推荐迁移顺序

建议按以下顺序推进：

1. 修复 README 和文档结构。
2. 增加 `configs/` 和 `outputs/`。
3. 抽取公共数据切分、设备选择、路径工具。
4. 抽取公共 H5 Dataset。
5. 抽取训练和评估公共模块。
6. 合并命令行入口。
7. 迁移包名为 `drone_rfa`。
8. 补充测试。

## 风险与注意事项

- 不建议一次性完成全部重构，训练脚本涉及数据路径、设备、checkpoint 和大文件 H5 读取，改动过大时定位问题困难。
- 包名迁移会影响所有导入路径，应单独作为一个阶段处理。
- 训练与测试集切分逻辑必须保持一致，否则会影响历史实验结果对比。
- 迁移 checkpoint 输出目录时，应保留旧路径兼容或在 README 中明确说明新路径。
- 当前 `docs` 下有 PDF 删除状态，整理文档前应确认该文件是有意删除还是误删。

## 总结

当前项目的核心优化方向不是增加更多目录，而是把重复的实验流程收敛成稳定的训练管线，把硬编码参数迁移到配置，把运行产物集中管理，并让文档反映真实结构。

短期内建议优先完成：

1. README 修复。
2. 配置目录建设。
3. 公共训练/评估逻辑抽取。
4. 统一输出目录。
5. 最小测试补充。

完成这些后，项目会更容易复现实验、扩展新特征、替换模型，并支持后续更系统的实验管理。
