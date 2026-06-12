# 项目结构规划优化方向

本文基于当前 `droneRFa` 项目的实际目录与代码组织情况，记录已经完成的结构优化，并给出剩余重构方向。

## 当前状态

项目目前仍保留早期实验入口：

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
  tests/
  docs/
  outputs/
```

当前已经完成一部分低风险整理：

- `readme.md` 已更新为当前真实结构和运行命令。
- `.gitignore` 已忽略 `outputs/*`，保留 `outputs/README.md`。
- 日志、checkpoint、图像和指标已按 `outputs/README.md` 归档：
  - `outputs/logs/`
  - `outputs/checkpoints/`
  - `outputs/figures/`
  - `outputs/metrics/`
- 已新增公共路径工具：`src/utils/paths.py`。
- 已新增默认设备选择工具：`src/utils/device.py`。
- 已新增数据集切分工具：`src/data/splits.py`。
- 已新增通用 H5 特征数据集：`src/data/h5_dataset.py`。
- `SpectrogramDataset` 和 `CppDataset` 已改为 `H5FeatureDataset` 的薄封装。
- 已新增最小测试：
  - `tests/test_device.py`
  - `tests/test_splits.py`
  - `tests/test_h5_dataset.py`

这些改动已经降低了入口脚本、数据读取和输出目录的重复度，但训练、评估、指标和配置仍然分散。

## 剩余主要问题

### 1. 训练与评估入口仍然重复

根目录仍有四个入口：

```text
train_stft.py
train_cpp.py
test_stft.py
test_cpp.py
```

它们仍重复承担：

- 参数解析。
- DataLoader 构造。
- 模型创建。
- 优化器和 loss 创建。
- 训练循环。
- 验证/测试推理。
- 指标计算。
- checkpoint 加载/保存。
- 混淆矩阵保存和绘图。

下一步优化重点应是抽取训练与评估公共模块，而不是直接移动包名或大规模改目录。

### 2. 配置仍硬编码在脚本中

如下配置仍在 `train_*.py` 和 `test_*.py` 中：

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
num_workers
epochs
device
```

这会影响实验复现，也不利于批量运行 STFT、CPP/FAM 或未来新特征。

### 3. 指标、混淆矩阵和评估循环还没有沉淀

`test_stft.py` 和 `test_cpp.py` 中仍重复包含：

- `compute_metrics`
- `save_confusion_matrix_image`
- 测试集推理循环
- 混淆矩阵 `.npy` 和 `.png` 保存逻辑

这些逻辑应抽到公共模块，避免后续新增特征时继续复制。

### 4. 预处理和可视化逻辑仍混在 `scripts/` 与 `src/utils/`

当前 `scripts/` 里既有 CLI 参数解析，也有较多业务逻辑：

- STFT 预计算。
- CPP/FAM 预计算。
- STFT PNG 生成。
- CPP/FAM PNG 生成。
- FAM demo 图生成。

`src/utils/` 中也包含不同职责：

- 日志和路径工具。
- FAM/SCF 算法。
- 图像绘制。
- 合成信号。

长期看应拆成 `preprocessing/` 和 `visualization/`，但这一步应放在训练/评估管线收敛之后。

### 5. 包名仍为泛化的 `src`

当前导入仍是：

```python
from src.data.stft_dataset import SpectrogramDataset
from src.models.resnet import DroneRFaResNet18
```

长期建议迁移为：

```python
from drone_rfa.data.stft_dataset import SpectrogramDataset
from drone_rfa.models.resnet import DroneRFaResNet18
```

但包名迁移会影响所有导入路径、脚本执行方式和测试，应单独作为后期阶段处理。

## 推荐目标结构

中期目标结构建议如下：

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
    test_device.py
    test_metrics.py
    test_shapes.py
    test_splits.py

  docs/
    dataset.md
    structure.md
    project_structure_optimization.md
    snr_accuracy_curve_implementation_plan.md

  outputs/
    README.md
    checkpoints/
    logs/
    figures/
    metrics/
```

注意：`src/drone_rfa/` 包名迁移不是下一步优先事项，应在公共训练/评估模块稳定后单独执行。

## 剩余优化方案

### 阶段 1：训练与评估公共模块

目标：减少 `train_stft.py`、`train_cpp.py`、`test_stft.py`、`test_cpp.py` 中的重复逻辑，但先保留现有入口文件。

建议新增：

```text
src/training/
  __init__.py
  evaluator.py
  metrics.py
  checkpoint.py
  trainer.py
```

建议拆分顺序：

1. `metrics.py`
   - 抽取 `compute_metrics`。
   - 抽取 confusion matrix 计算。
   - 成功标准：`test_stft.py` 和 `test_cpp.py` 不再各自定义 `compute_metrics`。

2. `visualization/confusion_matrix.py` 或 `training/metrics.py`
   - 抽取混淆矩阵图片保存。
   - 成功标准：混淆矩阵 `.npy` 仍进 `outputs/metrics/`，图片仍进 `outputs/figures/`。

3. `evaluator.py`
   - 抽取验证/测试推理循环。
   - 兼容只返回 loss/accuracy 和返回 preds/labels 两种场景。
   - 成功标准：训练验证和测试推理复用同一套 evaluator。

4. `checkpoint.py`
   - 抽取 checkpoint 路径、保存和加载。
   - 成功标准：训练脚本不直接调用 `torch.save`，测试脚本不直接拼 checkpoint 路径。

5. `trainer.py`
   - 抽取 epoch 训练循环、early stopping、best checkpoint 保存。
   - 成功标准：`train_stft.py` 和 `train_cpp.py` 只负责参数解析、选择 dataset、创建 model 并调用 trainer。

建议验证：

```bash
~/Desktop/venv/bin/python -m pytest
~/Desktop/venv/bin/python -m py_compile train_stft.py train_cpp.py test_stft.py test_cpp.py src/training/*.py
~/Desktop/venv/bin/python train_stft.py --help
~/Desktop/venv/bin/python test_stft.py --help
```

### 阶段 2：配置文件

目标：把硬编码训练参数迁移到可复现配置。

建议新增：

```text
configs/
  stft.yaml
  cpp.yaml
```

配置内容建议：

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
  seed: 42
data:
  dir: ~/Desktop/dataset/droneRFa/stft_h5
output:
  checkpoint: outputs/checkpoints/best_stft_model.pth
```

建议先支持：

```bash
python train_stft.py --config configs/stft.yaml
python train_cpp.py --config configs/cpp.yaml
```

暂时不建议一开始就引入复杂配置框架。可以优先使用 `yaml.safe_load` 和一个小型配置解析函数。

成功标准：

- 无 `--config` 时保持现有默认行为。
- 有 `--config` 时配置文件覆盖默认参数。
- README 中给出最小配置示例。
- 测试覆盖配置读取和默认值合并。

### 阶段 3：统一 CLI 入口

目标：在公共训练/评估模块稳定后，新增统一入口：

```bash
python scripts/train.py --feature stft --config configs/stft.yaml
python scripts/train.py --feature cpp --config configs/cpp.yaml
python scripts/evaluate.py --feature stft --config configs/stft.yaml
python scripts/evaluate.py --feature cpp --config configs/cpp.yaml
```

迁移策略：

1. 先新增 `scripts/train.py` 和 `scripts/evaluate.py`。
2. 保留 `train_stft.py`、`train_cpp.py`、`test_stft.py`、`test_cpp.py` 作为兼容入口。
3. README 先推荐新入口，但保留旧入口说明。
4. 等新入口稳定后，再决定是否删除旧入口。

成功标准：

- STFT/CPP 可以通过同一个训练入口运行。
- STFT/CPP 可以通过同一个评估入口运行。
- 旧入口仍可运行或明确标记为 deprecated。

### 阶段 4：预处理与可视化模块化

目标：让 `scripts/` 只负责 CLI，不承载大量业务逻辑。

建议迁移：

```text
src/preprocessing/
  stft.py
  cpp.py
  fam.py
  fam_torch.py
  fam_grid.py

src/visualization/
  stft_png.py
  cpp_png.py
  fam_plot.py
  plot_utils.py
  confusion_matrix.py
```

迁移顺序：

1. 先移动纯函数和算法函数。
2. 再让 `scripts/precompute_*.py` 调用模块函数。
3. 最后让图片生成脚本调用 `visualization` 模块。

注意：

- 这一步可能影响预处理性能和设备路径，应该在训练/评估入口稳定后做。
- 不建议同时改算法实现和文件结构。

### 阶段 5：模型工厂与形状测试

目标：为后续新增模型或特征输入尺寸做准备。

建议新增：

```text
src/models/factory.py
tests/test_shapes.py
```

初始只支持：

```python
create_model("resnet18", num_classes=25)
```

成功标准：

- ResNet 前向输出 shape 为 `(batch_size, num_classes)`。
- STFT/CPP 样本可被模型接受或在 README 中明确输入尺寸约束。

### 阶段 6：包名迁移

目标：将真实业务包名从 `src` 迁移到 `drone_rfa`。

建议迁移目标：

```text
src/
  drone_rfa/
    data/
    models/
    training/
    preprocessing/
    visualization/
    utils/
```

迁移策略：

1. 新增 `pyproject.toml`，配置 package discovery。
2. 移动模块到 `src/drone_rfa/`。
3. 全局替换导入路径。
4. 调整脚本入口的运行方式。
5. 跑全量测试和关键 CLI `--help`。

风险：

- 影响所有导入路径。
- 影响直接运行脚本的方式。
- 容易与未提交重构混在一起。

因此该阶段应最后单独进行。

## 推荐后续执行顺序

建议从最小风险到最大风险推进：

1. 抽取 `src/training/metrics.py`。
2. 抽取混淆矩阵保存和绘图逻辑。
3. 抽取 `src/training/evaluator.py`。
4. 抽取 `src/training/checkpoint.py`。
5. 抽取 `src/training/trainer.py`。
6. 增加 `configs/stft.yaml` 和 `configs/cpp.yaml`。
7. 新增统一 `scripts/train.py` 和 `scripts/evaluate.py`。
8. 整理 `preprocessing/` 和 `visualization/`。
9. 增加模型 factory 和 shape 测试。
10. 最后迁移包名为 `drone_rfa`。

## 风险与注意事项

- 不建议一次性完成全部重构。训练脚本涉及数据路径、设备、checkpoint 和大文件 H5 读取，改动过大时定位问题困难。
- 公共训练/评估模块应先服务现有 STFT/CPP 两条线，不要提前为未来模型做过度抽象。
- 训练与测试集切分必须继续使用同一个 `split_dataset(..., seed=42)`，否则会影响历史结果对比。
- `outputs/` 已作为默认运行产物目录，后续新增输出应优先归档到 `outputs/README.md` 中定义的子目录。
- 预计算 `.h5` 文件目前仍建议保留在数据集目录下，因为它们是训练输入数据，不是普通实验产物。
- 包名迁移应在公共模块和 CLI 稳定后独立执行。

## 总结

当前已经完成低风险基础整理：README、outputs、路径工具、设备选择、数据切分、H5 Dataset 抽象和最小测试。

剩余优化的主线应是：

1. 先收敛训练/评估公共逻辑。
2. 再引入配置文件提高复现能力。
3. 然后合并 CLI 入口。
4. 最后做目录语义升级和包名迁移。

这样可以在保持现有实验可运行的前提下，逐步减少重复代码，并为新增特征、模型和系统化实验管理打基础。
