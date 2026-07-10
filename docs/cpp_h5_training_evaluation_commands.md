# CPP H5 生成、训练与评估命令

本文给出完整的 CPP 工作流：原始 `.mat` → CPP `.h5` → 训练并生成 manifest → 常规评估 → SNR 评估。

## 1. 设置路径

```bash
cd /home/wurixin/uavIdentification

PY=/home/wurixin/venv-3.10/bin/python
RAW=/path/to/DroneRFa
H5=/path/to/DroneRFa_cpp_awgn_random_h5

MANIFEST=outputs/splits/cpp_random12_seed42.csv
CHECKPOINT=outputs/checkpoints/best_cpp_model.pth
```

将 `/path/to/DroneRFa` 替换为实际原始数据目录。

## 2. 每类生成 12 个 CPP H5 文件

```bash
$PY scripts/precompute_cpp_h5.py \
  --data-dir "$RAW" \
  --output-dir "$H5" \
  --files-per-class 12 \
  --sample-length 1000000 \
  --segment-samples 262144 \
  --fam-merge mean \
  --fam-nfft 256 \
  --fam-hop 256 \
  --cpp-normalization log-zscore-sample \
  --f-bins 257 \
  --alpha-bins 257 \
  --snr-min -5 \
  --snr-max 15 \
  --noise-seed 42 \
  --device cuda:0
```

该命令默认添加 `-5～15 dB` 的随机 AWGN。没有 CUDA 时将 `--device cuda:0` 改为 `--device cpu`。

## 3. 训练并生成划分 CSV

```bash
$PY scripts/train.py \
  --feature cpp \
  --model resnet18-small-stem \
  --data-dir "$H5" \
  --files-per-class 12 \
  --split-manifest "$MANIFEST" \
  --checkpoint-path "$CHECKPOINT" \
  --batch-size 64 \
  --lr 0.001 \
  --epochs 200 \
  --patience 10 \
  --num-workers 4 \
  --device cuda:0
```

每类有 12 个文件时，默认文件级划分为：

- train：每类 7 个
- val：每类 2 个
- test：每类 3 个

`train.py` 是唯一会创建 manifest 的入口。若 `$MANIFEST` 已存在，会严格校验并复用，不会重新划分。需要新划分时应使用新的 CSV 文件名。

## 4. 在预计算 H5 测试集上评估

```bash
$PY scripts/evaluate.py \
  --feature cpp \
  --model resnet18-small-stem \
  --data-dir "$H5" \
  --model-path "$CHECKPOINT" \
  --split-manifest "$MANIFEST" \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda:0 \
  --cm-image-path outputs/figures/cpp_confusion_matrix.png
```

该命令严格读取 manifest 中的 `test` 文件，不会创建或修改 CSV。

## 5. 在原始 IQ 上进行固定 SNR 评估

```bash
$PY scripts/eval_snr_accuracy_cpp.py \
  --data-dir "$RAW" \
  --model-path "$CHECKPOINT" \
  --model resnet18-small-stem \
  --split-manifest "$MANIFEST" \
  --snrs -15 -10 -7.5 -5 -2.5 0 2.5 5 7.5 10 \
  --sample-length 1000000 \
  --segment-samples 262144 \
  --fam-merge mean \
  --fam-nfft 256 \
  --fam-hop 256 \
  --cpp-normalization log-zscore-sample \
  --f-bins 257 \
  --alpha-bins 257 \
  --batch-size 1 \
  --device cuda:0 \
  --output-csv outputs/metrics/cpp_snr_accuracy.csv \
  --output-png outputs/figures/cpp_snr_accuracy.png \
  --output-cm-prefix cpp_snr_confusion_matrix \
  --predictions-csv outputs/metrics/cpp_snr_predictions.csv \
  --per-file-csv outputs/metrics/cpp_snr_per_file.csv
```

SNR 评估只会为 manifest 中的测试文件计算 CPP，但原始数据目录必须包含 manifest 引用的全部源文件，以便完成一致性校验。

预计算和 SNR 评估的 CPP 参数必须保持一致。如果预计算时启用了 `--use-rf-segmentation` 或修改了 FAM/CPP 参数，SNR 评估时也必须传入相同设置。
