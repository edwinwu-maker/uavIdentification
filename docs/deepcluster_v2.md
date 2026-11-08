# STFT DeepCluster-v2 适配

`--backbone resnet18` 使用面向小数据集的 DeepCluster-v2 适配；默认 CNN 继续使用原来的 PCA 白化与均衡采样流程。

```bash
~/Desktop/venv/bin/python scripts/train_stft_deepcluster.py \
  --data-dir /path/to/clean_stft_h5 --backbone resnet18 \
  --epochs 100 --n-clusters 16 --batch-size 8 \
  --learning-rate 0.0003 --min-learning-rate 0.000001 \
  --warmup-epochs 5 --temperature 0.1 \
  --work-dir outputs/stft_deepcluster_v2_resnet_epoch100_clusters16
```

ResNet 保留小卷积入口、GroupNorm 和 2×2 池化，投影使用两层 MLP、BatchNorm1d 和 L2 归一化。每轮在确定性输入的投影空间执行球面 KMeans，以固定单位中心计算温度交叉熵；两个时间平移视图使用相同伪标签。训练遍历全部样本，两视图合并前向，因此 batch size 表示原始样本数，实际前向包含两倍样本。初轮前无梯度初始化投影 BN 统计。

这里保留 AdamW，使用 warmup 和余弦学习率；单聚类头、全量重新编码替代官方多头和分布式特征库。时间增强仅循环平移最多 5%，保留频率位置和所有信号值，但存在循环接缝。`--pca-dim` 对 ResNet 不生效。新结构须从头训练，旧 ResNet 权重不能直接加载。

最终重新编码、聚类后保存 CSV、checkpoint 和复核材料。中心距离来自单位投影空间，不能与旧 PCA 空间距离直接比较。评估清洗收益时，使用 `--reuse-review-dir` 复用同一数据集的盲审身份，比较图传误删、其他信号误保留与隔离比例；loss 或 NMI 不代表清洗准确率。上述学习率是实验起点。

参考：[官方训练](https://github.com/facebookresearch/swav/blob/main/main_deepclusterv2.py)、[官方模型](https://github.com/facebookresearch/swav/blob/main/src/resnet50.py)。本实现是 STFT 适配，并非官方大规模配置的完整复现。
