"""DeepCluster 审核、导出和可视化共享的身份字段与合法取值。"""

# 来源身份字段，顺序固定，用于生成身份键和核验样本来源。
# 不包含 cluster_id，避免重新聚类后簇编号变化影响逐样本审核复用。
SOURCE_IDENTITY_FIELDS = (
    "source_file",        # 源 H5 文件名，不含目录。
    "source_row_idx",     # 样本在源 H5 数据集中的行索引。
    "original_label",     # 原始 DroneRFa 类别标签，未经模型类别重映射。
    "rf_channel",         # 原始 RF 通道编号。
    "source_sample_idx",  # 对应 RF 通道内的原始 IQ 样本索引。
)

# 人工判定的信号语义；空字符串表示尚未标注，不等同于 uncertain。
MANUAL_LABELS = (
    "video",      # 可确认的有效图传信号。
    "other",      # 可明确判定应剔除的其它信号。
    "uncertain",  # 无法确认是否满足有效图传要求。
)

# 仅用于已确认的开发逐样本审核；独立盲审标签不能决定处置。
MANUAL_LABEL_DECISIONS = {"video": "keep", "other": "drop", "uncertain": "quarantine"}

# 清洗处置的合法值；frozenset 防止集合被原地修改。
# keep：写入保留分区；drop：不导出 H5，仅记录清单；
# quarantine：写入独立隔离分区，留待进一步审核。
DECISIONS = frozenset(("keep", "drop", "quarantine"))
