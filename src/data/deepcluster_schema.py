"""DeepCluster 审核与导出共用的身份字段和人工处置取值。"""

SOURCE_IDENTITY_FIELDS = (
    "source_file", "source_row_idx", "original_label", "rf_channel", "source_sample_idx",
)
MANUAL_LABELS = (
    "video", "background", "control_only", "interference", "weak_or_mixed", "uncertain",
)
DECISIONS = frozenset(("keep", "drop", "quarantine"))
