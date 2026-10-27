def limited_samples(total_samples: int, max_samples: int | None) -> int:
    """返回限制后的样本数量；上限为 None 时返回总样本数。"""
    if max_samples is None:
        return total_samples
    return min(total_samples, max_samples)
