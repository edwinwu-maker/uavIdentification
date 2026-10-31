from collections.abc import Iterator

import numpy as np
import torch


def limited_samples(total_samples: int, max_samples: int | None) -> int:
    """返回按上限限制后的样本数量。

    输入：
        total_samples: 样本总数，整数。
        max_samples: 样本数量上限，整数或 None；None 表示不限制。
    输出：
        整数样本数量；上限为 None 时返回 total_samples，否则返回二者的较小值。
    """
    if max_samples is None:
        return total_samples
    return min(total_samples, max_samples)


def allocate_evenly(total: int, group_count: int) -> list[int]:
    """将整数总量尽量均匀地分配到各组。

    输入：
        total: 待分配的非负整数总量。
        group_count: 正整数分组数；输入约束由调用方保证。
    输出：
        长度为 group_count 的整数列表，第 i 项为组 i 的配额，总和为 total。
        余数依次分给前几个组，各组配额最多相差 1。
    示例：
        allocate_evenly(10, 3) 返回 [4, 3, 3]。
    """
    # base: 每组基础数量
    # remainder: 有 remainder 个组，需要多分配 1 个元素
    base, remainder = divmod(total, group_count)
    return [base + (group_id < remainder) for group_id in range(group_count)]


def prioritize_spread_indices(
    indices: np.ndarray, values: np.ndarray, front_count: int
) -> list[int]:
    """按数值排序候选索引，优先排列覆盖整个排序区间的索引。

    输入：
        indices: 一维、不重复的有效整数索引数组。
        values: 可按 indices 访问的一维数值数组；输入约束由调用方保证。
        front_count: 希望前置的索引数量，整数。
    输出：
        完整的 Python 索引列表；先按排序位置均匀取点并取整，前置最多 front_count 项，
        再按数值升序追加其余索引。front_count <= 0 时仅返回按数值排序的索引列表。
        均匀取点针对排序位置，数值间距未必相等；不修改输入数组。
    """
    ordered = indices[np.argsort(values[indices])]
    if len(ordered) <= 1 or front_count <= 0:
        return ordered.tolist()
    positions = np.linspace(
        0, len(ordered) - 1, min(front_count, len(ordered))
    ).round().astype(int)
    # 取点数不超过候选数，取整后的位置互异；掩码保留其余索引的排序顺序。
    remaining_mask = np.ones(len(ordered), dtype=bool)
    remaining_mask[positions] = False
    return np.concatenate((ordered[positions], ordered[remaining_mask])).tolist()


def iter_padded_segments(
    x: torch.Tensor,
    *,
    segment_samples: int,
    segment_hop_samples: int,
) -> Iterator[torch.Tensor]:
    """按固定步长逐段返回一维张量，不足段长的片段在右侧补零。

    输入：
        x: 形状为 (L,) 的一维 Tensor。
        segment_samples: 每段长度，须为正整数。
        segment_hop_samples: 相邻片段起点的间隔，须为正整数。
        一维输入及正参数要求由调用方保证。
    输出：
        惰性迭代器，每段形状为 (segment_samples,)，dtype 和设备与 x 一致。
        空输入不产生片段；重叠分段时可能有多个尾部片段需要补零。

    完整片段与输入共享存储，调用方应避免原地修改；补零片段为新张量。
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    for start in range(0, x.numel(), segment_hop_samples):
        segment = x[start : start + segment_samples]
        if segment.numel() == 0:
            continue
        if segment.numel() < segment_samples:
            pad = torch.zeros(
                segment_samples - segment.numel(),
                dtype=segment.dtype,
                device=segment.device,
            )
            segment = torch.cat((segment, pad))
        yield segment
