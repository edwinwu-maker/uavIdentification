from collections.abc import Iterator

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
