import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RFSegmentationConfig:
    frame_len: int                 # N: 每帧采样点数
    confidence: float = 0.95       # 论文中的 varrho
    eps: float = 1e-12             # 防止除零和 log(0)
    use_order_statistic: bool = True
    normalize_eser_for_mode: bool = True


class RFSignalAcquisitionSegmentation(nn.Module):
    """
    PyTorch implementation of RF Signal Acquisition and Segmentation
    from TASE-Net paper.

    Input:
        r: Tensor, shape [M] or [B, M]
           real or complex RF signal sequence

    Output:
        x: selected predominant signal sequence, shape [B, phi * N]
        selected_indices: indices of selected frames, shape [B, phi]
        eser: ST-ESER values of all frames, shape [B, I]
    """

    def __init__(self, config: RFSegmentationConfig):
        super().__init__()
        self.cfg = config

    def forward(
        self,
        r: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if r.dim() == 1:
            r = r.unsqueeze(0)

        B, M = r.shape
        N = self.cfg.frame_len
        I = M // N

        if I < 1:
            raise ValueError(f"Signal length M={M} is smaller than frame_len N={N}")

        # 截断到整数帧长度：I = floor(M / N)
        r = r[:, :I * N]

        # Eq. (1): ui(t') = r(t' + (i-1)N)
        frames = r.reshape(B, I, N)

        # Eq. (2): DFT
        U = torch.fft.fft(frames, dim=-1)

        # |Ui(m)|^2
        power = torch.abs(U) ** 2

        # Eq. (3): spectral probability Pi(m)
        power_sum = power.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        P = power / power_sum

        # Eq. (4): spectral entropy Hi
        H = -(P * torch.log(P.clamp_min(self.cfg.eps))).sum(dim=-1)
        H = H.clamp_min(self.cfg.eps)

        # Eq. (5): ST-ESER ei
        eser = power.sum(dim=-1) / H

        # 选择 predominant segments
        if top_k is not None:
            phi = min(top_k, I)
        elif self.cfg.use_order_statistic:
            phi = self._estimate_phi_by_order_statistic(eser)
        else:
            phi = max(1, int(round(0.1 * I)))

        # 选择 ST-ESER 最大的 phi 帧
        values, selected_indices = torch.topk(eser, k=phi, dim=1, largest=True)

        # 为保持时间顺序，可排序索引；若想严格按 ST-ESER 从大到小拼接，删除这行
        selected_indices, _ = torch.sort(selected_indices, dim=1)

        gather_idx = selected_indices.unsqueeze(-1).expand(-1, -1, N)
        selected_frames = torch.gather(frames, dim=1, index=gather_idx)

        # Eq. (10): concatenate retained frames into x(n)
        x = selected_frames.reshape(B, phi * N)

        return x, selected_indices, eser

    def _estimate_phi_by_order_statistic(self, eser: torch.Tensor) -> int:
        """
        Approximate Eq. (7)-(9).

        论文中:
            P(E_(iota) > e_bar) = confidence
            phi = I - iota + 1

        这里用经验分布 ECDF 近似。
        """
        B, I = eser.shape

        e = eser.detach()

        if self.cfg.normalize_eser_for_mode:
            e = e / e.max(dim=1, keepdim=True).values.clamp_min(self.cfg.eps)

        # 用直方图众数近似论文中的 e_bar: most probable ST-ESER value
        e_bar_list = []
        bins = min(64, max(8, I // 4))

        for b in range(B):
            hist = torch.histc(e[b].float(), bins=bins, min=float(e[b].min()), max=float(e[b].max()))
            mode_bin = torch.argmax(hist)
            e_min, e_max = e[b].min(), e[b].max()
            bin_width = (e_max - e_min) / bins
            e_bar = e_min + (mode_bin.float() + 0.5) * bin_width
            e_bar_list.append(e_bar)

        e_bar = torch.stack(e_bar_list)

        # 经验估计 F_E(e_bar)
        F = (e <= e_bar[:, None]).float().mean(dim=1)

        # 对每个 batch 求满足 1 - F_{E_(iota)}(e_bar) >= confidence 的 iota
        phis = []
        for b in range(B):
            Fb = F[b].clamp(0.0, 1.0)
            best_iota = I

            for iota in range(1, I + 1):
                # F_{E_(iota)}(e) = sum_{j=iota}^{I} C(I,j) F^j (1-F)^(I-j)
                prob_le = 0.0
                for j in range(iota, I + 1):
                    comb = torch.exp(
                        torch.lgamma(torch.tensor(I + 1.0, device=eser.device))
                        - torch.lgamma(torch.tensor(j + 1.0, device=eser.device))
                        - torch.lgamma(torch.tensor(I - j + 1.0, device=eser.device))
                    )
                    prob_le = prob_le + comb * (Fb ** j) * ((1 - Fb) ** (I - j))

                prob_gt = 1.0 - prob_le

                if prob_gt >= self.cfg.confidence:
                    best_iota = iota
                    break

            phi = I - best_iota + 1
            phis.append(max(1, min(I, phi)))

        # batch 中取中位数，保证输出定长，方便后续 CPP / DataLoader
        return int(torch.median(torch.tensor(phis)).item())


# -------------------------
# Example
# -------------------------
if __name__ == "__main__":
    fs = 100e6
    M = 100_000
    N = 1024

    # 模拟一段 RF 信号：噪声 + 若干高能 UAV-like burst
    r = 0.02 * torch.randn(M)
    r[20_000:35_000] += 0.5 * torch.randn(15_000)
    r[60_000:75_000] += 0.5 * torch.randn(15_000)

    segmenter = RFSignalAcquisitionSegmentation(
        RFSegmentationConfig(frame_len=N, confidence=0.95)
    )

    x, idx, eser = segmenter(r, top_k=20)

    print("Selected signal shape:", x.shape)
    print("Selected frame indices:", idx)
    print("ST-ESER shape:", eser.shape)