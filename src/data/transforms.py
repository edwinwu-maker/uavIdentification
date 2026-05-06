import numpy as np
import torch
from scipy.signal import stft

# Paper-specified parameters
FS = 100e6
NPERSEG = 1024
NOVERLAP = 512  # overlap rate 0.5
NFFT = 1024
SPEC_TIME_BINS = 1024  # truncate time dim to 1024


class IQToSpectrogram:
    """Transform complex IQ signals (2, N) into normalized spectrograms (2, 1024, 1024).

    Paper reference: Section 4.3 — STFT with window length 1024, overlap rate 0.5,
    input size (C, W, H) = (2, 1024, 1024).
    """

    def __call__(self, iq_data: np.ndarray, label: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            iq_data: complex IQ signals of shape (2, sample_length).
            label: integer drone class label.

        Returns:
            specs: (2, 1024, 1024) float32 tensor, z-score normalized per channel.
            label: int64 tensor.
        """
        specs = []
        for ch in range(2):
            spec = self._compute_spectrogram(iq_data[ch])
            spec = self._normalize(spec)
            specs.append(spec)

        specs = np.stack(specs, axis=0).astype(np.float32)
        return torch.from_numpy(specs), torch.tensor(label, dtype=torch.int64)

    def _compute_spectrogram(self, sig_iq: np.ndarray) -> np.ndarray:
        sig = sig_iq.flatten()
        _, _, Zxx = stft(
            sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
            return_onesided=False,  # 输出双边频谱（正负频率都要）
        )
        Zxx = np.fft.fftshift(Zxx, axes=0)  # 把0频移到中间
        eps = np.finfo(float).eps
        Zxx_db = 20.0 * np.log10(np.abs(Zxx) + eps)
        return Zxx_db[:, :SPEC_TIME_BINS]  # (1024, 1024)

    @staticmethod
    def _normalize(spec: np.ndarray) -> np.ndarray:
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + 1e-8)
