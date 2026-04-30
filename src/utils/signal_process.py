import numpy as np
from scipy.signal import stft
from .logger import logger
from typing import Optional, Tuple

def iq_to_spectrogram(
    sig_iq: np.ndarray,
    fs: float = 100e6,
    nperseg: int = 4096,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 IQ 复数信号转换为时频矩阵（频谱图）。

    Parameters
    ----------
    sig_iq : np.ndarray
        IQ 复数信号，I + 1j*Q 格式。
    fs : float
        采样频率（Hz），默认 100e6（100 MHz）。
    nperseg : int
        每个 STFT 段的样本数（窗长），默认 256。
    noverlap : int or None
        相邻段之间的重叠样本数，默认 nperseg // 2。
    nfft : int or None
        FFT 点数，默认 nperseg。

    Returns
    -------
    freqs : np.ndarray
        频率数组（Hz），形状 (n_freqs,)。
    times : np.ndarray
        时间数组（s），形状 (n_times,)。
    Sxx_dB : np.ndarray
        时频矩阵（功率谱密度，dB），形状 (n_freqs, n_times)。
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    sig = sig_iq.flatten()

    freqs, times, time_freq_matrix = stft(
        sig,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False
    )
    freqs = np.fft.fftshift(freqs)
    time_freq_matrix = np.fft.fftshift(time_freq_matrix, axes=0)
    # 功率谱密度 → dB
    eps = np.finfo(float).eps
    time_freq_matrix_dB = 20.0 * np.log10(np.abs(time_freq_matrix) + eps)
    logger.debug(
        "Spectrogram shape: (%d freq bins, %d time segments), ",
        time_freq_matrix.shape[0],
        time_freq_matrix.shape[1]
    )
    return freqs, times, np.abs(time_freq_matrix_dB).T