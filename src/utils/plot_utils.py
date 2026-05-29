import numpy as np
import matplotlib.pyplot as plt
from .logger import logger
from typing import Optional

def plot_iq_time_domain(sig_iq: np.ndarray,
                        sample_start: int = 0, sample_limit: int = 1000000):
    '''
    输入：复数信号 sig_iq = I + 1j*Q
    输出： 3 张子图(I路, Q路, abs)
    '''
    logger.debug("plot waveform in time domain")
    sample_end = sample_start + sample_limit
    sig = sig_iq.flatten()[sample_start:sample_end]

    sig_i = np.real(sig)   # I路
    sig_q = np.imag(sig)   # Q路
    sig_abs = np.abs(sig)  # 信号幅度

    plt.figure(figsize=(14, 10))
    # 子图1：I 路
    plt.subplot(3, 1, 1)
    plt.plot(sig_i, linewidth=0.6, color='#1f77b4')
    plt.title('I(real)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(alpha=0.3)

    # 子图2：Q 路
    plt.subplot(3, 1, 2)
    plt.plot(sig_q, linewidth=0.6, color='#ff7f0e')
    plt.title('Q(imag)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(alpha=0.3)

    # 子图3：abs 幅度
    plt.subplot(3, 1, 3)
    plt.plot(sig_abs, linewidth=0.6, color='#d62728')
    plt.title('abs(I+jQ)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlabel('Sample Index', fontsize=12)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_iq_frequency_domain(sig_iq: np.ndarray, 
                             sample_start: int = 0, sample_limit: int = 1000000):
    '''
    输入：复数信号 sig_iq = I + 1j*Q
    输出： 1 张子图(FFT abs)
    '''
    logger.info("plot spectrum in frequency domain")
    sample_end = sample_start + sample_limit
    sig = sig_iq.flatten()[sample_start:sample_end]
    N = len(sig)
    fft_sig = np.fft.fft(sig)
    fft_sig = np.fft.fftshift(fft_sig)
    freq = np.fft.fftshift(np.fft.fftfreq(N, 1.0))

    # 幅度谱
    mag = 20 * np.log10(np.abs(fft_sig) + 1e-10)  # dB 表示

    # 绘图
    plt.figure(figsize=(12, 5))
    plt.plot(freq, mag, linewidth=0.7, color="#9400d3")
    plt.title("I/Q frequency domain (FFT spectrum)")
    plt.xlabel("Normalized frequency")
    plt.ylabel("dB")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    time_freq_matrix_dB: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """
    绘制时频矩阵（频谱图）。

    Parameters
    ----------
    freqs : np.ndarray
        频率数组（Hz），形状 (n_freqs,)。
    times : np.ndarray
        时间数组（s），形状 (n_times,)。
    time_freq_matrix_dB : np.ndarray
        时频矩阵（功率谱密度，dB），形状 (n_times, n_freqs)。
    vmin, vmax : float or None
        colorbar 的 dB 范围，None 时自动根据数据确定。
    """
    title = "Spectrogram (STFT)"
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(
        freqs / 1e6,
        times,
        time_freq_matrix_dB,
        shading="auto",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Frequency (MHz)", fontsize=12)
    plt.ylabel("Time (s)", fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label("Power/Frequency (dB/Hz)", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Saved spectrogram to '%s'", save_path)
    plt.close()
