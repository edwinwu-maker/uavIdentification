from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import logger


def _import_pyplot(show: bool):
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_iq_time_domain(
    sig_iq: np.ndarray,
    sample_start: int = 0,
    sample_limit: int = 1000000,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    输入：复数信号 sig_iq = I + 1j*Q
    输出：3 张子图(I路, Q路, abs)
    """

    plt = _import_pyplot(show)

    logger.debug("plot waveform in time domain")
    sample_end = sample_start + sample_limit
    sig = sig_iq.flatten()[sample_start:sample_end]

    sig_i = np.real(sig)
    sig_q = np.imag(sig)
    sig_abs = np.abs(sig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(sig_i, linewidth=0.6, color="#1f77b4")
    axes[0].set_title("I(real)", fontsize=14)
    axes[0].set_ylabel("Amplitude", fontsize=12)
    axes[0].grid(alpha=0.3)

    axes[1].plot(sig_q, linewidth=0.6, color="#ff7f0e")
    axes[1].set_title("Q(imag)", fontsize=14)
    axes[1].set_ylabel("Amplitude", fontsize=12)
    axes[1].grid(alpha=0.3)

    axes[2].plot(sig_abs, linewidth=0.6, color="#d62728")
    axes[2].set_title("abs(I+jQ)", fontsize=14)
    axes[2].set_ylabel("Amplitude", fontsize=12)
    axes[2].set_xlabel("Sample Index", fontsize=12)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved time-domain plot to '%s'", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_iq_frequency_domain(
    sig_iq: np.ndarray,
    sample_start: int = 0,
    sample_limit: int = 1000000,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    输入：复数信号 sig_iq = I + 1j*Q
    输出：1 张子图(FFT abs)
    """

    plt = _import_pyplot(show)

    logger.info("plot spectrum in frequency domain")
    sample_end = sample_start + sample_limit
    sig = sig_iq.flatten()[sample_start:sample_end]
    n = len(sig)
    fft_sig = np.fft.fft(sig)
    fft_sig = np.fft.fftshift(fft_sig)
    freq = np.fft.fftshift(np.fft.fftfreq(n, 1.0))

    mag = 20 * np.log10(np.abs(fft_sig) + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freq, mag, linewidth=0.7, color="#9400d3")
    ax.set_title("I/Q frequency domain (FFT spectrum)")
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("dB")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved frequency-domain plot to '%s'", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_stft(
    freqs: np.ndarray,
    times: np.ndarray,
    time_freq_matrix_dB: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    *,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    绘制时频矩阵（频谱图）。
    """

    plt = _import_pyplot(show)

    title = "stft (STFT)"
    fig, ax = plt.subplots(figsize=(14, 6))
    mesh = ax.pcolormesh(
        freqs / 1e6,
        times,
        time_freq_matrix_dB,
        shading="auto",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Power/Frequency (dB/Hz)", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved stft to '%s'", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig
