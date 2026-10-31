from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import logger


def pooled_axis(values: np.ndarray, output_bins: int) -> np.ndarray:
    """生成长度为 output_bins 的绘图坐标，不修改输入。

    values 为非空一维坐标数组，output_bins 为正整数，约束由调用方保证。
    可整除时按等宽分组求均值；不可整除时在首尾端点间线性插值，
    作为显示用近似，并非精确的自适应平均池化中心。数据类型沿用 NumPy 规则。
    """
    if len(values) % output_bins != 0:
        return np.linspace(values[0], values[-1], output_bins)
    return values.reshape(output_bins, -1).mean(axis=1)


def stft_plot_axes(
    *,
    sample_rate: float,
    n_fft: int,
    hop_length: int,
    source_time_bins: int,
    output_freq_bins: int,
    output_time_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成池化 STFT 的显示坐标，分别返回 Hz 频率轴和秒时间轴。

    sample_rate、n_fft、hop_length 及各轴长度须为正数，由调用方保证。
    source_time_bins 为池化前的时间帧数；输出轴长度与特征图尺寸对应。
    可整除时取等宽分组均值，否则沿用端点线性插值作为显示近似。
    """
    source_freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / sample_rate))
    source_times = np.arange(source_time_bins, dtype=np.float64) * hop_length / sample_rate
    freqs = pooled_axis(source_freqs, output_freq_bins)
    times = pooled_axis(source_times, output_time_bins)
    return freqs, times


def import_pyplot(show: bool):
    """按显示需求选择绘图后端，并延迟导入、返回 pyplot 模块。

    show=False 时在导入 pyplot 前设置无界面的 Agg 后端，适用于服务器
    或批量保存图片；show=True 时沿用 Matplotlib 当前的后端配置。
    """
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_stft_sample(
    stft_sample: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    save_path: str | Path,
    *,
    title: str,
) -> None:
    """保存单通道 z-score STFT，形状为 (1, H, W)，不再做归一化。

    freqs、times 的长度分别为 H、W，单位为 Hz、秒，显示时换算为 MHz、ms。
    标题由调用方提供；自动创建输出目录，绘图或保存异常向调用方传播。
    """
    plt = import_pyplot(False)
    fig, ax = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)
    image = ax.pcolormesh(
        freqs / 1e6, times * 1e3, stft_sample[0].T,
        shading="auto", cmap="jet",
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Frequency (MHz)", fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=10)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label("Normalized Power (z-score)", fontsize=9)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cpp_sample(
    cpp_sample: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    save_path: str | Path,
    *,
    title: str,
) -> None:
    """保存形状为 (1, H, W) 的归一化 CPP/FAM，不再做归一化。

    f_axis、alpha_axis 的长度分别为 W、H，单位均为 cycles/sample。
    标题由调用方提供；自动创建输出目录，绘图或保存异常向调用方传播。
    """
    plt = import_pyplot(False)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    image = ax.imshow(
        cpp_sample[0],
        origin="lower",
        aspect="auto",
        extent=[f_axis[0], f_axis[-1], alpha_axis[0], alpha_axis[-1]],
        cmap="jet",
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("f (cycles/sample)", fontsize=10)
    ax.set_ylabel("alpha (cycles/sample)", fontsize=10)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label("normalized |SCF|", fontsize=9)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    plt = import_pyplot(show)

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

    plt = import_pyplot(show)

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

    plt = import_pyplot(show)

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
