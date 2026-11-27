from pathlib import Path

import numpy as np


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


def _uniform_axis_bounds(axis: np.ndarray) -> tuple[float, float]:
    """计算均匀中心坐标轴的外边界，使 imshow 像素中心与 pcolormesh 对齐。"""
    start = float(axis[0])
    end = float(axis[-1])
    if len(axis) <= 1:
        return start - 0.5, end + 0.5
    half_step = (end - start) / (2 * (len(axis) - 1))
    return start - half_step, end + half_step


def _slice_iq(sig_iq: np.ndarray, sample_start: int, sample_limit: int) -> np.ndarray:
    """将复数 IQ 展平并切片，要求起点非负、长度上限为正且切片非空。"""
    if sample_start < 0 or sample_limit <= 0:
        raise ValueError(
            f"Invalid IQ slice range: sample_start={sample_start}, sample_limit={sample_limit}"
        )
    # reshape(-1) 在数组连续时返回视图，避免对长 IQ 信号做全量拷贝。
    sig = np.asarray(sig_iq).reshape(-1)[sample_start : sample_start + sample_limit]
    if sig.size == 0:
        raise ValueError(
            f"Empty IQ slice for sample_start={sample_start}, sample_limit={sample_limit}"
        )
    return sig


class _ReusableFigurePlotter:
    """复用单个 Matplotlib Figure 批量保存 PNG 的公共生命周期基类。"""

    def __init__(self, *, dpi: int = 150) -> None:
        self._plt = import_pyplot(False)
        self._dpi = dpi
        self._created_dirs: set[Path] = set()
        self._closed = False
        self._fig = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is already closed")

    def _save_figure(self, save_path: str | Path) -> None:
        parent = Path(save_path).parent
        if parent not in self._created_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            self._created_dirs.add(parent)
        # compress_level=1 为无损快速 PNG 压缩，显著降低批量写盘编码耗时。
        self._fig.savefig(save_path, dpi=self._dpi, pil_kwargs={"compress_level": 1})

    def close(self) -> None:
        """关闭复用的 Figure，释放绘图资源。"""
        if not self._closed:
            if self._fig is not None:
                self._plt.close(self._fig)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class _SingleChannelImagePlotter(_ReusableFigurePlotter):
    """复用单个画布批量保存单通道二维特征图，避免重复建图与双重布局渲染。"""

    def __init__(
        self,
        *,
        figsize: tuple[float, float],
        extent: list[float],
        xlabel: str,
        ylabel: str,
        colorbar_label: str,
        dpi: int = 150,
        transpose: bool = False,
        margins: tuple[float, float, float, float] = (0.07, 0.96, 0.92, 0.12),
    ) -> None:
        super().__init__(dpi=dpi)
        self._transpose = transpose

        left, right, top, bottom = margins
        # 用固定边距替代 constrained_layout 与 bbox_inches="tight"，避免每帧双重布局渲染。
        self._fig, self._ax = self._plt.subplots(1, 1, figsize=figsize)
        self._fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
        self._image = self._ax.imshow(
            np.zeros((2, 2), dtype=np.float32),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="jet",
        )
        self._title = self._ax.set_title("", fontsize=12)
        self._ax.set_xlabel(xlabel, fontsize=10)
        self._ax.set_ylabel(ylabel, fontsize=10)
        self._colorbar = self._fig.colorbar(self._image, ax=self._ax, shrink=0.92, pad=0.02)
        self._colorbar.set_label(colorbar_label, fontsize=9)

    def save(
        self,
        sample: np.ndarray,
        save_path: str | Path,
        *,
        title: str,
    ) -> None:
        """更新数据、色阶与标题后保存 PNG；异常向调用方传播。"""
        self._ensure_open()

        data = sample[0].T if self._transpose else sample[0]
        self._image.set_data(data)
        # 每张图独立自适应色阶并同步刷新 colorbar 刻度，保持与单次绘图一致。
        self._image.autoscale()
        self._title.set_text(title)
        self._save_figure(save_path)


class StftSamplePlotter(_SingleChannelImagePlotter):
    """复用单个画布批量保存 z-score STFT 频谱图。

    freqs、times 须为均匀网格，单位分别为 Hz、秒，显示时换算为 MHz、ms；
    输入样本形状为 (1, H, W)，不再做归一化。推荐配合 with 语句使用。
    """

    def __init__(
        self,
        freqs: np.ndarray,
        times: np.ndarray,
        *,
        dpi: int = 150,
    ) -> None:
        x0, x1 = _uniform_axis_bounds(freqs / 1e6)
        y0, y1 = _uniform_axis_bounds(times * 1e3)
        super().__init__(
            figsize=(14, 5),
            extent=[x0, x1, y0, y1],
            xlabel="Frequency (MHz)",
            ylabel="Time (ms)",
            colorbar_label="Normalized Power (z-score)",
            dpi=dpi,
            transpose=True,
            margins=(0.06, 0.96, 0.92, 0.12),
        )


class CppSamplePlotter(_SingleChannelImagePlotter):
    """复用单个画布批量保存归一化 CPP/FAM 特征图，减少重复建图与布局开销。

    构造时固定 f_axis 与 alpha_axis（单位均为 cycles/sample）；
    输入样本形状为 (1, H, W)，不再做归一化。推荐配合 with 语句使用。
    """

    def __init__(
        self,
        f_axis: np.ndarray,
        alpha_axis: np.ndarray,
        *,
        dpi: int = 150,
    ) -> None:
        super().__init__(
            figsize=(12, 5),
            extent=[float(f_axis[0]), float(f_axis[-1]), float(alpha_axis[0]), float(alpha_axis[-1])],
            xlabel="f (cycles/sample)",
            ylabel="alpha (cycles/sample)",
            colorbar_label="normalized |SCF|",
            dpi=dpi,
            transpose=False,
        )


class IqTimeDomainPlotter(_ReusableFigurePlotter):
    """复用单个画布批量保存复数 IQ 信号的时域三联图（I 路、Q 路、幅值）。

    输入为复数信号 sig_iq = I + 1j*Q，固定使用无界面 Agg 后端；推荐配合 with 语句使用。
    """

    def __init__(
        self,
        *,
        dpi: int = 150,
        sample_start: int = 0,
        sample_limit: int = 1_000_000,
    ) -> None:
        super().__init__(dpi=dpi)
        self._sample_start = sample_start
        self._sample_limit = sample_limit
        self._cached_x: np.ndarray | None = None

        self._fig, self._axes = self._plt.subplots(3, 1, figsize=(14, 10))
        self._fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.06, hspace=0.32)
        self._suptitle = self._fig.suptitle("", fontsize=14, y=0.975)

        specs = (
            ("I(real)", "#1f77b4", None),
            ("Q(imag)", "#ff7f0e", None),
            ("abs(I+jQ)", "#d62728", "Sample Index"),
        )
        self._lines = []
        for ax, (subtitle, color, xlabel) in zip(self._axes, specs):
            (line,) = ax.plot([], [], linewidth=0.6, color=color)
            ax.set_title(subtitle, fontsize=14)
            ax.set_ylabel("Amplitude", fontsize=12)
            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=12)
            ax.grid(alpha=0.3)
            self._lines.append(line)

    def save(
        self,
        sig_iq: np.ndarray,
        save_path: str | Path,
        *,
        title: str = "",
        sample_start: int | None = None,
        sample_limit: int | None = None,
    ) -> None:
        """切片复数 IQ 信号并更新三联时域图后保存 PNG；异常向调用方传播。"""
        self._ensure_open()

        start = self._sample_start if sample_start is None else sample_start
        limit = self._sample_limit if sample_limit is None else sample_limit
        sig = _slice_iq(sig_iq, start, limit)
        n = sig.size
        if self._cached_x is None or self._cached_x.size != n:
            self._cached_x = np.arange(n)

        series = (np.real(sig), np.imag(sig), np.abs(sig))
        for ax, line, values in zip(self._axes, self._lines, series):
            line.set_data(self._cached_x, values)
            ax.relim()
            ax.autoscale_view()

        self._suptitle.set_text(title)
        self._save_figure(save_path)


class IqFrequencyDomainPlotter(_ReusableFigurePlotter):
    """复用单个画布批量保存复数 IQ 信号的 FFT 频谱图。

    输入为复数信号 sig_iq = I + 1j*Q，固定使用无界面 Agg 后端；推荐配合 with 语句使用。
    """

    DEFAULT_TITLE = "I/Q frequency domain (FFT spectrum)"

    def __init__(
        self,
        *,
        dpi: int = 150,
        sample_start: int = 0,
        sample_limit: int = 1_000_000,
    ) -> None:
        super().__init__(dpi=dpi)
        self._sample_start = sample_start
        self._sample_limit = sample_limit
        self._cached_freq: np.ndarray | None = None

        self._fig, self._ax = self._plt.subplots(1, 1, figsize=(12, 5))
        self._fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.12)
        (self._line,) = self._ax.plot([], [], linewidth=0.7, color="#9400d3")
        self._title = self._ax.set_title(self.DEFAULT_TITLE, fontsize=12)
        self._ax.set_xlabel("Normalized frequency", fontsize=10)
        self._ax.set_ylabel("dB", fontsize=10)
        self._ax.grid(alpha=0.3)

    def save(
        self,
        sig_iq: np.ndarray,
        save_path: str | Path,
        *,
        title: str | None = None,
        sample_start: int | None = None,
        sample_limit: int | None = None,
    ) -> None:
        """切片复数 IQ 信号，计算 FFT 幅度谱（dB）并保存 PNG；异常向调用方传播。"""
        self._ensure_open()

        start = self._sample_start if sample_start is None else sample_start
        limit = self._sample_limit if sample_limit is None else sample_limit
        sig = _slice_iq(sig_iq, start, limit)
        n = sig.size
        if self._cached_freq is None or self._cached_freq.size != n:
            self._cached_freq = np.fft.fftshift(np.fft.fftfreq(n, 1.0))

        fft_sig = np.fft.fftshift(np.fft.fft(sig))
        mag = 20.0 * np.log10(np.abs(fft_sig) + 1e-10)

        self._line.set_data(self._cached_freq, mag)
        self._ax.relim()
        self._ax.autoscale_view()
        self._title.set_text(self.DEFAULT_TITLE if title is None else title)
        self._save_figure(save_path)
