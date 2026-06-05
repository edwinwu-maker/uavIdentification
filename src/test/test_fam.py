import argparse
import sys
from pathlib import Path

import numpy as np

# Allow this script to import project modules when run from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.fam import fam_scf_points

SUPPORTED_SIGNAL_TYPES = ("bpsk", "bpsk_noise", "noise")
SUPPORTED_FAM_MERGE_MODES = ("mean", "max", "sum")
FAM_NFFT = 64
FAM_HOP = 64

DEFAULT_GRID_IMAGE_PATH = Path(__file__).with_name("signal_fam_grid.png")
DEFAULT_SURFACE_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface.png")
DEFAULT_SURFACE_X_VIEW_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface_x_view.png")
DEFAULT_SURFACE_Y_VIEW_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface_y_view.png")
DEFAULT_SURFACE_Z_VIEW_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface_z_view.png")
DEFAULT_TIME_DOMAIN_IMAGE_PATH = Path(__file__).with_name("signal_fam_time_domain.png")
DEFAULT_FREQUENCY_DOMAIN_IMAGE_PATH = Path(__file__).with_name("signal_fam_frequency_domain.png")
DEFAULT_SEGMENT_SAMPLES = 262144

def generate_bpsk(
    *,
    num_symbols: int = 1024,
    samples_per_symbol: int = 10,
    rolloff: float = 0.35,
    span_symbols: int = 8,
    seed: int = 7,
) -> np.ndarray:
    """Generate 10x oversampled BPSK IQ after root-raised-cosine pulse shaping."""

    rng = np.random.default_rng(seed)
    symbols = 2 * rng.integers(0, 2, size=num_symbols) - 1

    # Upsample by zero insertion, then apply the transmit pulse-shaping filter.
    upsampled = np.zeros(num_symbols * samples_per_symbol, dtype=float)
    upsampled[::samples_per_symbol] = symbols
    taps = root_raised_cosine(
        samples_per_symbol=samples_per_symbol,
        rolloff=rolloff,
        span_symbols=span_symbols,
    )
    shaped = np.convolve(upsampled, taps, mode="same")
    return shaped.astype(np.complex128)


def generate_awgn(
    *,
    num_samples: int,
    power: float = 1.0,
    seed: int = 7,
) -> np.ndarray:
    """Generate complex additive white Gaussian noise with the requested power."""

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(power / 2.0)
    noise = sigma * (
        rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)
    )
    return noise.astype(np.complex128)


def add_awgn_for_snr(
    x: np.ndarray,
    *,
    snr_db: float,
    seed: int = 7,
) -> np.ndarray:
    """Add complex AWGN to make the output match the requested SNR in dB."""

    signal_power = float(np.mean(np.abs(x) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = generate_awgn(num_samples=len(x), power=noise_power, seed=seed)
    return x + noise


def generate_signal(
    signal_type: str,
    *,
    num_symbols: int = 200000,
    samples_per_symbol: int = 10,
    snr_db: float = 10.0,
    seed: int = 7,
) -> np.ndarray:
    """Generate the selected IQ signal for FAM visualization."""

    if signal_type == "bpsk":
        return generate_bpsk(
            num_symbols=num_symbols,
            samples_per_symbol=samples_per_symbol,
            seed=seed,
        )
    if signal_type == "bpsk_noise":
        bpsk = generate_bpsk(
            num_symbols=num_symbols,
            samples_per_symbol=samples_per_symbol,
            seed=seed,
        )
        return add_awgn_for_snr(bpsk, snr_db=snr_db, seed=seed + 1)
    if signal_type == "noise":
        return generate_awgn(
            num_samples=num_symbols * samples_per_symbol,
            seed=seed,
        )
    raise ValueError(f"unsupported signal type: {signal_type}")


def root_raised_cosine(
    *,
    samples_per_symbol: int,
    rolloff: float,
    span_symbols: int,
) -> np.ndarray:
    """Return energy-normalized root-raised-cosine FIR taps."""

    half_len = span_symbols * samples_per_symbol // 2
    t = np.arange(-half_len, half_len + 1, dtype=float) / samples_per_symbol
    taps = np.empty_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 + rolloff * (4.0 / np.pi - 1.0)
        elif rolloff > 0 and np.isclose(abs(ti), 1.0 / (4.0 * rolloff)):
            taps[i] = (
                rolloff
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * rolloff))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * rolloff))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1.0 - rolloff))
                + 4.0 * rolloff * ti * np.cos(np.pi * ti * (1.0 + rolloff))
            )
            denominator = np.pi * ti * (1.0 - (4.0 * rolloff * ti) ** 2)
            taps[i] = numerator / denominator

    return taps / np.sqrt(np.sum(taps**2))


def points_to_grid(
    f: np.ndarray,
    alpha: np.ndarray,
    value: np.ndarray,
    *,
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = (-0.5, 0.5),
    alpha_range: tuple[float, float] = (-1.0, 1.0),
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 FAM 点估计聚合成二维 |SCF| 网格。"""

    # Step 1: 将每个 FAM 点的 f 坐标映射到二维网格的列索引。
    f_idx = np.floor(
        (f - f_range[0]) / (f_range[1] - f_range[0]) * (f_bins - 1)
    ).astype(int)

    # Step 1: 将每个 FAM 点的 alpha 坐标映射到二维网格的行索引。
    a_idx = np.floor(
        (alpha - alpha_range[0])
        / (alpha_range[1] - alpha_range[0])
        * (alpha_bins - 1)
    ).astype(int)

    # Step 2: 只保留落在 f/alpha 显示范围内的有效网格点。
    valid = (
        (f_idx >= 0)
        & (f_idx < f_bins)
        & (a_idx >= 0)
        & (a_idx < alpha_bins)
    )
    mag = np.abs(value)
    image_sum = np.zeros((alpha_bins, f_bins), dtype=float)
    image_count = np.zeros((alpha_bins, f_bins), dtype=int)

    # Step 3: 同一网格可能落入多个 FAM 点，先累加幅度并统计点数。
    np.add.at(image_sum, (a_idx[valid], f_idx[valid]), mag[valid])
    np.add.at(image_count, (a_idx[valid], f_idx[valid]), 1)

    # Step 3: 对每个有效网格内的多个点取幅度平均值。
    image = np.zeros((alpha_bins, f_bins), dtype=float)
    np.divide(image_sum, image_count, out=image, where=image_count > 0)

    # Step 4: 按需归一化。分段融合时关闭这里的归一化，等融合后统一处理。
    if normalize and image.max() > 0:
        image = image / image.max()

    # Step 5: 生成与 image 行列对应的 f 轴和 alpha 轴坐标。
    f_axis = np.linspace(f_range[0], f_range[1], f_bins)
    alpha_axis = np.linspace(alpha_range[0], alpha_range[1], alpha_bins)
    return image, f_axis, alpha_axis


def compute_fam_grid(
    x: np.ndarray,
    *,
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = (-0.5, 0.5),
    alpha_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 FAM 点估计，并聚合为可绘制的归一化 |SCF| 二维网格。

    输入参数：
        x:
            一维复数 IQ 序列，是 FAM 算法的待分析信号。
        f_bins:
            频率轴 f 的网格数量，对应输出图像的列数。
        alpha_bins:
            循环频率轴 alpha 的网格数量，对应输出图像的行数。
        f_range:
            频率轴 f 的显示范围，单位为 cycles/sample。
        alpha_range:
            循环频率轴 alpha 的显示范围，单位为 cycles/sample。

    输出参数：
        image:
            shape 为 (alpha_bins, f_bins) 的归一化 |SCF| 矩阵。
        f_axis:
            与 image 列方向对应的频率坐标。
        alpha_axis:
            与 image 行方向对应的循环频率坐标。

    """

    # Step 1: 调用 FAM 算法，得到稀疏的 (f, alpha, value) 点估计。
    result = fam_scf_points(
        x,
        nfft=FAM_NFFT,
        hop=FAM_HOP,
        window="hann",
        keep_principal_domain=True,
    )
    return points_to_grid(
        result.f,
        result.alpha,
        result.value,
        f_bins=f_bins,
        alpha_bins=alpha_bins,
        f_range=f_range,
        alpha_range=alpha_range,
        normalize=True,
    )


def compute_fam_grid_segmented(
    x: np.ndarray,
    *,
    segment_samples: int = DEFAULT_SEGMENT_SAMPLES,
    segment_hop_samples: int | None = None,
    merge: str = "mean",
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = (-0.5, 0.5),
    alpha_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对长 IQ 序列做分段 FAM，并在二维 |SCF| 网格上融合。"""

    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    if segment_hop_samples is None:
        segment_hop_samples = segment_samples
    if segment_hop_samples <= 0:
        raise ValueError("segment_hop_samples must be positive")
    if merge not in SUPPORTED_FAM_MERGE_MODES:
        raise ValueError(f"unsupported merge mode: {merge}")

    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional complex IQ sequence")

    merged = np.zeros((alpha_bins, f_bins), dtype=float)
    segment_count = 0

    if len(x) <= segment_samples:
        segment_starts = [0]
    else:
        last_full_start = len(x) - segment_samples
        segment_starts = range(0, last_full_start + 1, segment_hop_samples)

    # Step 1: 按固定长度和步长遍历 IQ；跳过不足完整长度的尾段，避免补零尾段引入边界泄漏。
    for start in segment_starts:
        segment = x[start : start + segment_samples]
        if len(segment) == 0:
            continue

        # Step 2: 每个片段独立做 FAM 点估计，控制单次第二阶段 FFT 的长度。
        result = fam_scf_points(
            segment,
            nfft=FAM_NFFT,
            hop=FAM_HOP,
            window="hann",
            keep_principal_domain=True,
        )

        # Step 3: 将当前片段的 FAM 点映射到固定 f/alpha 网格。这里不归一化，
        # 等所有片段融合完成后再统一归一化，保留片段之间的强弱差异。
        image, f_axis, alpha_axis = points_to_grid(
            result.f,
            result.alpha,
            result.value,
            f_bins=f_bins,
            alpha_bins=alpha_bins,
            f_range=f_range,
            alpha_range=alpha_range,
            normalize=False,
        )

        # Step 4: 在二维网格上融合每个片段的 |SCF| 结果。
        if merge == "max":
            merged = np.maximum(merged, image)
        else:
            merged += image
        segment_count += 1

    if segment_count == 0:
        raise ValueError("input signal is empty")

    # Step 5: mean 模式需要除以片段数；sum/max 模式直接使用融合结果。
    if merge == "mean":
        merged = merged / segment_count

    # Step 6: 对最终融合图做一次全局归一化，得到用于显示的 |SCF| 图。
    if merged.max() > 0:
        merged = merged / merged.max()

    return merged, f_axis, alpha_axis


def plot_fam_cpp(
    image: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    output_path: str | Path = DEFAULT_GRID_IMAGE_PATH,
    *,
    show: bool = False,
) -> str:
    """将归一化 |SCF| 网格绘制为二维 FAM 热力图。

    输入参数：
        image:
            compute_fam_grid 输出的归一化 |SCF| 矩阵。行方向对应 alpha，
            列方向对应 f。
        f_axis:
            频率轴坐标，单位为 cycles/sample。
        alpha_axis:
            循环频率轴坐标，单位为 cycles/sample。
        output_path:
            热力图保存路径。
        show:
            是否在保存图片后弹出 matplotlib 显示窗口。

    输出参数：
        str:
            已保存热力图文件的路径字符串。

    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    # Step 1: 使用 imshow 按 f/alpha 坐标范围绘制归一化 |SCF| 矩阵。
    im = ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=[f_axis[0], f_axis[-1], alpha_axis[0], alpha_axis[-1]],
        cmap="jet",
    )

    # Step 2: 添加坐标轴标签、标题和归一化 |SCF| 色条。
    ax.set_xlabel("f (cycles/sample)")
    ax.set_ylabel("alpha (cycles/sample)")
    ax.set_title("Signal FAM |SCF|")
    fig.colorbar(im, ax=ax, label="normalized |SCF|")
    fig.tight_layout()

    # Step 3: 保存图片，并在 show=True 时展示窗口。
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return str(output_path)


def plot_signal_time_frequency(
    x: np.ndarray,
    time_path: str | Path = DEFAULT_TIME_DOMAIN_IMAGE_PATH,
    frequency_path: str | Path = DEFAULT_FREQUENCY_DOMAIN_IMAGE_PATH,
    *,
    show: bool = False,
) -> tuple[str, str]:
    """Plot the filtered BPSK signal through the project's shared plot utilities."""

    if not show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from src.utils.plot_utils import plot_iq_frequency_domain, plot_iq_time_domain

    original_show = plt.show
    # plot_utils calls plt.show() internally. Suppress that call so figures are
    # saved before any interactive window can clear or close them.
    plt.show = lambda *args, **kwargs: None

    try:
        plot_iq_time_domain(x, sample_start=0, sample_limit=400)
        time_fig = plt.gcf()
        time_fig.savefig(time_path, dpi=150, bbox_inches="tight")
        if show:
            original_show()
        plt.close(time_fig)

        plot_iq_frequency_domain(x, sample_start=0, sample_limit=len(x))
        frequency_fig = plt.gcf()
        frequency_fig.savefig(frequency_path, dpi=150, bbox_inches="tight")
        if show:
            original_show()
        plt.close(frequency_fig)
    finally:
        plt.show = original_show

    return str(time_path), str(frequency_path)

def plot_fam_surface_views(
    image: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    surface_path: str | Path = DEFAULT_SURFACE_IMAGE_PATH,
    x_view_path: str | Path = DEFAULT_SURFACE_X_VIEW_IMAGE_PATH,
    y_view_path: str | Path = DEFAULT_SURFACE_Y_VIEW_IMAGE_PATH,
    z_view_path: str | Path = DEFAULT_SURFACE_Z_VIEW_IMAGE_PATH,
    *,
    show: bool = False,
) -> tuple[str, str, str, str]:
    """Draw 3D FAM |SCF| surface images from the default and axis views."""

    import matplotlib.pyplot as plt

    f_grid, alpha_grid = np.meshgrid(f_axis, alpha_axis)

    views = [
        (surface_path, 28, -135, "Signal FAM |SCF| Surface"),
        (x_view_path, 0, 0, "Signal FAM |SCF| Surface - X View"),
        (y_view_path, 0, 90, "Signal FAM |SCF| Surface - Y View"),
        (z_view_path, 90, -90, "Signal FAM |SCF| Surface - Z View"),
    ]

    saved_paths = []
    for output_path, elev, azim, title in views:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            f_grid,
            alpha_grid,
            image,
            cmap="jet",
            linewidth=0,
            antialiased=True,
            rcount=160,
            ccount=120,
        )
        ax.set_xlabel("f (cycles/sample)")
        ax.set_ylabel("alpha (cycles/sample)")
        ax.set_zlabel("normalized |SCF|")
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        saved_paths.append(str(output_path))

    return tuple(saved_paths)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate time-domain, frequency-domain, and FAM plots for a synthetic signal."
    )
    parser.add_argument(
        "--signal",
        choices=SUPPORTED_SIGNAL_TYPES,
        default="bpsk",
        help="Synthetic signal type to generate: bpsk, bpsk_noise, or complex Gaussian white noise.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=10.0,
        help="SNR in dB for --signal bpsk_noise.",
    )
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=200000,
        help="Number of generated symbols. With 10x oversampling, 1000000 symbols produce 10000000 IQ samples.",
    )
    parser.add_argument(
        "--segmented-fam",
        action="store_true",
        help="Compute FAM by segmenting the IQ sequence and merging |SCF| grids.",
    )
    parser.add_argument(
        "--segment-samples",
        type=int,
        default=DEFAULT_SEGMENT_SAMPLES,
        help="Number of IQ samples per FAM segment when --segmented-fam is used.",
    )
    parser.add_argument(
        "--segment-hop-samples",
        type=int,
        default=None,
        help="Hop size between FAM segments. Defaults to --segment-samples.",
    )
    parser.add_argument(
        "--fam-merge",
        choices=SUPPORTED_FAM_MERGE_MODES,
        default="mean",
        help="How to merge segmented FAM grids: mean, max, or sum.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save images without opening plot windows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    samples_per_symbol = 10
    x = generate_signal(
        args.signal,
        num_symbols=args.num_symbols,
        samples_per_symbol=samples_per_symbol,
        snr_db=args.snr_db,
    )
    if args.segmented_fam:
        image, f_axis, alpha_axis = compute_fam_grid_segmented(
            x,
            segment_samples=args.segment_samples,
            segment_hop_samples=args.segment_hop_samples,
            merge=args.fam_merge,
        )
    else:
        image, f_axis, alpha_axis = compute_fam_grid(x)

    time_path, frequency_path = plot_signal_time_frequency(x, show=not args.no_show)
    grid_path = plot_fam_cpp(image, f_axis, alpha_axis, show=not args.no_show)
    surface_path, x_view_path, y_view_path, z_view_path = plot_fam_surface_views(
        image,
        f_axis,
        alpha_axis,
        show=not args.no_show,
    )
    print(f"Saved time-domain image to {time_path}")
    print(f"Saved frequency-domain image to {frequency_path}")
    print(f"Saved FAM grid image to {grid_path}")
    print(f"Saved FAM surface image to {surface_path}")
    print(f"Saved FAM surface x-view image to {x_view_path}")
    print(f"Saved FAM surface y-view image to {y_view_path}")
    print(f"Saved FAM surface z-view image to {z_view_path}")
