import argparse
import sys
from pathlib import Path

import numpy as np

# Allow this script to import project modules when run from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.fam import fam_scf_points

SUPPORTED_SIGNAL_TYPES = ("bpsk", "noise")

DEFAULT_GRID_IMAGE_PATH = Path(__file__).with_name("signal_fam_grid.png")
DEFAULT_SURFACE_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface.png")
DEFAULT_SURFACE_X_VIEW_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface_x_view.png")
DEFAULT_SURFACE_Y_VIEW_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface_y_view.png")
DEFAULT_SURFACE_Z_VIEW_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface_z_view.png")
DEFAULT_TIME_DOMAIN_IMAGE_PATH = Path(__file__).with_name("signal_fam_time_domain.png")
DEFAULT_FREQUENCY_DOMAIN_IMAGE_PATH = Path(__file__).with_name("signal_fam_frequency_domain.png")

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


def generate_signal(
    signal_type: str,
    *,
    num_symbols: int = 10000,
    samples_per_symbol: int = 10,
    seed: int = 7,
) -> np.ndarray:
    """Generate the selected IQ signal for FAM visualization."""

    if signal_type == "bpsk":
        return generate_bpsk(
            num_symbols=num_symbols,
            samples_per_symbol=samples_per_symbol,
            seed=seed,
        )
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

def compute_fam_grid(
    x: np.ndarray,
    samples_per_symbol: int,
    *,
    f_bins: int = 257,
    alpha_bins: int = 513,
    f_range: tuple[float, float] = (-0.5, 0.5),
    alpha_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FAM once and return the reusable normalized |SCF| image grid."""

    result = fam_scf_points(
        x,
        nfft=64,
        hop=64,
        window="hann",
        keep_principal_domain=True,
    )
    f_idx = np.floor(
        (result.f - f_range[0]) / (f_range[1] - f_range[0]) * (f_bins - 1)
    ).astype(int)
    a_idx = np.floor(
        (result.alpha - alpha_range[0])
        / (alpha_range[1] - alpha_range[0])
        * (alpha_bins - 1)
    ).astype(int)

    valid = (
        (f_idx >= 0)
        & (f_idx < f_bins)
        & (a_idx >= 0)
        & (a_idx < alpha_bins)
    )

    mag = np.abs(result.value)
    image_sum = np.zeros((alpha_bins, f_bins), dtype=float)
    image_count = np.zeros((alpha_bins, f_bins), dtype=int)
    np.add.at(image_sum, (a_idx[valid], f_idx[valid]), mag[valid])
    np.add.at(image_count, (a_idx[valid], f_idx[valid]), 1)
    image = np.zeros((alpha_bins, f_bins), dtype=float)
    np.divide(image_sum, image_count, out=image, where=image_count > 0)

    if image.max() > 0:
        image = image / image.max()

    f_axis = np.linspace(f_range[0], f_range[1], f_bins)
    alpha_axis = np.linspace(alpha_range[0], alpha_range[1], alpha_bins)
    return image, f_axis, alpha_axis


def plot_fam_cpp(
    image: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    output_path: str | Path = DEFAULT_GRID_IMAGE_PATH,
    *,
    show: bool = False,
) -> str:
    """"cpp: cyclic-paw-print"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=[f_axis[0], f_axis[-1], alpha_axis[0], alpha_axis[-1]],
        cmap="jet",
    )
    ax.set_xlabel("f (cycles/sample)")
    ax.set_ylabel("alpha (cycles/sample)")
    ax.set_title("Signal FAM |SCF|")
    fig.colorbar(im, ax=ax, label="normalized |SCF|")
    fig.tight_layout()
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
        help="Synthetic signal type to generate: bpsk or complex Gaussian white noise.",
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
    x = generate_signal(args.signal, samples_per_symbol=samples_per_symbol)
    image, f_axis, alpha_axis = compute_fam_grid(x, samples_per_symbol)

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
