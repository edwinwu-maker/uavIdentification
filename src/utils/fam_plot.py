"""Plot helpers for FAM visualization scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_fam_cpp(
    image: np.ndarray,
    f_axis: np.ndarray,
    alpha_axis: np.ndarray,
    output_path: str | Path,
    *,
    show: bool = False,
) -> str:
    """Draw a normalized |SCF| grid as a 2D FAM heatmap."""

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
    time_path: str | Path,
    frequency_path: str | Path,
    *,
    show: bool = False,
) -> tuple[str, str]:
    """Plot the signal through the project's shared time/frequency utilities."""

    if not show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from src.utils.plot_utils import plot_iq_frequency_domain, plot_iq_time_domain

    original_show = plt.show
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
    surface_path: str | Path,
    x_view_path: str | Path,
    y_view_path: str | Path,
    z_view_path: str | Path,
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
