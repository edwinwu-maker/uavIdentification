"""Generate time-domain, frequency-domain, and FAM plots for synthetic signals.

Usage:
  python src/scripts/plot_fam_demo.py --no-show
  python src/scripts/plot_fam_demo.py --signal bpsk_noise --snr-db 5 --no-show
  python src/scripts/plot_fam_demo.py --segment-samples 131072 --fam-merge max --no-show
  python src/scripts/plot_fam_demo.py --full-fam --num-symbols 2000 --no-show
  python src/scripts/plot_fam_demo.py --device cuda --no-show
  python src/scripts/plot_fam_demo.py --device cuda:1 --gpu-pair-chunk-size 4096 --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow this script to import project modules when run from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.fam_grid import (
    DEFAULT_SEGMENT_SAMPLES,
    SUPPORTED_FAM_MERGE_MODES,
    compute_fam_grid,
    compute_fam_grid_segmented,
)
from src.utils.fam_plot import (
    plot_fam_cpp,
    plot_fam_surface_views,
    plot_signal_time_frequency,
)
from src.utils.synthetic_signal import (
    DEFAULT_SAMPLES_PER_SYMBOL,
    SUPPORTED_SIGNAL_TYPES,
    generate_signal,
)

DEFAULT_GRID_IMAGE_PATH = Path(__file__).with_name("signal_fam_grid.png")
DEFAULT_SURFACE_IMAGE_PATH = Path(__file__).with_name("signal_fam_surface.png")
DEFAULT_SURFACE_X_VIEW_IMAGE_PATH = Path(__file__).with_name(
    "signal_fam_surface_x_view.png"
)
DEFAULT_SURFACE_Y_VIEW_IMAGE_PATH = Path(__file__).with_name(
    "signal_fam_surface_y_view.png"
)
DEFAULT_SURFACE_Z_VIEW_IMAGE_PATH = Path(__file__).with_name(
    "signal_fam_surface_z_view.png"
)
DEFAULT_TIME_DOMAIN_IMAGE_PATH = Path(__file__).with_name(
    "signal_fam_time_domain.png"
)
DEFAULT_FREQUENCY_DOMAIN_IMAGE_PATH = Path(__file__).with_name(
    "signal_fam_frequency_domain.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate time-domain, frequency-domain, and FAM plots for a synthetic signal."
    )
    parser.add_argument(
        "--signal",
        choices=SUPPORTED_SIGNAL_TYPES,
        default="bpsk",
        help="Synthetic signal type to generate: bpsk, bpsk_noise, ofdm, ofdm_noise, or complex Gaussian white noise.",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        default=10.0,
        help="SNR in dB for --signal bpsk_noise or ofdm_noise.",
    )
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=200000,
        help="Number of generated symbols. BPSK uses samples-per-symbol expansion; OFDM uses one FFT block plus cyclic prefix per symbol.",
    )
    parser.add_argument(
        "--full-fam",
        action="store_true",
        help="Compute FAM on the full IQ sequence instead of segmented FAM.",
    )
    parser.add_argument(
        "--segment-samples",
        type=int,
        default=DEFAULT_SEGMENT_SAMPLES,
        help="Number of IQ samples per FAM segment.",
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
        help="How to merge segmented FAM grids: mean or max.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='FAM compute device: "cpu", "cuda", or an explicit device such as "cuda:1".',
    )
    parser.add_argument(
        "--gpu-pair-chunk-size",
        type=int,
        default=8192,
        help="Number of (k, l) channel pairs per batch for CUDA FAM computation.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save images without opening plot windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = generate_signal(
        args.signal,
        num_symbols=args.num_symbols,
        samples_per_symbol=DEFAULT_SAMPLES_PER_SYMBOL,
        snr_db=args.snr_db,
    )
    if args.full_fam:
        image, f_axis, alpha_axis = compute_fam_grid(
            x,
            device=args.device,
            pair_chunk_size=args.gpu_pair_chunk_size,
        )
    else:
        image, f_axis, alpha_axis = compute_fam_grid_segmented(
            x,
            segment_samples=args.segment_samples,
            segment_hop_samples=args.segment_hop_samples,
            merge=args.fam_merge,
            device=args.device,
            pair_chunk_size=args.gpu_pair_chunk_size,
        )

    time_path, frequency_path = plot_signal_time_frequency(
        x,
        time_path=DEFAULT_TIME_DOMAIN_IMAGE_PATH,
        frequency_path=DEFAULT_FREQUENCY_DOMAIN_IMAGE_PATH,
        show=not args.no_show,
    )
    grid_path = plot_fam_cpp(
        image,
        f_axis,
        alpha_axis,
        output_path=DEFAULT_GRID_IMAGE_PATH,
        show=not args.no_show,
    )
    surface_path, x_view_path, y_view_path, z_view_path = plot_fam_surface_views(
        image,
        f_axis,
        alpha_axis,
        surface_path=DEFAULT_SURFACE_IMAGE_PATH,
        x_view_path=DEFAULT_SURFACE_X_VIEW_IMAGE_PATH,
        y_view_path=DEFAULT_SURFACE_Y_VIEW_IMAGE_PATH,
        z_view_path=DEFAULT_SURFACE_Z_VIEW_IMAGE_PATH,
        show=not args.no_show,
    )
    print(f"Saved time-domain image to {time_path}")
    print(f"Saved frequency-domain image to {frequency_path}")
    print(f"Saved FAM grid image to {grid_path}")
    print(f"Saved FAM surface image to {surface_path}")
    print(f"Saved FAM surface x-view image to {x_view_path}")
    print(f"Saved FAM surface y-view image to {y_view_path}")
    print(f"Saved FAM surface z-view image to {z_view_path}")


if __name__ == "__main__":
    main()
