"""
Compatibility wrapper for CPP/FAM training.

Usage:
  python train_cpp.py --data-dir ~/Desktop/dataset/droneRFa/cpp_h5
"""

from scripts.train import parse_args as _parse_args
from scripts.train import train


def parse_args(argv=None):
    argv = [] if argv is None else list(argv)
    if "--feature" not in argv:
        argv = ["--feature", "cpp", *argv]
    return _parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
