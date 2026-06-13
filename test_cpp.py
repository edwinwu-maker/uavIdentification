"""
Compatibility wrapper for CPP/FAM evaluation.

Usage:
  python test_cpp.py --data-dir ~/Desktop/dataset/droneRFa/cpp_h5
"""

__test__ = False

from scripts.evaluate import evaluate
from scripts.evaluate import parse_args as _parse_args


def parse_args(argv=None):
    argv = [] if argv is None else list(argv)
    if "--feature" not in argv:
        argv = ["--feature", "cpp", *argv]
    return _parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
