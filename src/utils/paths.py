from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_dir() -> Path:
    return project_root() / "outputs"


def log_dir() -> Path:
    return output_dir() / "logs"


def model_dir() -> Path:
    return output_dir() / "models"


def figures_dir() -> Path:
    return output_dir() / "figures"


def metrics_dir() -> Path:
    return output_dir() / "metrics"
