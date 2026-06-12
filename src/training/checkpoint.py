from pathlib import Path

import torch

from src.utils.paths import checkpoint_dir


def default_checkpoint_path(checkpoint_name: str) -> Path:
    return checkpoint_dir() / checkpoint_name


def _resolve_checkpoint_path(checkpoint_path_or_name) -> Path:
    path = Path(checkpoint_path_or_name)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return default_checkpoint_path(path.name)


def save_checkpoint(state_dict, checkpoint_path_or_name):
    path = _resolve_checkpoint_path(checkpoint_path_or_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    return path


def load_checkpoint(checkpoint_path_or_name, map_location=None):
    path = _resolve_checkpoint_path(checkpoint_path_or_name)
    return torch.load(path, map_location=map_location)
