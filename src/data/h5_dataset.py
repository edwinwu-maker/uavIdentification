import os

import h5py
import torch
from torch.utils.data import Dataset


class H5FeatureDataset(Dataset):
    """Load one feature array and labels from pre-computed .h5 files."""

    def __init__(
        self,
        cache_dir: str,
        *,
        feature_key: str,
        dataset_name: str = "H5 feature",
    ) -> None:
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(f"{dataset_name} cache directory does not exist: {cache_dir}")

        self.cache_dir = cache_dir
        self.feature_key = feature_key
        self.index = []  # list of (file_path, row_idx, label)
        self.files = {}  # path -> opened h5py.File (lazy)

        for fname in sorted(os.listdir(cache_dir)):
            if fname.endswith(".h5"):
                path = os.path.join(cache_dir, fname)
                with h5py.File(path, "r") as f:
                    if feature_key not in f:
                        raise ValueError(f"{dataset_name} cache is missing dataset '{feature_key}': {path}")
                    feature_shape = f[feature_key].shape
                    if len(feature_shape) != 4 or feature_shape[1] != 1:
                        raise ValueError(
                            f"{dataset_name} cache must have shape (N, 1, H, W), got {feature_shape} in {path}. "
                            "Re-run precomputation; legacy dual-channel caches are not supported."
                        )
                    if "rf_channel" not in f.attrs or int(f.attrs["rf_channel"]) not in (0, 1):
                        raise ValueError(
                            f"{dataset_name} cache is missing a valid rf_channel attribute: {path}. "
                            "Re-run precomputation."
                        )
                    labels = f["labels"][:]
                for row_idx, label in enumerate(labels):
                    self.index.append((path, row_idx, int(label)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, row_idx, label = self.index[idx]
        f = self._get_file(path)
        feature = f[self.feature_key][row_idx]
        return torch.from_numpy(feature), torch.tensor(label, dtype=torch.int64)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["files"] = {}
        return state

    def _get_file(self, path: str) -> h5py.File:
        if path not in self.files:
            self.files[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024)
        return self.files[path]

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        self.files.clear()
