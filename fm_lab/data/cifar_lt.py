"""Deterministic long-tailed CIFAR target distributions."""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_CIFAR_SPECS = {
    "cifar10": {
        "num_classes": 10,
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        "archive": "cifar-10-binary.tar.gz",
        "directory": "cifar-10-batches-bin",
        "train_files": [f"data_batch_{index}.bin" for index in range(1, 6)],
        "test_files": ["test_batch.bin"],
        "record_size": 3073,
        "label_offset": 0,
        "pixel_offset": 1,
    },
    "cifar100": {
        "num_classes": 100,
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz",
        "archive": "cifar-100-binary.tar.gz",
        "directory": "cifar-100-binary",
        "train_files": ["train.bin"],
        "test_files": ["test.bin"],
        "record_size": 3074,
        "label_offset": 1,
        "pixel_offset": 2,
    },
}


@dataclass
class ImbalancedCIFARImages:
    """CIFAR-10/100 with the exponential LT split used by ImbDiff-CM.

    For class index ``c``, the retained training count is
    ``int(n_max * imbalance_factor ** (c / (num_classes - 1)))``. Class subsets
    are selected with NumPy's legacy seeded shuffle to match the reference code.
    """

    dataset: str
    root: str | Path
    train: bool = True
    download: bool = False
    imbalance_type: str = "exp"
    imbalance_factor: float = 0.01
    subset_seed: int = 0
    normalize: str = "minus_one_one"
    horizontal_flip: bool = True
    name: str = field(default="", init=False)
    dim: int = field(default=3 * 32 * 32, init=False)
    image_shape: tuple[int, int, int] = field(default=(3, 32, 32), init=False)
    num_classes: int = field(default=0, init=False)
    _raw_images: torch.Tensor | None = field(default=None, init=False, repr=False)
    _labels: torch.Tensor | None = field(default=None, init=False, repr=False)
    _selected_indices: np.ndarray | None = field(default=None, init=False, repr=False)
    _class_counts: tuple[int, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self.dataset = self.dataset.lower().replace("-", "")
        if self.dataset not in _CIFAR_SPECS:
            raise ValueError("dataset must be 'cifar10' or 'cifar100'.")
        if self.imbalance_type not in {"exp", "balanced"}:
            raise ValueError("imbalance_type must be 'exp' or 'balanced'.")
        if not 0.0 < self.imbalance_factor <= 1.0:
            raise ValueError("imbalance_factor must be in (0, 1].")
        self.name = f"{self.dataset}_lt" if self.train else self.dataset
        self.num_classes = int(_CIFAR_SPECS[self.dataset]["num_classes"])
        self._load()

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        samples, _ = self.sample_with_labels(n, device=device)
        return samples

    def sample_with_labels(
        self,
        n: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n < 1:
            raise ValueError("ImbalancedCIFARImages.sample requires n >= 1.")
        assert self._raw_images is not None
        assert self._labels is not None
        indices = torch.randint(self._raw_images.shape[0], (n,))
        images = self._raw_images[indices].float() / 255.0
        if self.train and self.horizontal_flip:
            flip = torch.rand(n) < 0.5
            images[flip] = images[flip].flip(-1)
        images = _normalize(images, self.normalize).reshape(n, -1)
        labels = self._labels[indices]
        if device is not None:
            resolved = torch.device(device)
            images = images.to(resolved)
            labels = labels.to(resolved)
        return images, labels

    def log_prob(self, x: torch.Tensor) -> None:
        del x
        return None

    def metadata(self) -> dict:
        assert self._selected_indices is not None
        return {
            "name": self.name,
            "dataset": self.dataset,
            "dim": self.dim,
            "image_shape": list(self.image_shape),
            "image_value_range": list(_image_value_range(self.normalize)),
            "num_classes": self.num_classes,
            "n_images": int(len(self.labels)),
            "train": self.train,
            "imbalance_type": self.imbalance_type,
            "imbalance_factor": self.imbalance_factor,
            "imbalance_ratio": float(1.0 / self.imbalance_factor),
            "subset_seed": self.subset_seed,
            "horizontal_flip": self.horizontal_flip,
            "class_counts": list(self.class_counts),
            "subset_sha256": hashlib.sha256(
                self._selected_indices.astype(np.int64).tobytes()
            ).hexdigest(),
        }

    @property
    def labels(self) -> torch.Tensor:
        assert self._labels is not None
        return self._labels

    @property
    def class_counts(self) -> tuple[int, ...]:
        return self._class_counts

    @property
    def selected_indices(self) -> np.ndarray:
        assert self._selected_indices is not None
        return self._selected_indices.copy()

    def _load(self) -> None:
        spec = _CIFAR_SPECS[self.dataset]
        data_dir = Path(self.root) / str(spec["directory"])
        filenames = spec["train_files"] if self.train else spec["test_files"]
        paths = [data_dir / str(filename) for filename in filenames]
        if not all(path.is_file() for path in paths):
            if not self.download:
                raise FileNotFoundError(
                    f"{self.dataset.upper()} files are missing under {data_dir}. "
                    "Set data.download: true to fetch them."
                )
            _download_and_extract(Path(self.root), self.dataset)
        images, labels = _read_binary_batches(paths, spec)
        if self.train:
            selected = _long_tail_indices(
                labels,
                num_classes=self.num_classes,
                imbalance_type=self.imbalance_type,
                imbalance_factor=self.imbalance_factor,
                seed=self.subset_seed,
            )
        else:
            selected = np.arange(len(labels), dtype=np.int64)
        selected_labels = labels[selected]
        self._selected_indices = selected
        self._raw_images = torch.from_numpy(images[selected].copy())
        self._labels = torch.from_numpy(selected_labels.astype(np.int64, copy=True))
        self._class_counts = tuple(
            int(np.sum(selected_labels == class_id))
            for class_id in range(self.num_classes)
        )


def _long_tail_indices(
    labels: np.ndarray,
    *,
    num_classes: int,
    imbalance_type: str,
    imbalance_factor: float,
    seed: int,
) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts != counts[0]):
        raise ValueError("CIFAR source split must be class-balanced before LT selection.")
    n_max = int(counts[0])
    rng = np.random.RandomState(seed)
    selected: list[np.ndarray] = []
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        if imbalance_type == "exp":
            exponent = class_id / (num_classes - 1.0)
            keep = int(n_max * imbalance_factor**exponent)
        else:
            keep = n_max
        selected.append(class_indices[:keep])
    return np.concatenate(selected).astype(np.int64, copy=False)


def _read_binary_batches(paths: list[Path], spec: dict) -> tuple[np.ndarray, np.ndarray]:
    image_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    record_size = int(spec["record_size"])
    for path in paths:
        records = np.fromfile(path, dtype=np.uint8)
        if records.size % record_size:
            raise ValueError(f"Malformed CIFAR batch: {path}")
        records = records.reshape(-1, record_size)
        labels = records[:, int(spec["label_offset"])]
        images = records[:, int(spec["pixel_offset"]):].reshape(-1, 3, 32, 32)
        image_parts.append(images)
        label_parts.append(labels)
    return np.concatenate(image_parts), np.concatenate(label_parts)


def _download_and_extract(root: Path, dataset: str) -> None:
    spec = _CIFAR_SPECS[dataset]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / str(spec["archive"])
    if not archive.exists():
        urllib.request.urlretrieve(str(spec["url"]), archive)  # noqa: S310
    with tarfile.open(archive, mode="r:gz") as handle:
        directory = str(spec["directory"])
        members = [
            member
            for member in handle.getmembers()
            if member.isfile()
            and member.name.startswith(f"{directory}/")
            and ".." not in Path(member.name).parts
        ]
        handle.extractall(root, members=members, filter="data")


def _normalize(images: torch.Tensor, mode: str) -> torch.Tensor:
    normalized = mode.lower()
    if normalized in {"zero_one", "01", "unit"}:
        return images
    if normalized in {"minus_one_one", "-1_1", "centered"}:
        return 2.0 * images - 1.0
    raise ValueError(f"Unsupported CIFAR normalization: {mode}")


def _image_value_range(mode: str) -> tuple[float, float]:
    if mode.lower() in {"zero_one", "01", "unit"}:
        return (0.0, 1.0)
    if mode.lower() in {"minus_one_one", "-1_1", "centered"}:
        return (-1.0, 1.0)
    raise ValueError(f"Unsupported CIFAR normalization: {mode}")
