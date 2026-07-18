"""Memory-mapped indexed targets for synthetic long-tail geometry data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.geometry_explorer.latent_factors import sample_values
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    BOUNDED_ROTATION_CONDITION_ID,
    BOUNDED_ROTATION_CONDITION_IDS,
    FACTOR_COLUMNS,
    FACTOR_IDENTITY_CONDITION_IDS,
    ConditionClass,
    ConditionManifest,
    bounded_rotation_condition_spec,
    bounded_rotation_followup_condition_specs,
    build_condition_specs,
    build_factor_space,
    canonical_factor_rows,
    factor_identity_condition_specs,
)


@dataclass
class SyntheticLongTailImages:
    """Sample condition prefixes from shared, immutable uint8 image pools."""

    condition_manifest: str | Path
    normalize: str = "minus_one_one"
    dequantize: bool = False
    sampling_policy: str = "empirical"
    name: str = "synthetic_long_tail_geometry"
    dim: int = field(default=0, init=False)
    image_shape: tuple[int, ...] = field(default=(), init=False)
    class_counts: tuple[int, ...] = field(default=(), init=False)
    _manifest_path: Path = field(init=False, repr=False)
    _manifest: ConditionManifest = field(init=False, repr=False)
    _arrays: tuple[np.memmap, ...] = field(init=False, repr=False)
    _offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.normalize not in {"zero_one", "minus_one_one"}:
            raise ValueError(f"Unsupported image normalization: {self.normalize}")
        if self.sampling_policy not in {"empirical", "class_balanced"}:
            raise ValueError("Synthetic sampling_policy must be 'empirical' or 'class_balanced'.")
        self._manifest_path = Path(self.condition_manifest).expanduser().resolve()
        if not self._manifest_path.is_file():
            raise ValueError(f"Synthetic condition manifest does not exist: {self._manifest_path}")
        raw_manifest = _read_raw_manifest(self._manifest_path)
        _validate_raw_manifest(raw_manifest)
        self._manifest = _manifest_from_raw(raw_manifest, self._manifest_path)
        self.image_shape = self._manifest.image_shape
        if any(size <= 0 for size in self.image_shape):
            raise ValueError("Synthetic manifest image_shape must be a positive CHW shape.")
        class_ids = tuple(entry.class_id for entry in self._manifest.classes)
        if class_ids != tuple(range(len(class_ids))):
            raise ValueError("Synthetic manifest class IDs must be contiguous from zero.")
        expected_manifest = _expected_condition_manifest(self._manifest)
        if self._manifest_path.stem != self._manifest.condition_id:
            raise ValueError("Synthetic manifest condition_id must match its filename stem.")

        arrays = []
        counts = []
        for entry, expected_entry in zip(
            self._manifest.classes,
            expected_manifest.classes,
            strict=True,
        ):
            _validate_class_identity(entry, expected_entry)
            if entry.count <= 0 or entry.index_start < 0:
                raise ValueError(
                    "Synthetic manifest class counts must be positive with non-negative "
                    "index_start values."
                )
            path = self._resolve_expected_pool_path(
                entry.image_path,
                entry.object_id,
                entry.dimension_id,
                "image",
            )
            if not path.is_file():
                raise ValueError(f"Synthetic class image path does not exist: {path}")
            array = _load_memmap(path, "image")
            if array.dtype != np.uint8:
                raise ValueError(
                    f"Synthetic image array must have dtype uint8: {path} has {array.dtype}."
                )
            if array.ndim != 4 or tuple(array.shape[1:]) != self.image_shape:
                raise ValueError(
                    "Synthetic image array shape must be (N, "
                    f"{', '.join(str(size) for size in self.image_shape)}): {path}."
                )
            if entry.index_start + entry.count > array.shape[0]:
                raise ValueError(
                    "Synthetic manifest prefix exceeds array length: "
                    f"class {entry.class_id} requests {entry.index_start + entry.count}, "
                    f"array has {array.shape[0]}."
                )
            factor_path = self._resolve_expected_pool_path(
                entry.factor_path,
                entry.object_id,
                entry.dimension_id,
                "factor",
            )
            if not factor_path.is_file():
                raise ValueError(f"Synthetic class factor path does not exist: {factor_path}")
            factors = _load_memmap(factor_path, "factor")
            _validate_factors(factors, array, entry, factor_path)
            arrays.append(array)
            counts.append(entry.count)

        self.dim = int(np.prod(self.image_shape))
        self.class_counts = tuple(counts)
        self._arrays = tuple(arrays)
        self._offsets = np.cumsum((0, *self.class_counts), dtype=np.int64)

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        images, _ = self.sample_with_labels(n, device=device)
        return images

    def sample_with_labels(
        self,
        n: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_sample_size(n)
        batch_size = int(n)
        if self.sampling_policy == "empirical":
            global_indices = np.random.randint(0, int(self._offsets[-1]), size=batch_size)
            labels = np.searchsorted(self._offsets[1:], global_indices, side="right")
        else:
            global_indices = None
            labels = np.random.randint(0, len(self.class_counts), size=batch_size)
        output = np.empty((batch_size, *self.image_shape), dtype=np.uint8)
        for class_id, (entry, array) in enumerate(
            zip(self._manifest.classes, self._arrays, strict=True)
        ):
            mask = labels == class_id
            if not np.any(mask):
                continue
            if global_indices is None:
                local = entry.index_start + np.random.randint(
                    0,
                    entry.count,
                    size=int(np.count_nonzero(mask)),
                )
            else:
                local = global_indices[mask] - self._offsets[class_id] + entry.index_start
            output[mask] = np.asarray(array[local], dtype=np.uint8)
        images = self._normalize(torch.from_numpy(output).reshape(batch_size, -1))
        label_tensor = torch.from_numpy(labels.astype(np.int64))
        if device is not None:
            images = images.to(device)
            label_tensor = label_tensor.to(device)
        return images, label_tensor

    def all_samples_with_labels(
        self,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        image_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        for class_id, (entry, array) in enumerate(
            zip(self._manifest.classes, self._arrays, strict=True)
        ):
            indices = np.arange(entry.index_start, entry.index_start + entry.count)
            image_parts.append(np.asarray(array[indices], dtype=np.uint8))
            label_parts.append(np.full(entry.count, class_id, dtype=np.int64))
            source_parts.append((np.int64(class_id) << np.int64(48)) | indices.astype(np.int64))
        images = self._normalize(
            torch.from_numpy(np.concatenate(image_parts)).reshape(-1, self.dim)
        )
        labels = torch.from_numpy(np.concatenate(label_parts))
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        return images, labels, np.concatenate(source_parts)

    def log_prob(self, x: torch.Tensor) -> None:
        del x
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition_id": self._manifest.condition_id,
            "condition_manifest": str(self._manifest_path),
            "dim": self.dim,
            "image_shape": list(self.image_shape),
            "class_counts": list(self.class_counts),
            "normalize": self.normalize,
            "dequantize": self.dequantize,
            "sampling_policy": self.sampling_policy,
            "config_hash": self._manifest.config_hash,
        }

    def _resolve_expected_pool_path(
        self,
        path: str,
        object_id: str,
        dimension_id: str,
        kind: str,
    ) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            raise ValueError(f"Synthetic {kind} path must be relative to the manifest.")
        suffix = "images.npy" if kind == "image" else "factors.npy"
        expected = (
            self._manifest_path.parent.parent / "pools" / object_id / dimension_id / suffix
        ).resolve()
        resolved = (self._manifest_path.parent / candidate).resolve()
        if not resolved.is_file():
            raise ValueError(f"Synthetic {kind} path does not exist: {resolved}")
        if resolved != expected:
            raise ValueError(
                f"Synthetic {kind} path must identify the expected "
                f"{object_id}/{dimension_id}/{suffix} pool cell."
            )
        return resolved

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        values = images.to(dtype=torch.float32) / 255.0
        if self.dequantize:
            values = torch.clamp(values + torch.rand_like(values) / 256.0, 0.0, 1.0)
        if self.normalize == "zero_one":
            return values
        return 2.0 * values - 1.0


def _read_raw_manifest(path: Path) -> dict[str, Any]:
    def reject_non_finite(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_non_finite)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid synthetic condition manifest: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Synthetic condition manifest must be a JSON object.")
    return raw


def _validate_raw_manifest(raw: dict[str, Any]) -> None:
    _require_string(raw, "condition_id")
    _require_integer(raw, "replicate")
    if raw["replicate"] < 0:
        raise ValueError("Synthetic manifest replicate must be non-negative.")
    _require_string(raw, "geometry_mapping")
    _require_string(raw, "frequency_mapping")
    _require_string(raw, "config_hash")
    image_shape = raw.get("image_shape")
    if not isinstance(image_shape, list) or len(image_shape) != 3:
        raise ValueError("Synthetic manifest image_shape must contain three integers.")
    for index, value in enumerate(image_shape):
        _require_integer({"value": value}, "value", f"image_shape[{index}]")
    classes = raw.get("classes")
    if not isinstance(classes, list) or len(classes) != 3:
        raise ValueError("Synthetic manifest must define exactly three classes.")
    for index, entry in enumerate(classes):
        if not isinstance(entry, dict):
            raise ValueError(f"Synthetic manifest class {index} must be an object.")
        for field_name in ("class_id", "true_dimension", "count", "index_start"):
            _require_integer(entry, field_name, f"classes[{index}].{field_name}")
        for field_name in ("object_id", "dimension_id", "image_path", "factor_path"):
            _require_string(entry, field_name, f"classes[{index}].{field_name}")


def _require_integer(
    values: dict[str, Any],
    key: str,
    label: str | None = None,
) -> None:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Synthetic manifest {label or key} must be an integer.")


def _require_string(
    values: dict[str, Any],
    key: str,
    label: str | None = None,
) -> None:
    if not isinstance(values.get(key), str):
        raise ValueError(f"Synthetic manifest {label or key} must be a string.")


def _manifest_from_raw(raw: dict[str, Any], path: Path) -> ConditionManifest:
    try:
        return ConditionManifest(
            condition_id=raw["condition_id"],
            replicate=raw["replicate"],
            geometry_mapping=raw["geometry_mapping"],
            frequency_mapping=raw["frequency_mapping"],
            image_shape=tuple(raw["image_shape"]),
            classes=tuple(ConditionClass(**entry) for entry in raw["classes"]),
            config_hash=raw["config_hash"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid synthetic condition manifest: {path}") from exc


def _expected_condition_manifest(manifest: ConditionManifest) -> ConditionManifest:
    if manifest.condition_id == BOUNDED_ROTATION_CONDITION_ID:
        expected = bounded_rotation_condition_spec(manifest.replicate)
    elif manifest.condition_id in BOUNDED_ROTATION_CONDITION_IDS:
        expected = {
            condition.condition_id: condition
            for condition in bounded_rotation_followup_condition_specs(manifest.replicate)
        }.get(manifest.condition_id)
    elif manifest.condition_id in FACTOR_IDENTITY_CONDITION_IDS:
        expected = {
            condition.condition_id: condition
            for condition in factor_identity_condition_specs(manifest.replicate)
        }.get(manifest.condition_id)
    else:
        expected = {
            condition.condition_id: condition
            for condition in build_condition_specs(manifest.replicate)
        }.get(manifest.condition_id)
    if expected is None:
        raise ValueError("Synthetic manifest condition_id is not an approved study condition.")
    if (
        manifest.geometry_mapping != expected.geometry_mapping
        or manifest.frequency_mapping != expected.frequency_mapping
    ):
        raise ValueError(
            "Synthetic manifest condition_id does not match its geometry/frequency mapping."
        )
    return expected


def _validate_class_identity(
    entry: ConditionClass,
    expected: ConditionClass,
) -> None:
    for field_name in ("class_id", "object_id", "dimension_id", "true_dimension"):
        if getattr(entry, field_name) != getattr(expected, field_name):
            raise ValueError(
                f"Synthetic manifest {field_name} does not match the Task 3 condition design."
            )


def _load_memmap(path: Path, kind: str) -> np.memmap:
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to memory-map synthetic {kind} array: {path}") from exc
    if not isinstance(array, np.memmap):
        raise ValueError(f"Synthetic {kind} array must be a memory-mappable .npy file: {path}")
    return array


def _validate_factors(
    factors: np.memmap,
    images: np.memmap,
    entry: ConditionClass,
    path: Path,
) -> None:
    if factors.dtype != np.float32:
        raise ValueError(
            f"Synthetic factor array must have dtype float32: {path} has {factors.dtype}."
        )
    if factors.ndim != 2 or factors.shape[1] != len(FACTOR_COLUMNS):
        raise ValueError(
            f"Synthetic factor array shape must be (N, {len(FACTOR_COLUMNS)}): {path}."
        )
    if factors.shape[0] != images.shape[0]:
        raise ValueError("Synthetic factor array length must align with its image array.")
    if entry.index_start + entry.count > factors.shape[0]:
        raise ValueError(
            f"Synthetic manifest prefix exceeds factor array length for class {entry.class_id}."
        )
    expected_mask = _expected_factor_mask(entry.dimension_id)
    finite_mask = np.isfinite(factors)
    if not np.all(finite_mask == expected_mask):
        raise ValueError("Synthetic factor array finite/NaN pattern does not match its dimension.")


def _expected_factor_mask(dimension_id: str) -> np.ndarray:
    factor = build_factor_space(dimension_id)
    template = canonical_factor_rows(
        factor,
        sample_values(factor.sample(1, seed=0)),
    )
    return np.isfinite(template[0])


def _validate_sample_size(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("SyntheticLongTailImages.sample requires a positive non-bool integer n.")
