"""Streaming exact-norm and fixed-sketch checkpoint measurements."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.checkpoints import (
    _validate_capacity_disabled,
    _validate_ordinary_objective,
)
from fm_lab.diagnostics.long_tail_geometry.gradients import resolve_probe_layers
from fm_lab.diagnostics.long_tail_geometry.manifest import ProbeBatch
from fm_lab.diagnostics.long_tail_geometry.sketch import CountSketchSpec
from fm_lab.utils.logging import write_json

_SCHEMA_VERSION = 1
_METADATA_COLUMNS = (
    "checkpoint_step",
    "probe_view",
    "target_split",
    "class_id",
    "stratum_id",
    "microbatch_id",
    "batch_size",
    "loss",
    "original_indices_sha256",
)


@dataclass(frozen=True)
class CheckpointMeasurements:
    """One checkpoint's row metadata, exact norms, and normalized sketches."""

    metadata: pd.DataFrame
    sketches: dict[str, torch.Tensor]
    exact_norms: dict[str, torch.Tensor]
    layer_shapes: dict[str, tuple[int, ...]]
    manifest_digests: dict[str, str]
    checkpoint_step: int
    checkpoint_sha256: str
    preregistration_sha256: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        metadata = self.metadata.copy(deep=True).reset_index(drop=True)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "sketches",
            {
                str(name): values.detach().float().cpu().contiguous().clone()
                for name, values in self.sketches.items()
            },
        )
        object.__setattr__(
            self,
            "exact_norms",
            {
                str(name): values.detach().float().cpu().contiguous().clone()
                for name, values in self.exact_norms.items()
            },
        )
        object.__setattr__(
            self,
            "layer_shapes",
            {
                str(name): tuple(int(value) for value in shape)
                for name, shape in self.layer_shapes.items()
            },
        )
        object.__setattr__(
            self,
            "manifest_digests",
            {str(name): str(value) for name, value in self.manifest_digests.items()},
        )
        self._validate()

    @property
    def digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(
                self._provenance(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        hasher.update(pd.util.hash_pandas_object(self.metadata, index=False).values.tobytes())
        for layer_name in sorted(self.sketches):
            hasher.update(layer_name.encode("utf-8"))
            hasher.update(self.sketches[layer_name].numpy().tobytes())
            hasher.update(self.exact_norms[layer_name].numpy().tobytes())
        return hasher.hexdigest()

    def save(self, directory: str | Path) -> Path:
        """Atomically write Parquet/NPZ data and a completion sentinel."""

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        complete_path = output / "complete.json"
        if complete_path.exists():
            existing = _read_complete(complete_path)
            if existing.get("measurement_digest") != self.digest:
                raise ValueError(
                    "Refusing to replace a completed measurement with different provenance."
                )
            return output

        layer_entries = []
        parquet_frame = self.metadata.copy()
        npz_payload: dict[str, np.ndarray] = {}
        for index, layer_name in enumerate(self.sketches):
            safe_name = f"layer_{index:02d}"
            norm_column = f"gradient_norm__{safe_name}"
            parquet_frame[norm_column] = self.exact_norms[layer_name].numpy()
            npz_payload[f"sketch__{safe_name}"] = self.sketches[layer_name].numpy()
            layer_entries.append(
                {
                    "name": layer_name,
                    "safe_name": safe_name,
                    "shape": list(self.layer_shapes[layer_name]),
                    "sketch_dimension": int(self.sketches[layer_name].shape[1]),
                    "norm_column": norm_column,
                }
            )

        parquet_path = output / "gradient_rows.parquet"
        parquet_temporary = output / "gradient_rows.parquet.tmp"
        parquet_frame.to_parquet(parquet_temporary, index=False)
        parquet_temporary.replace(parquet_path)
        npz_path = output / "gradient_sketches.npz"
        npz_temporary = output / "gradient_sketches.npz.tmp"
        with npz_temporary.open("wb") as handle:
            np.savez_compressed(handle, **npz_payload)
        npz_temporary.replace(npz_path)

        complete = {
            **self._provenance(),
            "passed": True,
            "measurement_digest": self.digest,
            "row_count": len(self.metadata),
            "probe_view_counts": {
                str(name): int(count)
                for name, count in self.metadata["probe_view"].value_counts().items()
            },
            "layers": layer_entries,
            "gradient_rows_sha256": _file_sha256(parquet_path),
            "gradient_sketches_sha256": _file_sha256(npz_path),
        }
        complete_temporary = output / "complete.json.tmp"
        write_json(complete, complete_temporary)
        complete_temporary.replace(complete_path)
        return output

    @classmethod
    def load(cls, directory: str | Path) -> CheckpointMeasurements:
        """Load a completed artifact after verifying both data-file hashes."""

        root = Path(directory)
        complete = _read_complete(root / "complete.json")
        parquet_path = root / "gradient_rows.parquet"
        npz_path = root / "gradient_sketches.npz"
        if _file_sha256(parquet_path) != complete["gradient_rows_sha256"]:
            raise ValueError("gradient_rows.parquet SHA-256 does not match complete.json.")
        if _file_sha256(npz_path) != complete["gradient_sketches_sha256"]:
            raise ValueError("gradient_sketches.npz SHA-256 does not match complete.json.")

        parquet_frame = pd.read_parquet(parquet_path)
        metadata = parquet_frame[list(_METADATA_COLUMNS)].copy()
        sketches: dict[str, torch.Tensor] = {}
        exact_norms: dict[str, torch.Tensor] = {}
        layer_shapes: dict[str, tuple[int, ...]] = {}
        with np.load(npz_path, allow_pickle=False) as payload:
            for layer in complete["layers"]:
                name = str(layer["name"])
                safe_name = str(layer["safe_name"])
                sketches[name] = torch.from_numpy(
                    payload[f"sketch__{safe_name}"].copy()
                )
                exact_norms[name] = torch.from_numpy(
                    parquet_frame[str(layer["norm_column"])].to_numpy(copy=True)
                ).float()
                layer_shapes[name] = tuple(int(value) for value in layer["shape"])
        result = cls(
            metadata=metadata,
            sketches=sketches,
            exact_norms=exact_norms,
            layer_shapes=layer_shapes,
            manifest_digests=dict(complete["manifest_digests"]),
            checkpoint_step=int(complete["checkpoint_step"]),
            checkpoint_sha256=str(complete["checkpoint_sha256"]),
            preregistration_sha256=str(complete["preregistration_sha256"]),
            schema_version=int(complete["schema_version"]),
        )
        if result.digest != complete["measurement_digest"]:
            raise ValueError("Measurement digest does not match the completed artifact.")
        return result

    def _provenance(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_sha256": self.checkpoint_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "manifest_digests": dict(sorted(self.manifest_digests.items())),
            "layer_shapes": {
                name: list(shape) for name, shape in sorted(self.layer_shapes.items())
            },
        }

    def _validate(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint-measurement schema version.")
        if tuple(self.metadata.columns) != _METADATA_COLUMNS or self.metadata.empty:
            raise ValueError("Measurement metadata columns are invalid or empty.")
        if self.checkpoint_step < 0:
            raise ValueError("Measurement checkpoint_step must be non-negative.")
        if not _is_sha256(self.checkpoint_sha256) or not _is_sha256(
            self.preregistration_sha256
        ):
            raise ValueError("Measurement provenance requires SHA-256 digests.")
        views = set(str(value) for value in self.metadata["probe_view"])
        if views != set(self.manifest_digests):
            raise ValueError("Measurement manifest digests must match every probe view.")
        if any(not _is_sha256(value) for value in self.manifest_digests.values()):
            raise ValueError("Measurement manifest digests must be SHA-256 values.")
        layer_names = set(self.sketches)
        if not layer_names or layer_names != set(self.exact_norms) or layer_names != set(
            self.layer_shapes
        ):
            raise ValueError("Measurement layers are incomplete or inconsistent.")
        row_count = len(self.metadata)
        for layer_name in layer_names:
            sketches = self.sketches[layer_name]
            norms = self.exact_norms[layer_name]
            if sketches.ndim != 2 or sketches.shape[0] != row_count:
                raise ValueError(f"Measurement sketch rows are invalid for {layer_name}.")
            if norms.shape != (row_count,) or torch.any(norms < 0):
                raise ValueError(f"Measurement exact norms are invalid for {layer_name}.")
            if not torch.isfinite(sketches).all() or not torch.isfinite(norms).all():
                raise ValueError(f"Measurement values are non-finite for {layer_name}.")
            sketch_norms = torch.linalg.vector_norm(sketches, dim=1)
            zero_rows = norms == 0
            if not torch.allclose(
                sketch_norms[zero_rows],
                torch.zeros_like(sketch_norms[zero_rows]),
                atol=1e-7,
            ) or not torch.allclose(
                sketch_norms[~zero_rows],
                torch.ones_like(sketch_norms[~zero_rows]),
                atol=1e-5,
            ):
                raise ValueError(f"Measurement sketches are not normalized for {layer_name}.")


def collect_checkpoint_measurements(
    *,
    model: nn.Module,
    objective: Any,
    path: Any,
    batches_by_view: Mapping[str, Iterable[ProbeBatch]],
    layer_names: Sequence[str],
    sketch_dim: int,
    sketch_seed: int,
    checkpoint_step: int,
    checkpoint_sha256: str,
    preregistration_sha256: str,
    manifest_digests: Mapping[str, str],
) -> CheckpointMeasurements:
    """Stream one gradient row at a time into fixed layer sketches."""

    _validate_ordinary_objective(objective)
    _validate_capacity_disabled(model)
    if sketch_dim < 2:
        raise ValueError("Measurement sketch_dim must be at least two.")
    if not batches_by_view:
        raise ValueError("Measurement collection requires probe views.")
    if set(batches_by_view) != set(manifest_digests):
        raise ValueError("Every measurement probe view requires a manifest digest.")
    layers = resolve_probe_layers(model, layer_names)
    specs: dict[str, CountSketchSpec | None] = {}
    for index, layer in enumerate(layers):
        input_dim = layer.parameter.numel()
        specs[layer.name] = (
            None
            if input_dim <= sketch_dim
            else CountSketchSpec.build(
                input_dim=input_dim,
                output_dim=sketch_dim,
                seed=int(sketch_seed) + index,
            )
        )

    metadata_rows: list[dict[str, Any]] = []
    sketches: dict[str, list[torch.Tensor]] = {layer.name: [] for layer in layers}
    exact_norms: dict[str, list[torch.Tensor]] = {layer.name: [] for layer in layers}
    was_training = model.training
    model.eval()
    try:
        for probe_view, batches in batches_by_view.items():
            target_split = _target_split(probe_view)
            for batch in batches:
                class_id = _single_value(batch.labels, "class")
                stratum_id = _single_value(batch.stratum_ids, "stratum")
                microbatch_id = _single_value(batch.microbatch_ids, "microbatch")
                loss, _ = objective(
                    model=model,
                    path=path,
                    x0=batch.x0,
                    x1=batch.x1,
                    t=batch.t,
                    compute_diagnostics=False,
                    class_labels=batch.labels,
                    original_class_labels=batch.labels,
                )
                gradients = torch.autograd.grad(
                    loss,
                    tuple(layer.parameter for layer in layers),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                for layer, gradient in zip(layers, gradients, strict=True):
                    row = gradient.detach().reshape(-1).float().cpu()
                    norm = torch.linalg.vector_norm(row)
                    if not torch.isfinite(row).all() or not torch.isfinite(norm):
                        raise ValueError(f"Invalid measurement gradient for {layer.name}.")
                    spec = specs[layer.name]
                    projected = row if spec is None else spec.apply(row[None])[0]
                    projected_norm = torch.linalg.vector_norm(projected)
                    if not torch.isfinite(projected_norm):
                        raise ValueError(f"Invalid measurement sketch for {layer.name}.")
                    if norm == 0:
                        sketches[layer.name].append(projected)
                    elif projected_norm == 0:
                        raise ValueError(f"Invalid measurement sketch for {layer.name}.")
                    else:
                        sketches[layer.name].append(projected / projected_norm)
                    exact_norms[layer.name].append(norm)
                metadata_rows.append(
                    {
                        "checkpoint_step": int(checkpoint_step),
                        "probe_view": str(probe_view),
                        "target_split": target_split,
                        "class_id": class_id,
                        "stratum_id": stratum_id,
                        "microbatch_id": microbatch_id,
                        "batch_size": int(len(batch.labels)),
                        "loss": float(loss.detach().cpu()),
                        "original_indices_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                batch.original_indices,
                                dtype=np.int64,
                            ).tobytes()
                        ).hexdigest(),
                    }
                )
    finally:
        model.train(was_training)

    if not metadata_rows:
        raise ValueError("Measurement collection produced no gradient rows.")
    return CheckpointMeasurements(
        metadata=pd.DataFrame(metadata_rows, columns=_METADATA_COLUMNS),
        sketches={name: torch.stack(rows) for name, rows in sketches.items()},
        exact_norms={name: torch.stack(rows) for name, rows in exact_norms.items()},
        layer_shapes={layer.name: layer.shape for layer in layers},
        manifest_digests=dict(manifest_digests),
        checkpoint_step=int(checkpoint_step),
        checkpoint_sha256=str(checkpoint_sha256),
        preregistration_sha256=str(preregistration_sha256),
    )


def _single_value(values: torch.Tensor | np.ndarray, label: str) -> int:
    if isinstance(values, torch.Tensor):
        unique = torch.unique(values.detach().cpu()).numpy()
    else:
        unique = np.unique(values)
    if len(unique) != 1:
        article = "one" if label == "class" else "one"
        raise ValueError(f"A measurement microbatch must contain {article} {label}.")
    return int(unique[0])


def _target_split(probe_view: str) -> str:
    if probe_view == "a" or probe_view.startswith("a_source_"):
        return "a"
    if probe_view == "b" or probe_view.startswith("b_source_"):
        return "b"
    raise ValueError(f"Unsupported measurement probe view: {probe_view}")


def _read_complete(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Measurement completion sentinel is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        complete = json.load(handle)
    if not isinstance(complete, dict) or complete.get("passed") is not True:
        raise ValueError("Measurement completion sentinel is invalid.")
    return complete


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
