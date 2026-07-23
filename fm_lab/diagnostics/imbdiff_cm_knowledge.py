"""Layerwise knowledge diagnostics for the official ImbDiff-CM expert branch."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from fm_lab.diagnostics.imbdiff_cm_probe import (
    ImbDiffCMProbeManifest,
    RestoredImbDiffCMCheckpoint,
    build_imbdiff_cm_probe_manifest,
    materialize_probe_noise,
    radial_spectral_fractions,
)
from fm_lab.evaluation.groups import frequency_ranked_groups

_SCHEMA_VERSION = 1
_FEATURE_NAMES = ("full", "low_pass", "high_pass")
_TASK_NAMES = ("fine_class", "coarse_class", "frequency_group")

# CIFAR-100 coarse labels in the canonical fine-label order used by the binary
# and Python archives. Coarse IDs follow coarse_label_names.txt.
_CIFAR100_COARSE_GROUPS = (
    (4, 30, 55, 72, 95),
    (1, 32, 67, 73, 91),
    (54, 62, 70, 82, 92),
    (9, 10, 16, 28, 61),
    (0, 51, 53, 57, 83),
    (22, 39, 40, 86, 87),
    (5, 20, 25, 84, 94),
    (6, 7, 14, 18, 24),
    (3, 42, 43, 88, 97),
    (12, 17, 37, 68, 76),
    (23, 33, 49, 60, 71),
    (15, 19, 21, 31, 38),
    (34, 63, 64, 66, 75),
    (26, 45, 77, 79, 99),
    (2, 11, 35, 46, 98),
    (27, 29, 44, 78, 93),
    (36, 50, 65, 74, 80),
    (47, 52, 56, 59, 96),
    (8, 13, 48, 58, 90),
    (41, 69, 81, 85, 89),
)


def cifar100_fine_to_coarse() -> tuple[int, ...]:
    """Return the canonical CIFAR-100 fine-to-coarse label mapping."""

    mapping = [-1] * 100
    for coarse_id, fine_ids in enumerate(_CIFAR100_COARSE_GROUPS):
        for fine_id in fine_ids:
            if mapping[fine_id] != -1:
                raise RuntimeError("CIFAR-100 fine label appears in two coarse groups.")
            mapping[fine_id] = coarse_id
    if any(value < 0 for value in mapping):
        raise RuntimeError("CIFAR-100 fine-to-coarse mapping is incomplete.")
    return tuple(mapping)


@dataclass(frozen=True)
class ImbDiffCMKnowledgeManifest:
    """Held-out examples, diffusion draws, coarse labels, and cross-fit folds."""

    probe: ImbDiffCMProbeManifest
    coarse_labels: np.ndarray
    crossfit_folds: np.ndarray
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported CM knowledge manifest version: {self.schema_version}.")
        coarse = np.asarray(self.coarse_labels, dtype=np.int64)
        folds = np.asarray(self.crossfit_folds, dtype=np.int64)
        if coarse.shape != (self.probe.num_rows,) or folds.shape != coarse.shape:
            raise ValueError("Knowledge-manifest labels and folds must match probe rows.")
        if not np.all(np.isin(folds, (0, 1))):
            raise ValueError("Knowledge-manifest crossfit_folds must contain only 0/1.")
        for class_id in np.unique(self.probe.labels):
            class_folds = folds[self.probe.labels == class_id]
            if len(np.unique(class_folds)) != 2:
                raise ValueError(f"Fine class {class_id} must occur in both cross-fit folds.")
        for name, values in (("coarse_labels", coarse), ("crossfit_folds", folds)):
            copied = np.ascontiguousarray(values).copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_digest=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "probe": self.probe.to_dict(include_digest=True),
            "coarse_labels": self.coarse_labels.tolist(),
            "crossfit_folds": self.crossfit_folds.tolist(),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> ImbDiffCMKnowledgeManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stored_digest = str(payload.pop("digest"))
        probe_payload = dict(payload.pop("probe"))
        probe_digest = str(probe_payload.pop("digest"))
        probe = ImbDiffCMProbeManifest(**probe_payload)
        if probe.digest != probe_digest:
            raise ValueError("Embedded CM probe manifest digest does not match.")
        manifest = cls(probe=probe, **payload)
        if manifest.digest != stored_digest:
            raise ValueError("CM knowledge manifest digest does not match its contents.")
        return manifest


def build_imbdiff_cm_knowledge_manifest(
    labels: np.ndarray | torch.Tensor,
    original_indices: Sequence[int | str],
    *,
    timesteps: Sequence[int],
    samples_per_class: int,
    seed: int,
    fine_to_coarse: Sequence[int] | None = None,
) -> ImbDiffCMKnowledgeManifest:
    """Build a balanced manifest with deterministic, class-stratified folds."""

    if int(samples_per_class) < 2:
        raise ValueError("Knowledge probes require at least two samples per class.")
    probe = build_imbdiff_cm_probe_manifest(
        labels,
        original_indices,
        timesteps=timesteps,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    mapping = (
        tuple(int(value) for value in fine_to_coarse)
        if fine_to_coarse is not None
        else cifar100_fine_to_coarse()
    )
    if int(probe.labels.max()) >= len(mapping):
        raise ValueError("fine_to_coarse does not cover every selected fine class.")
    coarse = np.asarray([mapping[int(value)] for value in probe.labels], dtype=np.int64)
    folds = np.empty(probe.num_rows, dtype=np.int64)
    rng = np.random.RandomState(int(seed) ^ 0x4B4E4F57)
    for class_id in np.unique(probe.labels):
        positions = np.flatnonzero(probe.labels == class_id)
        order = rng.permutation(positions)
        split = max(1, len(order) // 2)
        folds[order[:split]] = 0
        folds[order[split:]] = 1
    return ImbDiffCMKnowledgeManifest(
        probe=probe,
        coarse_labels=coarse,
        crossfit_folds=folds,
    )


@dataclass(frozen=True)
class _CaptureContext:
    manifest_rows: np.ndarray
    timestep: int


class ExpertResponseCollector:
    """Capture compact local responses from every active official LoRA convolution."""

    def __init__(
        self,
        model: nn.Module,
        *,
        sketch_dim: int,
        seed: int,
        layer_names: Sequence[str] | None = None,
    ) -> None:
        if int(sketch_dim) < 2:
            raise ValueError("sketch_dim must be at least two.")
        requested = None if layer_names is None else set(str(value) for value in layer_names)
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if module.__class__.__name__ == "Conv2d_LoRA"
            and int(getattr(module, "r", 0)) > 0
            and (requested is None or name in requested)
        ]
        if not modules:
            raise ValueError("Knowledge probe found no selected active Conv2d_LoRA layers.")
        if requested is not None:
            found = {name for name, _ in modules}
            missing = sorted(requested - found)
            if missing:
                raise ValueError(f"Unknown or inactive LoRA layers: {missing}")
        self.model = model
        self.sketch_dim = int(sketch_dim)
        self.seed = int(seed)
        self.modules = tuple(modules)
        self.layer_to_index = {name: index for index, (name, _) in enumerate(modules)}
        self._context: _CaptureContext | None = None
        self._handles: list[Any] = []
        self._projection_cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self._descriptor_rows: list[dict[str, Any]] = []
        self._metadata: dict[str, list[np.ndarray]] = {
            "layer_index": [],
            "manifest_row": [],
            "timestep": [],
        }
        self._sketches: dict[str, list[np.ndarray]] = {name: [] for name in _FEATURE_NAMES}
        self._layer_validation: dict[str, dict[str, float]] = {}

    def __enter__(self) -> ExpertResponseCollector:
        for layer_name, module in self.modules:
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._context = None

    def set_context(self, *, manifest_rows: np.ndarray, timestep: int) -> None:
        rows = np.asarray(manifest_rows, dtype=np.int64)
        if rows.ndim != 1 or len(rows) < 1:
            raise ValueError("Capture manifest_rows must be a non-empty vector.")
        self._context = _CaptureContext(manifest_rows=rows.copy(), timestep=int(timestep))

    def clear_context(self) -> None:
        self._context = None

    def finalize(self) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
        if not self._descriptor_rows:
            raise ValueError("Expert-response collector captured no responses.")
        atlas = {name: np.concatenate(parts, axis=0) for name, parts in self._metadata.items()}
        atlas.update(
            {
                f"{name}_sketch": np.concatenate(parts, axis=0)
                for name, parts in self._sketches.items()
            }
        )
        layer_rows: list[dict[str, Any]] = []
        for layer_name, module in self.modules:
            effective = _effective_expert_weight(module).detach().float()
            singular_values = torch.linalg.svdvals(effective.flatten(1))
            squared = singular_values.square()
            stable_rank = float(
                squared.sum() / squared.max().clamp_min(torch.finfo(squared.dtype).tiny)
            )
            validation = self._layer_validation[layer_name]
            layer_rows.append(
                {
                    "layer_index": self.layer_to_index[layer_name],
                    "layer_name": layer_name,
                    "in_channels": int(module.in_channels),
                    "out_channels": int(module.out_channels),
                    "kernel_height": int(module.kernel_size[0]),
                    "kernel_width": int(module.kernel_size[1]),
                    "declared_rank": int(module.r),
                    "effective_weight_rms": float(effective.square().mean().sqrt().cpu()),
                    "effective_weight_stable_rank": stable_rank,
                    **validation,
                }
            )
        return atlas, self._descriptor_rows, layer_rows

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            context = self._context
            if context is None:
                return
            capacity_enabled = True if len(inputs) < 2 else bool(inputs[1])
            if not capacity_enabled:
                return
            activations = inputs[0]
            if not isinstance(activations, torch.Tensor):
                raise TypeError("LoRA hook expected a tensor input activation.")
            if activations.shape[0] != len(context.manifest_rows):
                raise ValueError("LoRA hook batch size does not match capture context.")
            with torch.no_grad():
                self._capture_layer(
                    layer_name,
                    module,
                    activations,
                    output,
                    context,
                )

        return hook

    def _capture_layer(
        self,
        layer_name: str,
        module: nn.Module,
        activations: torch.Tensor,
        output: torch.Tensor,
        context: _CaptureContext,
    ) -> None:
        expert_weight = _effective_expert_weight(module)
        expert = F.conv2d(
            activations,
            expert_weight,
            None,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
        general = F.conv2d(
            activations,
            module.weight,
            module.bias,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
        )
        expert_float = expert.float()
        general_float = general.float()
        output_float = output.float()
        residual = output_float - general_float - expert_float
        residual_flat = residual.flatten(1)
        output_scale = output_float.flatten(1).square().mean(1).sqrt()
        reconstruction_max = residual_flat.abs().amax(1)
        reconstruction_relative = residual_flat.square().mean(1).sqrt() / output_scale.clamp_min(
            torch.finfo(output_scale.dtype).tiny
        )
        previous = self._layer_validation.get(
            layer_name,
            {
                "reconstruction_max_abs": 0.0,
                "reconstruction_max_relative_rms": 0.0,
                "captures": 0.0,
            },
        )
        previous["reconstruction_max_abs"] = max(
            previous["reconstruction_max_abs"],
            float(reconstruction_max.max().cpu()),
        )
        previous["reconstruction_max_relative_rms"] = max(
            previous["reconstruction_max_relative_rms"],
            float(reconstruction_relative.max().cpu()),
        )
        previous["captures"] += 1.0
        self._layer_validation[layer_name] = previous

        descriptors = _response_descriptors(expert_float, general_float)
        filtered = _low_high_filtered(expert_float)
        sketches = {
            "full": self._compact_sketch(layer_name, "full", expert_float),
            "low_pass": self._compact_sketch(
                layer_name,
                "low_pass",
                filtered["low_pass"],
            ),
            "high_pass": self._compact_sketch(
                layer_name,
                "high_pass",
                filtered["high_pass"],
            ),
        }
        batch_size = expert.shape[0]
        layer_index = self.layer_to_index[layer_name]
        self._metadata["layer_index"].append(np.full(batch_size, layer_index, dtype=np.int64))
        self._metadata["manifest_row"].append(context.manifest_rows.copy())
        self._metadata["timestep"].append(np.full(batch_size, context.timestep, dtype=np.int64))
        for feature_name, values in sketches.items():
            self._sketches[feature_name].append(values.detach().cpu().numpy())

        detached_descriptors = {
            name: values.detach().cpu().numpy() for name, values in descriptors.items()
        }
        for batch_index, manifest_row in enumerate(context.manifest_rows):
            row = {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "manifest_row": int(manifest_row),
                "timestep": int(context.timestep),
                "output_height": int(expert.shape[-2]),
                "output_width": int(expert.shape[-1]),
                "out_channels": int(expert.shape[1]),
                "reconstruction_max_abs": float(reconstruction_max[batch_index].cpu()),
                "reconstruction_relative_rms": float(reconstruction_relative[batch_index].cpu()),
            }
            row.update(
                {name: float(values[batch_index]) for name, values in detached_descriptors.items()}
            )
            self._descriptor_rows.append(row)

    def _compact_sketch(
        self,
        layer_name: str,
        feature_name: str,
        response: torch.Tensor,
    ) -> torch.Tensor:
        normalized = response.float()
        scale = normalized.flatten(1).square().mean(1).sqrt()
        normalized = normalized / scale[:, None, None, None].clamp_min(
            torch.finfo(normalized.dtype).tiny
        )
        channel_mean = normalized.mean(dim=(-2, -1))
        channel_rms = normalized.square().mean(dim=(-2, -1)).sqrt()
        spatial_mean = normalized.mean(dim=1, keepdim=True)
        spatial_rms = normalized.square().mean(dim=1, keepdim=True).sqrt()
        target_height = min(8, normalized.shape[-2])
        target_width = min(8, normalized.shape[-1])
        spatial_mean = F.adaptive_avg_pool2d(
            spatial_mean,
            (target_height, target_width),
        ).flatten(1)
        spatial_rms = F.adaptive_avg_pool2d(
            spatial_rms,
            (target_height, target_width),
        ).flatten(1)
        compact = torch.cat(
            (channel_mean, channel_rms, spatial_mean, spatial_rms),
            dim=1,
        )
        key = (layer_name, feature_name, compact.shape[1])
        projection = self._projection_cache.get(key)
        if projection is None or projection.device != compact.device:
            projection_seed = _stable_seed(self.seed, *key)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(projection_seed)
            projection = torch.randint(
                0,
                2,
                (compact.shape[1], self.sketch_dim),
                generator=generator,
                dtype=torch.int8,
            )
            projection = projection.to(device=compact.device, dtype=compact.dtype).mul_(2).sub_(
                1
            ) / math.sqrt(compact.shape[1])
            self._projection_cache[key] = projection
        sketch = compact @ projection
        return sketch / sketch.norm(dim=1, keepdim=True).clamp_min(torch.finfo(sketch.dtype).tiny)


def probe_imbdiff_cm_knowledge(
    restored: RestoredImbDiffCMCheckpoint,
    *,
    clean_images: torch.Tensor,
    manifest: ImbDiffCMKnowledgeManifest,
    class_counts: Sequence[int],
    batch_size: int,
    sketch_dim: int,
    seed: int,
    permutation_repeats: int = 10,
    ridge_alpha: float = 1.0,
    subspace_rank: int = 3,
    layer_names: Sequence[str] | None = None,
    compute_linear_probes: bool = True,
    compute_subspaces: bool = True,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    """Capture K1 expert responses and compute K1/K2 knowledge diagnostics."""

    if int(batch_size) < 1:
        raise ValueError("Knowledge-probe batch_size must be positive.")
    if int(permutation_repeats) < 1:
        raise ValueError("permutation_repeats must be positive.")
    if float(ridge_alpha) <= 0:
        raise ValueError("ridge_alpha must be positive.")
    if int(subspace_rank) < 1:
        raise ValueError("subspace_rank must be positive.")
    model = restored.model
    objective = restored.objective
    device = next(model.parameters()).device
    image_shape = tuple(int(value) for value in objective.image_shape)
    probe = manifest.probe
    if clean_images.shape != (probe.num_rows, *image_shape):
        raise ValueError("Knowledge-probe images do not match the manifest.")
    if tuple(int(value) for value in class_counts) != tuple(objective.class_counts):
        raise ValueError("Knowledge-probe class counts do not match the objective.")

    groups = frequency_ranked_groups(class_counts)
    group_by_class = {
        int(class_id): group_name
        for group_name, class_ids in groups.items()
        for class_id in class_ids
    }
    group_to_id = {"many": 0, "medium": 1, "few": 2}
    clean = clean_images.to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(probe.labels.copy()).to(device=device, dtype=torch.long)
    was_training = model.training
    model.eval()
    collector = ExpertResponseCollector(
        model,
        sketch_dim=sketch_dim,
        seed=seed,
        layer_names=layer_names,
    )
    try:
        with collector, torch.no_grad():
            for timestep_index, timestep in enumerate(probe.timesteps):
                if timestep >= objective.timesteps:
                    raise ValueError(f"Knowledge-probe timestep {timestep} exceeds the schedule.")
                noise = materialize_probe_noise(
                    probe.noise_seeds[timestep_index],
                    image_shape,
                ).to(device)
                for start in range(0, probe.num_rows, int(batch_size)):
                    stop = min(start + int(batch_size), probe.num_rows)
                    row_indices = np.arange(start, stop, dtype=np.int64)
                    discrete_t = torch.full(
                        (stop - start,),
                        int(timestep),
                        device=device,
                        dtype=torch.long,
                    )
                    noisy = _materialize_noisy(
                        objective,
                        clean[start:stop],
                        noise[start:stop],
                        discrete_t,
                    )
                    collector.set_context(
                        manifest_rows=row_indices,
                        timestep=int(timestep),
                    )
                    model(
                        noisy,
                        discrete_t,
                        y=labels[start:stop],
                        use_cm=True,
                    )
                    collector.clear_context()
    finally:
        model.train(was_training)

    atlas, descriptor_rows, layer_rows = collector.finalize()
    _attach_manifest_metadata(
        descriptor_rows,
        manifest=manifest,
        group_by_class=group_by_class,
    )
    atlas = _attach_atlas_metadata(
        atlas,
        manifest=manifest,
        group_by_class=group_by_class,
        group_to_id=group_to_id,
    )
    probe_rows = (
        linear_probe_rows(
            atlas,
            permutation_repeats=permutation_repeats,
            ridge_alpha=ridge_alpha,
            seed=seed,
        )
        if compute_linear_probes
        else []
    )
    if compute_subspaces:
        subspace_rows, subspace_pairs = class_subspace_rows(
            atlas,
            class_counts=class_counts,
            rank=subspace_rank,
        )
    else:
        subspace_rows, subspace_pairs = [], {}

    max_reconstruction = max(float(row["reconstruction_max_relative_rms"]) for row in layer_rows)
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint": {
            "path": str(restored.checkpoint_path),
            "step": int(restored.checkpoint_step),
            "method": restored.method,
            "weights": restored.weights,
        },
        "manifest_digest": manifest.digest,
        "dropout_mode": "disabled_eval",
        "timesteps": [int(value) for value in probe.timesteps],
        "samples_per_class": int(probe.samples_per_class),
        "num_manifest_rows": int(probe.num_rows),
        "num_response_rows": int(len(descriptor_rows)),
        "sketch_dim": int(sketch_dim),
        "feature_names": list(_FEATURE_NAMES),
        "linear_probe_tasks": list(_TASK_NAMES),
        "linear_probes_computed": bool(compute_linear_probes),
        "subspaces_computed": bool(compute_subspaces),
        "subspace_rank": int(subspace_rank),
        "max_reconstruction_relative_rms": max_reconstruction,
        "layers": layer_rows,
        "linear_probe_summary": _summarize_linear_probes(probe_rows),
        "subspace_summary": _summarize_subspaces(subspace_rows),
        "interpretation_boundary": (
            "Normalized sketches and linear/subspace probes test reproducible "
            "class, superclass, frequency-group, and band-selective structure. "
            "They do not by themselves establish causal importance; stable "
            "directions require matched interventions in K4."
        ),
    }
    return (
        summary,
        descriptor_rows,
        atlas,
        probe_rows,
        subspace_rows,
        subspace_pairs,
    )


def linear_probe_rows(
    atlas: Mapping[str, np.ndarray],
    *,
    permutation_repeats: int,
    ridge_alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Run deterministic two-way cross-fit ridge probes and null controls."""

    layer_indices = np.asarray(atlas["layer_index"], dtype=np.int64)
    timesteps = np.asarray(atlas["timestep"], dtype=np.int64)
    folds = np.asarray(atlas["crossfit_fold"], dtype=np.int64)
    task_targets = {
        "fine_class": np.asarray(atlas["class_id"], dtype=np.int64),
        "coarse_class": np.asarray(atlas["coarse_id"], dtype=np.int64),
        "frequency_group": np.asarray(atlas["frequency_group_id"], dtype=np.int64),
    }
    rows: list[dict[str, Any]] = []
    for layer_index in np.unique(layer_indices):
        for timestep in np.unique(timesteps):
            mask = (layer_indices == layer_index) & (timesteps == timestep)
            for feature_name in _FEATURE_NAMES:
                features = np.asarray(atlas[f"{feature_name}_sketch"])[mask]
                feature_folds = folds[mask]
                for task_name, targets in task_targets.items():
                    feature_targets = targets[mask]
                    classes = np.unique(feature_targets)
                    if len(classes) < 2 or not _folds_cover_classes(
                        feature_targets,
                        feature_folds,
                    ):
                        continue
                    accuracy = _crossfit_ridge_accuracy(
                        features,
                        feature_targets,
                        feature_folds,
                        alpha=ridge_alpha,
                    )
                    rng = np.random.RandomState(
                        _stable_seed(
                            seed,
                            "linear",
                            int(layer_index),
                            int(timestep),
                            feature_name,
                            task_name,
                        )
                    )
                    permutation_scores: list[float] = []
                    random_feature_scores: list[float] = []
                    for _ in range(int(permutation_repeats)):
                        permuted = feature_targets.copy()
                        for fold in (0, 1):
                            positions = np.flatnonzero(feature_folds == fold)
                            permuted[positions] = rng.permutation(permuted[positions])
                        permutation_scores.append(
                            _crossfit_ridge_accuracy(
                                features,
                                permuted,
                                feature_folds,
                                alpha=ridge_alpha,
                            )
                        )
                        random_features = rng.standard_normal(features.shape)
                        random_features *= features.std(axis=0, keepdims=True) + 1e-8
                        random_features += features.mean(axis=0, keepdims=True)
                        random_feature_scores.append(
                            _crossfit_ridge_accuracy(
                                random_features,
                                feature_targets,
                                feature_folds,
                                alpha=ridge_alpha,
                            )
                        )
                    permutation_mean = float(np.mean(permutation_scores))
                    rows.append(
                        {
                            "layer_index": int(layer_index),
                            "timestep": int(timestep),
                            "feature": feature_name,
                            "task": task_name,
                            "num_rows": int(mask.sum()),
                            "num_classes": int(len(classes)),
                            "ridge_alpha": float(ridge_alpha),
                            "accuracy": float(accuracy),
                            "chance_accuracy": float(1.0 / len(classes)),
                            "permutation_mean": permutation_mean,
                            "permutation_std": float(np.std(permutation_scores)),
                            "random_feature_mean": float(np.mean(random_feature_scores)),
                            "random_feature_std": float(np.std(random_feature_scores)),
                            "accuracy_minus_permutation": float(accuracy - permutation_mean),
                        }
                    )
    return rows


def class_subspace_rows(
    atlas: Mapping[str, np.ndarray],
    *,
    class_counts: Sequence[int],
    rank: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Build class response subspaces and compact pairwise transfer graphs."""

    layer_indices = np.asarray(atlas["layer_index"], dtype=np.int64)
    class_ids = np.asarray(atlas["class_id"], dtype=np.int64)
    coarse_ids = np.asarray(atlas["coarse_id"], dtype=np.int64)
    group_ids = np.asarray(atlas["frequency_group_id"], dtype=np.int64)
    features = np.asarray(atlas["full_sketch"], dtype=np.float64)
    aggregate_rows: list[dict[str, Any]] = []
    pair_parts: dict[str, list[np.ndarray]] = {
        "layer_index": [],
        "class_a": [],
        "class_b": [],
        "coarse_a": [],
        "coarse_b": [],
        "frequency_group_a": [],
        "frequency_group_b": [],
        "overlap": [],
        "mean_principal_angle_degrees": [],
        "frequency_log_distance": [],
    }
    for layer_index in np.unique(layer_indices):
        layer_mask = layer_indices == layer_index
        layer_classes = np.unique(class_ids[layer_mask])
        bases: dict[int, np.ndarray] = {}
        for class_id in layer_classes:
            class_features = features[layer_mask & (class_ids == class_id)]
            normalized = class_features / np.maximum(
                np.linalg.norm(class_features, axis=1, keepdims=True),
                np.finfo(np.float64).tiny,
            )
            _, _, right = np.linalg.svd(normalized, full_matrices=False)
            actual_rank = min(int(rank), len(right))
            bases[int(class_id)] = right[:actual_rank].T

        local_pairs: list[dict[str, Any]] = []
        for position, class_a in enumerate(layer_classes):
            for class_b in layer_classes[position + 1 :]:
                basis_a = bases[int(class_a)]
                basis_b = bases[int(class_b)]
                singular_values = np.linalg.svd(
                    basis_a.T @ basis_b,
                    compute_uv=False,
                )
                clipped = np.clip(singular_values, 0.0, 1.0)
                overlap = float(np.mean(clipped**2))
                angle = float(np.degrees(np.arccos(clipped)).mean())
                representative_a = np.flatnonzero(layer_mask & (class_ids == class_a))[0]
                representative_b = np.flatnonzero(layer_mask & (class_ids == class_b))[0]
                coarse_a = int(coarse_ids[representative_a])
                coarse_b = int(coarse_ids[representative_b])
                group_a = int(group_ids[representative_a])
                group_b = int(group_ids[representative_b])
                frequency_distance = float(
                    abs(
                        math.log(max(int(class_counts[int(class_a)]), 1))
                        - math.log(max(int(class_counts[int(class_b)]), 1))
                    )
                )
                local_pairs.append(
                    {
                        "overlap": overlap,
                        "angle": angle,
                        "same_superclass": coarse_a == coarse_b,
                        "frequency_relation": _frequency_relation(group_a, group_b),
                        "frequency_distance": frequency_distance,
                    }
                )
                for name, value in (
                    ("layer_index", int(layer_index)),
                    ("class_a", int(class_a)),
                    ("class_b", int(class_b)),
                    ("coarse_a", coarse_a),
                    ("coarse_b", coarse_b),
                    ("frequency_group_a", group_a),
                    ("frequency_group_b", group_b),
                    ("overlap", overlap),
                    ("mean_principal_angle_degrees", angle),
                    ("frequency_log_distance", frequency_distance),
                ):
                    dtype = np.float64 if isinstance(value, float) else np.int64
                    pair_parts[name].append(np.asarray([value], dtype=dtype))

        for semantic_relation, semantic_filter in (
            ("within_superclass", lambda row: row["same_superclass"]),
            ("across_superclass", lambda row: not row["same_superclass"]),
        ):
            selected = [row for row in local_pairs if semantic_filter(row)]
            if selected:
                aggregate_rows.append(
                    _aggregate_subspace_rows(
                        layer_index=int(layer_index),
                        relation_type="semantic",
                        relation=semantic_relation,
                        rows=selected,
                    )
                )
        for relation in sorted({row["frequency_relation"] for row in local_pairs}):
            selected = [row for row in local_pairs if row["frequency_relation"] == relation]
            aggregate_rows.append(
                _aggregate_subspace_rows(
                    layer_index=int(layer_index),
                    relation_type="frequency",
                    relation=relation,
                    rows=selected,
                )
            )
        overlaps = np.asarray([row["overlap"] for row in local_pairs])
        distances = np.asarray([row["frequency_distance"] for row in local_pairs])
        correlation_value = float(spearmanr(distances, overlaps).statistic)
        correlation = correlation_value if math.isfinite(correlation_value) else None
        aggregate_rows.append(
            {
                "layer_index": int(layer_index),
                "relation_type": "frequency_distance",
                "relation": "spearman",
                "num_pairs": int(len(local_pairs)),
                "overlap_mean": float(overlaps.mean()),
                "overlap_std": float(overlaps.std()),
                "angle_mean_degrees": float(np.mean([row["angle"] for row in local_pairs])),
                "frequency_distance_overlap_spearman": float(correlation),
            }
        )
    pairs = {
        name: np.concatenate(parts, axis=0) if parts else np.asarray([])
        for name, parts in pair_parts.items()
    }
    return aggregate_rows, pairs


def _materialize_noisy(
    objective: Any,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    signal = objective._sqrt_alpha_bars.to(  # noqa: SLF001
        device=clean.device,
        dtype=clean.dtype,
    )[timesteps]
    sigma = objective._sqrt_one_minus_alpha_bars.to(  # noqa: SLF001
        device=clean.device,
        dtype=clean.dtype,
    )[timesteps]
    expansion = (slice(None),) + (None,) * (clean.ndim - 1)
    return signal[expansion] * clean + sigma[expansion] * noise


def _effective_expert_weight(module: nn.Module) -> torch.Tensor:
    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        raise ValueError("Active LoRA module is missing factor parameters.")
    return (module.lora_B @ module.lora_A).reshape_as(module.weight)


def _response_descriptors(
    expert: torch.Tensor,
    general: torch.Tensor,
) -> dict[str, torch.Tensor]:
    expert_energy = expert.square()
    general_energy = general.square()
    expert_rms = expert_energy.flatten(1).mean(1).sqrt()
    general_rms = general_energy.flatten(1).mean(1).sqrt()
    tiny = torch.finfo(expert.dtype).tiny

    channel_energy = expert_energy.sum(dim=(-2, -1))
    channel_probability = channel_energy / channel_energy.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(tiny)
    channel_participation = (
        1.0 / channel_probability.square().sum(dim=1).clamp_min(tiny) / expert.shape[1]
    )
    top_channels = max(1, math.ceil(expert.shape[1] * 0.1))
    top_channel_fraction = torch.topk(
        channel_probability,
        k=top_channels,
        dim=1,
    ).values.sum(dim=1)

    spatial_energy = expert_energy.sum(dim=1).flatten(1)
    spatial_probability = spatial_energy / spatial_energy.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(tiny)
    spatial_participation = (
        1.0 / spatial_probability.square().sum(dim=1).clamp_min(tiny) / spatial_probability.shape[1]
    )
    spectral = radial_spectral_fractions(expert)
    return {
        "expert_rms": expert_rms,
        "general_rms": general_rms,
        "expert_to_general_rms": expert_rms / general_rms.clamp_min(tiny),
        "channel_participation_ratio": channel_participation,
        "top10_channel_energy_fraction": top_channel_fraction,
        "spatial_participation_ratio": spatial_participation,
        **{f"spectral_{name}_energy_fraction": values for name, values in spectral.items()},
    }


def _low_high_filtered(response: torch.Tensor) -> dict[str, torch.Tensor]:
    height, width = response.shape[-2:]
    fy = torch.fft.fftfreq(height, device=response.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=response.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    radius = radius / math.sqrt(0.5**2 + 0.5**2)
    frequency = torch.fft.fft2(response.float(), norm="ortho")
    low_mask = (radius < 0.25).to(frequency.dtype)
    high_mask = (radius >= 0.5).to(frequency.dtype)
    return {
        "low_pass": torch.fft.ifft2(
            frequency * low_mask,
            norm="ortho",
        ).real,
        "high_pass": torch.fft.ifft2(
            frequency * high_mask,
            norm="ortho",
        ).real,
    }


def _attach_manifest_metadata(
    descriptor_rows: list[dict[str, Any]],
    *,
    manifest: ImbDiffCMKnowledgeManifest,
    group_by_class: Mapping[int, str],
) -> None:
    probe = manifest.probe
    for row in descriptor_rows:
        manifest_row = int(row["manifest_row"])
        class_id = int(probe.labels[manifest_row])
        row.update(
            {
                "class_id": class_id,
                "coarse_id": int(manifest.coarse_labels[manifest_row]),
                "frequency_group": group_by_class[class_id],
                "crossfit_fold": int(manifest.crossfit_folds[manifest_row]),
                "dataset_position": int(probe.dataset_positions[manifest_row]),
                "original_index": int(probe.original_indices[manifest_row]),
            }
        )


def _attach_atlas_metadata(
    atlas: dict[str, np.ndarray],
    *,
    manifest: ImbDiffCMKnowledgeManifest,
    group_by_class: Mapping[int, str],
    group_to_id: Mapping[str, int],
) -> dict[str, np.ndarray]:
    rows = np.asarray(atlas["manifest_row"], dtype=np.int64)
    class_ids = manifest.probe.labels[rows]
    atlas.update(
        {
            "class_id": class_ids.astype(np.int64),
            "coarse_id": manifest.coarse_labels[rows].astype(np.int64),
            "frequency_group_id": np.asarray(
                [group_to_id[group_by_class[int(value)]] for value in class_ids],
                dtype=np.int64,
            ),
            "crossfit_fold": manifest.crossfit_folds[rows].astype(np.int64),
            "original_index": manifest.probe.original_indices[rows].astype(np.int64),
        }
    )
    return atlas


def _folds_cover_classes(targets: np.ndarray, folds: np.ndarray) -> bool:
    expected = set(int(value) for value in np.unique(targets))
    return all(
        set(int(value) for value in np.unique(targets[folds == fold])) == expected
        for fold in (0, 1)
    )


def _crossfit_ridge_accuracy(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    alpha: float,
) -> float:
    predictions = np.full_like(targets, fill_value=-1)
    classes = np.unique(targets)
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    for test_fold in (0, 1):
        train = folds != test_fold
        test = folds == test_fold
        train_features = np.asarray(features[train], dtype=np.float64)
        test_features = np.asarray(features[test], dtype=np.float64)
        mean = train_features.mean(axis=0, keepdims=True)
        scale = train_features.std(axis=0, keepdims=True)
        train_features = (train_features - mean) / np.maximum(scale, 1e-8)
        test_features = (test_features - mean) / np.maximum(scale, 1e-8)
        train_features = np.concatenate(
            (train_features, np.ones((len(train_features), 1))),
            axis=1,
        )
        test_features = np.concatenate(
            (test_features, np.ones((len(test_features), 1))),
            axis=1,
        )
        one_hot = np.zeros((train.sum(), len(classes)), dtype=np.float64)
        for row, target in enumerate(targets[train]):
            one_hot[row, class_to_column[int(target)]] = 1.0
        penalty = np.eye(train_features.shape[1], dtype=np.float64) * float(alpha)
        penalty[-1, -1] = 0.0
        system = train_features.T @ train_features + penalty
        right = train_features.T @ one_hot
        try:
            weights = np.linalg.solve(system, right)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(system) @ right
        predictions[test] = classes[np.argmax(test_features @ weights, axis=1)]
    class_accuracies = [
        float(np.mean(predictions[targets == class_id] == class_id)) for class_id in classes
    ]
    return float(np.mean(class_accuracies))


def _frequency_relation(group_a: int, group_b: int) -> str:
    names = ("many", "medium", "few")
    first, second = sorted((int(group_a), int(group_b)))
    return f"{names[first]}-{names[second]}"


def _aggregate_subspace_rows(
    *,
    layer_index: int,
    relation_type: str,
    relation: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    overlaps = np.asarray([float(row["overlap"]) for row in rows])
    angles = np.asarray([float(row["angle"]) for row in rows])
    return {
        "layer_index": int(layer_index),
        "relation_type": relation_type,
        "relation": relation,
        "num_pairs": int(len(rows)),
        "overlap_mean": float(overlaps.mean()),
        "overlap_std": float(overlaps.std()),
        "angle_mean_degrees": float(angles.mean()),
        "frequency_distance_overlap_spearman": None,
    }


def _summarize_linear_probes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for task in _TASK_NAMES:
        for feature in _FEATURE_NAMES:
            selected = [row for row in rows if row["task"] == task and row["feature"] == feature]
            if not selected:
                continue
            best = max(selected, key=lambda row: float(row["accuracy_minus_permutation"]))
            summaries.append(
                {
                    "task": task,
                    "feature": feature,
                    "best_layer_index": int(best["layer_index"]),
                    "best_timestep": int(best["timestep"]),
                    "best_accuracy": float(best["accuracy"]),
                    "best_permutation_mean": float(best["permutation_mean"]),
                    "best_excess": float(best["accuracy_minus_permutation"]),
                    "mean_excess": float(
                        np.mean([float(row["accuracy_minus_permutation"]) for row in selected])
                    ),
                }
            )
    return summaries


def _summarize_subspaces(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for layer_index in sorted({int(row["layer_index"]) for row in rows}):
        selected = [row for row in rows if int(row["layer_index"]) == layer_index]
        within = next(
            (row for row in selected if row["relation"] == "within_superclass"),
            None,
        )
        across = next(
            (row for row in selected if row["relation"] == "across_superclass"),
            None,
        )
        correlation = next(
            (row for row in selected if row["relation_type"] == "frequency_distance"),
            None,
        )
        result.append(
            {
                "layer_index": layer_index,
                "within_minus_across_superclass_overlap": (
                    None
                    if within is None or across is None
                    else float(within["overlap_mean"]) - float(across["overlap_mean"])
                ),
                "frequency_distance_overlap_spearman": (
                    None
                    if correlation is None
                    or correlation["frequency_distance_overlap_spearman"] is None
                    else float(correlation["frequency_distance_overlap_spearman"])
                ),
            }
        )
    return result


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(seed)), *(str(value) for value in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)
