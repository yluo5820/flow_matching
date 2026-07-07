"""CIFAR-10 background-removal ablation datasets and metrics."""

from __future__ import annotations

import itertools
import logging
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

from fm_lab.geometry_explorer.photometric import _relative_to_project
from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import (
    CIFAR10_LABELS,
    DatasetBundle,
    load_dataset,
)
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.utils.config import ConfigError, save_config
from fm_lab.utils.logging import write_json

LOGGER = logging.getLogger("fm_lab.geometry_explorer.cifar_background")

CIFAR_BACKGROUND_COMPONENTS = ("raw", "fg", "bg", "mask", "crop")
CIFAR_BACKGROUND_PIPELINES = ("rembg", "grounded_sam")
DEFAULT_PROMPTS = {
    "airplane": "airplane",
    "automobile": "car",
    "bird": "bird",
    "cat": "cat",
    "deer": "deer",
    "dog": "dog",
    "frog": "frog",
    "horse": "horse",
    "ship": "ship",
    "truck": "truck",
}


@dataclass(frozen=True)
class CifarBackgroundConfig:
    dataset_root: str = "data/cifar10"
    output_root: str = "data/cifar10/background_ablation"
    split: str = "train"
    max_samples: int | None = None
    sample_seed: int = 42
    sample_strategy: str = "random"
    pipelines: tuple[str, ...] = ("rembg",)
    upsample_size: int = 256
    fill: str = "mean"
    min_mask_area: float = 0.03
    max_mask_area: float = 0.90
    max_components: int = 8
    min_largest_component_ratio: float = 0.45
    metrics_max_samples: int = 5_000
    metrics_pairs: int = 100_000
    overwrite: bool = False
    rembg_model: str = "u2netp"
    device: str = "cpu"
    grounded_sam_config: str = ""
    grounded_sam_checkpoint: str = ""
    sam_checkpoint: str = ""
    sam_model_type: str = "vit_b"
    box_threshold: float = 0.20
    text_threshold: float = 0.20


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    confidence: float = 1.0
    failure_reason: str = ""


Segmenter = Callable[[np.ndarray, pd.Series, int], MaskResult]


def build_cifar_background_ablation(
    config: CifarBackgroundConfig,
    *,
    project_root: str | Path | None = None,
    segmenters: dict[str, Segmenter] | None = None,
) -> dict[str, Any]:
    """Generate CIFAR-10 foreground/background/crop/mask ablation variants."""

    root = Path(project_root or Path.cwd()).expanduser().resolve()
    _validate_config(config, segmenters=segmenters)
    source = _load_source(config, project_root=root)
    positions = _select_positions(
        source.metadata,
        maximum=config.max_samples,
        strategy=config.sample_strategy,
        seed=config.sample_seed,
    )
    metadata = source.metadata.iloc[positions].reset_index(drop=True).copy()
    vectors = np.asarray(source.vectors[positions], dtype=np.float32)
    images = np.clip(vectors.reshape(-1, 32, 32, 3) / 255.0, 0.0, 1.0)
    output_base = root / config.output_root
    output_base.mkdir(parents=True, exist_ok=True)

    datasets: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    masks_by_pipeline: dict[str, np.ndarray] = {}
    mask_metadata_by_pipeline: dict[str, pd.DataFrame] = {}
    segmenters = segmenters or {}

    for pipeline in config.pipelines:
        pipeline_dir = output_base / pipeline / config.split
        if pipeline_dir.exists():
            if not config.overwrite:
                raise ConfigError(f"Output already exists: {pipeline_dir}. Use --overwrite.")
            shutil.rmtree(pipeline_dir)
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        segmenter = segmenters.get(pipeline) or _make_segmenter(pipeline, config)
        masks, mask_metadata = _segment_masks(
            images,
            metadata,
            segmenter,
            pipeline=pipeline,
            config=config,
        )
        masks_by_pipeline[pipeline] = masks
        mask_metadata_by_pipeline[pipeline] = mask_metadata
        np.save(pipeline_dir / "masks.npy", masks.astype(np.float32))
        write_parquet(mask_metadata, pipeline_dir / "mask_metadata.parquet")
        _write_mask_preview(masks, pipeline_dir / "mask_preview_grid.png")

        components = _component_images(
            images,
            masks,
            mask_metadata,
            fill_color=_fill_color(images, config.fill),
        )
        datasets.extend(
            _write_component_datasets(
                config,
                pipeline=pipeline,
                output_dir=pipeline_dir,
                components=components,
                source_metadata=metadata,
                mask_metadata=mask_metadata,
                project_root=root,
            )
        )
        metrics_rows.extend(
            _pipeline_metric_rows(
                config,
                pipeline=pipeline,
                components=components,
                metadata=mask_metadata,
            )
        )

    metrics_rows.extend(
        _agreement_metric_rows(
            masks_by_pipeline,
            mask_metadata_by_pipeline,
            config=config,
        )
    )
    metrics_path = output_base / "metrics" / "summary.csv"
    if metrics_rows:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(metrics_rows).to_csv(metrics_path, index=False)
    write_json(
        {
            "dataset": "cifar10",
            "source_dataset_root": config.dataset_root,
            "split": config.split,
            "rows": int(len(images)),
            "pipelines": list(config.pipelines),
            "components": list(CIFAR_BACKGROUND_COMPONENTS),
            "output_root": str(output_base),
            "datasets": [
                {
                    "variant_id": row["variant_id"],
                    "rows": row["rows"],
                    "dataset_config_path": str(row["dataset_config_path"]),
                }
                for row in datasets
            ],
            "metrics_path": str(metrics_path) if metrics_rows else None,
        },
        output_base / "manifest.json",
    )
    return {
        "output_dir": output_base,
        "datasets": datasets,
        "metrics_path": metrics_path if metrics_rows else None,
    }


def analyze_mask(mask: np.ndarray, *, config: CifarBackgroundConfig) -> dict[str, Any]:
    """Compute compact quality metadata for one soft mask."""

    soft = np.clip(np.asarray(mask, dtype=np.float32).reshape(32, 32), 0.0, 1.0)
    hard = soft >= 0.5
    area = int(np.count_nonzero(hard))
    area_ratio = float(area / hard.size)
    labeled, component_count = connected_components(hard)
    component_sizes = np.bincount(labeled.reshape(-1), minlength=component_count + 1)[1:]
    largest_component = int(component_sizes.max()) if component_sizes.size else 0
    largest_component_ratio = float(largest_component / area) if area else 0.0
    touches_boundary = bool(
        np.any(hard[0, :])
        or np.any(hard[-1, :])
        or np.any(hard[:, 0])
        or np.any(hard[:, -1])
    )
    bbox = _mask_bbox(hard)
    if bbox is None:
        bbox_y0 = bbox_x0 = bbox_y1 = bbox_x1 = -1
        bbox_area_ratio = 0.0
    else:
        bbox_y0, bbox_x0, bbox_y1, bbox_x1 = bbox
        bbox_area_ratio = float((bbox_y1 - bbox_y0) * (bbox_x1 - bbox_x0) / hard.size)
    failures = []
    if area == 0:
        failures.append("empty_mask")
    if area_ratio < config.min_mask_area:
        failures.append("mask_too_small")
    if area_ratio > config.max_mask_area:
        failures.append("mask_too_large")
    if component_count > config.max_components and (
        largest_component_ratio < config.min_largest_component_ratio
    ):
        failures.append("fragmented_mask")
    return {
        "mask_area": area,
        "mask_area_ratio": area_ratio,
        "soft_mask_mass": float(np.mean(soft)),
        "num_components": int(component_count),
        "largest_component_ratio": largest_component_ratio,
        "touches_boundary": touches_boundary,
        "bbox_y0": int(bbox_y0),
        "bbox_x0": int(bbox_x0),
        "bbox_y1": int(bbox_y1),
        "bbox_x1": int(bbox_x1),
        "bbox_area_ratio": bbox_area_ratio,
        "valid_mask": not failures,
        "failure_reason": ";".join(failures),
    }


def _segment_masks(
    images: np.ndarray,
    metadata: pd.DataFrame,
    segmenter: Segmenter,
    *,
    pipeline: str,
    config: CifarBackgroundConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    masks = np.zeros((len(images), 32, 32), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for index, image in _progress(
        enumerate(images),
        total=len(images),
        description=f"{pipeline} masks",
    ):
        row = metadata.iloc[index]
        result = segmenter(image, row, index)
        mask = np.clip(np.asarray(result.mask, dtype=np.float32).reshape(32, 32), 0.0, 1.0)
        masks[index] = mask
        label_id = _label_id(row)
        quality = analyze_mask(mask, config=config)
        failure_parts = [
            value
            for value in (result.failure_reason, quality["failure_reason"])
            if value
        ]
        rows.append(
            {
                "row_id": int(index),
                "pipeline": pipeline,
                "split": str(row.get("split", config.split)),
                "label": str(row.get("label", CIFAR10_LABELS[label_id])),
                "label_id": int(label_id),
                "class_label": str(row.get("label", CIFAR10_LABELS[label_id])),
                "class_label_id": int(label_id),
                "source_index": int(row.get("source_index", index)),
                "original_index": int(row.get("original_index", index)),
                "mask_confidence": float(result.confidence),
                **quality,
                "valid_mask": bool(quality["valid_mask"]) and not result.failure_reason,
                "failure_reason": ";".join(failure_parts),
            }
        )
    return masks, pd.DataFrame.from_records(rows)


def _component_images(
    images: np.ndarray,
    masks: np.ndarray,
    mask_metadata: pd.DataFrame,
    *,
    fill_color: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = np.clip(masks[..., None], 0.0, 1.0)
    fill = np.asarray(fill_color, dtype=np.float32).reshape(1, 1, 1, 3)
    filled = np.broadcast_to(fill, images.shape)
    return {
        "raw": images.astype(np.float32, copy=True),
        "fg": (mask * images + (1.0 - mask) * filled).astype(np.float32),
        "bg": ((1.0 - mask) * images + mask * filled).astype(np.float32),
        "mask": np.repeat(masks[..., None], 3, axis=-1).astype(np.float32),
        "crop": np.stack(
            [
                _crop_and_resize(image, meta)
                for image, meta in zip(
                    images,
                    mask_metadata.to_dict("records"),
                    strict=False,
                )
            ],
            axis=0,
        ).astype(np.float32),
    }


def _write_component_datasets(
    config: CifarBackgroundConfig,
    *,
    pipeline: str,
    output_dir: Path,
    components: dict[str, np.ndarray],
    source_metadata: pd.DataFrame,
    mask_metadata: pd.DataFrame,
    project_root: Path,
) -> list[dict[str, Any]]:
    results = []
    for component in CIFAR_BACKGROUND_COMPONENTS:
        metadata = _component_metadata(
            source_metadata,
            mask_metadata,
            pipeline=pipeline,
            component=component,
            combined=False,
        )
        labels = metadata["label_id"].to_numpy(dtype=np.int64)
        results.append(
            _write_dataset(
                config,
                pipeline=pipeline,
                component=component,
                output_dir=output_dir / component,
                images=components[component],
                labels=labels,
                metadata=metadata,
                project_root=project_root,
            )
        )

    combined_images, combined_metadata, combined_labels = _combined_component_rows(
        components,
        source_metadata=source_metadata,
        mask_metadata=mask_metadata,
        pipeline=pipeline,
    )
    results.append(
        _write_dataset(
            config,
            pipeline=pipeline,
            component="combined",
            output_dir=output_dir / "combined",
            images=combined_images,
            labels=combined_labels,
            metadata=combined_metadata,
            project_root=project_root,
        )
    )
    return results


def _write_dataset(
    config: CifarBackgroundConfig,
    *,
    pipeline: str,
    component: str,
    output_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    project_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = np.clip(np.asarray(images, dtype=np.float32), 0.0, 1.0).reshape(len(images), -1)
    metadata = metadata.reset_index(drop=True).copy()
    metadata["row_id"] = np.arange(len(metadata), dtype=int)

    images_path = output_dir / "images.npy"
    labels_path = output_dir / "labels.npy"
    metadata_path = output_dir / "metadata.parquet"
    dataset_config_path = output_dir / "dataset.yaml"
    np.save(images_path, rows)
    np.save(labels_path, labels)
    write_parquet(metadata, metadata_path)
    _write_rgb_preview_grid(rows, output_dir / "preview_grid.png")

    variant = f"background_{pipeline}_{config.split}_{component}"
    save_config(
        _dataset_config(
            config,
            variant=variant,
            images_path=images_path,
            labels_path=labels_path,
            metadata_path=metadata_path,
            project_root=project_root,
        ),
        dataset_config_path,
    )
    write_json(
        {
            "family": "cifar10",
            "variant_id": f"cifar10/{variant}",
            "pipeline": pipeline,
            "component": component,
            "rows": int(len(rows)),
            "images_path": str(images_path),
            "labels_path": str(labels_path),
            "metadata_path": str(metadata_path),
            "dataset_config_path": str(dataset_config_path),
        },
        output_dir / "manifest.json",
    )
    return {
        "variant_id": f"cifar10/{variant}",
        "rows": int(len(rows)),
        "output_dir": output_dir,
        "dataset_config_path": dataset_config_path,
    }


def _component_metadata(
    source_metadata: pd.DataFrame,
    mask_metadata: pd.DataFrame,
    *,
    pipeline: str,
    component: str,
    combined: bool,
) -> pd.DataFrame:
    metadata = mask_metadata.copy()
    metadata["dataset"] = "cifar10_background"
    metadata["pipeline"] = pipeline
    metadata["component"] = component
    metadata["sample_type"] = "dataset"
    metadata["status"] = "success"
    metadata["base_label"] = source_metadata["label"].astype(str).to_numpy()
    metadata["base_label_id"] = source_metadata["label_id"].astype(int).to_numpy()
    metadata["class_label"] = source_metadata["label"].astype(str).to_numpy()
    metadata["class_label_id"] = source_metadata["label_id"].astype(int).to_numpy()
    if combined:
        metadata["label"] = component
        metadata["label_id"] = CIFAR_BACKGROUND_COMPONENTS.index(component)
        metadata["family"] = component
        metadata["prompt_id"] = f"cifar10_background_{component}"
        metadata["prompt"] = f"CIFAR-10 background ablation {component}"
    else:
        metadata["label"] = source_metadata["label"].astype(str).to_numpy()
        metadata["label_id"] = source_metadata["label_id"].astype(int).to_numpy()
        metadata["family"] = source_metadata["label"].astype(str).to_numpy()
        metadata["prompt_id"] = source_metadata["prompt_id"].astype(str).to_numpy()
        metadata["prompt"] = source_metadata["prompt"].astype(str).to_numpy()
    return metadata


def _combined_component_rows(
    components: dict[str, np.ndarray],
    *,
    source_metadata: pd.DataFrame,
    mask_metadata: pd.DataFrame,
    pipeline: str,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    image_rows: list[np.ndarray] = []
    metadata_rows: list[pd.DataFrame] = []
    label_rows: list[str] = []
    for index in range(len(source_metadata)):
        for component in CIFAR_BACKGROUND_COMPONENTS:
            image_rows.append(components[component][index])
            row = _component_metadata(
                source_metadata.iloc[[index]].reset_index(drop=True),
                mask_metadata.iloc[[index]].reset_index(drop=True),
                pipeline=pipeline,
                component=component,
                combined=True,
            )
            metadata_rows.append(row)
            label_rows.append(component)
    return (
        np.stack(image_rows, axis=0).astype(np.float32),
        pd.concat(metadata_rows, ignore_index=True),
        np.asarray(label_rows, dtype="<U16"),
    )


def _pipeline_metric_rows(
    config: CifarBackgroundConfig,
    *,
    pipeline: str,
    components: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for subset_name, positions in _metric_subsets(metadata, config).items():
        if len(positions) < 3:
            continue
        sampled = metadata.iloc[positions].reset_index(drop=True)
        sampled_components = {
            key: value[positions].reshape(len(positions), -1)
            for key, value in components.items()
        }
        labels = sampled["class_label"].astype(str).to_numpy()
        rows.append(
            {
                "pipeline": pipeline,
                "subset": subset_name,
                "metric_type": "component",
                "n_samples": int(len(positions)),
                "valid_mask_rate": float(sampled["valid_mask"].mean()),
                "mask_area_ratio_median": float(sampled["mask_area_ratio"].median()),
                "mask_confidence_median": float(sampled["mask_confidence"].median()),
                **_distance_pair_metrics(
                    sampled_components,
                    num_pairs=config.metrics_pairs,
                    seed=config.sample_seed,
                ),
                **_knn_metrics(sampled_components, labels),
            }
        )
    return rows


def _agreement_metric_rows(
    masks_by_pipeline: dict[str, np.ndarray],
    metadata_by_pipeline: dict[str, pd.DataFrame],
    *,
    config: CifarBackgroundConfig,
) -> list[dict[str, Any]]:
    rows = []
    for left, right in itertools.combinations(sorted(masks_by_pipeline), 2):
        left_masks = masks_by_pipeline[left]
        right_masks = masks_by_pipeline[right]
        if left_masks.shape != right_masks.shape:
            continue
        valid = (
            metadata_by_pipeline[left]["valid_mask"].to_numpy(dtype=bool)
            & metadata_by_pipeline[right]["valid_mask"].to_numpy(dtype=bool)
        )
        for subset_name, positions in {
            "all": np.arange(len(left_masks), dtype=int),
            "valid_both": np.flatnonzero(valid),
        }.items():
            if len(positions) == 0:
                continue
            iou = _mask_iou(left_masks[positions] >= 0.5, right_masks[positions] >= 0.5)
            rows.append(
                {
                    "pipeline": f"{left}|{right}",
                    "subset": subset_name,
                    "metric_type": "mask_agreement",
                    "n_samples": int(len(positions)),
                    "mask_iou_mean": float(np.mean(iou)),
                    "mask_iou_median": float(np.median(iou)),
                    "valid_mask_rate": float(np.mean(valid[positions])),
                    "metrics_pairs": int(config.metrics_pairs),
                }
            )
    return rows


def _distance_pair_metrics(
    components: dict[str, np.ndarray],
    *,
    num_pairs: int,
    seed: int,
) -> dict[str, float]:
    raw = components["raw"]
    rng = np.random.default_rng(seed)
    n = len(raw)
    pair_count = min(num_pairs, max(1, n * max(1, n - 1)))
    left = rng.integers(0, n, size=pair_count)
    right = rng.integers(0, n, size=pair_count)
    same = left == right
    if np.any(same):
        right[same] = (right[same] + 1) % n
    d_raw = _squared_distances(raw, left, right)
    metrics: dict[str, float] = {}
    eps = float(np.finfo(np.float32).eps)
    for component in ("fg", "bg", "mask", "crop"):
        distances = _squared_distances(components[component], left, right)
        metrics[f"{component}_distance_fraction_median"] = float(
            np.median(distances / (d_raw + eps))
        )
        metrics[f"pearson_raw_{component}"] = _pearson(d_raw, distances)
        metrics[f"spearman_raw_{component}"] = _spearman(d_raw, distances)
    return metrics


def _knn_metrics(components: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    k = min(15, max(1, len(labels) - 1))
    raw_neighbors = _knn_indices(components["raw"], k)
    metrics = {"knn_class_purity_raw": _neighbor_purity(raw_neighbors, labels)}
    for component in ("fg", "bg", "mask", "crop"):
        neighbors = _knn_indices(components[component], k)
        metrics[f"knn_class_purity_{component}"] = _neighbor_purity(neighbors, labels)
        metrics[f"knn_overlap_raw_{component}"] = _neighbor_overlap(raw_neighbors, neighbors)
    return metrics


def _metric_subsets(
    metadata: pd.DataFrame,
    config: CifarBackgroundConfig,
) -> dict[str, np.ndarray]:
    positions = _metric_positions(
        len(metadata),
        max_samples=config.metrics_max_samples,
        seed=config.sample_seed,
    )
    sampled = metadata.iloc[positions].reset_index(drop=True)
    valid = sampled["valid_mask"].to_numpy(dtype=bool)
    area = sampled["mask_area_ratio"].to_numpy(dtype=float)
    confidence = sampled["mask_confidence"].to_numpy(dtype=float)
    subsets = {"all": positions}
    valid_positions = positions[valid]
    if len(valid_positions):
        subsets["valid"] = valid_positions
        threshold = float(np.quantile(confidence[valid], 0.75))
        subsets["high_confidence"] = positions[valid & (confidence >= threshold)]
        subsets["mid_area_valid"] = positions[
            valid & (area >= config.min_mask_area * 2.0) & (area <= config.max_mask_area * 0.85)
        ]
    return {key: value for key, value in subsets.items() if len(value) > 0}


def _make_segmenter(pipeline: str, config: CifarBackgroundConfig) -> Segmenter:
    if pipeline == "rembg":
        return _make_rembg_segmenter(config)
    if pipeline == "grounded_sam":
        return _make_grounded_sam_segmenter(config)
    raise ConfigError(f"Unknown CIFAR background pipeline: {pipeline}")


def _make_rembg_segmenter(config: CifarBackgroundConfig) -> Segmenter:
    try:
        from rembg import new_session, remove
    except ImportError as error:
        raise ConfigError(
            "The rembg pipeline requires the optional 'rembg' package. "
            "Install it in the fm_lab environment before running --pipeline rembg."
        ) from error

    session = new_session(config.rembg_model)

    def segment(image: np.ndarray, row: pd.Series, index: int) -> MaskResult:
        del row, index
        try:
            upsampled = _pil_rgb(image).resize(
                (config.upsample_size, config.upsample_size),
                resample=_pil_resampling("bicubic"),
            )
            output = remove(upsampled, session=session, only_mask=True)
            mask = _pil_mask(output).resize((32, 32), resample=_pil_resampling("box"))
            values = np.asarray(mask, dtype=np.float32) / 255.0
            return MaskResult(mask=values, confidence=float(np.max(values)))
        except Exception as error:  # pragma: no cover - exercised by integration runs.
            LOGGER.warning("rembg failed for CIFAR sample: %s", error)
            return MaskResult(
                mask=np.zeros((32, 32), dtype=np.float32),
                confidence=0.0,
                failure_reason=f"rembg_error:{type(error).__name__}",
            )

    return segment


def _make_grounded_sam_segmenter(config: CifarBackgroundConfig) -> Segmenter:
    missing = [
        name
        for name, value in {
            "--grounded-sam-config": config.grounded_sam_config,
            "--grounded-sam-checkpoint": config.grounded_sam_checkpoint,
            "--sam-checkpoint": config.sam_checkpoint,
        }.items()
        if not value
    ]
    if missing:
        raise ConfigError(
            "The grounded_sam pipeline requires checkpoint/config paths: "
            + ", ".join(missing)
        )
    try:
        import torch
        from groundingdino.util.inference import load_image, load_model, predict
        from segment_anything import SamPredictor, sam_model_registry
        from torchvision.ops import box_convert
    except ImportError as error:
        raise ConfigError(
            "The grounded_sam pipeline requires optional GroundingDINO, "
            "segment-anything, torch, and torchvision packages."
        ) from error

    device = config.device
    dino = load_model(
        config.grounded_sam_config,
        config.grounded_sam_checkpoint,
        device=device,
    )
    sam = sam_model_registry[config.sam_model_type](checkpoint=config.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    def segment(image: np.ndarray, row: pd.Series, index: int) -> MaskResult:
        label_name = str(row.get("label", CIFAR10_LABELS[_label_id(row)]))
        prompt = DEFAULT_PROMPTS.get(label_name, label_name).strip()
        if not prompt.endswith("."):
            prompt = f"{prompt}."
        upsampled = _pil_rgb(image).resize(
            (config.upsample_size, config.upsample_size),
            resample=_pil_resampling("bicubic"),
        )
        with tempfile.NamedTemporaryFile(suffix=f"_cifar_{index}.png") as handle:
            upsampled.save(handle.name)
            image_source, image_tensor = load_image(handle.name)
        boxes, logits, _phrases = predict(
            model=dino,
            image=image_tensor,
            caption=prompt,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
            device=device,
        )
        if len(boxes) == 0:
            return MaskResult(
                mask=np.zeros((32, 32), dtype=np.float32),
                confidence=0.0,
                failure_reason="no_detection",
            )
        best = int(torch.argmax(logits).item())
        h, w, _ = image_source.shape
        box = boxes[best : best + 1] * torch.tensor(
            [w, h, w, h],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        box_xyxy = box_convert(boxes=box, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        predictor.set_image(image_source)
        masks, scores, _logits = predictor.predict(
            box=box_xyxy[0],
            multimask_output=True,
        )
        mask_index = int(np.argmax(scores))
        mask = _resize_mask(np.asarray(masks[mask_index], dtype=np.float32), size=32)
        confidence = float(scores[mask_index]) * float(logits[best].detach().cpu().item())
        return MaskResult(mask=mask, confidence=confidence)

    return segment


def _load_source(
    config: CifarBackgroundConfig,
    *,
    project_root: Path,
) -> DatasetBundle:
    dataset = load_dataset(
        InputConfig(
            type="cifar10",
            dataset_root=config.dataset_root,
            split=config.split,
            order="source",
            color_mode="rgb",
            thumbnail_mode="none",
            max_samples=None,
            download=False,
        ),
        project_root=project_root,
        thumbnail_dir=None,
    )
    if dataset.vectors is None:
        raise ConfigError("CIFAR background ablations require vector-backed data.")
    if dataset.vectors.shape[1] != 32 * 32 * 3:
        raise ConfigError(f"Expected flattened 32x32 RGB data, got {dataset.vectors.shape}.")
    return dataset


def _select_positions(
    metadata: pd.DataFrame,
    *,
    maximum: int | None,
    strategy: str,
    seed: int,
) -> np.ndarray:
    total = len(metadata)
    if maximum is None or maximum >= total:
        return np.arange(total, dtype=int)
    if maximum < 1:
        raise ConfigError("--max-samples must be positive when provided.")
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return np.sort(rng.choice(total, size=maximum, replace=False))
    if strategy != "stratified":
        raise ConfigError("--sample-strategy must be random or stratified.")
    labels = metadata["label_id"].astype(int).to_numpy()
    unique = np.sort(np.unique(labels))
    per_class = maximum // len(unique)
    remainder = maximum % len(unique)
    selected = []
    for offset, label in enumerate(unique):
        candidates = np.flatnonzero(labels == label)
        take = min(len(candidates), per_class + (1 if offset < remainder else 0))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < maximum:
        missing = maximum - len(selected)
        remaining = np.setdiff1d(np.arange(total, dtype=int), np.asarray(selected, dtype=int))
        selected.extend(rng.choice(remaining, size=missing, replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def _validate_config(
    config: CifarBackgroundConfig,
    *,
    segmenters: dict[str, Segmenter] | None,
) -> None:
    if config.split not in {"train", "test", "all"}:
        raise ConfigError("--split must be train, test, or all.")
    if config.upsample_size < 32:
        raise ConfigError("--upsample-size must be at least 32.")
    if config.fill not in {"mean", "gray"}:
        raise ConfigError("--fill must be mean or gray.")
    if not config.pipelines:
        raise ConfigError("At least one --pipeline is required.")
    custom = set(segmenters or {})
    unknown = set(config.pipelines) - set(CIFAR_BACKGROUND_PIPELINES) - custom
    if unknown:
        raise ConfigError(f"Unknown CIFAR background pipelines: {sorted(unknown)}")


def _dataset_config(
    config: CifarBackgroundConfig,
    *,
    variant: str,
    images_path: Path,
    labels_path: Path,
    metadata_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "family": "cifar10",
        "variant": variant,
        "base": "original",
        "split": config.split,
        "seed": config.sample_seed,
        "input": {
            "type": "numpy",
            "data_path": _relative_to_project(images_path, project_root),
            "labels_path": _relative_to_project(labels_path, project_root),
            "metadata_path": _relative_to_project(metadata_path, project_root),
            "image_shape": [32, 32, 3],
            "value_range": [0.0, 1.0],
            "thumbnail_mode": "atlas",
        },
        "selection": {},
    }


def _fill_color(images: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gray":
        return np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
    return np.asarray(np.mean(images, axis=(0, 1, 2)), dtype=np.float32)


def _crop_and_resize(image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    if not bool(metadata.get("valid_mask", False)):
        return image.astype(np.float32, copy=True)
    y0 = int(metadata.get("bbox_y0", -1))
    x0 = int(metadata.get("bbox_x0", -1))
    y1 = int(metadata.get("bbox_y1", -1))
    x1 = int(metadata.get("bbox_x1", -1))
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        return image.astype(np.float32, copy=True)
    pad = 2
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(32, y1 + pad)
    x1 = min(32, x1 + pad)
    crop = image[y0:y1, x0:x1]
    return np.asarray(
        _pil_rgb(crop).resize((32, 32), resample=_pil_resampling("bicubic")),
        dtype=np.float32,
    ) / 255.0


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.where(mask)
    if len(y) == 0:
        return None
    return int(y.min()), int(x.min()), int(y.max()) + 1, int(x.max()) + 1


def _label_id(row: pd.Series) -> int:
    if "label_id" in row and not pd.isna(row["label_id"]):
        return int(row["label_id"])
    label = str(row.get("label", "0"))
    if label in CIFAR10_LABELS:
        return CIFAR10_LABELS.index(label)
    return int(label)


def _write_rgb_preview_grid(
    rows: np.ndarray,
    output_path: Path,
    *,
    columns: int = 10,
    max_images: int = 100,
) -> None:
    from PIL import Image

    images = np.asarray(rows, dtype=np.float32).reshape(-1, 32, 32, 3)
    count = min(len(images), max_images)
    if count == 0:
        return
    columns = min(columns, count)
    grid_rows = int(np.ceil(count / columns))
    grid = np.zeros((grid_rows * 32, columns * 32, 3), dtype=np.uint8)
    pixels = np.asarray(np.round(np.clip(images[:count], 0.0, 1.0) * 255.0), dtype=np.uint8)
    for index, tile in enumerate(pixels):
        row, column = divmod(index, columns)
        grid[row * 32 : (row + 1) * 32, column * 32 : (column + 1) * 32] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="RGB").save(output_path)


def _write_mask_preview(masks: np.ndarray, output_path: Path) -> None:
    _write_rgb_preview_grid(
        np.repeat(np.asarray(masks, dtype=np.float32)[..., None], 3, axis=-1).reshape(
            len(masks),
            -1,
        ),
        output_path,
    )


def _resize_mask(mask: np.ndarray, *, size: int) -> np.ndarray:
    from PIL import Image

    pixels = np.asarray(np.round(np.clip(mask, 0.0, 1.0) * 255.0), dtype=np.uint8)
    resized = Image.fromarray(pixels, mode="L").resize(
        (size, size),
        resample=_pil_resampling("box"),
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _pil_rgb(image: np.ndarray):
    from PIL import Image

    pixels = np.asarray(np.round(np.clip(image, 0.0, 1.0) * 255.0), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _pil_mask(value: Any):
    from io import BytesIO

    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("L")
    if isinstance(value, bytes | bytearray):
        return Image.open(BytesIO(value)).convert("L")
    return Image.fromarray(np.asarray(value), mode="L")


def _pil_resampling(name: str) -> Any:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    return {
        "bicubic": resampling.BICUBIC,
        "box": resampling.BOX,
    }[name]


def _progress(
    values: Iterable[tuple[int, np.ndarray]],
    *,
    total: int,
    description: str,
) -> Iterable[tuple[int, np.ndarray]]:
    try:
        from tqdm import tqdm
    except ImportError:
        return values
    return tqdm(values, total=total, desc=description)


def _metric_positions(total: int, *, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_samples, replace=False))


def _knn_indices(values: np.ndarray, k: int) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(values)
    return neighbors.kneighbors(return_distance=False)


def _neighbor_purity(neighbors: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(labels[neighbors] == labels[:, None]))


def _neighbor_overlap(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            [
                len(set(left_row.tolist()) & set(right_row.tolist())) / len(left_row)
                for left_row, right_row in zip(left, right, strict=False)
            ]
        )
    )


def _squared_distances(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = values[left] - values[right]
    return np.einsum("ij,ij->i", diff, diff)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(pearsonr(left, right).statistic)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(left, right).reshape(len(left), -1).sum(axis=1)
    union = np.logical_or(left, right).reshape(len(left), -1).sum(axis=1)
    return np.divide(
        intersection,
        np.maximum(union, 1),
        out=np.ones_like(intersection, dtype=np.float32),
        where=union > 0,
    )
