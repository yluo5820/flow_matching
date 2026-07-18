"""Stage-0 geometry selection for the Fashion-MNIST frequency bridge."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from fm_lab.data import LongTailedFashionMNIST
from fm_lab.image_diagnostics.config import FeatureConfig
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle
from fm_lab.image_diagnostics.feature_models import ImageFeatureExtractor, load_feature_model
from fm_lab.image_diagnostics.feature_runner import compute_or_load_features
from fm_lab.image_diagnostics.id_estimators import mle_lid_local, two_nn_global
from fm_lab.utils.config import load_config

_CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
_REPRESENTATIONS = ("raw_pca50", "dinov2_pca50")
_ESTIMATORS = (
    "two_nn",
    "mle_lid_k10",
    "mle_lid_k20",
    "participation_ratio",
    "pca_dim_90",
)
_COMPARISON_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class SelectionThresholds:
    min_finite_fraction: float = 0.80
    max_probe_score_gap: float = 0.10
    max_cell_iqr: float = 0.25
    min_adjacent_score_gap: float = 0.15


@dataclass(frozen=True)
class FashionGeometryStage0Config:
    raw: dict[str, Any]
    config_hash: str
    project_root: Path
    data_root: Path
    output_dir: Path
    download: bool
    partition_seed: int
    diagnostic_pool_per_class: int
    pca_components: int
    pca_seed: int
    dinov2_repo_id: str
    dinov2_batch_size: int
    dinov2_dtype: str
    subsamples: int
    subsample_fraction: float
    subsample_seed: int
    thresholds: SelectionThresholds


def load_stage0_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> FashionGeometryStage0Config:
    """Load and validate the frozen Fashion-MNIST Stage-0 configuration."""

    config_path = Path(path).expanduser().resolve()
    raw = load_config(config_path)
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    data = _mapping(raw, "data")
    features = _mapping(raw, "features")
    dinov2 = _mapping(features, "dinov2")
    selection = _mapping(raw, "selection")
    stability = _mapping(selection, "stability")
    output = _mapping(raw, "output")

    diagnostic_pool = _positive_int(
        "data.diagnostic_pool_per_class",
        data.get("diagnostic_pool_per_class", 1000),
    )
    if diagnostic_pool % 2:
        raise ValueError("data.diagnostic_pool_per_class must be even.")
    if diagnostic_pool >= 6000:
        raise ValueError("The diagnostic pool must leave Fashion-MNIST training images.")
    pca_components = _positive_int(
        "features.pca_components",
        features.get("pca_components", 50),
    )
    if pca_components != 50:
        raise ValueError("The frozen Stage-0 protocol requires features.pca_components=50.")
    if pca_components > diagnostic_pool // 2:
        raise ValueError("features.pca_components cannot exceed one probe's class count.")
    subsamples = _positive_int("selection.subsamples", selection.get("subsamples", 100))
    subsample_fraction = _probability(
        "selection.subsample_fraction",
        selection.get("subsample_fraction", 0.8),
    )
    if int((diagnostic_pool // 2) * subsample_fraction) <= 20:
        raise ValueError("Each stability subsample must contain more than 20 images.")
    thresholds = SelectionThresholds(
        min_finite_fraction=_probability(
            "selection.stability.min_finite_fraction",
            stability.get("min_finite_fraction", 0.8),
        ),
        max_probe_score_gap=_nonnegative_float(
            "selection.stability.max_probe_score_gap",
            stability.get("max_probe_score_gap", 0.1),
        ),
        max_cell_iqr=_nonnegative_float(
            "selection.stability.max_cell_iqr",
            stability.get("max_cell_iqr", 0.25),
        ),
        min_adjacent_score_gap=_nonnegative_float(
            "selection.stability.min_adjacent_score_gap",
            stability.get("min_adjacent_score_gap", 0.15),
        ),
    )
    output_root = _resolve(root, output.get("root_dir", "outputs/fashion_mnist_geometry_frequency"))
    stage_name = str(output.get("stage_name", "stage0_class_selection"))
    if not stage_name or Path(stage_name).name != stage_name:
        raise ValueError("output.stage_name must be one path component.")

    return FashionGeometryStage0Config(
        raw=raw,
        config_hash=_json_digest(raw),
        project_root=root,
        data_root=_resolve(root, data.get("root", "data/fashion_mnist")),
        output_dir=output_root / stage_name,
        download=bool(data.get("download", False)),
        partition_seed=_nonnegative_int("data.partition_seed", data.get("partition_seed", 0)),
        diagnostic_pool_per_class=diagnostic_pool,
        pca_components=pca_components,
        pca_seed=_nonnegative_int("features.pca_seed", features.get("pca_seed", 42)),
        dinov2_repo_id=str(dinov2.get("repo_id", "facebook/dinov2-base")),
        dinov2_batch_size=_positive_int(
            "features.dinov2.batch_size",
            dinov2.get("batch_size", 16),
        ),
        dinov2_dtype=str(dinov2.get("dtype", "float16")),
        subsamples=subsamples,
        subsample_fraction=subsample_fraction,
        subsample_seed=_nonnegative_int(
            "selection.seed",
            selection.get("seed", 26072026),
        ),
        thresholds=thresholds,
    )


def run_stage0(
    config: FashionGeometryStage0Config,
    *,
    device: str = "auto",
    dry_run: bool = False,
    model_loader: Callable[[FeatureConfig], ImageFeatureExtractor] = load_feature_model,
) -> dict[str, Any]:
    """Build independent probes and select a stable low/middle/high class trio."""

    probe_per_class = config.diagnostic_pool_per_class // 2
    subsample_size = int(probe_per_class * config.subsample_fraction)
    if dry_run:
        return {
            "stage": "fashion_mnist_geometry_frequency_stage0",
            "config_hash": config.config_hash,
            "output_dir": str(config.output_dir),
            "data_root": str(config.data_root),
            "download": config.download,
            "probe_images": 2 * probe_per_class * len(_CLASS_NAMES),
            "training_candidates_per_class": 6000 - config.diagnostic_pool_per_class,
            "representations": list(_REPRESENTATIONS),
            "pca_components": config.pca_components,
            "subsamples_per_cell": config.subsamples,
            "subsample_size": subsample_size,
            "estimator_records": (
                len(_REPRESENTATIONS)
                * 2
                * len(_CLASS_NAMES)
                * config.subsamples
            ),
            "learned_feature_device": device,
            "estimated_runtime": (
                "DINOv2 extraction over 10,000 images dominates; benchmark locally before "
                "deciding whether the 30-minute handoff rule applies"
            ),
            "outcome_training_enabled": False,
        }

    completed = _completed_result(config)
    if completed is not None:
        return completed | {"reused": True}
    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise FileExistsError(f"Incomplete Stage-0 output already exists: {config.output_dir}")

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}-", dir=config.output_dir.parent)
    )
    try:
        dataset = LongTailedFashionMNIST(
            root=config.data_root,
            train=True,
            download=config.download,
            imbalance_type="balanced",
            imbalance_factor=1.0,
            subset_seed=config.partition_seed,
            normalize="zero_one",
            dequantize=False,
            frequency_mapping_offset=0,
            frequency_mapping_multiplier=3,
            diagnostic_pool_per_class=config.diagnostic_pool_per_class,
        )
        probe = _materialize_probes(dataset, config)
        _write_partition_artifacts(temporary, dataset, probe, config)
        features = _build_representations(
            probe,
            config,
            output_dir=temporary,
            device=device,
            model_loader=model_loader,
        )
        records = estimate_geometry_records(
            features,
            labels=probe["labels"],
            probe_splits=probe["probe_splits"],
            subsamples=config.subsamples,
            subsample_fraction=config.subsample_fraction,
            seed=config.subsample_seed,
        )
        records.to_csv(temporary / "geometry_estimates.csv", index=False)
        gate, ranks = select_geometry_trio(records, config.thresholds)
        ranks.to_csv(temporary / "geometry_percentile_ranks.csv", index=False)
        gate.update(
            {
                "schema_version": 1,
                "stage": "fashion_mnist_geometry_frequency_stage0",
                "config_hash": config.config_hash,
                "outcome_training_enabled": False,
                "artifacts": {
                    "partition_manifest": "partition_manifest.json",
                    "partition_indices": "partition_indices.npz",
                    "geometry_estimates": "geometry_estimates.csv",
                    "geometry_percentile_ranks": "geometry_percentile_ranks.csv",
                    "raw_features": "features/raw_pca50.npy",
                    "dinov2_features": "features/dinov2_pca50.npy",
                },
            }
        )
        if gate["passed"]:
            selected = [
                {
                    "role": role,
                    "original_class_id": int(class_id),
                    "class_name": _CLASS_NAMES[int(class_id)],
                }
                for role, class_id in gate["selected_class_ids"].items()
            ]
            selection = {
                "schema_version": 1,
                "config_hash": config.config_hash,
                "classes": selected,
                "selection_digest": _json_digest(selected),
            }
            _write_json(temporary / "selected_classes.json", selection)
            gate["artifacts"]["selected_classes"] = "selected_classes.json"
            gate["selection_digest"] = selection["selection_digest"]
        _write_json(temporary / "selection_gate.json", gate)
        temporary.replace(config.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return _completed_result(config) or {}


def estimate_geometry_records(
    representations: dict[str, np.ndarray],
    *,
    labels: np.ndarray,
    probe_splits: np.ndarray,
    subsamples: int,
    subsample_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Estimate five ID proxies on paired subsamples of every class and probe."""

    labels = np.asarray(labels, dtype=np.int64)
    probe_splits = np.asarray(probe_splits).astype(str)
    if set(representations) != set(_REPRESENTATIONS):
        raise ValueError(f"Representations must be exactly {_REPRESENTATIONS}.")
    if any(len(values) != len(labels) for values in representations.values()):
        raise ValueError("Every representation must align with labels.")
    if len(probe_splits) != len(labels) or set(probe_splits) != {"a", "b"}:
        raise ValueError("probe_splits must align and contain exactly 'a' and 'b'.")
    if set(np.unique(labels).tolist()) != set(range(len(_CLASS_NAMES))):
        raise ValueError("Geometry records require all ten Fashion-MNIST classes.")
    if subsamples < 1 or not 0.0 < subsample_fraction <= 1.0:
        raise ValueError("Invalid stability subsampling settings.")

    rows: list[dict[str, Any]] = []
    for representation in _REPRESENTATIONS:
        values = np.asarray(representations[representation], dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"Representation {representation} must be a finite matrix.")
        for split in ("a", "b"):
            for class_id in range(len(_CLASS_NAMES)):
                positions = np.flatnonzero((probe_splits == split) & (labels == class_id))
                sample_size = int(len(positions) * subsample_fraction)
                if sample_size <= 20:
                    raise ValueError("Every class-probe subsample must contain more than 20 rows.")
                class_features = values[positions]
                distance_matrix = pairwise_distances(class_features, metric="euclidean")
                np.fill_diagonal(distance_matrix, np.inf)
                for draw in range(subsamples):
                    generator = np.random.default_rng(
                        _subsample_seed(seed, split=split, class_id=class_id, draw=draw)
                    )
                    selected = np.sort(
                        generator.choice(len(positions), size=sample_size, replace=False)
                    )
                    estimates = _estimate_one_subsample(
                        class_features[selected],
                        distance_matrix[np.ix_(selected, selected)],
                    )
                    rows.append(
                        {
                            "representation": representation,
                            "probe_split": split,
                            "subsample": draw,
                            "class_id": class_id,
                            **estimates,
                        }
                    )
    return pd.DataFrame(rows)


def select_geometry_trio(
    records: pd.DataFrame,
    thresholds: SelectionThresholds,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Rank estimator cells and apply the preregistered stability gate."""

    id_columns = ["representation", "probe_split", "subsample", "class_id"]
    required = set(id_columns) | set(_ESTIMATORS)
    if not required <= set(records.columns):
        raise ValueError(f"Geometry records are missing columns: {sorted(required - set(records))}")
    long = records.melt(
        id_vars=id_columns,
        value_vars=list(_ESTIMATORS),
        var_name="estimator",
        value_name="estimate",
    )
    long["percentile_rank"] = long.groupby(
        ["representation", "probe_split", "subsample", "estimator"],
        sort=False,
    )["estimate"].rank(method="average", pct=True)

    summaries = []
    for class_id in range(len(_CLASS_NAMES)):
        class_rows = long[long["class_id"] == class_id]
        finite_fraction = float(np.isfinite(class_rows["estimate"]).mean())
        probe_scores = class_rows.groupby("probe_split")["percentile_rank"].median()
        probe_a = float(probe_scores.get("a", np.nan))
        probe_b = float(probe_scores.get("b", np.nan))
        probe_gap = abs(probe_a - probe_b)
        cell_scores = class_rows.groupby(["representation", "estimator"])[
            "percentile_rank"
        ].median()
        cell_values = cell_scores.to_numpy(dtype=float)
        cell_iqr = float(np.nanquantile(cell_values, 0.75) - np.nanquantile(cell_values, 0.25))
        representation_probe = class_rows.groupby(["representation", "probe_split"])[
            "percentile_rank"
        ].median()
        reversal = False
        for representation in _REPRESENTATIONS:
            first = float(representation_probe.get((representation, "a"), np.nan))
            second = float(representation_probe.get((representation, "b"), np.nan))
            if (first <= 1.0 / 3.0 and second >= 2.0 / 3.0) or (
                second <= 1.0 / 3.0 and first >= 2.0 / 3.0
            ):
                reversal = True
        score = float(class_rows["percentile_rank"].median())
        reasons = []
        if finite_fraction + _COMPARISON_TOLERANCE < thresholds.min_finite_fraction:
            reasons.append("finite_fraction")
        if not math.isfinite(probe_gap) or probe_gap > (
            thresholds.max_probe_score_gap + _COMPARISON_TOLERANCE
        ):
            reasons.append("probe_gap")
        if not math.isfinite(cell_iqr) or cell_iqr > (
            thresholds.max_cell_iqr + _COMPARISON_TOLERANCE
        ):
            reasons.append("cell_iqr")
        if reversal:
            reasons.append("probe_third_reversal")
        summaries.append(
            {
                "class_id": class_id,
                "geometry_score": score,
                "finite_fraction": finite_fraction,
                "probe_a_score": probe_a,
                "probe_b_score": probe_b,
                "probe_score_gap": probe_gap,
                "cell_iqr": cell_iqr,
                "third_reversal": reversal,
                "eligible": not reasons,
                "reasons": reasons,
            }
        )

    eligible = [row for row in summaries if row["eligible"]]
    low_candidates = [row for row in eligible if row["geometry_score"] <= 1.0 / 3.0]
    middle_candidates = [
        row for row in eligible if 1.0 / 3.0 < row["geometry_score"] < 2.0 / 3.0
    ]
    high_candidates = [row for row in eligible if row["geometry_score"] >= 2.0 / 3.0]
    gate_reasons = []
    selected_ids: dict[str, int] = {}
    if not low_candidates:
        gate_reasons.append("no_stable_low_class")
    if not middle_candidates:
        gate_reasons.append("no_stable_middle_class")
    if not high_candidates:
        gate_reasons.append("no_stable_high_class")
    if not gate_reasons:
        low = min(low_candidates, key=lambda row: (row["geometry_score"], row["class_id"]))
        middle = min(
            middle_candidates,
            key=lambda row: (abs(row["geometry_score"] - 0.5), row["class_id"]),
        )
        high = max(
            high_candidates,
            key=lambda row: (row["geometry_score"], -row["class_id"]),
        )
        if middle["geometry_score"] - low["geometry_score"] + (
            _COMPARISON_TOLERANCE
        ) < thresholds.min_adjacent_score_gap:
            gate_reasons.append("low_middle_gap")
        if high["geometry_score"] - middle["geometry_score"] + (
            _COMPARISON_TOLERANCE
        ) < thresholds.min_adjacent_score_gap:
            gate_reasons.append("middle_high_gap")
        selected_ids = {
            "low": int(low["class_id"]),
            "middle": int(middle["class_id"]),
            "high": int(high["class_id"]),
        }

    gate = {
        "passed": not gate_reasons,
        "reasons": gate_reasons,
        "selected_class_ids": selected_ids if not gate_reasons else {},
        "class_summaries": summaries,
        "thresholds": {
            "min_finite_fraction": thresholds.min_finite_fraction,
            "max_probe_score_gap": thresholds.max_probe_score_gap,
            "max_cell_iqr": thresholds.max_cell_iqr,
            "min_adjacent_score_gap": thresholds.min_adjacent_score_gap,
        },
    }
    return gate, long


def _materialize_probes(
    dataset: LongTailedFashionMNIST,
    config: FashionGeometryStage0Config,
) -> dict[str, np.ndarray]:
    if dataset.class_counts != (6000 - config.diagnostic_pool_per_class,) * 10:
        raise ValueError(f"Unexpected training candidate counts: {dataset.class_counts}")
    images_a, labels_a, ids_a = dataset.diagnostic_samples("a")
    images_b, labels_b, ids_b = dataset.diagnostic_samples("b")
    expected = config.diagnostic_pool_per_class // 2
    for labels in (labels_a.numpy(), labels_b.numpy()):
        counts = np.bincount(labels, minlength=10)
        if not np.array_equal(counts, np.full(10, expected, dtype=np.int64)):
            raise ValueError(f"Unexpected diagnostic counts: {counts.tolist()}")
    return {
        "images": np.concatenate(
            [
                images_a.numpy().reshape(len(images_a), -1),
                images_b.numpy().reshape(len(images_b), -1),
            ]
        ).astype(np.float32),
        "labels": np.concatenate([labels_a.numpy(), labels_b.numpy()]).astype(np.int64),
        "original_indices": np.concatenate(
            [ids_a.astype(np.int64), ids_b.astype(np.int64)]
        ),
        "probe_splits": np.asarray(["a"] * len(ids_a) + ["b"] * len(ids_b)),
    }


def _build_representations(
    probe: dict[str, np.ndarray],
    config: FashionGeometryStage0Config,
    *,
    output_dir: Path,
    device: str,
    model_loader: Callable[[FeatureConfig], ImageFeatureExtractor],
) -> dict[str, np.ndarray]:
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    images = probe["images"]
    raw, raw_pca = _pca_features(images, config.pca_components, config.pca_seed)
    np.save(feature_dir / "raw_pca50.npy", raw)
    np.savez_compressed(
        feature_dir / "raw_pca50_transform.npz",
        components=raw_pca.components_,
        mean=raw_pca.mean_,
        explained_variance=raw_pca.explained_variance_,
    )

    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(images)),
            "label": probe["labels"],
            "probe_split": probe["probe_splits"],
            "original_index": probe["original_indices"],
        }
    )
    bundle = DatasetBundle(
        metadata=metadata,
        vectors=images,
        source_id=_array_digest(probe["original_indices"]),
        source_description="Fashion-MNIST disjoint Stage-0 geometry probes",
        total_rows=len(images),
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
    )
    dino_config = FeatureConfig(
        mode="dinov2",
        name="dinov2_base_cls",
        normalize=True,
        skip_existing=False,
        repo_id=config.dinov2_repo_id,
        batch_size=config.dinov2_batch_size,
        device=device,
        dtype=config.dinov2_dtype,
    )
    dino_result = compute_or_load_features(
        config=dino_config,
        dataset=bundle,
        output_dir=output_dir / "dinov2_source",
        save=True,
        model_loader=model_loader,
    )
    dino, dino_pca = _pca_features(
        dino_result.features,
        config.pca_components,
        config.pca_seed,
    )
    np.save(feature_dir / "dinov2_pca50.npy", dino)
    np.savez_compressed(
        feature_dir / "dinov2_pca50_transform.npz",
        components=dino_pca.components_,
        mean=dino_pca.mean_,
        explained_variance=dino_pca.explained_variance_,
    )
    metadata.to_csv(feature_dir / "probe_metadata.csv", index=False)
    return {"raw_pca50": raw, "dinov2_pca50": dino}


def _pca_features(
    features: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, PCA]:
    model = PCA(
        n_components=components,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    )
    values = model.fit_transform(np.asarray(features, dtype=np.float32))
    return np.asarray(values, dtype=np.float32), model


def _estimate_one_subsample(
    features: np.ndarray,
    distances: np.ndarray,
) -> dict[str, float]:
    nearest = np.partition(distances, kth=19, axis=1)[:, :20]
    nearest.sort(axis=1)
    mle10 = mle_lid_local(nearest[:, :10])
    mle20 = mle_lid_local(nearest[:, :20])
    centered = features - features.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(features) - 1)
    spectrum = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    total = float(spectrum.sum())
    squared = float(np.square(spectrum).sum())
    participation = total**2 / squared if squared > np.finfo(float).eps else float("nan")
    if total > np.finfo(float).eps:
        pca90 = float(np.searchsorted(np.cumsum(spectrum) / total, 0.9) + 1)
    else:
        pca90 = float("nan")
    return {
        "two_nn": two_nn_global(nearest),
        "mle_lid_k10": _finite_median(mle10),
        "mle_lid_k20": _finite_median(mle20),
        "participation_ratio": participation,
        "pca_dim_90": pca90,
    }


def _write_partition_artifacts(
    output_dir: Path,
    dataset: LongTailedFashionMNIST,
    probe: dict[str, np.ndarray],
    config: FashionGeometryStage0Config,
) -> None:
    training_indices = dataset.selected_indices.astype(np.int64)
    training_labels = dataset.labels.numpy().astype(np.int64)
    np.savez_compressed(
        output_dir / "partition_indices.npz",
        training_indices=training_indices,
        training_labels=training_labels,
        probe_indices=probe["original_indices"],
        probe_labels=probe["labels"],
        probe_splits=probe["probe_splits"],
    )
    classes = []
    for class_id in range(10):
        train = training_indices[training_labels == class_id]
        probe_a = probe["original_indices"][
            (probe["labels"] == class_id) & (probe["probe_splits"] == "a")
        ]
        probe_b = probe["original_indices"][
            (probe["labels"] == class_id) & (probe["probe_splits"] == "b")
        ]
        if set(train) & set(probe_a) or set(train) & set(probe_b) or set(probe_a) & set(probe_b):
            raise ValueError(f"Partition overlap detected for class {class_id}.")
        classes.append(
            {
                "class_id": class_id,
                "training_candidates": len(train),
                "probe_a": len(probe_a),
                "probe_b": len(probe_b),
                "training_sha256": _array_digest(train),
                "probe_a_sha256": _array_digest(probe_a),
                "probe_b_sha256": _array_digest(probe_b),
            }
        )
    _write_json(
        output_dir / "partition_manifest.json",
        {
            "schema_version": 1,
            "config_hash": config.config_hash,
            "partition_seed": config.partition_seed,
            "diagnostic_pool_per_class": config.diagnostic_pool_per_class,
            "classes": classes,
            "probe_image_sha256": _array_digest(probe["images"]),
        },
    )


def _completed_result(config: FashionGeometryStage0Config) -> dict[str, Any] | None:
    gate_path = config.output_dir / "selection_gate.json"
    if not gate_path.is_file():
        return None
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("config_hash") != config.config_hash:
        raise ValueError("Completed Stage-0 output uses a different config hash.")
    return gate | {"output_dir": str(config.output_dir), "reused": False}


def _subsample_seed(seed: int, *, split: str, class_id: int, draw: int) -> int:
    split_offset = 0 if split == "a" else 1_000_000_007
    return int((seed + split_offset + 1_000_003 * class_id + 97_003 * draw) % (2**63 - 1))


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    result = values.get(key, {})
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be a mapping.")
    return dict(result)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _probability(name: str, value: object) -> float:
    number = _nonnegative_float(name, value)
    if number <= 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return number


def _nonnegative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite non-negative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return number


def _json_digest(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    payload = f"{array.dtype}:{array.shape}".encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
