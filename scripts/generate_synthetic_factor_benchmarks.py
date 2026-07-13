"""Generate synthetic factor-geometry benchmark datasets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from fm_lab.geometry_explorer.latent_factors import (
    AmbientLightInterval,
    AzimuthCircle,
    BoundedTranslation,
    CameraDepthTranslationInterval,
    CameraIntrinsicsFactor,
    CameraLocalTranslation,
    CameraLogAspectRatioInterval,
    CameraLogFocalScaleInterval,
    CameraPrincipalPointOffset,
    CameraRadiusInterval,
    CameraRollInterval,
    CameraSE3Factor,
    CameraSkewInterval,
    DiffuseLightInterval,
    FullAppearanceFactor,
    FullCameraFactor,
    IlluminationFactor,
    ImageLogExposureInterval,
    LatentFactorSpace,
    LightingDirectionSphere,
    LightLogEnergyInterval,
    LookAtViewSphere,
    OrientationSO3,
    PhotometryFactor,
    ProductFactorSpace,
    ZoomInterval,
    sample_values,
)
from fm_lab.geometry_explorer.latent_pixel_diagnostics import (
    analyze_latent_pixel_diagnostics,
)
from fm_lab.geometry_explorer.product_structure import analyze_product_structure
from fm_lab.geometry_explorer.pullback_metric import analyze_pullback_metrics
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.render_maps import RenderConfig, RenderMap
from fm_lab.geometry_explorer.views import build_projection_view
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.utils.config import ConfigError, load_config, save_config
from fm_lab.utils.logging import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW_CONFIG = PROJECT_ROOT / "configs" / "geometry_explorer" / "views" / "raw_pixels.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Benchmark YAML config.")
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / DEFAULT_WORKSPACE),
        help="Geometry explorer workspace root.",
    )
    parser.add_argument(
        "--view-config",
        default=str(DEFAULT_VIEW_CONFIG),
        help="Projection view config used when diagnostics.umap is true.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)
    workspace = Path(args.workspace).expanduser()
    plan = _build_plan(config)
    if args.dry_run:
        print("Synthetic factor benchmark plan")
        for item in plan:
            print(
                "  - "
                f"{item['object']} | {item['factor'].name} | "
                f"{item['render_mode']} | n={item['n_samples']:,}"
            )
        return
    results = [
        _generate_dataset(
            item,
            config=config,
            workspace=workspace,
            config_path=config_path,
            view_config=Path(args.view_config).expanduser(),
        )
        for item in plan
    ]
    print("Generated synthetic factor benchmarks:")
    for result in results:
        print(f"  - {result['variant_id']} ({result['rows']:,} rows): {result['output_dir']}")


def _build_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    objects = list(config.get("objects", ("cube",)))
    factor_specs = list(config.get("factor_spaces", ("AzimuthCircle",)))
    render_modes = list(config.get("render_modes", ("colored",)))
    n_samples = int(config.get("n_samples", 1441))
    seed = int(config.get("seed", 42))
    plan = []
    for object_index, object_name in enumerate(objects):
        for factor_index, factor_spec in enumerate(factor_specs):
            factor = _factor_from_spec(factor_spec)
            for render_index, render_mode in enumerate(render_modes):
                plan.append(
                    {
                        "object": str(object_name),
                        "factor": factor,
                        "render_mode": str(render_mode),
                        "n_samples": n_samples,
                        "seed": seed
                        + object_index * 10_000
                        + factor_index * 100
                        + render_index,
                    }
                )
    return plan


def _generate_dataset(
    item: dict[str, Any],
    *,
    config: dict[str, Any],
    workspace: Path,
    config_path: Path,
    view_config: Path,
) -> dict[str, Any]:
    registry = GeometryRegistry(workspace)
    factor: LatentFactorSpace = item["factor"]
    object_name = str(item["object"])
    render_mode = str(item["render_mode"])
    n_samples = int(item["n_samples"])
    variant = _variant_name(object_name, factor.name, render_mode, n_samples)
    family = "synthetic_factor"
    variant_id = f"{family}/{variant}"
    output_dir = registry.workspace / "datasets" / family / variant
    image_dir = output_dir / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pngs = bool(config.get("output", {}).get("save_pngs", True))
    if save_pngs:
        image_dir.mkdir(parents=True, exist_ok=True)

    render_config = RenderConfig(
        image_size=int(config.get("image_size", 64)),
        render_mode=render_mode,
        background=config.get("background", "white"),
        antialias=bool(config.get("antialias", True)),
        normalize_pixels=True,
        object_config=config.get("object_config", {}),
        camera_config=config.get("camera_config", {}),
        light_config=config.get("light_config", {}),
    )
    render_map = RenderMap(factor, object_name=object_name, config=render_config)
    sample = factor.sample(n_samples, seed=int(item["seed"]))
    values = sample_values(sample)
    vectors = []
    rows = []
    for row_id, z_value in enumerate(values):
        image = render_map.render(z_value)
        vectors.append(image.reshape(-1).astype(np.float32))
        image_path = ""
        if save_pngs:
            path = image_dir / f"{row_id:05d}.png"
            _save_float_image(image, path)
            image_path = str(path.resolve())
        rows.append(
            _metadata_row(
                row_id=row_id,
                z_value=z_value,
                render_map=render_map,
                factor=factor,
                sample_metadata=sample.metadata,
                image_path=image_path,
                family=family,
                variant=variant,
                variant_id=variant_id,
                experiment=str(config.get("experiment", "synthetic_factor_benchmark")),
            )
        )
    vector_array = np.asarray(vectors, dtype=np.float32)
    metadata = pd.DataFrame(rows)
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    dataset_path = write_parquet(metadata, output_dir / "dataset_index.parquet")
    data_path = output_dir / "data.npy"
    labels_path = output_dir / "labels.npy"
    np.save(data_path, vector_array)
    np.save(labels_path, labels)
    save_config(config, output_dir / "config_used.yaml")

    diagnostics = dict(config.get("diagnostics", {}))
    summary = {
        "dataset_name": variant,
        "variant_id": variant_id,
        "object_name": object_name,
        "factor_space": factor.name,
        "factor_names": list(factor.factor_names),
        "factor_dims": list(factor.factor_dims),
        "true_latent_dim": int(factor.dim),
        "render_mode": render_mode,
        "n_samples": n_samples,
        "image_size": int(render_config.image_size),
    }
    if diagnostics.get("pullback_metric", False):
        pullback = analyze_pullback_metrics(
            render_map,
            factor,
            values,
            eps=float(diagnostics.get("pullback_eps", 1.0e-3)),
            max_points=int(diagnostics.get("pullback_metric_points", 256)),
            seed=int(item["seed"]),
            output_path=output_dir / "diagnostics" / "pullback_metric.json",
        )
        summary["pullback_metric"] = _json_ready(asdict(pullback))
    if diagnostics.get("product_structure", False) and isinstance(factor, ProductFactorSpace):
        product = analyze_product_structure(
            render_map,
            factor,
            values,
            eps=float(diagnostics.get("pullback_eps", 1.0e-3)),
            max_points=int(diagnostics.get("pullback_metric_points", 256)),
            seed=int(item["seed"]),
            output_path=output_dir / "diagnostics" / "product_structure.json",
        )
        summary["product_structure"] = _json_ready(asdict(product))
    if diagnostics.get("latent_pixel", False):
        latent_pixel = analyze_latent_pixel_diagnostics(
            render_map,
            factor,
            values,
            max_samples=int(diagnostics.get("knn_samples", 2000)),
            pair_count=int(diagnostics.get("latent_pixel_distance_pairs", 100_000)),
            ks=tuple(int(value) for value in diagnostics.get("knn_k", (15,))),
            seed=int(item["seed"]),
        )
        summary["latent_pixel_diagnostics"] = _json_ready(asdict(latent_pixel))
    write_json(summary, output_dir / "benchmark_summary.json")

    label_counts = metadata["label"].astype(str).value_counts().sort_index().to_dict()
    label_counts = {str(key): int(value) for key, value in label_counts.items()}
    registry.register_dataset_variant(
        variant_id=variant_id,
        family=family,
        variant=variant,
        base="analytic",
        split=str(config.get("experiment", "synthetic_factor_benchmark")),
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=config_path,
        row_count=len(metadata),
        label_counts=label_counts,
        image_shape=(render_config.image_size, render_config.image_size, 3),
        value_range=(0.0, 1.0),
    )
    if diagnostics.get("umap", False):
        build_projection_view(
            variant_id=variant_id,
            config_path=view_config,
            workspace=workspace,
            project_root=PROJECT_ROOT,
        )
    return {"variant_id": variant_id, "rows": len(metadata), "output_dir": output_dir}


def _metadata_row(
    *,
    row_id: int,
    z_value: Any,
    render_map: RenderMap,
    factor: LatentFactorSpace,
    sample_metadata: dict[str, Any],
    image_path: str,
    family: str,
    variant: str,
    variant_id: str,
    experiment: str,
) -> dict[str, Any]:
    bins = factor.bins(z_value)
    label = str(bins.get("label", factor.name))
    label_id = _label_id(bins.get("label_id", row_id))
    row = {
        "row_id": int(row_id),
        "image_path": image_path,
        "dataset": family,
        "split": experiment,
        "label": label,
        "label_id": label_id,
        "family": family,
        "variant_id": variant_id,
        "variant": variant,
        "base_variant": "analytic",
        "prompt_id": render_map.object_name,
        "prompt": f"{render_map.object_name} under {factor.name} sample {row_id}",
        "tags": ["synthetic", "factor_geometry", render_map.object_name, factor.name],
        "source_index": int(row_id),
        "sample_type": "synthetic_factor_render",
        "object_name": render_map.object_name,
        "object_kind": render_map.object_spec.kind,
        "factor_space": factor.name,
        "factor_names": list(factor.factor_names),
        "factor_dims": list(factor.factor_dims),
        "true_latent_dim": int(factor.dim),
        "render_mode": render_map.render_mode,
        "latent_value_json": json.dumps(_json_ready(z_value), sort_keys=True),
    }
    row.update(render_map.coordinates(z_value))
    row.update({key: _column_value(values, row_id) for key, values in sample_metadata.items()})
    return row


def _factor_from_spec(spec: Any) -> LatentFactorSpace:
    if isinstance(spec, str):
        name = spec
        values: dict[str, Any] = {}
    elif isinstance(spec, dict) and len(spec) == 1:
        name, values = next(iter(spec.items()))
        values = dict(values or {})
    else:
        raise ConfigError(f"Invalid factor-space spec: {spec!r}")
    if name == "AzimuthCircle":
        return AzimuthCircle()
    if name == "LookAtViewSphere":
        return LookAtViewSphere()
    if name == "LightingDirectionSphere":
        return LightingDirectionSphere()
    if name == "LightLogEnergyInterval":
        return LightLogEnergyInterval(bounds=tuple(values.get("bounds", (-0.8, 0.8))))
    if name == "AmbientLightInterval":
        return AmbientLightInterval(bounds=tuple(values.get("bounds", (0.05, 0.75))))
    if name == "DiffuseLightInterval":
        return DiffuseLightInterval(bounds=tuple(values.get("bounds", (0.05, 1.10))))
    if name == "ImageLogExposureInterval":
        return ImageLogExposureInterval(
            bounds=tuple(values.get("bounds", (-0.6, 0.4)))
        )
    if name == "BoundedTranslation2D":
        return BoundedTranslation(dim=2, bounds=tuple(values.get("bounds", (-0.5, 0.5))))
    if name == "BoundedTranslation3D":
        return BoundedTranslation(dim=3, bounds=tuple(values.get("bounds", (-0.5, 0.5))))
    if name == "CameraLocalTranslation":
        return CameraLocalTranslation(bounds=tuple(values.get("bounds", (-0.5, 0.5))))
    if name == "CameraRollInterval":
        return CameraRollInterval(bounds=tuple(values.get("bounds", (-0.5, 0.5))))
    if name == "CameraRadiusInterval":
        return CameraRadiusInterval(bounds=tuple(values.get("bounds", (2.0, 8.0))))
    if name == "CameraDepthTranslationInterval":
        return CameraDepthTranslationInterval(
            bounds=tuple(values.get("bounds", (-1.5, 2.5)))
        )
    if name == "ZoomInterval":
        return ZoomInterval(bounds=tuple(values.get("bounds", (40.0, 100.0))))
    if name == "CameraLogFocalScaleInterval":
        return CameraLogFocalScaleInterval(
            bounds=tuple(values.get("bounds", (-0.4, 0.4)))
        )
    if name == "CameraLogAspectRatioInterval":
        return CameraLogAspectRatioInterval(
            bounds=tuple(values.get("bounds", (-0.25, 0.25)))
        )
    if name == "CameraPrincipalPointOffset":
        return CameraPrincipalPointOffset(bounds=tuple(values.get("bounds", (-8.0, 8.0))))
    if name == "CameraSkewInterval":
        return CameraSkewInterval(bounds=tuple(values.get("bounds", (-0.15, 0.15))))
    if name == "CameraSE3Factor":
        return CameraSE3Factor(
            roll_bounds=tuple(values.get("roll_bounds", (-0.5, 0.5))),
            translation_bounds=tuple(values.get("translation_bounds", (-0.5, 0.5))),
            name=str(values.get("name", "camera_se3")),
        )
    if name == "CameraIntrinsicsFactor":
        return CameraIntrinsicsFactor(
            focal_log_bounds=tuple(values.get("focal_log_bounds", (-0.4, 0.4))),
            aspect_log_bounds=tuple(values.get("aspect_log_bounds", (-0.25, 0.25))),
            principal_point_bounds=tuple(values.get("principal_point_bounds", (-8.0, 8.0))),
            skew_bounds=tuple(values.get("skew_bounds", (-0.15, 0.15))),
            name=str(values.get("name", "camera_intrinsics_k")),
        )
    if name == "FullCameraFactor":
        return FullCameraFactor(
            roll_bounds=tuple(values.get("roll_bounds", (-0.5, 0.5))),
            translation_bounds=tuple(values.get("translation_bounds", (-0.5, 0.5))),
            focal_log_bounds=tuple(values.get("focal_log_bounds", (-0.4, 0.4))),
            aspect_log_bounds=tuple(values.get("aspect_log_bounds", (-0.25, 0.25))),
            principal_point_bounds=tuple(values.get("principal_point_bounds", (-8.0, 8.0))),
            skew_bounds=tuple(values.get("skew_bounds", (-0.15, 0.15))),
            name=str(values.get("name", "full_camera")),
        )
    if name == "IlluminationFactor":
        return IlluminationFactor(
            energy_log_bounds=tuple(values.get("energy_log_bounds", (-0.8, 0.8))),
            ambient_bounds=tuple(values.get("ambient_bounds", (0.05, 0.75))),
            diffuse_bounds=tuple(values.get("diffuse_bounds", (0.05, 1.10))),
            name=str(values.get("name", "illumination")),
        )
    if name == "PhotometryFactor":
        return PhotometryFactor(
            energy_log_bounds=tuple(values.get("energy_log_bounds", (-0.8, 0.8))),
            ambient_bounds=tuple(values.get("ambient_bounds", (0.05, 0.75))),
            diffuse_bounds=tuple(values.get("diffuse_bounds", (0.05, 1.10))),
            exposure_log_bounds=tuple(values.get("exposure_log_bounds", (-0.6, 0.4))),
            name=str(values.get("name", "photometry")),
        )
    if name == "FullAppearanceFactor":
        return FullAppearanceFactor(
            energy_log_bounds=tuple(values.get("energy_log_bounds", (-0.8, 0.8))),
            ambient_bounds=tuple(values.get("ambient_bounds", (0.05, 0.75))),
            diffuse_bounds=tuple(values.get("diffuse_bounds", (0.05, 1.10))),
            exposure_log_bounds=tuple(values.get("exposure_log_bounds", (-0.6, 0.4))),
            name=str(values.get("name", "full_appearance")),
        )
    if name == "OrientationSO3":
        return OrientationSO3()
    if name == "ProductFactorSpace":
        return ProductFactorSpace(
            [_factor_from_spec(item) for item in values.get("factors", [])],
            name=values.get("name"),
        )
    raise ConfigError(f"Unsupported factor-space spec: {name}.")


def _variant_name(
    object_name: str,
    factor_name: str,
    render_mode: str,
    n_samples: int,
) -> str:
    label = f"{object_name}_{factor_name}_{render_mode}_{_sample_count_label(n_samples)}"
    return re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_").lower()


def _sample_count_label(n_samples: int) -> str:
    if n_samples % 1000 == 0:
        return f"{n_samples // 1000}k"
    return str(n_samples)


def _save_float_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _column_value(values: Any, row_id: int) -> Any:
    if isinstance(values, list | tuple):
        return values[row_id]
    array = np.asarray(values)
    value = array[row_id]
    if np.isscalar(value) or getattr(value, "shape", ()) == ():
        return value.item() if hasattr(value, "item") else value
    return _json_ready(value)


def _label_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return abs(hash(str(value))) % 1_000_000


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    main()
