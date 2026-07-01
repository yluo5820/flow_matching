"""Model-dependent diagnostics for registered geometry explorer datasets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from fm_lab.diagnostics.fm_lid import FMJacobianSpectrumEstimator
from fm_lab.experiments.factory import build_model, build_source, resolve_device
from fm_lab.geometry_explorer.bundles import build_and_register_projection_payload_index
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.variants import load_variant_bundle
from fm_lab.image_diagnostics.save_utils import read_parquet, write_parquet
from fm_lab.solvers import EulerSolver, HeunSolver, MidpointSolver, RK4Solver
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import ConfigError, load_config
from fm_lab.utils.logging import write_json


def build_model_diagnostics(
    *,
    variant_id: str,
    run_dir: str | Path,
    workspace: str | Path = DEFAULT_WORKSPACE,
    estimators: tuple[str, ...] = ("fm_jacobian",),
    t_values: tuple[float, ...] = (0.6, 0.8, 0.9, 0.95),
    eps: float = 1e-2,
    num_directions: int = 64,
    threshold: float = 1e-2,
    nfe: int = 32,
    solver: str = "rk4",
    max_samples: int | None = None,
    sample_seed: int = 0,
    device: str = "auto",
    normalize: str = "auto",
    view_id: str | None = None,
    rebuild_payload: bool = True,
) -> dict[str, Any]:
    """Compute and merge model-dependent diagnostics for one dataset variant."""

    unsupported = set(estimators) - {"fm_jacobian"}
    if unsupported:
        raise ConfigError(
            "Only the fm_jacobian model diagnostic is wired into the explorer "
            f"pipeline for now. Unsupported: {', '.join(sorted(unsupported))}"
        )
    started = time.perf_counter()
    registry = GeometryRegistry(workspace)
    variant = registry.get_dataset_variant(variant_id)
    family = str(variant["family"])
    variant_name = str(variant["variant"])
    source_run_dir = Path(run_dir).expanduser().resolve()
    checkpoint_path = source_run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise ConfigError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config = load_config(source_run_dir / "config.yaml")
    run_id = source_run_dir.name
    registry.register_model_run(
        run_id=run_id,
        run_dir=source_run_dir,
        variant_id=variant_id,
        family=family,
        variant=variant_name,
        config_path=source_run_dir / "config.yaml",
        metrics_path=source_run_dir / "metrics.json",
    )

    dataset = load_variant_bundle(variant_id, workspace=registry.workspace)
    if dataset.vectors is None:
        raise ConfigError(f"Dataset variant {variant_id} has no saved vectors.")
    metadata = dataset.metadata.reset_index(drop=True).copy()
    positions = _selected_positions(len(metadata), max_samples=max_samples, seed=sample_seed)
    input_vectors = _normalize_vectors(
        np.asarray(dataset.vectors, dtype=np.float32)[positions],
        normalize=_resolved_normalize(config, normalize),
        value_range=dataset.value_range or (0.0, 1.0),
    )
    selected_metadata = metadata.iloc[positions].reset_index(drop=True).copy()
    local = selected_metadata[["row_id"]].copy()

    torch_device = resolve_device(device)
    source = build_source(config)
    model = build_model(config, dim=source.dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()
    if bool(getattr(model, "requires_source_label", False)):
        raise ConfigError("FM Jacobian diagnostics do not support source-label models yet.")

    diagnostic_dir = (
        registry.workspace
        / "model_runs"
        / family
        / variant_name
        / run_id
        / "model_diagnostics"
        / "fm_jacobian"
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    local, spectra = _compute_fm_jacobian(
        model=model,
        values=input_vectors,
        local=local,
        t_values=t_values,
        eps=eps,
        num_directions=num_directions,
        threshold=threshold,
        nfe=nfe,
        solver=solver,
        device=torch_device,
    )
    local_path = write_parquet(local, diagnostic_dir / "local_fm_jacobian.parquet")
    spectra_path = diagnostic_dir / "spectra_fm_jacobian.npz"
    np.savez_compressed(
        spectra_path,
        row_id=local["row_id"].to_numpy(dtype=np.int64),
        t_values=np.asarray(t_values, dtype=np.float32),
        singular_values=spectra,
    )
    group = _group_summary(
        selected_metadata.merge(local, on="row_id", how="left", validate="one_to_one"),
        metric_columns=[column for column in local.columns if column != "row_id"],
        feature_space=f"fm_jacobian:{run_id}",
    )
    group_path = diagnostic_dir / "group_id_fm_jacobian.csv"
    group.to_csv(group_path, index=False)
    write_parquet(group, group_path.with_suffix(".parquet"))

    projection_views = _selected_projection_views(registry, variant_id, view_id=view_id)
    merged_views = []
    for projection_view in projection_views:
        merged_views.append(
            _merge_into_projection_view(
                registry=registry,
                view_id=projection_view.view_id,
                local=local,
                group=group,
                run_id=run_id,
                rebuild_payload=rebuild_payload,
            )
        )

    manifest = {
        "variant_id": variant_id,
        "run_id": run_id,
        "run_dir": str(source_run_dir),
        "checkpoint": str(checkpoint_path),
        "estimators": list(estimators),
        "t_values": list(t_values),
        "eps": eps,
        "num_directions": num_directions,
        "threshold": threshold,
        "nfe": nfe,
        "solver": solver,
        "rows_total": int(len(metadata)),
        "rows_computed": int(len(local)),
        "normalize": _resolved_normalize(config, normalize),
        "local_path": str(local_path),
        "spectra_path": str(spectra_path),
        "group_id_path": str(group_path),
        "merged_views": merged_views,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(manifest, diagnostic_dir / "manifest.json")
    return manifest


def _compute_fm_jacobian(
    *,
    model: torch.nn.Module,
    values: np.ndarray,
    local: pd.DataFrame,
    t_values: tuple[float, ...],
    eps: float,
    num_directions: int,
    threshold: float,
    nfe: int,
    solver: str,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    estimator = FMJacobianSpectrumEstimator(
        model=model,
        ode_solver=_solver_from_name(solver),
        t_values=t_values,
        eps=eps,
        num_directions=num_directions,
        threshold=threshold,
        nfe=nfe,
        device=device,
    )
    rows: list[dict[str, float]] = []
    spectra: list[np.ndarray] = []
    for value in tqdm(values, desc="fm-jacobian", dynamic_ncols=True):
        point = torch.as_tensor(value, dtype=torch.float32)
        estimate = estimator.estimate_point(point)
        row: dict[str, float] = {}
        for index, t_value in enumerate(t_values):
            suffix = _time_suffix(t_value)
            row[f"fm_jacobian_participation_rank_{suffix}"] = float(
                estimate.participation_rank[index].detach().cpu()
            )
            row[f"fm_jacobian_entropy_rank_{suffix}"] = float(
                estimate.entropy_rank[index].detach().cpu()
            )
            row[f"fm_jacobian_threshold_rank_{suffix}"] = float(
                estimate.threshold_rank[index].detach().cpu()
            )
        rows.append(row)
        spectra.append(
            np.stack(
                [values.detach().cpu().numpy() for values in estimate.singular_values],
                axis=0,
            )
        )
    metrics = pd.DataFrame(rows)
    return pd.concat([local.reset_index(drop=True), metrics], axis=1), np.stack(spectra)


def _merge_into_projection_view(
    *,
    registry: GeometryRegistry,
    view_id: str,
    local: pd.DataFrame,
    group: pd.DataFrame,
    run_id: str,
    rebuild_payload: bool,
) -> dict[str, Any]:
    row = registry.get_projection_view(view_id)
    explorer_path = registry.resolve(row["explorer_data_path"])
    explorer = read_parquet(explorer_path)
    metric_columns = [column for column in local.columns if column != "row_id"]
    merged = explorer.drop(columns=metric_columns, errors="ignore").merge(
        local,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    write_parquet(merged, explorer_path)

    output_dir = registry.resolve(row["output_dir"])
    summary_dir = (
        output_dir
        / "id_estimation"
        / f"model_diagnostics_{_safe_name(run_id)}"
        / "intrinsic_dimension"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    group_path = summary_dir / f"group_id_fm_jacobian_{_safe_name(run_id)}.csv"
    group.to_csv(group_path, index=False)
    write_parquet(group, group_path.with_suffix(".parquet"))
    if rebuild_payload:
        build_and_register_projection_payload_index(view_id, workspace=registry.workspace)
    return {
        "view_id": view_id,
        "explorer_data_path": str(explorer_path),
        "group_id_path": str(group_path),
        "rows": int(len(merged)),
        "metrics": metric_columns,
        "payload_rebuilt": rebuild_payload,
    }


def _group_summary(
    frame: pd.DataFrame,
    *,
    metric_columns: list[str],
    feature_space: str,
) -> pd.DataFrame:
    rows = [
        _summary_row(
            frame,
            groupby_column="__all__",
            group_value="__all__",
            metric_columns=metric_columns,
            feature_space=feature_space,
        )
    ]
    if "label" in frame:
        for label, group in frame.groupby(frame["label"].astype(str), sort=True):
            rows.append(
                _summary_row(
                    group,
                    groupby_column="label",
                    group_value=str(label),
                    metric_columns=metric_columns,
                    feature_space=feature_space,
                )
            )
    return pd.DataFrame(rows)


def _summary_row(
    frame: pd.DataFrame,
    *,
    groupby_column: str,
    group_value: str,
    metric_columns: list[str],
    feature_space: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "groupby_column": groupby_column,
        "group_value": group_value,
        "n_samples": int(len(frame)),
        "feature_space": feature_space,
    }
    for column in metric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        row[f"mean_{column}"] = float(values.mean())
        row[f"median_{column}"] = float(values.median())
    return row


def _selected_positions(
    row_count: int,
    *,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    if max_samples is None or max_samples >= row_count:
        return np.arange(row_count, dtype=int)
    if max_samples < 1:
        raise ConfigError("--max-samples must be positive.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(row_count, size=max_samples, replace=False)).astype(int)


def _selected_projection_views(
    registry: GeometryRegistry,
    variant_id: str,
    *,
    view_id: str | None,
):
    if view_id is None:
        views = registry.projection_views(variant_id)
    else:
        view = registry.get_projection_view(view_id)
        if view["variant_id"] != variant_id:
            raise ConfigError(f"Projection view {view_id} does not belong to {variant_id}.")
        views = [
            record for record in registry.projection_views(variant_id) if record.view_id == view_id
        ]
    if not views:
        raise ConfigError(f"No projection views are registered for {variant_id}.")
    return views


def _resolved_normalize(config: dict[str, Any], normalize: str) -> str:
    if normalize != "auto":
        return normalize
    return str(config.get("data", {}).get("normalize", "zero_one"))


def _normalize_vectors(
    values: np.ndarray,
    *,
    normalize: str,
    value_range: tuple[float, ...],
) -> np.ndarray:
    normalized = normalize.lower().replace("-", "_")
    low = float(value_range[0]) if len(value_range) >= 1 else 0.0
    high = float(value_range[1]) if len(value_range) >= 2 else 1.0
    scale = high - low
    zero_one = (values - low) / scale if scale > 0 else values
    if normalized in {"zero_one", "01", "unit", "none"}:
        return np.asarray(zero_one, dtype=np.float32)
    if normalized in {"minus_one_one", "_1_1", "centered"}:
        return np.asarray(2.0 * zero_one - 1.0, dtype=np.float32)
    raise ConfigError(f"Unsupported model diagnostic normalization: {normalize}")


def _solver_from_name(name: str):
    normalized = name.lower()
    if normalized == "euler":
        return EulerSolver()
    if normalized == "heun":
        return HeunSolver()
    if normalized == "midpoint":
        return MidpointSolver()
    if normalized == "rk4":
        return RK4Solver()
    raise ConfigError("Model diagnostics support solver values: euler, heun, midpoint, rk4.")


def _time_suffix(value: float) -> str:
    return f"t{int(round(float(value) * 1000)):04d}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
