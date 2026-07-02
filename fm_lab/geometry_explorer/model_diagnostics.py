"""Model-dependent diagnostics for registered geometry explorer datasets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from fm_lab.diagnostics.diffusion_lid import flipd_dimension, normal_bundle_dimension
from fm_lab.diagnostics.fm_lid import (
    FMFLIPDEstimator,
    FMJacobianSpectrumEstimator,
    GaussianFMSchedule,
)
from fm_lab.experiments.factory import build_model, build_path, build_source, resolve_device
from fm_lab.geometry_explorer.bundles import build_and_register_projection_payload_index
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.variants import load_variant_bundle
from fm_lab.image_diagnostics.save_utils import read_parquet, write_parquet
from fm_lab.solvers import EulerSolver, HeunSolver, MidpointSolver, RK4Solver
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import ConfigError, load_config
from fm_lab.utils.logging import write_json

MODEL_DIAGNOSTIC_ESTIMATORS = {
    "fm_jacobian",
    "fm_flipd",
    "diffusion_normal_bundle",
    "diffusion_flipd",
}
FM_ESTIMATORS = {"fm_jacobian", "fm_flipd"}
DIFFUSION_ESTIMATORS = {"diffusion_normal_bundle", "diffusion_flipd"}


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
    num_trace_samples: int | None = 1,
    num_perturbations: int = 64,
    batch_size: int = 64,
    fm_schedule: str = "auto",
    diffusion_sigmas: tuple[float, ...] | None = None,
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

    estimators = tuple(dict.fromkeys(estimators))
    if not estimators:
        raise ConfigError("At least one model diagnostic estimator is required.")
    unsupported = set(estimators) - MODEL_DIAGNOSTIC_ESTIMATORS
    if unsupported:
        raise ConfigError(
            "Unsupported model diagnostic estimator(s): "
            f"{', '.join(sorted(unsupported))}. Supported: "
            f"{', '.join(sorted(MODEL_DIAGNOSTIC_ESTIMATORS))}"
        )
    if batch_size < 1:
        raise ConfigError("--batch-size must be positive.")
    if num_trace_samples is not None and num_trace_samples < 1:
        raise ConfigError("--num-trace-samples must be positive, or 0 for exact divergence.")
    if num_perturbations < 1:
        raise ConfigError("--num-perturbations must be positive.")

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
    _validate_checkpoint_estimator_match(config, estimators)
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
        raise ConfigError("Model diagnostics do not support source-label models yet.")

    torch.manual_seed(sample_seed)
    diagnostic_root = (
        registry.workspace / "model_runs" / family / variant_name / run_id / "model_diagnostics"
    )
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    local_parts = [local]
    local_paths: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    resolved_diffusion_sigmas: tuple[float, ...] | None = None

    if "fm_jacobian" in estimators:
        estimator_dir = diagnostic_root / "fm_jacobian"
        estimator_dir.mkdir(parents=True, exist_ok=True)
        fm_local, spectra = _compute_fm_jacobian(
            model=model,
            values=input_vectors,
            local=selected_metadata[["row_id"]].copy(),
            t_values=t_values,
            eps=eps,
            num_directions=num_directions,
            threshold=threshold,
            nfe=nfe,
            solver=solver,
            device=torch_device,
        )
        local_paths["fm_jacobian"] = str(
            write_parquet(fm_local, estimator_dir / "local_fm_jacobian.parquet")
        )
        spectra_path = estimator_dir / "spectra_fm_jacobian.npz"
        np.savez_compressed(
            spectra_path,
            row_id=fm_local["row_id"].to_numpy(dtype=np.int64),
            t_values=np.asarray(t_values, dtype=np.float32),
            singular_values=spectra,
        )
        artifact_paths["fm_jacobian_spectra"] = str(spectra_path)
        local_parts.append(fm_local.drop(columns=["row_id"]))

    if "fm_flipd" in estimators:
        estimator_dir = diagnostic_root / "fm_flipd"
        estimator_dir.mkdir(parents=True, exist_ok=True)
        fm_flipd_local = _compute_fm_flipd(
            model=model,
            values=input_vectors,
            local=selected_metadata[["row_id"]].copy(),
            t_values=t_values,
            num_trace_samples=num_trace_samples,
            schedule=_resolved_fm_schedule(config, fm_schedule),
            batch_size=batch_size,
            device=torch_device,
        )
        local_paths["fm_flipd"] = str(
            write_parquet(fm_flipd_local, estimator_dir / "local_fm_flipd.parquet")
        )
        local_parts.append(fm_flipd_local.drop(columns=["row_id"]))

    if "diffusion_normal_bundle" in estimators or "diffusion_flipd" in estimators:
        score_model = _DiffusionScoreModel(model, config)
        sigma_values = _resolved_diffusion_sigmas(
            config,
            t_values=t_values,
            diffusion_sigmas=diffusion_sigmas,
            device=torch_device,
        )
        resolved_diffusion_sigmas = sigma_values
        if "diffusion_normal_bundle" in estimators:
            estimator_dir = diagnostic_root / "diffusion_normal_bundle"
            estimator_dir.mkdir(parents=True, exist_ok=True)
            normal_local, normal_spectra = _compute_diffusion_normal_bundle(
                score_model=score_model,
                values=input_vectors,
                local=selected_metadata[["row_id"]].copy(),
                t_values=t_values,
                sigma_values=sigma_values,
                num_perturbations=num_perturbations,
                threshold=threshold,
                batch_size=batch_size,
                device=torch_device,
            )
            local_paths["diffusion_normal_bundle"] = str(
                write_parquet(
                    normal_local,
                    estimator_dir / "local_diffusion_normal_bundle.parquet",
                )
            )
            spectra_path = estimator_dir / "spectra_diffusion_normal_bundle.npz"
            np.savez_compressed(
                spectra_path,
                row_id=normal_local["row_id"].to_numpy(dtype=np.int64),
                t_values=np.asarray(t_values, dtype=np.float32),
                sigma_values=np.asarray(sigma_values, dtype=np.float32),
                singular_values=normal_spectra,
            )
            artifact_paths["diffusion_normal_bundle_spectra"] = str(spectra_path)
            local_parts.append(normal_local.drop(columns=["row_id"]))

        if "diffusion_flipd" in estimators:
            estimator_dir = diagnostic_root / "diffusion_flipd"
            estimator_dir.mkdir(parents=True, exist_ok=True)
            diffusion_flipd_local = _compute_diffusion_flipd(
                score_model=score_model,
                values=input_vectors,
                local=selected_metadata[["row_id"]].copy(),
                t_values=t_values,
                sigma_values=sigma_values,
                num_trace_samples=num_trace_samples,
                batch_size=batch_size,
                device=torch_device,
            )
            local_paths["diffusion_flipd"] = str(
                write_parquet(
                    diffusion_flipd_local,
                    estimator_dir / "local_diffusion_flipd.parquet",
                )
            )
            local_parts.append(diffusion_flipd_local.drop(columns=["row_id"]))

    local = pd.concat([part.reset_index(drop=True) for part in local_parts], axis=1)
    local_path = write_parquet(local, diagnostic_root / "local_model_diagnostics.parquet")
    group = _group_summary(
        selected_metadata.merge(local, on="row_id", how="left", validate="one_to_one"),
        metric_columns=[column for column in local.columns if column != "row_id"],
        feature_space=f"model_diagnostics:{run_id}:{'+'.join(estimators)}",
    )
    group_path = diagnostic_root / "group_id_model_diagnostics.csv"
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
                estimator_slug=_estimator_slug(estimators),
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
        "num_trace_samples": num_trace_samples,
        "num_perturbations": num_perturbations,
        "batch_size": batch_size,
        "fm_schedule": _resolved_fm_schedule(config, fm_schedule)
        if "fm_flipd" in estimators
        else None,
        "diffusion_sigmas": (
            list(resolved_diffusion_sigmas)
            if resolved_diffusion_sigmas is not None
            else None
        ),
        "nfe": nfe,
        "solver": solver,
        "rows_total": int(len(metadata)),
        "rows_computed": int(len(local)),
        "normalize": _resolved_normalize(config, normalize),
        "local_path": str(local_path),
        "local_paths": local_paths,
        "artifact_paths": artifact_paths,
        "group_id_path": str(group_path),
        "merged_views": merged_views,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(manifest, diagnostic_root / "manifest.json")
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


def _compute_fm_flipd(
    *,
    model: torch.nn.Module,
    values: np.ndarray,
    local: pd.DataFrame,
    t_values: tuple[float, ...],
    num_trace_samples: int | None,
    schedule: str,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    estimator = FMFLIPDEstimator(
        model=model,
        path_schedule=GaussianFMSchedule(schedule),
        t_values=t_values,
        num_trace_samples=num_trace_samples,
        device=device,
    )
    rows: list[dict[str, float]] = []
    for start in tqdm(range(0, len(values), batch_size), desc="fm-flipd", dynamic_ncols=True):
        batch = torch.as_tensor(
            values[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        estimate = estimator.estimate_batch(batch)
        for batch_index in range(batch.shape[0]):
            row: dict[str, float] = {}
            for time_index, t_value in enumerate(t_values):
                suffix = _time_suffix(t_value)
                row[f"fm_flipd_lid_{suffix}"] = float(
                    estimate.lid[time_index, batch_index].detach().cpu()
                )
                row[f"fm_flipd_divergence_{suffix}"] = float(
                    estimate.divergence[time_index, batch_index].detach().cpu()
                )
                row[f"fm_flipd_score_norm_{suffix}"] = float(
                    estimate.recovered_score_norm[time_index, batch_index].detach().cpu()
                )
            rows.append(row)
    return pd.concat([local.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _compute_diffusion_normal_bundle(
    *,
    score_model: torch.nn.Module,
    values: np.ndarray,
    local: pd.DataFrame,
    t_values: tuple[float, ...],
    sigma_values: tuple[float, ...],
    num_perturbations: int,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, float]] = []
    spectra_by_time: list[list[np.ndarray]] = [[] for _ in t_values]
    for start in tqdm(
        range(0, len(values), batch_size),
        desc="diffusion-normal-bundle",
        dynamic_ncols=True,
    ):
        batch = torch.as_tensor(
            values[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        batch_rows = [{} for _ in range(batch.shape[0])]
        for time_index, (t_value, sigma_value) in enumerate(zip(t_values, sigma_values)):
            query = _diffusion_query_batch(score_model, batch, t_value)
            estimate = normal_bundle_dimension(
                score_model,
                query,
                sigma=sigma_value,
                t=t_value,
                n_perturbations=num_perturbations,
                rank_threshold=threshold,
            )
            suffix = _time_suffix(t_value)
            lid = estimate.intrinsic_dimension.detach().cpu().numpy()
            normal_dim = estimate.normal_dimension.detach().cpu().numpy()
            spectra_by_time[time_index].append(estimate.singular_values.detach().cpu().numpy())
            for batch_index, row in enumerate(batch_rows):
                row[f"diffusion_normal_bundle_lid_{suffix}"] = float(lid[batch_index])
                row[f"diffusion_normal_bundle_normal_dim_{suffix}"] = float(
                    normal_dim[batch_index]
                )
        rows.extend(batch_rows)
    spectra = np.stack([np.concatenate(parts, axis=0) for parts in spectra_by_time], axis=1)
    return pd.concat([local.reset_index(drop=True), pd.DataFrame(rows)], axis=1), spectra


def _compute_diffusion_flipd(
    *,
    score_model: torch.nn.Module,
    values: np.ndarray,
    local: pd.DataFrame,
    t_values: tuple[float, ...],
    sigma_values: tuple[float, ...],
    num_trace_samples: int | None,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for start in tqdm(
        range(0, len(values), batch_size),
        desc="diffusion-flipd",
        dynamic_ncols=True,
    ):
        batch = torch.as_tensor(
            values[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        batch_rows = [{} for _ in range(batch.shape[0])]
        for t_value, sigma_value in zip(t_values, sigma_values):
            query = _diffusion_query_batch(score_model, batch, t_value)
            estimate = flipd_dimension(
                score_model,
                query,
                sigma=sigma_value,
                t=t_value,
                hutchinson_samples=num_trace_samples,
            )
            suffix = _time_suffix(t_value)
            lid = estimate.intrinsic_dimension.detach().cpu().numpy()
            divergence = estimate.divergence.detach().cpu().numpy()
            for batch_index, row in enumerate(batch_rows):
                row[f"diffusion_flipd_lid_{suffix}"] = float(lid[batch_index])
                row[f"diffusion_flipd_divergence_{suffix}"] = float(divergence[batch_index])
        rows.extend(batch_rows)
    return pd.concat([local.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _merge_into_projection_view(
    *,
    registry: GeometryRegistry,
    view_id: str,
    local: pd.DataFrame,
    group: pd.DataFrame,
    run_id: str,
    estimator_slug: str,
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
    group_path = summary_dir / f"group_id_{_safe_name(estimator_slug)}_{_safe_name(run_id)}.csv"
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
        row[f"std_{column}"] = float(values.std())
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


def _validate_checkpoint_estimator_match(config: dict[str, Any], estimators: tuple[str, ...]) -> None:
    requested = set(estimators)
    is_diffusion = _is_diffusion_checkpoint(config)
    if requested & DIFFUSION_ESTIMATORS and not is_diffusion:
        raise ConfigError(
            "Diffusion model diagnostics require a checkpoint trained with "
            "path.name: gaussian_diffusion and objective.name: diffusion."
        )
    if requested & FM_ESTIMATORS and is_diffusion:
        raise ConfigError(
            "FM diagnostics require a flow-matching velocity checkpoint. "
            "Use diffusion_normal_bundle or diffusion_flipd for diffusion checkpoints."
        )
    if "fm_flipd" in requested and not _has_independent_gaussian_coupling(config):
        coupling = str(config.get("coupling", {}).get("name", "independent"))
        raise ConfigError(
            "fm_flipd requires an independent Gaussian source/data coupling so the "
            "velocity-to-score identity is valid. This checkpoint uses "
            f"coupling.name={coupling!r}. For OT-coupled flow matching runs, use "
            "fm_jacobian and omit fm_flipd."
        )


def _is_diffusion_checkpoint(config: dict[str, Any]) -> bool:
    path_name = str(config.get("path", {}).get("name", "")).lower()
    objective_name = str(config.get("objective", {}).get("name", "")).lower()
    return path_name in {"gaussian_diffusion", "diffusion", "stochastic_interpolant"} or (
        objective_name.startswith("diffusion")
    )


def _has_independent_gaussian_coupling(config: dict[str, Any]) -> bool:
    coupling_name = str(config.get("coupling", {}).get("name", "independent")).lower()
    source_name = str(config.get("source", {}).get("name", "gaussian")).lower()
    return coupling_name in {"independent", "product"} and source_name == "gaussian"


def _resolved_fm_schedule(config: dict[str, Any], schedule: str) -> str:
    if schedule != "auto":
        return schedule
    path_name = str(config.get("path", {}).get("name", "linear")).lower()
    if path_name in {"trig", "cosine"}:
        return "trig"
    return "linear"


def _resolved_diffusion_sigmas(
    config: dict[str, Any],
    *,
    t_values: tuple[float, ...],
    diffusion_sigmas: tuple[float, ...] | None,
    device: torch.device,
) -> tuple[float, ...]:
    if diffusion_sigmas is not None:
        if len(diffusion_sigmas) == 1:
            values = tuple(float(diffusion_sigmas[0]) for _ in t_values)
        elif len(diffusion_sigmas) == len(t_values):
            values = tuple(float(value) for value in diffusion_sigmas)
        else:
            raise ConfigError("--diffusion-sigmas must have one value or match --t-values.")
        if any(value <= 0.0 for value in values):
            raise ConfigError("--diffusion-sigmas values must be positive.")
        return values

    path = build_path(config)
    if not hasattr(path, "_schedule"):
        raise ConfigError("Diffusion diagnostics require a Gaussian diffusion path schedule.")
    t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device)
    _, sigma, _, _ = path._schedule(t_tensor)  # noqa: SLF001 - no public schedule API yet.
    sigma_min = float(getattr(path, "sigma_min", 1e-4))
    return tuple(float(value) for value in sigma.clamp_min(sigma_min).detach().cpu())


def _diffusion_query_batch(
    score_model: torch.nn.Module,
    clean_batch: torch.Tensor,
    t_value: float,
) -> torch.Tensor:
    if not hasattr(score_model, "path"):
        return clean_batch
    time = torch.full(
        (clean_batch.shape[0],),
        float(t_value),
        dtype=clean_batch.dtype,
        device=clean_batch.device,
    )
    alpha, _, _, _ = score_model.path._schedule(time)  # noqa: SLF001 - no public schedule API yet.
    shape = (clean_batch.shape[0],) + (1,) * (clean_batch.ndim - 1)
    return alpha.reshape(shape) * clean_batch


class _DiffusionScoreModel(torch.nn.Module):
    """Convert diffusion checkpoint outputs into score estimates."""

    def __init__(self, model: torch.nn.Module, config: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.path = build_path(config)
        if not hasattr(self.path, "_schedule"):
            raise ConfigError("Diffusion diagnostics require path.name: gaussian_diffusion.")
        self.prediction_type = _diffusion_prediction_type(config)
        self.sigma_min = float(getattr(self.path, "sigma_min", 1e-4))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        prediction = self.model(x, t)
        alpha, sigma, alpha_dot, sigma_dot = self._expanded_schedule(t, x)
        sigma = sigma.clamp_min(self.sigma_min)
        if self.prediction_type == "score":
            return prediction
        if self.prediction_type == "epsilon":
            return -prediction / sigma
        if self.prediction_type == "velocity":
            sigma_ratio = sigma_dot / sigma
            coefficient = (alpha_dot - sigma_ratio * alpha).clamp_min(
                torch.finfo(x.dtype).eps
            )
            posterior_mean = (prediction - sigma_ratio * x) / coefficient
            return (alpha * posterior_mean - x) / sigma.square()
        raise ConfigError(f"Unsupported diffusion prediction type: {self.prediction_type}")

    def _expanded_schedule(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time = t.to(device=x.device, dtype=x.dtype)
        if time.ndim == 0:
            time = time.expand(x.shape[0])
        elif time.numel() == 1:
            time = time.reshape(1).expand(x.shape[0])
        elif time.shape[0] != x.shape[0]:
            raise ConfigError("Diffusion diagnostic time tensor must match batch size.")
        alpha, sigma, alpha_dot, sigma_dot = self.path._schedule(time)  # noqa: SLF001
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return (
            alpha.reshape(shape),
            sigma.reshape(shape),
            alpha_dot.reshape(shape),
            sigma_dot.reshape(shape),
        )


def _diffusion_prediction_type(config: dict[str, Any]) -> str:
    objective = config.get("objective", {})
    name = str(objective.get("name", "diffusion")).lower()
    aliases = {
        "diffusion_epsilon": "epsilon",
        "epsilon_prediction": "epsilon",
        "noise_prediction": "epsilon",
        "diffusion_score": "score",
        "score_matching": "score",
        "diffusion_velocity": "velocity",
    }
    prediction = str(objective.get("prediction_type", aliases.get(name, "epsilon"))).lower()
    normalized = {
        "eps": "epsilon",
        "epsilon": "epsilon",
        "noise": "epsilon",
        "score": "score",
        "velocity": "velocity",
        "v": "velocity",
    }.get(prediction)
    if normalized is None:
        raise ConfigError(f"Unsupported diffusion prediction type: {prediction}")
    return normalized


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


def _estimator_slug(estimators: tuple[str, ...]) -> str:
    if len(estimators) == 1:
        return estimators[0]
    return "model_diagnostics"
