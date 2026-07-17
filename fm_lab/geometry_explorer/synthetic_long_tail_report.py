"""Effect aggregation and living report for the synthetic long-tail study."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fm_lab.image_diagnostics.save_utils import write_parquet


@dataclass(frozen=True)
class EffectEstimate:
    log_count: float
    dimension: float
    interaction: float
    replicate_interactions: tuple[float, ...]
    interval: tuple[float, float] | None = None


def fit_frequency_dimension_effect(
    frame: pd.DataFrame,
    outcome: str = "wasserstein_error",
) -> EffectEstimate:
    """Fit the preregistered object/replicate fixed-effects interaction model."""

    clean = _effect_frame(frame, outcome)
    coefficients, names = _fit_coefficients(clean, outcome, include_replicate=True)
    replicate_interactions = tuple(
        float(_fit_coefficients(group, outcome, include_replicate=False)[0][3])
        for _, group in clean.groupby("replicate", sort=True)
    )
    by_name = dict(zip(names, coefficients, strict=True))
    return EffectEstimate(
        log_count=float(by_name["log10_count"]),
        dimension=float(by_name["dimension"]),
        interaction=float(by_name["log10_count:dimension"]),
        replicate_interactions=replicate_interactions,
    )


def paired_hierarchical_bootstrap(
    frame: pd.DataFrame,
    draws: int,
    seed: int,
    outcome: str = "wasserstein_error",
) -> np.ndarray:
    """Resample replicate bundles, then object-dimension blocks within bundles."""

    clean = _effect_frame(frame, outcome)
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    replicates = np.asarray(sorted(clean["replicate"].unique()))
    if len(replicates) < 2:
        raise ValueError("paired bootstrap requires at least two replicate bundles.")
    object_values = sorted(clean["object_id"].astype(str).unique())
    object_codes = (
        clean["object_id"]
        .astype(str)
        .map({value: index for index, value in enumerate(object_values)})
        .to_numpy()
    )
    log_count = np.log10(clean["count"].to_numpy(dtype=np.float64))
    dimensions = clean["dimension"].to_numpy(dtype=np.float64)
    outcomes = clean[outcome].to_numpy(dtype=np.float64)
    blocks_by_replicate: dict[Any, tuple[np.ndarray, ...]] = {}
    for replicate in replicates:
        source = clean[clean["replicate"] == replicate]
        blocks_by_replicate[replicate] = tuple(
            group.index.to_numpy(dtype=np.int64)
            for _, group in source.groupby(["object_id", "dimension"], sort=True)
        )
    generator = np.random.default_rng(seed)
    interactions = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        for _ in range(100):
            sampled_indices = []
            sampled_replicates = []
            selected_replicates = generator.choice(replicates, size=len(replicates), replace=True)
            for bootstrap_replicate, replicate in enumerate(selected_replicates):
                blocks = blocks_by_replicate[replicate]
                selected_blocks = generator.integers(0, len(blocks), size=len(blocks))
                for block_index in selected_blocks:
                    indices = blocks[int(block_index)]
                    sampled_indices.append(indices)
                    sampled_replicates.append(
                        np.full(len(indices), bootstrap_replicate, dtype=np.int64)
                    )
            indices = np.concatenate(sampled_indices)
            bootstrap_codes = np.concatenate(sampled_replicates)
            columns = [
                np.ones(len(indices)),
                log_count[indices],
                dimensions[indices],
                log_count[indices] * dimensions[indices],
            ]
            columns.extend(
                (object_codes[indices] == code).astype(float)
                for code in range(1, len(object_values))
            )
            columns.extend(
                (bootstrap_codes == code).astype(float) for code in range(1, len(replicates))
            )
            design = np.column_stack(columns)
            if np.linalg.matrix_rank(design) != design.shape[1]:
                continue
            interactions[draw] = float(np.linalg.lstsq(design, outcomes[indices], rcond=None)[0][3])
            break
        else:
            raise ValueError("Unable to draw a full-rank paired bootstrap sample.")
    return interactions


def aggregate_experiment(
    root: str | Path,
    bootstrap_draws: int = 10_000,
    *,
    seed: int = 17_072_026,
) -> dict[str, Any]:
    """Aggregate all discovered class metric files into immutable analysis artifacts."""

    experiment_root = Path(root).expanduser().resolve()
    frame = _load_experiment_frame(experiment_root)
    conditions = _condition_records(experiment_root)
    calibration = _calibration_records(experiment_root)
    effects: dict[str, Any] = {
        "H1": {"status": "not_evaluable", "reason": "no completed class evaluations"},
        "H2": {"status": "not_evaluable", "reason": "no completed geometry evaluations"},
        "H3": {"status": "exploratory"},
        "H4": {"status": "not_evaluable", "reason": "no paired context evaluations"},
    }
    if not frame.empty:
        estimate = fit_frequency_dimension_effect(frame, outcome="wasserstein_error")
        draws = paired_hierarchical_bootstrap(
            frame,
            draws=bootstrap_draws,
            seed=seed,
            outcome="wasserstein_error",
        )
        interval = tuple(float(value) for value in np.percentile(draws, [2.5, 97.5]))
        estimate = EffectEstimate(**{**asdict(estimate), "interval": interval})
        signs_agree = all(value < 0.0 for value in estimate.replicate_interactions)
        supported = signs_agree and interval[1] < 0.0
        effects["H1"] = {
            "status": "supported" if supported else "inconclusive",
            "estimate": asdict(estimate),
            "all_replicate_signs_negative": signs_agree,
            "interval_excludes_zero_in_predicted_direction": interval[1] < 0.0,
        }
        effects["H2"] = _h2_decision(frame)
        effects["H4"] = _balanced_context_effect(frame)
    summary = {
        "schema_version": 1,
        "effects": effects,
        "calibration": calibration,
        "conditions": conditions,
        "rows": int(len(frame)),
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(seed),
    }
    destination = experiment_root / "analysis"
    _publish_analysis(destination, frame, summary)
    return summary


def render_research_report(
    summary: dict[str, Any],
    ledger: dict[str, Any],
    destination: str | Path,
) -> Path:
    """Create or update generated report sections without touching handwritten prose."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else _report_template()
    generated = {
        "calibration": _json_block(summary.get("calibration", {})),
        "design": _condition_table(summary.get("conditions", [])),
        "ledger": _ledger_table(ledger.get("entries", [])),
        "effects": _json_block(summary.get("effects", {})),
    }
    for name, content in generated.items():
        text = _replace_generated(text, name, content)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _fit_coefficients(
    frame: pd.DataFrame,
    outcome: str,
    *,
    include_replicate: bool,
) -> tuple[np.ndarray, list[str]]:
    log_count = np.log10(frame["count"].to_numpy(dtype=np.float64))
    dimension = frame["dimension"].to_numpy(dtype=np.float64)
    columns = [
        np.ones(len(frame)),
        log_count,
        dimension,
        log_count * dimension,
    ]
    names = ["intercept", "log10_count", "dimension", "log10_count:dimension"]
    object_indicators = pd.get_dummies(
        frame["object_id"].astype(str), prefix="object", drop_first=True, dtype=float
    )
    columns.extend(object_indicators[column].to_numpy() for column in object_indicators)
    names.extend(object_indicators.columns.tolist())
    if include_replicate:
        replicate_indicators = pd.get_dummies(
            frame["replicate"].astype(str), prefix="replicate", drop_first=True, dtype=float
        )
        columns.extend(replicate_indicators[column].to_numpy() for column in replicate_indicators)
        names.extend(replicate_indicators.columns.tolist())
    design = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError(
            f"Fixed-effects design must have full column rank; found {rank}/{design.shape[1]}."
        )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, frame[outcome].to_numpy(dtype=np.float64), rcond=None
    )
    return coefficients, names


def _effect_frame(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    required = {"replicate", "object_id", "dimension", "count", outcome}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Effect frame missing columns: {', '.join(missing)}")
    clean = frame[list(required)].copy()
    for column in ("replicate", "dimension", "count", outcome):
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    values = clean[["replicate", "dimension", "count", outcome]].to_numpy(dtype=float)
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("Effect frame numeric columns must be finite.")
    if bool((clean["count"] <= 0).any()):
        raise ValueError("Effect frame counts must be positive.")
    if clean["replicate"].nunique() < 1 or clean["object_id"].nunique() < 2:
        raise ValueError("Effect frame needs replicate and object variation.")
    return clean.reset_index(drop=True)


def _load_experiment_frame(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("factor_metrics_by_class.csv")):
        metadata = _path_metadata(path)
        if metadata is None:
            continue
        replicate, condition_id = metadata
        manifest_path = root / f"replicate_{replicate:02d}" / "conditions" / f"{condition_id}.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        classes = {int(item["class_id"]): item for item in manifest["classes"]}
        metrics = pd.read_csv(path)
        for record in metrics.to_dict(orient="records"):
            class_info = classes[int(record["requested_class"])]
            active = _active_factors(str(class_info["dimension_id"]))
            factor_values = [float(record[f"all_{name}_normalized_wasserstein"]) for name in active]
            rows.append(
                {
                    **record,
                    "replicate": replicate,
                    "condition_id": condition_id,
                    "dimension": int(class_info["true_dimension"]),
                    "count": int(class_info["count"]),
                    "wasserstein_error": float(np.mean(factor_values)),
                }
            )
    return pd.DataFrame(rows)


def _path_metadata(path: Path) -> tuple[int, str] | None:
    replicate = next(
        (
            int(part.removeprefix("replicate_"))
            for part in path.parts
            if part.startswith("replicate_") and part.removeprefix("replicate_").isdigit()
        ),
        None,
    )
    condition = next(
        (part for part in path.parts if part.startswith("g") and "_" in part),
        None,
    )
    return None if replicate is None or condition is None else (replicate, condition)


def _active_factors(dimension_id: str) -> tuple[str, ...]:
    if dimension_id == "low":
        return ("tz",)
    if dimension_id == "medium":
        return ("tx", "ty", "tz")
    if dimension_id == "high":
        return ("tx", "ty", "tz", "azimuth", "elevation")
    raise ValueError(f"Unknown dimension_id: {dimension_id}")


def _condition_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("replicate_*/conditions/*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "replicate": raw.get("replicate"),
                "condition_id": raw.get("condition_id"),
                "geometry_mapping": raw.get("geometry_mapping"),
                "frequency_mapping": raw.get("frequency_mapping"),
            }
        )
    return records


def _calibration_records(root: Path) -> dict[str, Any]:
    records = {}
    for name, relative in {
        "renderer": Path("calibration/renderer/renderer_gate.json"),
        "oracle": Path("calibration/oracle/oracle_gate.json"),
        "metric": Path("calibration/metric_gate.json"),
        "pilot": Path("calibration/pilot_gate.json"),
    }.items():
        path = root / relative
        records[name] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"status": "not_run", "path": str(path)}
        )
    return records


def _h2_decision(frame: pd.DataFrame) -> dict[str, Any]:
    geometry_columns = {"tangent_alignment_loss", "fm_flipd_deficit"}
    available = geometry_columns & set(frame.columns)
    if not available:
        return {"status": "not_evaluable", "reason": "local geometry metrics absent"}
    factor_loss = bool((frame["wasserstein_error"] > 0.05).any())
    geometry_loss = any(bool((frame[column] > 0.0).any()) for column in available)
    explained_by_invalid = bool(
        (
            (frame.get("class_leakage_rate", 0.0) > 0.01)
            | (frame.get("off_renderer_rate", 0.0) > 0.01)
        ).all()
    )
    return {
        "status": "supported"
        if factor_loss and geometry_loss and not explained_by_invalid
        else "inconclusive",
        "factor_loss": factor_loss,
        "geometry_loss": geometry_loss,
        "explained_entirely_by_invalid_mass": explained_by_invalid,
    }


def _balanced_context_effect(frame: pd.DataFrame) -> dict[str, Any]:
    maximum = frame["count"].max()
    selected = frame[frame["count"] == maximum]
    balanced = selected[selected["condition_id"].str.endswith("balanced")]
    long_tail = selected[~selected["condition_id"].str.endswith("balanced")]
    if balanced.empty or long_tail.empty:
        return {"status": "not_evaluable", "reason": "paired 5000-count contexts absent"}
    left = balanced.groupby(["replicate", "object_id", "dimension"])["wasserstein_error"].mean()
    right = long_tail.groupby(["replicate", "object_id", "dimension"])["wasserstein_error"].mean()
    paired = left.to_frame("balanced").join(right.to_frame("long_tail"), how="inner")
    return {
        "status": "exploratory",
        "pairs": int(len(paired)),
        "mean_long_tail_minus_balanced": float((paired["long_tail"] - paired["balanced"]).mean()),
    }


def _publish_analysis(destination: Path, frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Analysis destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".analysis-", dir=destination.parent))
    published = False
    try:
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        frame.to_csv(staging / "class_effects.csv", index=False)
        write_parquet(frame, staging / "class_effects.parquet")
        _effect_figure(frame, staging / "frequency_dimension_interaction.png")
        os.symlink(
            os.path.relpath(staging, destination.parent), destination, target_is_directory=True
        )
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _effect_figure(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    if frame.empty:
        axis.text(0.5, 0.5, "No completed evaluations", ha="center", va="center")
        axis.set_axis_off()
    else:
        grouped = frame.groupby(["dimension", "count"], as_index=False)["wasserstein_error"].mean()
        for dimension, group in grouped.groupby("dimension"):
            axis.plot(
                group["count"], group["wasserstein_error"], marker="o", label=f"d={dimension}"
            )
        axis.set_xscale("log")
        axis.set_xlabel("Class examples")
        axis.set_ylabel("Normalized Wasserstein error")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _report_template() -> str:
    return """# Synthetic Long-Tail Geometry Report

## Research question

When class examples become scarce, does a conditional generator selectively lose valid
directions of variation, especially for higher-dimensional class manifolds?

## Frozen hypotheses

- **H1:** log class frequency and calibrated data dimension interact on factor coverage.
- **H2:** factor loss is accompanied by tangent-alignment or FM-FLIPD deficit before exact copying.
- **H3:** shared directions may be protected by cross-class capacity borrowing (exploratory).
- **H4:** long-tail context may alter an otherwise well-sampled class.

## Frozen design

<!-- GENERATED:design:START -->
Not generated yet.
<!-- GENERATED:design:END -->

## Calibration record

<!-- GENERATED:calibration:START -->
Not generated yet.
<!-- GENERATED:calibration:END -->

## Run ledger

<!-- GENERATED:ledger:START -->
Not generated yet.
<!-- GENERATED:ledger:END -->

## Observations

Record observations here without changing the frozen hypotheses.

<!-- GENERATED:effects:START -->
No effect estimates yet.
<!-- GENERATED:effects:END -->

## Interpretation

Interpret observations only after the generated effect record is available.

## Limitations

This synthetic renderer establishes internal causality, not natural-image generality.

## Next decision

Advance to a natural-image probe only after the preregistered gates and falsification checks.
"""


def _replace_generated(text: str, name: str, content: str) -> str:
    start = f"<!-- GENERATED:{name}:START -->"
    end = f"<!-- GENERATED:{name}:END -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        raise ValueError(f"Report generated section is missing or malformed: {name}")
    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{content.rstrip()}\n{end}{suffix}"


def _json_block(value: Any) -> str:
    return f"```json\n{json.dumps(value, indent=2, sort_keys=True)}\n```"


def _condition_table(conditions: list[dict[str, Any]]) -> str:
    if not conditions:
        return "No frozen condition manifests have been built yet."
    rows = ["| Replicate | Condition | Geometry | Frequency |", "|---:|---|---|---|"]
    rows.extend(
        f"| {item['replicate']} | {item['condition_id']} | {item['geometry_mapping']} | "
        f"{item['frequency_mapping']} |"
        for item in conditions
    )
    return "\n".join(rows)


def _ledger_table(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "No stages have completed yet."
    rows = ["| Stage | Condition | Replicate | Status | Output |", "|---|---|---:|---|---|"]
    rows.extend(
        f"| {item.get('stage', '')} | {item.get('condition', '')} | "
        f"{item.get('replicate', '')} | {item.get('status', '')} | "
        f"{item.get('output_path', '')} |"
        for item in entries
    )
    return "\n".join(rows)
