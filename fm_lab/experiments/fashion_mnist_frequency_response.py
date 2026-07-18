"""All-class Fashion-MNIST frequency-response experiment.

The protocol treats every frozen Stage-0 representation/estimator pair as a
separate geometry hypothesis. Ten cyclic frequency mappings give every class
every unique-support rank exactly once.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fm_lab.data.long_tail import frequency_rank_mapping
from fm_lab.utils.config import deep_update, load_config, save_config
from fm_lab.utils.logging import create_run_dir, write_json
from fm_lab.utils.seeding import seed_everything

_NUM_CLASSES = 10
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
_PRIMARY_POLICY = "class_balanced"
_SUPPORTED_POLICIES = frozenset({_PRIMARY_POLICY, "empirical"})
_REQUIRED_REPRESENTATIONS = frozenset({"raw_pca50", "dinov2_pca50"})
_REQUIRED_ESTIMATORS = frozenset(
    {
        "two_nn",
        "mle_lid_k10",
        "mle_lid_k20",
        "participation_ratio",
        "pca_dim_90",
    }
)


@dataclass(frozen=True)
class Stage1Condition:
    condition_id: str
    sampling_policy: str
    balanced: bool
    offset: int
    class_ranks: tuple[int, ...]
    class_counts: tuple[int, ...]


@dataclass(frozen=True)
class FashionFrequencyStage1Config:
    raw: dict[str, Any]
    config_hash: str
    project_root: Path
    base_config_path: Path
    stage0_dir: Path
    data_root: Path
    download: bool
    subset_seed: int
    diagnostic_pool_per_class: int
    imbalance_factor: float
    frequency_multiplier: int
    rotation_offsets: tuple[int, ...]
    sampling_policies: tuple[str, ...]
    experiment_seed: int
    calibration_steps: tuple[int, ...]
    min_class_accuracy: float
    max_macro_fid_relative_improvement: float
    max_accuracy_improvement: float
    samples_per_class: int
    classifier_checkpoint: Path
    classifier_steps: int
    classifier_minimum_accuracy: float
    evaluation_repeats: int
    evaluation_kid_subsets: int
    output_dir: Path
    runs_dir: Path


def load_stage1_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> FashionFrequencyStage1Config:
    """Load and validate the frozen all-class outcome protocol."""

    config_path = Path(path).expanduser().resolve()
    raw = load_config(config_path)
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    experiment = _mapping(raw, "experiment")
    stage0 = _mapping(raw, "stage0")
    data = _mapping(raw, "data")
    design = _mapping(raw, "design")
    training = _mapping(raw, "training")
    calibration = _mapping(training, "calibration")
    evaluation = _mapping(raw, "evaluation")
    output = _mapping(raw, "output")

    offsets = tuple(int(value) for value in design.get("rotation_offsets", range(10)))
    if offsets != tuple(range(_NUM_CLASSES)):
        raise ValueError("design.rotation_offsets must be exactly 0..9 in order.")
    multiplier = int(design.get("frequency_multiplier", 3))
    if math.gcd(multiplier, _NUM_CLASSES) != 1:
        raise ValueError("design.frequency_multiplier must be coprime with ten classes.")
    policies = tuple(str(value) for value in design.get("sampling_policies", []))
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("design.sampling_policies must be a non-empty unique list.")
    if _PRIMARY_POLICY not in policies or not set(policies) <= _SUPPORTED_POLICIES:
        raise ValueError(
            "design.sampling_policies must include class_balanced and may add empirical."
        )
    diagnostic_pool = _positive_int(
        "data.diagnostic_pool_per_class",
        data.get("diagnostic_pool_per_class", 1000),
    )
    if diagnostic_pool != 1000:
        raise ValueError("Stage 1 must reuse the frozen 1,000-image Stage-0 probe pool.")
    imbalance_factor = float(data.get("imbalance_factor", 0.01))
    if imbalance_factor != 0.01:
        raise ValueError("The frozen all-class protocol requires imbalance_factor=0.01.")
    calibration_steps = tuple(int(value) for value in calibration.get("steps", ()))
    if calibration_steps != (2000, 5000, 10000):
        raise ValueError("training.calibration.steps must be [2000, 5000, 10000].")
    samples_per_class = _positive_int(
        "evaluation.samples_per_class",
        evaluation.get("samples_per_class", 1000),
    )
    if samples_per_class != 1000:
        raise ValueError("The frozen evaluation requires 1,000 generated samples per class.")
    base_config_path = _resolve(root, training.get("base_config"))
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Stage-1 base config does not exist: {base_config_path}")

    return FashionFrequencyStage1Config(
        raw=raw,
        config_hash=_json_digest(raw),
        project_root=root,
        base_config_path=base_config_path,
        stage0_dir=_resolve(root, stage0.get("output_dir")),
        data_root=_resolve(root, data.get("root", "data/fashion_mnist")),
        download=bool(data.get("download", False)),
        subset_seed=_nonnegative_int("data.subset_seed", data.get("subset_seed", 0)),
        diagnostic_pool_per_class=diagnostic_pool,
        imbalance_factor=imbalance_factor,
        frequency_multiplier=multiplier,
        rotation_offsets=offsets,
        sampling_policies=policies,
        experiment_seed=_nonnegative_int("experiment.seed", experiment.get("seed", 0)),
        calibration_steps=calibration_steps,
        min_class_accuracy=_probability(
            "training.calibration.min_class_accuracy",
            calibration.get("min_class_accuracy", 0.8),
        ),
        max_macro_fid_relative_improvement=_nonnegative_float(
            "training.calibration.max_macro_fid_relative_improvement",
            calibration.get("max_macro_fid_relative_improvement", 0.1),
        ),
        max_accuracy_improvement=_nonnegative_float(
            "training.calibration.max_accuracy_improvement",
            calibration.get("max_accuracy_improvement", 0.02),
        ),
        samples_per_class=samples_per_class,
        classifier_checkpoint=_resolve(
            root,
            evaluation.get(
                "classifier_checkpoint",
                "artifacts/fashion_mnist_lt_evaluator.pt",
            ),
        ),
        classifier_steps=_positive_int(
            "evaluation.classifier_steps",
            evaluation.get("classifier_steps", 1000),
        ),
        classifier_minimum_accuracy=_probability(
            "evaluation.classifier_minimum_accuracy",
            evaluation.get("classifier_minimum_accuracy", 0.9),
        ),
        evaluation_repeats=_positive_int(
            "evaluation.repeats",
            evaluation.get("repeats", 2),
        ),
        evaluation_kid_subsets=_positive_int(
            "evaluation.kid_subsets",
            evaluation.get("kid_subsets", 20),
        ),
        output_dir=_resolve(
            root,
            output.get(
                "protocol_dir",
                "outputs/fashion_mnist_geometry_frequency/stage1_all_classes",
            ),
        ),
        runs_dir=_resolve(
            root,
            output.get(
                "runs_dir",
                "runs/fashion_mnist_geometry_frequency/stage1_all_classes",
            ),
        ),
    )


def stage1_conditions(config: FashionFrequencyStage1Config) -> tuple[Stage1Condition, ...]:
    """Return the balanced reference and complete cyclic rotations."""

    n_max = 6000 - config.diagnostic_pool_per_class
    balanced_ranks = frequency_rank_mapping(
        _NUM_CLASSES,
        multiplier=config.frequency_multiplier,
        offset=0,
    )
    balanced = Stage1Condition(
        condition_id="balanced_class_balanced",
        sampling_policy=_PRIMARY_POLICY,
        balanced=True,
        offset=0,
        class_ranks=tuple(int(value) for value in balanced_ranks),
        class_counts=(n_max,) * _NUM_CLASSES,
    )
    rotations = []
    for policy in config.sampling_policies:
        for offset in config.rotation_offsets:
            ranks = frequency_rank_mapping(
                _NUM_CLASSES,
                multiplier=config.frequency_multiplier,
                offset=offset,
            )
            counts = tuple(
                max(
                    1,
                    int(
                        n_max
                        * config.imbalance_factor
                        ** (float(rank) / (_NUM_CLASSES - 1.0))
                    ),
                )
                for rank in ranks
            )
            rotations.append(
                Stage1Condition(
                    condition_id=f"{policy}_offset_{offset:02d}",
                    sampling_policy=policy,
                    balanced=False,
                    offset=offset,
                    class_ranks=tuple(int(value) for value in ranks),
                    class_counts=counts,
                )
            )
    return (balanced, *rotations)


def prepare_stage1(
    config: FashionFrequencyStage1Config,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Freeze geometry predictors, conditions, and generated training configs."""

    gate, rank_path = _validated_stage0(config)
    geometry = _frozen_geometry_predictors(rank_path)
    conditions = stage1_conditions(config)
    payload = _protocol_payload(config, gate, geometry, conditions)
    if dry_run:
        return payload | {"dry_run": True}

    existing = _existing_protocol(config)
    if existing is not None:
        return existing | {"reused": True}
    if config.output_dir.exists() or config.output_dir.is_symlink():
        raise FileExistsError(f"Incomplete Stage-1 protocol exists: {config.output_dir}")

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}-", dir=config.output_dir.parent)
    )
    try:
        geometry.to_csv(temporary / "frozen_geometry_predictors.csv", index=False)
        config_dir = temporary / "condition_configs"
        config_dir.mkdir()
        manifest_rows = []
        for condition in conditions:
            condition_config = _condition_config(config, condition)
            config_path = config_dir / f"{condition.condition_id}.yaml"
            save_config(condition_config, config_path)
            manifest_rows.append(
                _condition_row(condition)
                | {
                    "config": str(config_path.relative_to(temporary)),
                    "config_sha256": _sha256_file(config_path),
                }
            )
        write_json(
            {
                "schema_version": 1,
                "config_hash": config.config_hash,
                "conditions": manifest_rows,
            },
            temporary / "condition_manifest.json",
        )
        write_json(payload, temporary / "protocol.json")
        config.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(config.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def _protocol_payload(
    config: FashionFrequencyStage1Config,
    gate: dict[str, Any],
    geometry: pd.DataFrame,
    conditions: tuple[Stage1Condition, ...],
) -> dict[str, Any]:
    predictor_count = len(geometry)
    return {
        "schema_version": 1,
        "stage": "fashion_mnist_frequency_response_stage1",
        "config_hash": config.config_hash,
        "stage0_config_hash": gate["config_hash"],
        "stage0_gate_passed": bool(gate["passed"]),
        "stage0_gate_reasons": gate["reasons"],
        "primary_sampling_policy": _PRIMARY_POLICY,
        "condition_count": len(conditions),
        "rotation_count_per_policy": len(config.rotation_offsets),
        "geometry_predictor_count": predictor_count,
        "geometry_predictors_are_competing": True,
        "geometry_selection_after_outcomes_allowed": False,
        "frequency_effect_is_causal_within_class": True,
        "geometry_effect_is_observational_across_classes": True,
        "no_model_seed_uncertainty_claim": True,
        "no_correlation_p_values": True,
        "evidence_rule": {
            "support_effect": (
                "at least 8/10 classes have positive tail-minus-head FID degradation "
                "and positive head-minus-tail recall degradation"
            ),
            "geometry_effect_per_representation": (
                "at least 4/5 estimators have positive Spearman correlations for both "
                "FID and recall degradation, with median rho >= 0.5 for both"
            ),
        },
        "calibration_steps": list(config.calibration_steps),
        "outcome_rotations_blocked_until_budget_gate": True,
        "estimated_full_primary_training_runs": 1 + len(config.rotation_offsets),
        "artifacts": {
            "geometry": "frozen_geometry_predictors.csv",
            "conditions": "condition_manifest.json",
            "configs": "condition_configs/",
            "budget_gate": "budget_gate.json",
        },
    }


def _condition_config(
    config: FashionFrequencyStage1Config,
    condition: Stage1Condition,
) -> dict[str, Any]:
    base = load_config(config.base_config_path)
    updates = {
        "experiment": {
            "name": f"fashion_geometry_{condition.condition_id}",
            "seed": config.experiment_seed,
            "output_dir": str(config.runs_dir / condition.condition_id),
        },
        "data": {
            "name": "fashion_mnist_lt",
            "root": str(config.data_root),
            "train": True,
            "download": config.download,
            "normalize": "minus_one_one",
            "dequantize": True,
            "imbalance_type": "balanced" if condition.balanced else "exp",
            "imbalance_factor": 1.0 if condition.balanced else config.imbalance_factor,
            "subset_seed": config.subset_seed,
            "sampling_policy": condition.sampling_policy,
            "frequency_mapping": {
                "offset": condition.offset,
                "multiplier": config.frequency_multiplier,
                "diagnostic_pool_per_class": config.diagnostic_pool_per_class,
            },
        },
        "training": {
            "steps": config.calibration_steps[0],
            "checkpoint_every": 0,
            "checkpoint_steps": [],
            "early_stopping": {"enabled": False},
        },
        "sampling": {
            "n_samples": config.samples_per_class * _NUM_CLASSES,
            "n_trajectories": 20,
            "nfe": 64,
            "sample_batch_size": 500,
            "classes": list(range(_NUM_CLASSES)),
            "seed": config.experiment_seed + 1,
            "classifier_free_guidance": {"scale": 1.0},
        },
    }
    return deep_update(base, updates)


def _condition_row(condition: Stage1Condition) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "sampling_policy": condition.sampling_policy,
        "balanced": condition.balanced,
        "offset": condition.offset,
        "class_ranks": list(condition.class_ranks),
        "class_counts": list(condition.class_counts),
    }


def run_calibration(
    config: FashionFrequencyStage1Config,
    *,
    training_steps: int,
    device: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one sequential balanced budget candidate."""

    prepare_stage1(config)
    if training_steps not in config.calibration_steps:
        raise ValueError(
            f"Calibration steps must be one of {list(config.calibration_steps)}."
        )
    index = config.calibration_steps.index(training_steps)
    resume_from: Path | None = None
    if index:
        previous = config.calibration_steps[index - 1]
        previous_dir = _calibration_run_dir(config, previous)
        resume_from = previous_dir / "checkpoint.pt"
        if not _completed_evaluation(previous_dir):
            raise RuntimeError(
                f"Calibration at {previous} steps must complete before {training_steps}."
            )
    condition = stage1_conditions(config)[0]
    run_dir = _calibration_run_dir(config, training_steps)
    condition_config = _load_generated_condition_config(config, condition.condition_id)
    condition_config = deep_update(
        condition_config,
        {
            "experiment": {"output_dir": str(run_dir)},
            "training": {
                "steps": training_steps,
                "resume_from": str(resume_from) if resume_from is not None else None,
            },
        },
    )
    if resume_from is None:
        condition_config["training"].pop("resume_from", None)
    if dry_run:
        return _run_preview(
            config,
            condition,
            run_dir=run_dir,
            training_steps=training_steps,
            resume_from=resume_from,
            device=device,
        )

    result = _run_training_and_evaluation(
        config,
        condition,
        condition_config,
        run_dir=run_dir,
        device=device,
    )
    result["budget_gate"] = evaluate_calibration_gate(config, write=True)
    return result


def evaluate_calibration_gate(
    config: FashionFrequencyStage1Config,
    *,
    write: bool = False,
) -> dict[str, Any]:
    """Apply the preregistered sequential budget rule to completed evaluations."""

    completed: dict[int, dict[str, Any]] = {}
    for steps in config.calibration_steps:
        run_dir = _calibration_run_dir(config, steps)
        report_path = run_dir / "evaluation" / "metrics.json"
        if report_path.is_file():
            completed[steps] = _read_json(report_path)

    candidates = []
    selected: int | None = None
    for earlier, later in zip(
        config.calibration_steps,
        config.calibration_steps[1:],
        strict=False,
    ):
        if earlier not in completed or later not in completed:
            continue
        earlier_summary = _calibration_summary(completed[earlier])
        later_summary = _calibration_summary(completed[later])
        fid_denominator = max(abs(earlier_summary["macro_fid"]), 1.0e-12)
        fid_improvement = (
            earlier_summary["macro_fid"] - later_summary["macro_fid"]
        ) / fid_denominator
        accuracy_improvement = (
            later_summary["requested_accuracy"]
            - earlier_summary["requested_accuracy"]
        )
        qualifies = (
            earlier_summary["minimum_class_accuracy"]
            >= config.min_class_accuracy
            and fid_improvement < config.max_macro_fid_relative_improvement
            and accuracy_improvement < config.max_accuracy_improvement
        )
        candidates.append(
            {
                "earlier_steps": earlier,
                "later_steps": later,
                "earlier": earlier_summary,
                "later": later_summary,
                "macro_fid_relative_improvement": fid_improvement,
                "requested_accuracy_improvement": accuracy_improvement,
                "qualifies": qualifies,
            }
        )
        if qualifies:
            selected = earlier
            break

    terminal = config.calibration_steps[-1]
    if selected is None and terminal in completed:
        terminal_summary = _calibration_summary(completed[terminal])
        if terminal_summary["minimum_class_accuracy"] >= config.min_class_accuracy:
            selected = terminal

    if selected is not None:
        status = "passed"
        next_steps = None
    else:
        missing = [step for step in config.calibration_steps if step not in completed]
        if missing:
            status = "pending"
            next_steps = missing[0]
        else:
            status = "failed"
            next_steps = None
    payload = {
        "schema_version": 1,
        "stage": "fashion_mnist_frequency_response_budget_gate",
        "config_hash": config.config_hash,
        "status": status,
        "selected_steps": selected,
        "next_steps": next_steps,
        "completed_steps": sorted(completed),
        "thresholds": {
            "minimum_class_accuracy": config.min_class_accuracy,
            "max_macro_fid_relative_improvement": (
                config.max_macro_fid_relative_improvement
            ),
            "max_requested_accuracy_improvement": config.max_accuracy_improvement,
        },
        "comparisons": candidates,
    }
    if write:
        path = config.output_dir / "budget_gate.json"
        existing = _read_json(path) if path.is_file() else None
        if existing is not None and existing.get("status") == "passed":
            if existing.get("selected_steps") != selected:
                raise RuntimeError("A passed budget gate is immutable.")
            return existing
        write_json(payload, path)
    return payload


def run_frequency_condition(
    config: FashionFrequencyStage1Config,
    *,
    condition_id: str,
    device: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one rotation at the frozen calibrated budget."""

    prepare_stage1(config)
    gate = evaluate_calibration_gate(config, write=True)
    if gate["status"] != "passed":
        raise RuntimeError(
            "Frequency outcomes are blocked until the balanced budget gate passes. "
            f"Next calibration budget: {gate['next_steps']}."
        )
    conditions = {item.condition_id: item for item in stage1_conditions(config)}
    if condition_id not in conditions or conditions[condition_id].balanced:
        raise ValueError(f"Unknown rotation condition: {condition_id}")
    condition = conditions[condition_id]
    training_steps = int(gate["selected_steps"])
    run_dir = _condition_run_dir(config, condition, training_steps)
    condition_config = _load_generated_condition_config(config, condition.condition_id)
    condition_config = deep_update(
        condition_config,
        {
            "experiment": {"output_dir": str(run_dir)},
            "training": {"steps": training_steps},
        },
    )
    if dry_run:
        return _run_preview(
            config,
            condition,
            run_dir=run_dir,
            training_steps=training_steps,
            resume_from=None,
            device=device,
        )
    return _run_training_and_evaluation(
        config,
        condition,
        condition_config,
        run_dir=run_dir,
        device=device,
    )


def run_all_frequency_conditions(
    config: FashionFrequencyStage1Config,
    *,
    device: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or preview every non-balanced condition in manifest order."""

    results = []
    for condition in stage1_conditions(config):
        if condition.balanced:
            continue
        results.append(
            run_frequency_condition(
                config,
                condition_id=condition.condition_id,
                device=device,
                dry_run=dry_run,
            )
        )
    return {
        "stage": "fashion_mnist_frequency_response_all_conditions",
        "dry_run": dry_run,
        "condition_count": len(results),
        "results": results,
    }


def _run_training_and_evaluation(
    config: FashionFrequencyStage1Config,
    condition: Stage1Condition,
    condition_config: dict[str, Any],
    *,
    run_dir: Path,
    device: str,
) -> dict[str, Any]:
    if _completed_evaluation(run_dir):
        return {
            "condition_id": condition.condition_id,
            "run_dir": str(run_dir),
            "evaluation": str(run_dir / "evaluation" / "metrics.json"),
            "reused": True,
        }
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.is_file():
        if run_dir.exists() or run_dir.is_symlink():
            raise FileExistsError(f"Incomplete Stage-1 training run exists: {run_dir}")
        seed_everything(config.experiment_seed)
        created = create_run_dir(condition_config, root=run_dir, unique=False)
        _train_condition(condition_config, created, device=device)
    _evaluate_condition(
        config,
        condition,
        run_dir=run_dir,
        device=device,
    )
    return {
        "condition_id": condition.condition_id,
        "run_dir": str(run_dir),
        "evaluation": str(run_dir / "evaluation" / "metrics.json"),
        "reused": False,
    }


def _train_condition(config: dict[str, Any], run_dir: Path, *, device: str) -> None:
    from fm_lab.experiments.factory import (
        build_coupling,
        build_model,
        build_path,
        build_solvers,
        build_source,
        build_target,
        resolve_device,
    )
    from fm_lab.training.trainer import (
        train_flow_matching,
        validate_resume_checkpoint_before_model,
    )

    target = build_target(config)
    path = build_path(config)
    validate_resume_checkpoint_before_model(config=config, target=target, path=path)
    source = build_source(config)
    coupling = build_coupling(config)
    model = build_model(config, dim=source.dim)
    solvers = build_solvers(config)
    train_flow_matching(
        config=config,
        run_dir=run_dir,
        target=target,
        source=source,
        coupling=coupling,
        path=path,
        model=model,
        solvers=solvers,
        device=resolve_device(device),
    )


def _evaluate_condition(
    config: FashionFrequencyStage1Config,
    condition: Stage1Condition,
    *,
    run_dir: Path,
    device: str,
) -> None:
    from fm_lab.experiments.run_fashion_mnist_lt_eval import main as evaluate_main

    arguments = [
        "--generated-samples",
        str(run_dir / "samples" / "euler_nfe64.npy"),
        "--generated-labels",
        str(run_dir / "samples" / "generated_labels.npy"),
        "--generative-checkpoint",
        str(run_dir / "checkpoint.pt"),
        "--generation-method",
        "flow_matching",
        "--generative-weights",
        "raw",
        "--data-root",
        str(config.data_root),
        "--classifier-checkpoint",
        str(config.classifier_checkpoint),
        "--classifier-steps",
        str(config.classifier_steps),
        "--minimum-accuracy",
        str(config.classifier_minimum_accuracy),
        "--device",
        device,
        "--samples-per-class",
        str(config.samples_per_class),
        "--overall-samples",
        str(config.samples_per_class * _NUM_CLASSES),
        "--class-counts",
        ",".join(str(value) for value in condition.class_counts),
        "--imbalance-factor",
        str(1.0 if condition.balanced else config.imbalance_factor),
        "--repeats",
        str(config.evaluation_repeats),
        "--kid-subsets",
        str(config.evaluation_kid_subsets),
        "--output-dir",
        str(run_dir / "evaluation"),
    ]
    if config.download:
        arguments.append("--download")
    exit_code = evaluate_main(arguments)
    if exit_code:
        raise RuntimeError(f"Fashion-MNIST evaluation failed with exit code {exit_code}.")


def _run_preview(
    config: FashionFrequencyStage1Config,
    condition: Stage1Condition,
    *,
    run_dir: Path,
    training_steps: int,
    resume_from: Path | None,
    device: str,
) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "sampling_policy": condition.sampling_policy,
        "training_steps": training_steps,
        "class_counts": list(condition.class_counts),
        "class_ranks": list(condition.class_ranks),
        "run_dir": str(run_dir),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "device": device,
        "estimated_runtime": (
            "benchmark with the balanced 2,000-step calibration; hand the full rotation "
            "command to the user if projected above 30 minutes"
        ),
        "dry_run": True,
        "config_hash": config.config_hash,
    }


def analyze_frequency_response(
    config: FashionFrequencyStage1Config,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Analyze complete rotations without selecting a geometry definition post hoc."""

    gate = evaluate_calibration_gate(config, write=False)
    if gate["status"] != "passed":
        raise RuntimeError("The calibrated training budget has not passed.")
    selected_steps = int(gate["selected_steps"])
    geometry = pd.read_csv(config.output_dir / "frozen_geometry_predictors.csv")
    balanced_report = _read_json(
        _calibration_run_dir(config, selected_steps) / "evaluation" / "metrics.json"
    )
    response_rows = []
    conditions = stage1_conditions(config)
    for condition in conditions:
        if condition.balanced:
            report = balanced_report
            run_dir = _calibration_run_dir(config, selected_steps)
        else:
            run_dir = _condition_run_dir(config, condition, selected_steps)
            report_path = run_dir / "evaluation" / "metrics.json"
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"Missing evaluation for {condition.condition_id}: {report_path}"
                )
            report = _read_json(report_path)
        response_rows.extend(
            _response_rows(
                condition,
                report,
                run_dir=run_dir,
                selected_steps=selected_steps,
            )
        )
    responses = pd.DataFrame(response_rows)
    degradations = _class_degradations(responses, config.sampling_policies)
    correlations = _geometry_correlations(geometry, degradations)
    evidence = _evidence_summary(degradations, correlations)
    payload = {
        "schema_version": 1,
        "stage": "fashion_mnist_frequency_response_analysis",
        "config_hash": config.config_hash,
        "selected_training_steps": selected_steps,
        "primary_sampling_policy": _PRIMARY_POLICY,
        "condition_count": len(conditions),
        "complete": True,
        "evidence": evidence,
        "interpretation_boundaries": {
            "frequency": "causal within class under cyclic support assignment",
            "geometry": "observational across natural classes",
            "model_seed": "one discovery seed; evaluation repeats are not model seeds",
            "multiple_geometry_predictors": (
                "all ten frozen predictors reported; none selected by outcome"
            ),
        },
        "artifacts": {
            "response_rows": "analysis/response_rows.csv",
            "class_degradations": "analysis/class_degradations.csv",
            "geometry_correlations": "analysis/geometry_correlations.csv",
        },
    }
    if write:
        analysis_dir = config.output_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        responses.to_csv(analysis_dir / "response_rows.csv", index=False)
        degradations.to_csv(analysis_dir / "class_degradations.csv", index=False)
        correlations.to_csv(analysis_dir / "geometry_correlations.csv", index=False)
        write_json(payload, analysis_dir / "analysis.json")
    return payload


def _response_rows(
    condition: Stage1Condition,
    report: dict[str, Any],
    *,
    run_dir: Path,
    selected_steps: int,
) -> list[dict[str, Any]]:
    metrics = report["metrics"]
    conditional = report["conditional"]["per_class"]
    rows = []
    for class_id in range(_NUM_CLASSES):
        class_key = f"class_{class_id}"
        rows.append(
            {
                "condition_id": condition.condition_id,
                "sampling_policy": condition.sampling_policy,
                "balanced": condition.balanced,
                "offset": condition.offset,
                "class_id": class_id,
                "class_name": _CLASS_NAMES[class_id],
                "frequency_rank": condition.class_ranks[class_id],
                "unique_support": condition.class_counts[class_id],
                "log10_unique_support": math.log10(condition.class_counts[class_id]),
                "classwise_fid": float(metrics["classwise_fid"][class_key]["mean"]),
                "classwise_recall": float(
                    metrics["classwise_recall"][class_key]["mean"]
                ),
                "requested_class_accuracy": float(
                    conditional[class_key]["requested_class_accuracy"]
                ),
                "mean_requested_class_probability": float(
                    conditional[class_key]["mean_requested_class_probability"]
                ),
                "training_steps": selected_steps,
                "run_dir": str(run_dir),
            }
        )
    return rows


def _class_degradations(
    responses: pd.DataFrame,
    policies: tuple[str, ...],
) -> pd.DataFrame:
    balanced = responses[responses["balanced"]].set_index("class_id")
    rows = []
    for policy in policies:
        subset = responses[
            (~responses["balanced"]) & (responses["sampling_policy"] == policy)
        ]
        for class_id in range(_NUM_CLASSES):
            class_rows = subset[subset["class_id"] == class_id].sort_values(
                "unique_support"
            )
            if len(class_rows) != _NUM_CLASSES:
                raise ValueError(
                    f"Class {class_id} under {policy} lacks all ten support ranks."
                )
            tail = class_rows.iloc[0]
            head = class_rows.iloc[-1]
            balanced_row = balanced.loc[class_id]
            log_support = class_rows["log10_unique_support"].to_numpy(dtype=float)
            fid = class_rows["classwise_fid"].to_numpy(dtype=float)
            recall = class_rows["classwise_recall"].to_numpy(dtype=float)
            accuracy = class_rows["requested_class_accuracy"].to_numpy(dtype=float)
            rows.append(
                {
                    "sampling_policy": policy,
                    "class_id": class_id,
                    "class_name": _CLASS_NAMES[class_id],
                    "head_support": int(head["unique_support"]),
                    "tail_support": int(tail["unique_support"]),
                    "fid_tail_minus_head": float(
                        tail["classwise_fid"] - head["classwise_fid"]
                    ),
                    "recall_head_minus_tail": float(
                        head["classwise_recall"] - tail["classwise_recall"]
                    ),
                    "accuracy_head_minus_tail": float(
                        head["requested_class_accuracy"]
                        - tail["requested_class_accuracy"]
                    ),
                    "fid_slope_per_log10_support": float(
                        np.polyfit(log_support, fid, 1)[0]
                    ),
                    "recall_slope_per_log10_support": float(
                        np.polyfit(log_support, recall, 1)[0]
                    ),
                    "accuracy_slope_per_log10_support": float(
                        np.polyfit(log_support, accuracy, 1)[0]
                    ),
                    "head_context_fid_minus_balanced": float(
                        head["classwise_fid"] - balanced_row["classwise_fid"]
                    ),
                    "head_context_recall_minus_balanced": float(
                        head["classwise_recall"] - balanced_row["classwise_recall"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _geometry_correlations(
    geometry: pd.DataFrame,
    degradations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for policy in sorted(degradations["sampling_policy"].unique()):
        outcomes = degradations[degradations["sampling_policy"] == policy].sort_values(
            "class_id"
        )
        for (representation, estimator), predictor in geometry.groupby(
            ["representation", "estimator"],
            sort=True,
        ):
            predictor = predictor.sort_values("class_id")
            if not np.array_equal(
                predictor["class_id"].to_numpy(),
                outcomes["class_id"].to_numpy(),
            ):
                raise ValueError("Frozen geometry predictors do not cover classes 0..9.")
            ranks = predictor["median_percentile_rank"].to_numpy(dtype=float)
            rows.append(
                {
                    "sampling_policy": policy,
                    "representation": representation,
                    "estimator": estimator,
                    "rho_fid_degradation": _spearman(
                        ranks,
                        outcomes["fid_tail_minus_head"].to_numpy(dtype=float),
                    ),
                    "rho_recall_degradation": _spearman(
                        ranks,
                        outcomes["recall_head_minus_tail"].to_numpy(dtype=float),
                    ),
                    "rho_accuracy_degradation": _spearman(
                        ranks,
                        outcomes["accuracy_head_minus_tail"].to_numpy(dtype=float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _evidence_summary(
    degradations: pd.DataFrame,
    correlations: pd.DataFrame,
) -> dict[str, Any]:
    primary = degradations[degradations["sampling_policy"] == _PRIMARY_POLICY]
    positive_fid = int(np.sum(primary["fid_tail_minus_head"] > 0))
    positive_recall = int(np.sum(primary["recall_head_minus_tail"] > 0))
    both = int(
        np.sum(
            (primary["fid_tail_minus_head"] > 0)
            & (primary["recall_head_minus_tail"] > 0)
        )
    )
    support_effect = positive_fid >= 8 and positive_recall >= 8 and both >= 8
    geometry_evidence = {}
    primary_correlations = correlations[
        correlations["sampling_policy"] == _PRIMARY_POLICY
    ]
    for representation, group in primary_correlations.groupby("representation"):
        positive_both = int(
            np.sum(
                (group["rho_fid_degradation"] > 0)
                & (group["rho_recall_degradation"] > 0)
            )
        )
        median_fid = float(group["rho_fid_degradation"].median())
        median_recall = float(group["rho_recall_degradation"].median())
        geometry_evidence[str(representation)] = {
            "positive_both_estimators": positive_both,
            "median_rho_fid_degradation": median_fid,
            "median_rho_recall_degradation": median_recall,
            "strong_support": (
                positive_both >= 4 and median_fid >= 0.5 and median_recall >= 0.5
            ),
        }
    return {
        "support_effect": {
            "positive_fid_classes": positive_fid,
            "positive_recall_classes": positive_recall,
            "positive_both_classes": both,
            "supported": support_effect,
        },
        "geometry_effect_by_representation": geometry_evidence,
    }


def _validated_stage0(
    config: FashionFrequencyStage1Config,
) -> tuple[dict[str, Any], Path]:
    gate_path = config.stage0_dir / "selection_gate.json"
    rank_path = config.stage0_dir / "geometry_percentile_ranks.csv"
    if not gate_path.is_file() or not rank_path.is_file():
        raise FileNotFoundError("Stage 1 requires completed Stage-0 gate and rank artifacts.")
    gate = _read_json(gate_path)
    if gate.get("stage") != "fashion_mnist_geometry_frequency_stage0":
        raise ValueError("Stage-0 gate has the wrong experiment stage.")
    if gate.get("passed") is not False:
        raise ValueError("The all-class fallback is only valid after the trio gate fails.")
    if gate.get("outcome_training_enabled") is not False:
        raise ValueError("Stage-0 gate must explicitly disable trio outcome training.")
    return gate, rank_path


def _frozen_geometry_predictors(rank_path: Path) -> pd.DataFrame:
    ranks = pd.read_csv(rank_path)
    required = {
        "representation",
        "probe_split",
        "subsample",
        "class_id",
        "estimator",
        "percentile_rank",
    }
    if not required <= set(ranks.columns):
        raise ValueError("Stage-0 rank artifact is missing required columns.")
    if set(ranks["representation"]) != _REQUIRED_REPRESENTATIONS:
        raise ValueError("Stage-0 rank artifact has unexpected representations.")
    if set(ranks["estimator"]) != _REQUIRED_ESTIMATORS:
        raise ValueError("Stage-0 rank artifact has unexpected estimators.")
    grouped = (
        ranks.groupby(["representation", "estimator", "class_id"], as_index=False)
        .agg(
            median_percentile_rank=("percentile_rank", "median"),
            percentile_rank_iqr=(
                "percentile_rank",
                lambda values: float(np.percentile(values, 75) - np.percentile(values, 25)),
            ),
            records=("percentile_rank", "size"),
        )
        .sort_values(["representation", "estimator", "class_id"])
        .reset_index(drop=True)
    )
    expected_rows = len(_REQUIRED_REPRESENTATIONS) * len(_REQUIRED_ESTIMATORS) * 10
    if len(grouped) != expected_rows or set(grouped["class_id"]) != set(range(10)):
        raise ValueError("Stage-0 ranks do not provide all ten frozen predictors per class.")
    return grouped


def _calibration_summary(report: dict[str, Any]) -> dict[str, Any]:
    per_class = report["conditional"]["per_class"]
    accuracies = [
        float(per_class[f"class_{class_id}"]["requested_class_accuracy"])
        for class_id in range(_NUM_CLASSES)
    ]
    return {
        "macro_fid": float(report["metrics"]["macro_classwise_fid"]["mean"]),
        "requested_accuracy": float(report["conditional"]["requested_class_accuracy"]),
        "minimum_class_accuracy": float(min(accuracies)),
        "class_accuracies": accuracies,
    }


def _existing_protocol(config: FashionFrequencyStage1Config) -> dict[str, Any] | None:
    protocol_path = config.output_dir / "protocol.json"
    if not protocol_path.is_file():
        return None
    payload = _read_json(protocol_path)
    if payload.get("config_hash") != config.config_hash:
        raise RuntimeError("Existing Stage-1 protocol was created from a different config.")
    required = (
        config.output_dir / "condition_manifest.json",
        config.output_dir / "frozen_geometry_predictors.csv",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("Existing Stage-1 protocol is incomplete.")
    return payload


def _load_generated_condition_config(
    config: FashionFrequencyStage1Config,
    condition_id: str,
) -> dict[str, Any]:
    path = config.output_dir / "condition_configs" / f"{condition_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Generated condition config does not exist: {path}")
    return load_config(path)


def _calibration_run_dir(
    config: FashionFrequencyStage1Config,
    steps: int,
) -> Path:
    return config.runs_dir / "calibration" / f"steps_{steps:08d}"


def _condition_run_dir(
    config: FashionFrequencyStage1Config,
    condition: Stage1Condition,
    steps: int,
) -> Path:
    return config.runs_dir / "rotations" / condition.condition_id / f"steps_{steps:08d}"


def _completed_evaluation(run_dir: Path) -> bool:
    return (
        (run_dir / "checkpoint.pt").is_file()
        and (run_dir / "samples" / "euler_nfe64.npy").is_file()
        and (run_dir / "samples" / "generated_labels.npy").is_file()
        and (run_dir / "evaluation" / "metrics.json").is_file()
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else float("nan")


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _resolve(root: Path, value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Configured path must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _nonnegative_float(name: str, value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return parsed


def _probability(name: str, value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError(f"{name} must be between zero and one.")
    return parsed


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload
