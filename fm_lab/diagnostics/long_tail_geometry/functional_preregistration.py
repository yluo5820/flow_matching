"""Immutable protocol for the Observation-0 Probe-A functional calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fm_lab.utils.config import load_config, save_config


def _require_exact_keys(
    values: dict[str, Any],
    expected: set[str],
    *,
    section: str,
) -> None:
    missing = expected - set(values)
    unknown = set(values) - expected
    if missing:
        raise ValueError(
            f"Functional preregistration {section} is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise ValueError(
            f"Functional preregistration {section} has unknown fields: {sorted(unknown)}"
        )


@dataclass(frozen=True)
class FunctionalCalibrationPreregistration:
    """Validated, digest-addressed contract for the Probe-A calibration."""

    schema_version: int
    observation0_preregistration_sha256: str
    required_observation0_status: str
    required_observation0_next_action: str
    observation0_phase: str
    checkpoint_steps: tuple[int, ...]
    primary_checkpoint_step: int
    positive_control_checkpoint_step: int
    probe_view: str
    stratum_id: int
    stratum_bounds: tuple[float, float]
    rank: int
    layers: tuple[str, ...]
    classes: tuple[int, ...]
    fit_positions: tuple[int, ...]
    scale_positions: tuple[int, ...]
    evaluation_positions: tuple[int, ...]
    relative_step_grid: tuple[float, ...]
    target_loss_change_fraction: float
    target_benefit_interval: tuple[float, float]
    max_relative_layer_step: float
    local_linearity_relative_error_max: float
    minimum_projection_fraction: float
    random_controls: int
    random_seed: int
    random_control_quantile: float
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float
    required_seed_repeats: int
    maximum_harm_to_benefit_ratio: float

    def __post_init__(self) -> None:
        for field_name in (
            "checkpoint_steps",
            "layers",
            "classes",
            "fit_positions",
            "scale_positions",
            "evaluation_positions",
            "relative_step_grid",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        object.__setattr__(
            self,
            "stratum_bounds",
            tuple(float(value) for value in self.stratum_bounds),
        )
        object.__setattr__(
            self,
            "target_benefit_interval",
            tuple(float(value) for value in self.target_benefit_interval),
        )
        self._validate()

    @property
    def microbatches_per_cell(self) -> int:
        return len(
            self.fit_positions + self.scale_positions + self.evaluation_positions
        )

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> FunctionalCalibrationPreregistration:
        return cls.from_dict(load_config(path))

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
    ) -> FunctionalCalibrationPreregistration:
        if not isinstance(config, dict):
            raise ValueError("Functional preregistration must be a mapping.")
        _require_exact_keys(
            config,
            {
                "schema_version",
                "observation0",
                "checkpoints",
                "probe",
                "perturbation",
                "controls",
                "decision",
            },
            section="root",
        )
        observation0 = config["observation0"]
        checkpoints = config["checkpoints"]
        probe = config["probe"]
        perturbation = config["perturbation"]
        controls = config["controls"]
        decision = config["decision"]
        sections = {
            "observation0": (
                observation0,
                {
                    "preregistration_sha256",
                    "required_status",
                    "required_next_action",
                    "phase",
                },
            ),
            "checkpoints": (
                checkpoints,
                {"steps", "primary_step", "positive_control_step"},
            ),
            "probe": (
                probe,
                {
                    "view",
                    "stratum_id",
                    "stratum_bounds",
                    "rank",
                    "layers",
                    "classes",
                    "fit_positions",
                    "scale_positions",
                    "evaluation_positions",
                },
            ),
            "perturbation": (
                perturbation,
                {
                    "relative_step_grid",
                    "target_loss_change_fraction",
                    "target_benefit_interval",
                    "max_relative_layer_step",
                    "local_linearity_relative_error_max",
                    "minimum_projection_fraction",
                },
            ),
            "controls": (
                controls,
                {"random_controls", "random_seed", "random_control_quantile"},
            ),
            "decision": (
                decision,
                {
                    "bootstrap_resamples",
                    "bootstrap_seed",
                    "confidence_level",
                    "required_seed_repeats",
                    "maximum_harm_to_benefit_ratio",
                },
            ),
        }
        for name, (values, keys) in sections.items():
            if not isinstance(values, dict):
                raise ValueError(f"Functional preregistration {name} must be a mapping.")
            _require_exact_keys(values, keys, section=name)
        try:
            return cls(
                schema_version=int(config["schema_version"]),
                observation0_preregistration_sha256=str(
                    observation0["preregistration_sha256"]
                ),
                required_observation0_status=str(observation0["required_status"]),
                required_observation0_next_action=str(
                    observation0["required_next_action"]
                ),
                observation0_phase=str(observation0["phase"]),
                checkpoint_steps=tuple(int(value) for value in checkpoints["steps"]),
                primary_checkpoint_step=int(checkpoints["primary_step"]),
                positive_control_checkpoint_step=int(
                    checkpoints["positive_control_step"]
                ),
                probe_view=str(probe["view"]),
                stratum_id=int(probe["stratum_id"]),
                stratum_bounds=tuple(
                    float(value) for value in probe["stratum_bounds"]
                ),
                rank=int(probe["rank"]),
                layers=tuple(str(value) for value in probe["layers"]),
                classes=tuple(int(value) for value in probe["classes"]),
                fit_positions=tuple(int(value) for value in probe["fit_positions"]),
                scale_positions=tuple(
                    int(value) for value in probe["scale_positions"]
                ),
                evaluation_positions=tuple(
                    int(value) for value in probe["evaluation_positions"]
                ),
                relative_step_grid=tuple(
                    float(value) for value in perturbation["relative_step_grid"]
                ),
                target_loss_change_fraction=float(
                    perturbation["target_loss_change_fraction"]
                ),
                target_benefit_interval=tuple(
                    float(value)
                    for value in perturbation["target_benefit_interval"]
                ),
                max_relative_layer_step=float(
                    perturbation["max_relative_layer_step"]
                ),
                local_linearity_relative_error_max=float(
                    perturbation["local_linearity_relative_error_max"]
                ),
                minimum_projection_fraction=float(
                    perturbation["minimum_projection_fraction"]
                ),
                random_controls=int(controls["random_controls"]),
                random_seed=int(controls["random_seed"]),
                random_control_quantile=float(
                    controls["random_control_quantile"]
                ),
                bootstrap_resamples=int(decision["bootstrap_resamples"]),
                bootstrap_seed=int(decision["bootstrap_seed"]),
                confidence_level=float(decision["confidence_level"]),
                required_seed_repeats=int(decision["required_seed_repeats"]),
                maximum_harm_to_benefit_ratio=float(
                    decision["maximum_harm_to_benefit_ratio"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid functional preregistration value: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation0": {
                "preregistration_sha256": self.observation0_preregistration_sha256,
                "required_status": self.required_observation0_status,
                "required_next_action": self.required_observation0_next_action,
                "phase": self.observation0_phase,
            },
            "checkpoints": {
                "steps": list(self.checkpoint_steps),
                "primary_step": self.primary_checkpoint_step,
                "positive_control_step": self.positive_control_checkpoint_step,
            },
            "probe": {
                "view": self.probe_view,
                "stratum_id": self.stratum_id,
                "stratum_bounds": list(self.stratum_bounds),
                "rank": self.rank,
                "layers": list(self.layers),
                "classes": list(self.classes),
                "fit_positions": list(self.fit_positions),
                "scale_positions": list(self.scale_positions),
                "evaluation_positions": list(self.evaluation_positions),
            },
            "perturbation": {
                "relative_step_grid": list(self.relative_step_grid),
                "target_loss_change_fraction": self.target_loss_change_fraction,
                "target_benefit_interval": list(self.target_benefit_interval),
                "max_relative_layer_step": self.max_relative_layer_step,
                "local_linearity_relative_error_max": (
                    self.local_linearity_relative_error_max
                ),
                "minimum_projection_fraction": self.minimum_projection_fraction,
            },
            "controls": {
                "random_controls": self.random_controls,
                "random_seed": self.random_seed,
                "random_control_quantile": self.random_control_quantile,
            },
            "decision": {
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "required_seed_repeats": self.required_seed_repeats,
                "maximum_harm_to_benefit_ratio": (
                    self.maximum_harm_to_benefit_ratio
                ),
            },
        }

    def lock(self, path: str | Path) -> Path:
        output = Path(path)
        if output.exists():
            existing = FunctionalCalibrationPreregistration.load(output)
            if existing.digest != self.digest:
                raise ValueError("Refusing to replace a locked functional preregistration.")
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        save_config(self.to_dict(), output)
        return output

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Functional preregistration schema_version must be 1.")
        digest = self.observation0_preregistration_sha256
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("Observation-0 identity must be a lowercase SHA-256 digest.")
        if self.required_observation0_status != "network_wide_measurable":
            raise ValueError("Calibration requires a network-wide measurable pilot.")
        if (
            self.required_observation0_next_action
            != "calibrate_probe_a_functional_overlap_before_stage1"
        ):
            raise ValueError("Calibration must be the pilot's only allowed next action.")
        if self.observation0_phase != "primary":
            raise ValueError("Functional calibration uses primary Observation-0 artifacts.")
        if self.primary_checkpoint_step == self.positive_control_checkpoint_step:
            raise ValueError("Primary and positive-control checkpoints must be distinct.")
        if set(self.checkpoint_steps) != {
            self.primary_checkpoint_step,
            self.positive_control_checkpoint_step,
        }:
            raise ValueError("Functional calibration requires both calibration checkpoints.")
        if self.probe_view != "a":
            raise ValueError("Functional calibration is Probe-A only.")
        if len(self.stratum_bounds) != 2 or not (
            0 <= self.stratum_bounds[0] < self.stratum_bounds[1] <= 1
        ):
            raise ValueError("Functional calibration stratum bounds are invalid.")
        if self.stratum_id < 0:
            raise ValueError("Functional calibration stratum ID must be non-negative.")
        if self.rank != 1:
            raise ValueError("This calibration is locked to rank-1 directions.")
        if len(self.layers) != 2 or len(set(self.layers)) != 2:
            raise ValueError("Functional calibration requires two adjacent layers.")
        if not all(layer.endswith(".weight") for layer in self.layers):
            raise ValueError("Functional calibration layers must name weights.")
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("Functional calibration classes must be unique and nontrivial.")
        partitions = (
            self.fit_positions,
            self.scale_positions,
            self.evaluation_positions,
        )
        if any(not values for values in partitions):
            raise ValueError("Every Probe-A partition must be non-empty.")
        combined = self.fit_positions + self.scale_positions + self.evaluation_positions
        if len(set(combined)) != len(combined):
            raise ValueError("Probe-A partition positions must not overlap.")
        if tuple(sorted(combined)) != tuple(range(len(combined))):
            raise ValueError("Probe-A partition positions must be consecutive from zero.")
        if len(self.fit_positions) < 2:
            raise ValueError("Direction fitting requires at least two microbatches.")
        grid = self.relative_step_grid
        if not grid or tuple(sorted(set(grid))) != grid or any(value <= 0 for value in grid):
            raise ValueError("Relative perturbation steps must be positive and sorted.")
        if self.target_loss_change_fraction != 0.01:
            raise ValueError("The functional calibration target must remain a 1% change.")
        if len(self.target_benefit_interval) != 2 or not (
            self.target_benefit_interval[0]
            <= self.target_loss_change_fraction
            <= self.target_benefit_interval[1]
        ):
            raise ValueError("The target interval must contain the 1% target.")
        if self.max_relative_layer_step != 0.01:
            raise ValueError("The selected relative layer step must be capped at 1%.")
        if max(grid) > self.max_relative_layer_step:
            raise ValueError("The relative-step grid exceeds its safety cap.")
        if self.local_linearity_relative_error_max != 0.10:
            raise ValueError("The local-linearity tolerance must remain 10%.")
        if not 0 < self.minimum_projection_fraction < 1:
            raise ValueError("The projection-fraction floor must lie in (0, 1).")
        if self.random_controls < 3:
            raise ValueError("At least three matched random controls are required.")
        if self.random_control_quantile != 0.99:
            raise ValueError("The random-control quantile must remain 0.99.")
        if self.bootstrap_resamples < 99:
            raise ValueError("At least 99 bootstrap resamples are required.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("Bootstrap confidence level must lie in (0, 1).")
        if self.required_seed_repeats != 2:
            raise ValueError("The decision requires a two seed repeat.")
        if self.maximum_harm_to_benefit_ratio != 0.5:
            raise ValueError("Non-target harm must remain below half the target benefit.")
