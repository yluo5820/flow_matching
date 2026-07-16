"""Immutable protocol for the representation-matched functional geometry audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fm_lab.utils.config import load_config, save_config


_OBSERVATION0_DIGEST = (
    "6cd1bcf18692dc947573dbfb0da7b4d98b16fd291ea8d436a32e3e9abec78e24"
)
_FUNCTIONAL_PREREGISTRATION_DIGEST = (
    "f40251443426eab2f24d89cdf359f07615ccda788341d16719c1f0ec40836bdf"
)
_LAYERS = (
    "down2_block.conv2.weight",
    "middle.conv2.weight",
)
_CLASSES = (0, 2, 3, 4, 6, 9)


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
            f"Functional geometry audit {section} is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise ValueError(
            f"Functional geometry audit {section} has unknown fields: {sorted(unknown)}"
        )


@dataclass(frozen=True)
class FunctionalGeometryAuditPreregistration:
    """Validated, digest-addressed contract for the post-calibration audit."""

    schema_version: int
    observation0_preregistration_sha256: str
    functional_preregistration_sha256: str
    required_stage1_unlocked: bool
    required_functional_next_action: str
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
    microbatches_per_cell: int
    fold_offsets: tuple[int, ...]
    fit_size: int
    scale_size: int
    evaluation_size: int
    basis_kinds: tuple[str, ...]
    orientation_gradient: str
    minimum_projection_fraction: float
    relative_step_grid: tuple[float, ...]
    local_linearity_relative_error_max: float
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float
    required_seed_repeats: int

    def __post_init__(self) -> None:
        for field_name in (
            "checkpoint_steps",
            "layers",
            "classes",
            "fold_offsets",
            "basis_kinds",
            "relative_step_grid",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        object.__setattr__(
            self,
            "stratum_bounds",
            tuple(float(value) for value in self.stratum_bounds),
        )
        self._validate()

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def fold_positions(self) -> tuple[dict[str, tuple[int, ...]], ...]:
        folds = []
        for offset in self.fold_offsets:
            order = tuple(
                (offset + index) % self.microbatches_per_cell
                for index in range(self.microbatches_per_cell)
            )
            fit_end = self.fit_size
            scale_end = fit_end + self.scale_size
            folds.append(
                {
                    "fit": order[:fit_end],
                    "scale": order[fit_end:scale_end],
                    "evaluation": order[scale_end:],
                }
            )
        return tuple(folds)

    @classmethod
    def load(cls, path: str | Path) -> FunctionalGeometryAuditPreregistration:
        return cls.from_dict(load_config(path))

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
    ) -> FunctionalGeometryAuditPreregistration:
        if not isinstance(config, dict):
            raise ValueError("Functional geometry audit preregistration must be a mapping.")
        _require_exact_keys(
            config,
            {
                "schema_version",
                "inputs",
                "checkpoints",
                "probe",
                "crossfit",
                "directions",
                "perturbation",
                "inference",
            },
            section="root",
        )
        inputs = config["inputs"]
        checkpoints = config["checkpoints"]
        probe = config["probe"]
        crossfit = config["crossfit"]
        directions = config["directions"]
        perturbation = config["perturbation"]
        inference = config["inference"]
        sections = {
            "inputs": (
                inputs,
                {
                    "observation0_preregistration_sha256",
                    "functional_preregistration_sha256",
                    "required_stage1_unlocked",
                    "required_functional_next_action",
                    "observation0_phase",
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
                    "microbatches_per_cell",
                },
            ),
            "crossfit": (
                crossfit,
                {"fold_offsets", "fit_size", "scale_size", "evaluation_size"},
            ),
            "directions": (
                directions,
                {
                    "basis_kinds",
                    "orientation_gradient",
                    "minimum_projection_fraction",
                },
            ),
            "perturbation": (
                perturbation,
                {
                    "relative_step_grid",
                    "local_linearity_relative_error_max",
                },
            ),
            "inference": (
                inference,
                {
                    "bootstrap_resamples",
                    "bootstrap_seed",
                    "confidence_level",
                    "required_seed_repeats",
                },
            ),
        }
        for name, (values, expected) in sections.items():
            if not isinstance(values, dict):
                raise ValueError(f"Functional geometry audit {name} must be a mapping.")
            _require_exact_keys(values, expected, section=name)
        try:
            return cls(
                schema_version=int(config["schema_version"]),
                observation0_preregistration_sha256=str(
                    inputs["observation0_preregistration_sha256"]
                ),
                functional_preregistration_sha256=str(
                    inputs["functional_preregistration_sha256"]
                ),
                required_stage1_unlocked=bool(inputs["required_stage1_unlocked"]),
                required_functional_next_action=str(
                    inputs["required_functional_next_action"]
                ),
                observation0_phase=str(inputs["observation0_phase"]),
                checkpoint_steps=tuple(int(value) for value in checkpoints["steps"]),
                primary_checkpoint_step=int(checkpoints["primary_step"]),
                positive_control_checkpoint_step=int(
                    checkpoints["positive_control_step"]
                ),
                probe_view=str(probe["view"]),
                stratum_id=int(probe["stratum_id"]),
                stratum_bounds=tuple(float(value) for value in probe["stratum_bounds"]),
                rank=int(probe["rank"]),
                layers=tuple(str(value) for value in probe["layers"]),
                classes=tuple(int(value) for value in probe["classes"]),
                microbatches_per_cell=int(probe["microbatches_per_cell"]),
                fold_offsets=tuple(int(value) for value in crossfit["fold_offsets"]),
                fit_size=int(crossfit["fit_size"]),
                scale_size=int(crossfit["scale_size"]),
                evaluation_size=int(crossfit["evaluation_size"]),
                basis_kinds=tuple(str(value) for value in directions["basis_kinds"]),
                orientation_gradient=str(directions["orientation_gradient"]),
                minimum_projection_fraction=float(
                    directions["minimum_projection_fraction"]
                ),
                relative_step_grid=tuple(
                    float(value) for value in perturbation["relative_step_grid"]
                ),
                local_linearity_relative_error_max=float(
                    perturbation["local_linearity_relative_error_max"]
                ),
                bootstrap_resamples=int(inference["bootstrap_resamples"]),
                bootstrap_seed=int(inference["bootstrap_seed"]),
                confidence_level=float(inference["confidence_level"]),
                required_seed_repeats=int(inference["required_seed_repeats"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid functional geometry audit value: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inputs": {
                "observation0_preregistration_sha256": (
                    self.observation0_preregistration_sha256
                ),
                "functional_preregistration_sha256": (
                    self.functional_preregistration_sha256
                ),
                "required_stage1_unlocked": self.required_stage1_unlocked,
                "required_functional_next_action": (
                    self.required_functional_next_action
                ),
                "observation0_phase": self.observation0_phase,
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
                "microbatches_per_cell": self.microbatches_per_cell,
            },
            "crossfit": {
                "fold_offsets": list(self.fold_offsets),
                "fit_size": self.fit_size,
                "scale_size": self.scale_size,
                "evaluation_size": self.evaluation_size,
            },
            "directions": {
                "basis_kinds": list(self.basis_kinds),
                "orientation_gradient": self.orientation_gradient,
                "minimum_projection_fraction": self.minimum_projection_fraction,
            },
            "perturbation": {
                "relative_step_grid": list(self.relative_step_grid),
                "local_linearity_relative_error_max": (
                    self.local_linearity_relative_error_max
                ),
            },
            "inference": {
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "required_seed_repeats": self.required_seed_repeats,
            },
        }

    def lock(self, path: str | Path) -> Path:
        output = Path(path)
        if output.exists():
            existing = FunctionalGeometryAuditPreregistration.load(output)
            if existing.digest != self.digest:
                raise ValueError(
                    "Refusing to replace a locked functional geometry audit."
                )
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        save_config(self.to_dict(), output)
        return output

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Functional geometry audit schema_version must be 1.")
        if self.observation0_preregistration_sha256 != _OBSERVATION0_DIGEST:
            raise ValueError("Functional geometry audit Observation-0 identity changed.")
        if self.functional_preregistration_sha256 != _FUNCTIONAL_PREREGISTRATION_DIGEST:
            raise ValueError("functional-calibration identity changed.")
        if self.required_stage1_unlocked:
            raise ValueError("Functional geometry audit requires a blocked calibration.")
        if (
            self.required_functional_next_action
            != "stop_stage1_and_revise_functional_geometry"
        ):
            raise ValueError("Functional geometry audit requires the failed-lock action.")
        if self.observation0_phase != "primary":
            raise ValueError("Functional geometry audit uses primary Observation-0 data.")
        if self.checkpoint_steps != (500, 20_000) or {
            self.primary_checkpoint_step,
            self.positive_control_checkpoint_step,
        } != {500, 20_000}:
            raise ValueError("Functional geometry audit requires both locked checkpoints.")
        if self.primary_checkpoint_step != 20_000:
            raise ValueError("Functional geometry audit primary checkpoint must be 20,000.")
        if self.positive_control_checkpoint_step != 500:
            raise ValueError("Functional geometry audit positive control must be step 500.")
        if self.probe_view != "a":
            raise ValueError("Functional geometry audit is Probe-A only.")
        if self.stratum_id != 0 or self.stratum_bounds != (0.02, 0.10):
            raise ValueError("Functional geometry audit requires locked stratum 0.")
        if self.rank != 1:
            raise ValueError("Functional geometry audit is locked to rank-1 directions.")
        if self.layers != _LAYERS:
            raise ValueError("Functional geometry audit requires the two locked layers.")
        if self.classes != _CLASSES:
            raise ValueError("Functional geometry audit requires the six locked classes.")
        if self.microbatches_per_cell != 16:
            raise ValueError("Functional geometry audit requires 16 microbatches per cell.")
        if (self.fit_size, self.scale_size, self.evaluation_size) != (8, 4, 4):
            raise ValueError("Functional geometry audit requires an 8/4/4 cross-fit.")
        if self.fold_offsets != (0, 4, 8, 12):
            raise ValueError("Functional geometry audit fold offsets have changed.")
        if len(set(self.fold_offsets)) != len(self.fold_offsets):
            raise ValueError("Functional geometry audit fold offsets must be unique.")
        fit_counts: Counter[int] = Counter()
        scale_counts: Counter[int] = Counter()
        evaluation_counts: Counter[int] = Counter()
        for fold in self.fold_positions:
            fit_counts.update(fold["fit"])
            scale_counts.update(fold["scale"])
            evaluation_counts.update(fold["evaluation"])
        expected_positions = set(range(self.microbatches_per_cell))
        if (
            set(fit_counts) != expected_positions
            or set(scale_counts) != expected_positions
            or set(evaluation_counts) != expected_positions
            or set(fit_counts.values()) != {2}
            or set(scale_counts.values()) != {1}
            or set(evaluation_counts.values()) != {1}
        ):
            raise ValueError("Functional geometry audit folds are not balanced.")
        if self.basis_kinds != ("raw", "row_normalized"):
            raise ValueError("Functional geometry audit basis comparison has changed.")
        if self.orientation_gradient != "raw_scale_mean":
            raise ValueError("Functional geometry audit must use the raw scale mean.")
        if not 0 < self.minimum_projection_fraction < 1:
            raise ValueError("Functional geometry audit projection floor is invalid.")
        if self.relative_step_grid != (1e-4, 3e-4, 1e-3):
            raise ValueError("Functional geometry audit finite-step grid has changed.")
        if self.local_linearity_relative_error_max != 0.10:
            raise ValueError("Functional geometry audit must retain the 10% tolerance.")
        if self.bootstrap_resamples < 99:
            raise ValueError("Functional geometry audit requires at least 99 bootstraps.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("Functional geometry audit confidence level is invalid.")
        if self.required_seed_repeats != 2:
            raise ValueError("Functional geometry audit requires two seed repeats.")
