"""Immutable protocol for the CIFAR-10-LT transport falsification."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fm_lab.utils.config import load_config, save_config

_OBSERVATION0_DIGEST = (
    "4a87fbc8b3a0e3e67a3f71080ce8702cdc524f35a4b7dc8d997200fc601067a7"
)
_LAYERS = (
    "down2_block.conv2.weight",
    "middle.conv2.weight",
)
_CLASSES = tuple(range(10))


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
            f"Natural-image transport {section} is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise ValueError(
            f"Natural-image transport {section} has unknown fields: {sorted(unknown)}"
        )


@dataclass(frozen=True)
class NaturalImageTransportPreregistration:
    """Validated, digest-addressed natural-image transport contract."""

    schema_version: int
    observation0_preregistration_sha256: str
    observation0_phase: str
    checkpoint_steps: tuple[int, ...]
    baseline_checkpoint_step: int
    early_checkpoint_step: int
    primary_checkpoint_step: int
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
    maximum_final_to_baseline_loss_ratio: float
    minimum_reliable_common_classes: int

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
    def load(cls, path: str | Path) -> NaturalImageTransportPreregistration:
        return cls.from_dict(load_config(path))

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
    ) -> NaturalImageTransportPreregistration:
        if not isinstance(config, dict):
            raise ValueError("Natural-image transport preregistration must be a mapping.")
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
                "decision",
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
        decision = config["decision"]
        sections = {
            "inputs": (
                inputs,
                {"observation0_preregistration_sha256", "observation0_phase"},
            ),
            "checkpoints": (
                checkpoints,
                {"steps", "baseline_step", "early_step", "primary_step"},
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
                {"relative_step_grid", "local_linearity_relative_error_max"},
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
            "decision": (
                decision,
                {
                    "maximum_final_to_baseline_loss_ratio",
                    "minimum_reliable_common_classes",
                },
            ),
        }
        for name, (values, expected) in sections.items():
            if not isinstance(values, dict):
                raise ValueError(f"Natural-image transport {name} must be a mapping.")
            _require_exact_keys(values, expected, section=name)
        try:
            return cls(
                schema_version=int(config["schema_version"]),
                observation0_preregistration_sha256=str(
                    inputs["observation0_preregistration_sha256"]
                ),
                observation0_phase=str(inputs["observation0_phase"]),
                checkpoint_steps=tuple(int(value) for value in checkpoints["steps"]),
                baseline_checkpoint_step=int(checkpoints["baseline_step"]),
                early_checkpoint_step=int(checkpoints["early_step"]),
                primary_checkpoint_step=int(checkpoints["primary_step"]),
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
                maximum_final_to_baseline_loss_ratio=float(
                    decision["maximum_final_to_baseline_loss_ratio"]
                ),
                minimum_reliable_common_classes=int(
                    decision["minimum_reliable_common_classes"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid natural-image transport value: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inputs": {
                "observation0_preregistration_sha256": (
                    self.observation0_preregistration_sha256
                ),
                "observation0_phase": self.observation0_phase,
            },
            "checkpoints": {
                "steps": list(self.checkpoint_steps),
                "baseline_step": self.baseline_checkpoint_step,
                "early_step": self.early_checkpoint_step,
                "primary_step": self.primary_checkpoint_step,
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
            "decision": {
                "maximum_final_to_baseline_loss_ratio": (
                    self.maximum_final_to_baseline_loss_ratio
                ),
                "minimum_reliable_common_classes": (
                    self.minimum_reliable_common_classes
                ),
            },
        }

    def lock(self, path: str | Path) -> Path:
        output = Path(path)
        if output.exists():
            existing = NaturalImageTransportPreregistration.load(output)
            if existing.digest != self.digest:
                raise ValueError(
                    "Refusing to replace a locked natural-image transport contract."
                )
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        save_config(self.to_dict(), output)
        return output

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Natural-image transport schema_version must be 1.")
        if self.observation0_preregistration_sha256 != _OBSERVATION0_DIGEST:
            raise ValueError("Natural-image transport Observation-0 identity changed.")
        if self.observation0_phase != "primary":
            raise ValueError("Natural-image transport uses primary Observation-0 data.")
        if self.checkpoint_steps != (0, 10_000, 100_000):
            raise ValueError("Natural-image transport requires three locked checkpoints.")
        if self.baseline_checkpoint_step != 0:
            raise ValueError("Natural-image transport baseline checkpoint must be step zero.")
        if self.early_checkpoint_step != 10_000:
            raise ValueError("Natural-image transport early checkpoint must be step 10,000.")
        if self.primary_checkpoint_step != 100_000:
            raise ValueError("Natural-image transport primary checkpoint must be step 100,000.")
        if self.probe_view != "a":
            raise ValueError("Natural-image transport is Probe-A only.")
        if self.stratum_id != 0 or self.stratum_bounds != (0.02, 0.10):
            raise ValueError("Natural-image transport requires locked stratum 0.")
        if self.rank != 1:
            raise ValueError("Natural-image transport requires rank-1 directions.")
        if self.layers != _LAYERS:
            raise ValueError("Natural-image transport requires the two locked layers.")
        if self.classes != _CLASSES:
            raise ValueError("Natural-image transport requires all ten classes.")
        if self.microbatches_per_cell != 16:
            raise ValueError("Natural-image transport requires 16 microbatches per cell.")
        if (self.fit_size, self.scale_size, self.evaluation_size) != (8, 4, 4):
            raise ValueError("Natural-image transport requires an 8/4/4 cross-fit.")
        if self.fold_offsets != (0, 4, 8, 12) or len(set(self.fold_offsets)) != 4:
            raise ValueError("Natural-image transport fold offsets have changed.")
        fit_counts: Counter[int] = Counter()
        scale_counts: Counter[int] = Counter()
        evaluation_counts: Counter[int] = Counter()
        for fold in self.fold_positions:
            fit_counts.update(fold["fit"])
            scale_counts.update(fold["scale"])
            evaluation_counts.update(fold["evaluation"])
        expected = set(range(self.microbatches_per_cell))
        if (
            set(fit_counts) != expected
            or set(scale_counts) != expected
            or set(evaluation_counts) != expected
            or set(fit_counts.values()) != {2}
            or set(scale_counts.values()) != {1}
            or set(evaluation_counts.values()) != {1}
        ):
            raise ValueError("Natural-image transport folds are not balanced.")
        if self.basis_kinds != ("raw", "row_normalized"):
            raise ValueError("Natural-image transport requires paired bases.")
        if self.orientation_gradient != "raw_scale_mean":
            raise ValueError("Natural-image transport must use the raw scale mean.")
        if not 0 < self.minimum_projection_fraction < 1:
            raise ValueError("Natural-image transport projection floor is invalid.")
        if self.relative_step_grid != (3e-5, 1e-4, 3e-4, 1e-3):
            raise ValueError("Natural-image transport finite-step grid has changed.")
        if self.local_linearity_relative_error_max != 0.10:
            raise ValueError("Natural-image transport retains a 10% local tolerance.")
        if self.bootstrap_resamples < 99:
            raise ValueError("Natural-image transport requires at least 99 bootstraps.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("Natural-image transport confidence level is invalid.")
        if self.required_seed_repeats != 2:
            raise ValueError("Natural-image transport requires two seed repeats.")
        if self.maximum_final_to_baseline_loss_ratio != 0.70:
            raise ValueError("Natural-image transport retains the 70% learning guard.")
        if self.minimum_reliable_common_classes != 5:
            raise ValueError("Natural-image transport requires five common classes.")
