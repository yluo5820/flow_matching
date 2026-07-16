"""Immutable scientific protocol for the long-tail Observation-0 pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fm_lab.utils.config import load_config, save_config


@dataclass(frozen=True)
class Observation0Preregistration:
    """Validated, digest-addressed Observation-0 protocol."""

    schema_version: int
    study_name: str
    dataset: str
    base_config: str
    training_seeds: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    frequency_multiplier: int
    observation0_mapping_offsets: tuple[int, ...]
    stage1_mapping_offsets: tuple[int, ...]
    balanced_control: bool
    manifest_seed: int
    probe_splits: tuple[str, ...]
    source_noise_replicas: int
    microbatch_size: int
    primary_microbatches_per_cell: int
    escalation_microbatches_per_cell: int
    time_strata: tuple[tuple[float, float], ...]
    layers: tuple[str, ...]
    sketch_dim: int
    max_sketch_dim: int
    sketch_seed: int
    storage_dtype: str
    reliability_representation: str
    gate_ranks: tuple[int, ...]
    descriptive_ranks: tuple[int, ...]
    null_generator: str
    null_permutations: int
    null_quantile: float
    required_seed_repeats: int
    minimum_common_classes: int
    exclude_checkpoint_zero_from_gate: bool
    stage1_requires_functional_lock: bool
    functional_loss_change_fraction: float

    def __post_init__(self) -> None:
        tuple_fields = (
            "training_seeds",
            "checkpoint_steps",
            "observation0_mapping_offsets",
            "stage1_mapping_offsets",
            "probe_splits",
            "layers",
            "gate_ranks",
            "descriptive_ranks",
        )
        for field_name in tuple_fields:
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        object.__setattr__(
            self,
            "time_strata",
            tuple((float(low), float(high)) for low, high in self.time_strata),
        )
        self._validate()

    @property
    def primary_rows_per_class_per_stratum(self) -> int:
        return self.microbatch_size * self.primary_microbatches_per_cell

    @property
    def escalation_rows_per_class_per_stratum(self) -> int:
        return self.microbatch_size * self.escalation_microbatches_per_cell

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> Observation0Preregistration:
        return cls.from_dict(load_config(path))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Observation0Preregistration:
        try:
            study = config["study"]
            training = config["training"]
            frequency = config["frequency"]
            probe = config["probe"]
            gradient = config["gradient"]
            reliability = config["reliability"]
            stage1_lock = config["stage1_lock"]
            return cls(
                schema_version=int(config["schema_version"]),
                study_name=str(study["name"]),
                dataset=str(study["dataset"]),
                base_config=str(study["base_config"]),
                training_seeds=tuple(int(value) for value in training["seeds"]),
                checkpoint_steps=tuple(
                    int(value) for value in training["checkpoint_steps"]
                ),
                frequency_multiplier=int(frequency["multiplier"]),
                observation0_mapping_offsets=tuple(
                    int(value) for value in frequency["observation0_mapping_offsets"]
                ),
                stage1_mapping_offsets=tuple(
                    int(value) for value in frequency["stage1_mapping_offsets"]
                ),
                balanced_control=bool(frequency["balanced_control"]),
                manifest_seed=int(probe["manifest_seed"]),
                probe_splits=tuple(str(value) for value in probe["splits"]),
                source_noise_replicas=int(probe["source_noise_replicas"]),
                microbatch_size=int(probe["microbatch_size"]),
                primary_microbatches_per_cell=int(
                    probe["primary_microbatches_per_cell"]
                ),
                escalation_microbatches_per_cell=int(
                    probe["escalation_microbatches_per_cell"]
                ),
                time_strata=tuple(
                    (float(low), float(high)) for low, high in probe["time_strata"]
                ),
                layers=tuple(str(value) for value in gradient["layers"]),
                sketch_dim=int(gradient["sketch_dim"]),
                max_sketch_dim=int(gradient["max_sketch_dim"]),
                sketch_seed=int(gradient["sketch_seed"]),
                storage_dtype=str(gradient["storage_dtype"]),
                reliability_representation=str(reliability["representation"]),
                gate_ranks=tuple(int(value) for value in reliability["gate_ranks"]),
                descriptive_ranks=tuple(
                    int(value) for value in reliability["descriptive_ranks"]
                ),
                null_generator=str(reliability["null_generator"]),
                null_permutations=int(reliability["null_permutations"]),
                null_quantile=float(reliability["null_quantile"]),
                required_seed_repeats=int(reliability["required_seed_repeats"]),
                minimum_common_classes=int(reliability["minimum_common_classes"]),
                exclude_checkpoint_zero_from_gate=bool(
                    reliability["exclude_checkpoint_zero_from_gate"]
                ),
                stage1_requires_functional_lock=bool(stage1_lock["required"]),
                functional_loss_change_fraction=float(
                    stage1_lock["functional_loss_change_fraction"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Observation-0 preregistration is missing required field: {exc}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study": {
                "name": self.study_name,
                "dataset": self.dataset,
                "base_config": self.base_config,
            },
            "training": {
                "seeds": list(self.training_seeds),
                "checkpoint_steps": list(self.checkpoint_steps),
            },
            "frequency": {
                "multiplier": self.frequency_multiplier,
                "observation0_mapping_offsets": list(
                    self.observation0_mapping_offsets
                ),
                "stage1_mapping_offsets": list(self.stage1_mapping_offsets),
                "balanced_control": self.balanced_control,
            },
            "probe": {
                "manifest_seed": self.manifest_seed,
                "splits": list(self.probe_splits),
                "source_noise_replicas": self.source_noise_replicas,
                "microbatch_size": self.microbatch_size,
                "primary_microbatches_per_cell": (
                    self.primary_microbatches_per_cell
                ),
                "escalation_microbatches_per_cell": (
                    self.escalation_microbatches_per_cell
                ),
                "time_strata": [list(interval) for interval in self.time_strata],
            },
            "gradient": {
                "layers": list(self.layers),
                "sketch_dim": self.sketch_dim,
                "max_sketch_dim": self.max_sketch_dim,
                "sketch_seed": self.sketch_seed,
                "storage_dtype": self.storage_dtype,
            },
            "reliability": {
                "representation": self.reliability_representation,
                "gate_ranks": list(self.gate_ranks),
                "descriptive_ranks": list(self.descriptive_ranks),
                "null_generator": self.null_generator,
                "null_permutations": self.null_permutations,
                "null_quantile": self.null_quantile,
                "required_seed_repeats": self.required_seed_repeats,
                "minimum_common_classes": self.minimum_common_classes,
                "exclude_checkpoint_zero_from_gate": (
                    self.exclude_checkpoint_zero_from_gate
                ),
            },
            "stage1_lock": {
                "required": self.stage1_requires_functional_lock,
                "functional_loss_change_fraction": (
                    self.functional_loss_change_fraction
                ),
            },
        }

    def lock(self, path: str | Path) -> Path:
        output = Path(path)
        if output.exists():
            existing = Observation0Preregistration.load(output)
            if existing.digest != self.digest:
                raise ValueError("Refusing to replace a locked preregistration.")
            return output
        output.parent.mkdir(parents=True, exist_ok=True)
        save_config(self.to_dict(), output)
        return output

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Observation-0 preregistration schema_version must be 1.")
        if not self.study_name or self.dataset != "fashion_mnist_lt":
            raise ValueError("Observation 0 requires the Fashion-MNIST long-tail study.")
        if not self.base_config:
            raise ValueError("Observation 0 requires a base config path.")
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("Observation-0 training seeds must be non-empty and unique.")
        if any(seed < 0 for seed in self.training_seeds):
            raise ValueError("Observation-0 training seeds must be non-negative.")
        if not self.checkpoint_steps or self.checkpoint_steps[0] != 0:
            raise ValueError("Observation-0 checkpoints must include step zero first.")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("Observation-0 checkpoint steps must be unique and sorted.")
        if self.observation0_mapping_offsets != (0,):
            raise ValueError("Observation 0 is restricted to frequency offset 0.")
        if self.frequency_multiplier != 3:
            raise ValueError("Observation 0 requires frequency multiplier 3.")
        if self.stage1_mapping_offsets != tuple(range(10)):
            raise ValueError("Stage-1 mapping offsets must contain the complete Latin set.")
        if not self.balanced_control:
            raise ValueError("The Stage-1 design must retain a balanced control.")
        if self.probe_splits != ("a", "b"):
            raise ValueError("Observation 0 requires Probe-A and Probe-B.")
        if self.source_noise_replicas != 2:
            raise ValueError("Observation 0 requires exactly two source-noise replicas.")
        if self.microbatch_size < 1 or self.primary_microbatches_per_cell < 2:
            raise ValueError("Observation-0 probe microbatch counts must be positive.")
        if self.escalation_microbatches_per_cell != 2 * self.primary_microbatches_per_cell:
            raise ValueError("Probe escalation must exactly double the microbatch count.")
        if not self.time_strata or any(
            not 0 <= low < high <= 1 for low, high in self.time_strata
        ):
            raise ValueError("Observation-0 timestep strata must be valid intervals.")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Observation-0 gradient layers must be non-empty and unique.")
        if not all(layer.endswith(".weight") for layer in self.layers):
            raise ValueError("Every Observation-0 gradient layer must name a weight.")
        if self.sketch_dim < 2 or self.max_sketch_dim < self.sketch_dim:
            raise ValueError("Observation-0 sketch dimensions are invalid.")
        if self.storage_dtype != "float32":
            raise ValueError("Observation-0 sketches must use float32 storage.")
        if self.reliability_representation != "centered_covariance":
            raise ValueError("The primary reliability representation must be centered covariance.")
        if not self.gate_ranks or any(
            rank < 1 or rank >= self.primary_microbatches_per_cell
            for rank in self.gate_ranks
        ):
            raise ValueError("Centered gate ranks must be below the primary sample rank.")
        if any(rank not in self.descriptive_ranks for rank in self.gate_ranks):
            raise ValueError("Every gate rank must also be a descriptive rank.")
        if tuple(sorted(set(self.descriptive_ranks))) != self.descriptive_ranks:
            raise ValueError("Descriptive ranks must be positive, unique, and sorted.")
        if self.null_generator != "class_label_permutation":
            raise ValueError("Observation 0 requires the class-label permutation null.")
        if self.null_permutations < 99:
            raise ValueError("Observation 0 requires at least 99 null permutations.")
        if self.null_quantile != 0.99:
            raise ValueError("Observation-0 null_quantile must remain 0.99.")
        if not 1 <= self.required_seed_repeats <= len(self.training_seeds):
            raise ValueError("Required repeats must fit within the training seeds.")
        if not 1 <= self.minimum_common_classes <= 10:
            raise ValueError("minimum_common_classes must lie in [1, 10].")
        if not self.exclude_checkpoint_zero_from_gate:
            raise ValueError("Checkpoint zero must be excluded from the primary gate.")
        if not self.stage1_requires_functional_lock:
            raise ValueError("Stage 1 must remain blocked on its functional overlap lock.")
        if self.functional_loss_change_fraction != 0.01:
            raise ValueError("The functional calibration target must remain a 1% loss change.")
