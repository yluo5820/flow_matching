"""Deterministic measurement tools for long-tail gradient geometry."""

from fm_lab.diagnostics.long_tail_geometry.controls import (
    PermutationResult,
    PlantedControlResult,
    permutation_null,
    planted_low_rank_control,
    projection_overlap,
)
from fm_lab.diagnostics.long_tail_geometry.gradients import (
    GradientRows,
    ProbeLayer,
    collect_gradient_rows,
    resolve_probe_layers,
)
from fm_lab.diagnostics.long_tail_geometry.manifest import (
    ProbeBatch,
    ProbeManifest,
    build_probe_manifest,
    build_source_noise_replica,
    materialize_probe_batch,
)
from fm_lab.diagnostics.long_tail_geometry.measurements import (
    CheckpointMeasurements,
    collect_checkpoint_measurements,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.diagnostics.long_tail_geometry.registry import (
    Observation0Run,
    prepare_observation0_registry,
    update_observation0_run,
)
from fm_lab.diagnostics.long_tail_geometry.sketch import (
    CountSketchSpec,
    SketchValidation,
    validate_sketch,
)

__all__ = [
    "ProbeBatch",
    "ProbeLayer",
    "ProbeManifest",
    "GradientRows",
    "CountSketchSpec",
    "CheckpointMeasurements",
    "PermutationResult",
    "PlantedControlResult",
    "Observation0Preregistration",
    "Observation0Run",
    "SketchValidation",
    "build_probe_manifest",
    "build_source_noise_replica",
    "collect_checkpoint_measurements",
    "collect_gradient_rows",
    "materialize_probe_batch",
    "permutation_null",
    "planted_low_rank_control",
    "prepare_observation0_registry",
    "projection_overlap",
    "resolve_probe_layers",
    "validate_sketch",
    "update_observation0_run",
]
