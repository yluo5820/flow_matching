"""Deterministic, hypothesis-neutral model probing utilities."""

from fm_lab.diagnostics.probes.checkpoints import (
    ProbeLossResult,
    evaluate_probe_batches,
    evaluate_probe_loss,
    restore_probe_model,
)
from fm_lab.diagnostics.probes.controls import (
    PermutationResult,
    PlantedControlResult,
    permutation_null,
    planted_low_rank_control,
    projection_overlap,
)
from fm_lab.diagnostics.probes.gradients import (
    GradientRows,
    ProbeLayer,
    collect_gradient_rows,
    resolve_probe_layers,
)
from fm_lab.diagnostics.probes.manifest import (
    ProbeBatch,
    ProbeManifest,
    build_probe_manifest,
    build_source_noise_replica,
    materialize_probe_batch,
)
from fm_lab.diagnostics.probes.perturbations import virtual_layer_update
from fm_lab.diagnostics.probes.sketch import (
    CountSketchSpec,
    SketchValidation,
    validate_sketch,
)
from fm_lab.diagnostics.probes.subspaces import (
    PrincipalDirection,
    ProjectedDirection,
    deterministic_random_unit_direction,
    projected_descent_direction,
    top_centered_covariance_direction,
)

__all__ = [
    "CountSketchSpec",
    "GradientRows",
    "PermutationResult",
    "PlantedControlResult",
    "PrincipalDirection",
    "ProbeBatch",
    "ProbeLayer",
    "ProbeLossResult",
    "ProbeManifest",
    "ProjectedDirection",
    "SketchValidation",
    "build_probe_manifest",
    "build_source_noise_replica",
    "collect_gradient_rows",
    "deterministic_random_unit_direction",
    "evaluate_probe_batches",
    "evaluate_probe_loss",
    "materialize_probe_batch",
    "permutation_null",
    "planted_low_rank_control",
    "projected_descent_direction",
    "projection_overlap",
    "resolve_probe_layers",
    "restore_probe_model",
    "top_centered_covariance_direction",
    "validate_sketch",
    "virtual_layer_update",
]
