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
from fm_lab.diagnostics.probes.sketch import (
    CountSketchSpec,
    SketchValidation,
    validate_sketch,
)

__all__ = [
    "CountSketchSpec",
    "GradientRows",
    "PermutationResult",
    "PlantedControlResult",
    "ProbeBatch",
    "ProbeLayer",
    "ProbeLossResult",
    "ProbeManifest",
    "SketchValidation",
    "build_probe_manifest",
    "build_source_noise_replica",
    "collect_gradient_rows",
    "evaluate_probe_batches",
    "evaluate_probe_loss",
    "materialize_probe_batch",
    "permutation_null",
    "planted_low_rank_control",
    "projection_overlap",
    "resolve_probe_layers",
    "restore_probe_model",
    "validate_sketch",
]
