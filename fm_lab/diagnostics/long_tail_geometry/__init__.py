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
    materialize_probe_batch,
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
    "PermutationResult",
    "PlantedControlResult",
    "SketchValidation",
    "build_probe_manifest",
    "collect_gradient_rows",
    "materialize_probe_batch",
    "permutation_null",
    "planted_low_rank_control",
    "projection_overlap",
    "resolve_probe_layers",
    "validate_sketch",
]
