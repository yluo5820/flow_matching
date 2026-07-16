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
from fm_lab.diagnostics.long_tail_geometry.natural_image import (
    NaturalImageTransportAnalysis,
    NaturalImageTransportDecision,
    NaturalImageTransportResult,
    analyze_natural_image_transport,
    run_natural_image_transport_falsification,
)
from fm_lab.diagnostics.long_tail_geometry.natural_image_preregistration import (
    NaturalImageTransportPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.observation0 import (
    Observation0CollectionSummary,
    Observation0Preparation,
    analyze_observation0_study,
    collect_observation0_run,
    prepare_observation0_study,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.diagnostics.long_tail_geometry.registry import (
    Observation0Run,
    prepare_observation0_registry,
    update_observation0_run,
)
from fm_lab.diagnostics.long_tail_geometry.reliability import (
    Observation0Decision,
    aggregate_observation0_reliability,
    analyze_seed_reliability,
    centered_cell_statistics,
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
    "Observation0Decision",
    "Observation0CollectionSummary",
    "Observation0Preparation",
    "Observation0Run",
    "NaturalImageTransportAnalysis",
    "NaturalImageTransportDecision",
    "NaturalImageTransportPreregistration",
    "NaturalImageTransportResult",
    "SketchValidation",
    "build_probe_manifest",
    "build_source_noise_replica",
    "aggregate_observation0_reliability",
    "analyze_seed_reliability",
    "analyze_observation0_study",
    "analyze_natural_image_transport",
    "centered_cell_statistics",
    "collect_checkpoint_measurements",
    "collect_observation0_run",
    "collect_gradient_rows",
    "materialize_probe_batch",
    "permutation_null",
    "planted_low_rank_control",
    "prepare_observation0_registry",
    "prepare_observation0_study",
    "projection_overlap",
    "resolve_probe_layers",
    "run_natural_image_transport_falsification",
    "validate_sketch",
    "update_observation0_run",
]
