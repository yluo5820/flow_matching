"""Training loops, objectives, and callbacks."""

from fm_lab.training.long_tail import (
    CBDMModifier,
    ContinuousObjectiveContext,
    ContinuousObjectiveModifier,
    build_continuous_modifiers,
)
from fm_lab.training.losses import (
    DiffusionObjective,
    DirectionOnlyStraightObjective,
    FlowMatchingObjective,
    KernelVStarConfig,
    build_objective,
    flow_matching_loss,
    kernel_vstar_estimate,
    learned_flow_straightness_loss,
    sample_uniform_time,
)
from fm_lab.training.trainer import train_flow_matching

__all__ = [
    "CBDMModifier",
    "ContinuousObjectiveContext",
    "ContinuousObjectiveModifier",
    "FlowMatchingObjective",
    "DiffusionObjective",
    "DirectionOnlyStraightObjective",
    "KernelVStarConfig",
    "kernel_vstar_estimate",
    "build_objective",
    "build_continuous_modifiers",
    "flow_matching_loss",
    "learned_flow_straightness_loss",
    "sample_uniform_time",
    "train_flow_matching",
]
