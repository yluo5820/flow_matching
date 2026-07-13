"""Source-target coupling strategies."""

from fm_lab.couplings.base import Coupling, pair_with_condition
from fm_lab.couplings.independent import IndependentCoupling
from fm_lab.couplings.minibatch_ot import MinibatchOTCoupling
from fm_lab.couplings.reflow import ModelGeneratedCoupling, ReflowCouplingPlaceholder

__all__ = [
    "Coupling",
    "pair_with_condition",
    "IndependentCoupling",
    "MinibatchOTCoupling",
    "ModelGeneratedCoupling",
    "ReflowCouplingPlaceholder",
]
