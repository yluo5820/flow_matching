"""Source-target coupling strategies."""

from fm_lab.couplings.base import Coupling
from fm_lab.couplings.independent import IndependentCoupling
from fm_lab.couplings.minibatch_ot import MinibatchOTCoupling
from fm_lab.couplings.reflow import ReflowCouplingPlaceholder

__all__ = [
    "Coupling",
    "IndependentCoupling",
    "MinibatchOTCoupling",
    "ReflowCouplingPlaceholder",
]
