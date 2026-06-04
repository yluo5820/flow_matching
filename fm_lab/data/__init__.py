"""Data distributions for flow matching experiments."""

from fm_lab.data.base import TargetDistribution
from fm_lab.data.toy_2d import Annulus, Checkerboard, ConcentricCircles, GaussianMixture2D, TwoMoons

__all__ = [
    "Annulus",
    "Checkerboard",
    "ConcentricCircles",
    "GaussianMixture2D",
    "TargetDistribution",
    "TwoMoons",
]
