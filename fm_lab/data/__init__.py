"""Data distributions for flow matching experiments."""

from fm_lab.data.base import TargetDistribution
from fm_lab.data.manifold_toys import (
    GaussianMixture3D,
    HelixMixture,
    MultiSwissRoll,
    MultiTorus,
    NestedSphericalShells,
    SphericalShell,
    SwissRoll,
    Torus,
)
from fm_lab.data.toy_2d import Annulus, Checkerboard, ConcentricCircles, GaussianMixture2D, TwoMoons

__all__ = [
    "Annulus",
    "Checkerboard",
    "ConcentricCircles",
    "GaussianMixture3D",
    "GaussianMixture2D",
    "HelixMixture",
    "MultiSwissRoll",
    "MultiTorus",
    "NestedSphericalShells",
    "SphericalShell",
    "SwissRoll",
    "TargetDistribution",
    "Torus",
    "TwoMoons",
]
