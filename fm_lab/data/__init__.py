"""Data distributions for flow matching experiments."""

from fm_lab.data.base import TargetDistribution
from fm_lab.data.manifold_toys import (
    GaussianMixture3D,
    HelixMixture,
    LineSegment3D,
    MoebiusStrip,
    MultiSwissRoll,
    MultiTorus,
    NestedSphericalShells,
    PlanarDisk,
    SphericalShell,
    SwissRoll,
    Torus,
    TrefoilKnot,
)
from fm_lab.data.mnist import MNISTImages
from fm_lab.data.toy_2d import Annulus, Checkerboard, ConcentricCircles, GaussianMixture2D, TwoMoons

__all__ = [
    "Annulus",
    "Checkerboard",
    "ConcentricCircles",
    "GaussianMixture3D",
    "GaussianMixture2D",
    "HelixMixture",
    "LineSegment3D",
    "MoebiusStrip",
    "MultiSwissRoll",
    "MultiTorus",
    "MNISTImages",
    "NestedSphericalShells",
    "PlanarDisk",
    "SphericalShell",
    "SwissRoll",
    "TargetDistribution",
    "Torus",
    "TrefoilKnot",
    "TwoMoons",
]
