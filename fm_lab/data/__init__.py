"""Data distributions for flow matching experiments."""

from typing import TYPE_CHECKING

from fm_lab.data.base import TargetDistribution
from fm_lab.data.cifar_lt import ImbalancedCIFARImages
from fm_lab.data.fashion_mnist import LongTailedFashionMNIST
from fm_lab.data.image_variant import ImageVariantImages
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
from fm_lab.data.mnist_variant import MNISTVariantImages
from fm_lab.data.toy_2d import Annulus, Checkerboard, ConcentricCircles, GaussianMixture2D, TwoMoons

if TYPE_CHECKING:
    from fm_lab.data.synthetic_long_tail import SyntheticLongTailImages

__all__ = [
    "Annulus",
    "Checkerboard",
    "ConcentricCircles",
    "GaussianMixture3D",
    "GaussianMixture2D",
    "HelixMixture",
    "ImageVariantImages",
    "ImbalancedCIFARImages",
    "LineSegment3D",
    "LongTailedFashionMNIST",
    "MoebiusStrip",
    "MultiSwissRoll",
    "MultiTorus",
    "MNISTImages",
    "MNISTVariantImages",
    "NestedSphericalShells",
    "PlanarDisk",
    "SphericalShell",
    "SwissRoll",
    "SyntheticLongTailImages",
    "TargetDistribution",
    "Torus",
    "TrefoilKnot",
    "TwoMoons",
]


def __getattr__(name: str):
    if name == "SyntheticLongTailImages":
        from fm_lab.data.synthetic_long_tail import SyntheticLongTailImages

        return SyntheticLongTailImages
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
