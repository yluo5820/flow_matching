"""Source distributions for flow matching experiments."""

from fm_lab.sources.base import SourceDistribution
from fm_lab.sources.gaussian import GaussianSource
from fm_lab.sources.spherical import SphericalShellSource

__all__ = ["GaussianSource", "SourceDistribution", "SphericalShellSource"]
