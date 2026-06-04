"""Research diagnostics for ambiguity, curvature, geometry, and solvers."""

from fm_lab.diagnostics.ambiguity import (
    GridAmbiguityResult,
    bayes_regression_gap_knn,
    grid_ambiguity,
    knn_ambiguity,
)
from fm_lab.diagnostics.metrics import sliced_wasserstein, squared_mmd

__all__ = [
    "GridAmbiguityResult",
    "bayes_regression_gap_knn",
    "grid_ambiguity",
    "knn_ambiguity",
    "sliced_wasserstein",
    "squared_mmd",
]
