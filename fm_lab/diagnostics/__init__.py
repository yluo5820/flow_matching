"""Research diagnostics for ambiguity, curvature, geometry, and solvers."""

from fm_lab.diagnostics.ambiguity import (
    GridAmbiguityResult,
    bayes_regression_gap_knn,
    grid_ambiguity,
    knn_ambiguity,
)
from fm_lab.diagnostics.curvature import curvature_stats, material_acceleration
from fm_lab.diagnostics.geometry import radial_deviation, radial_tangent_velocity_2d
from fm_lab.diagnostics.jacobian import exact_jacobian, jacobian_stats
from fm_lab.diagnostics.metrics import sliced_wasserstein, squared_mmd
from fm_lab.diagnostics.solver_sensitivity import (
    generate_solver_samples,
    pairwise_solver_distances,
    solver_sensitivity_summary,
)
from fm_lab.diagnostics.trajectory_umap import (
    TrajectoryUMAPConfig,
    project_saved_trajectories,
)

__all__ = [
    "GridAmbiguityResult",
    "TrajectoryUMAPConfig",
    "bayes_regression_gap_knn",
    "curvature_stats",
    "exact_jacobian",
    "generate_solver_samples",
    "grid_ambiguity",
    "jacobian_stats",
    "knn_ambiguity",
    "material_acceleration",
    "pairwise_solver_distances",
    "project_saved_trajectories",
    "radial_deviation",
    "radial_tangent_velocity_2d",
    "sliced_wasserstein",
    "solver_sensitivity_summary",
    "squared_mmd",
]
