"""Research diagnostics for ambiguity, curvature, geometry, and solvers."""

from fm_lab.diagnostics.ambiguity import (
    GridAmbiguityResult,
    bayes_regression_gap_knn,
    grid_ambiguity,
    knn_ambiguity,
)
from fm_lab.diagnostics.curvature import curvature_stats, material_acceleration
from fm_lab.diagnostics.diffusion_lid import (
    FlipdEstimate,
    NormalBundleEstimate,
    flipd_dimension,
    normal_bundle_dimension,
)
from fm_lab.diagnostics.fm_lid import (
    FMFLIPDEstimate,
    FMFLIPDEstimator,
    FMJacobianSpectrumEstimate,
    FMJacobianSpectrumEstimator,
    GaussianFMSchedule,
    entropy_rank,
    participation_rank,
    sample_unit_directions,
    summarize_lid_values,
    threshold_rank,
)
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
    "FlipdEstimate",
    "FMFLIPDEstimate",
    "FMFLIPDEstimator",
    "FMJacobianSpectrumEstimate",
    "FMJacobianSpectrumEstimator",
    "GaussianFMSchedule",
    "NormalBundleEstimate",
    "TrajectoryUMAPConfig",
    "bayes_regression_gap_knn",
    "curvature_stats",
    "entropy_rank",
    "exact_jacobian",
    "flipd_dimension",
    "generate_solver_samples",
    "grid_ambiguity",
    "jacobian_stats",
    "knn_ambiguity",
    "material_acceleration",
    "normal_bundle_dimension",
    "pairwise_solver_distances",
    "participation_rank",
    "project_saved_trajectories",
    "radial_deviation",
    "radial_tangent_velocity_2d",
    "sample_unit_directions",
    "sliced_wasserstein",
    "solver_sensitivity_summary",
    "squared_mmd",
    "summarize_lid_values",
    "threshold_rank",
]
