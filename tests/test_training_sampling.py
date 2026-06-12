import numpy as np
import torch
from torch import nn

from fm_lab.couplings import IndependentCoupling, MinibatchOTCoupling, ReflowCouplingPlaceholder
from fm_lab.data import GaussianMixture3D
from fm_lab.paths import LinearPath, SphericalPath
from fm_lab.solvers import EulerSolver, HeunSolver
from fm_lab.sources import GaussianSource
from fm_lab.training.losses import build_objective
from fm_lab.training.trainer import _validate_training_compatibility, sample_and_plot


class ZeroVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class UnitXDirectionSpeed(nn.Module):
    requires_source_label = True

    def direction(self, source_label: torch.Tensor) -> torch.Tensor:
        direction = torch.zeros_like(source_label)
        direction[:, 0] = 1.0
        return direction

    def speed(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        source_label: torch.Tensor,
    ) -> torch.Tensor:
        del x, t
        return torch.ones(source_label.shape[0], device=source_label.device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = context["source_label"]
        return self.speed(x, t, source_label)[:, None] * self.direction(source_label)


def test_sample_and_plot_reuses_trajectory_sources_across_solvers(tmp_path) -> None:
    config = _sampling_config(seed=123)

    sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=ZeroVelocity(),
        solvers=[EulerSolver(), HeunSolver()],
        device=torch.device("cpu"),
    )

    reference = np.load(tmp_path / "trajectories" / "source_reference_nfe3.npy")
    euler = np.load(tmp_path / "trajectories" / "euler_nfe3.npy")
    heun = np.load(tmp_path / "trajectories" / "heun_nfe3.npy")

    assert np.allclose(euler[0], reference)
    assert np.allclose(heun[0], reference)
    assert np.allclose(euler[0], heun[0])


def test_sampling_seed_makes_plot_sources_reproducible(tmp_path) -> None:
    config = _sampling_config(seed=321)

    torch.manual_seed(1)
    sample_and_plot(
        config=config,
        run_dir=tmp_path / "run_a",
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=ZeroVelocity(),
        solvers=[EulerSolver(), HeunSolver()],
        device=torch.device("cpu"),
    )

    torch.manual_seed(999)
    sample_and_plot(
        config=config,
        run_dir=tmp_path / "run_b",
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=ZeroVelocity(),
        solvers=[EulerSolver(), HeunSolver()],
        device=torch.device("cpu"),
    )

    source_a = np.load(tmp_path / "run_a" / "samples" / "source_reference.npy")
    source_b = np.load(tmp_path / "run_b" / "samples" / "source_reference.npy")
    trajectories_a = np.load(tmp_path / "run_a" / "trajectories" / "source_reference_nfe3.npy")
    trajectories_b = np.load(tmp_path / "run_b" / "trajectories" / "source_reference_nfe3.npy")

    assert np.allclose(source_a, source_b)
    assert np.allclose(trajectories_a, trajectories_b)


def test_sample_and_plot_supports_source_label_conditioned_models(tmp_path) -> None:
    config = _sampling_config(seed=777)

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=UnitXDirectionSpeed(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    trajectory = np.load(tmp_path / "trajectories" / "euler_nfe3.npy")

    assert trajectory.shape == (4, 5, 3)
    assert summary["line_containment"]["euler"]["off_line_max"] < 1.0e-6
    assert (tmp_path / "samples" / "euler_nfe3.npy").exists()


def test_direction_only_training_guard_accepts_minibatch_ot_coupling() -> None:
    objective = build_objective({"name": "direction_only_straight"})

    _validate_training_compatibility(
        objective,
        MinibatchOTCoupling(),
        LinearPath(),
        UnitXDirectionSpeed(),
    )


def test_direction_only_training_guard_rejects_unsupported_coupling() -> None:
    objective = build_objective({"name": "direction_only_straight"})

    try:
        _validate_training_compatibility(
            objective,
            ReflowCouplingPlaceholder(),
            LinearPath(),
            UnitXDirectionSpeed(),
        )
    except ValueError as exc:
        assert "independent" in str(exc)
        assert "minibatch_ot" in str(exc)
    else:
        raise AssertionError("Expected unsupported coupling to be rejected.")


def test_direction_only_training_guard_rejects_non_linear_path() -> None:
    objective = build_objective({"name": "direction_only_straight"})

    try:
        _validate_training_compatibility(
            objective,
            IndependentCoupling(),
            SphericalPath(),
            UnitXDirectionSpeed(),
        )
    except ValueError as exc:
        assert "linear" in str(exc)
    else:
        raise AssertionError("Expected non-linear path to be rejected.")


def _sampling_config(seed: int) -> dict:
    return {
        "experiment": {"seed": 0},
        "sampling": {
            "n_samples": 8,
            "n_trajectories": 5,
            "nfe": 3,
            "seed": seed,
        },
        "solvers": {"schedule": "uniform"},
    }
