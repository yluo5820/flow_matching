import numpy as np
import torch
from torch import nn

from fm_lab.couplings import IndependentCoupling, MinibatchOTCoupling
from fm_lab.data import GaussianMixture3D, TwoMoons
from fm_lab.paths import LearnedAccelerationPath, LinearPath, SphericalPath
from fm_lab.solvers import EulerSolver, HeunSolver
from fm_lab.sources import GaussianSource
from fm_lab.training.losses import build_objective
from fm_lab.training.trainer import (
    _validate_training_compatibility,
    sample_and_plot,
    train_flow_matching,
)
from fm_lab.utils.checkpoints import load_checkpoint


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


class CustomCoupling:
    name = "custom_coupling"

    def pair(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x0, x1


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


def test_direction_only_training_guard_accepts_arbitrary_coupling_name() -> None:
    objective = build_objective({"name": "direction_only_straight"})

    _validate_training_compatibility(
        objective,
        CustomCoupling(),
        LinearPath(),
        UnitXDirectionSpeed(),
    )


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


def test_train_flow_matching_updates_trainable_learned_acceleration_path(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {
            "name": "flow_matching",
            "straightness": {"weight": 0.1, "sample_size": 4},
            "interpolant_acceleration": {"weight": 0.001},
        },
        "training": {
            "batch_size": 8,
            "steps": 2,
            "log_every": 1,
            "lr": 1.0e-3,
            "learned_acceleration": {
                "warmup_steps": 0,
                "theta_steps": 1,
                "psi_steps": 1,
                "psi_lr": 1.0e-3,
            },
        },
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }
    path = LearnedAccelerationPath(dim=2, hidden_dim=8, depth=1)
    initial_path_state = {key: value.detach().clone() for key, value in path.state_dict().items()}

    train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=TwoMoons(noise=0.0),
        source=GaussianSource(dim=2),
        coupling=IndependentCoupling(),
        path=path,
        model=TrainableTimeScaledVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    checkpoint = load_checkpoint(tmp_path / "checkpoint.pt")

    assert "path_state_dict" in checkpoint
    assert "theta" in checkpoint["optimizer_state_dict"]
    assert "psi" in checkpoint["optimizer_state_dict"]
    assert any(
        not torch.allclose(value, initial_path_state[key])
        for key, value in path.state_dict().items()
    )
    assert (tmp_path / "diagnostics" / "training_history.csv").exists()
    assert (tmp_path / "plots" / "training_loss.png").exists()
    assert (tmp_path / "plots" / "generated_samples_nfe3.png").exists()


class TrainableTimeScaledVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * t[:, None] * x


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
