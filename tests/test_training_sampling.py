import numpy as np
import torch
from torch import nn

from fm_lab.couplings import IndependentCoupling, MinibatchOTCoupling
from fm_lab.data import GaussianMixture3D, TwoMoons
from fm_lab.paths import GaussianDiffusionPath, LearnedAccelerationPath, LinearPath, SphericalPath
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


class RecordingZeroVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        self.batch_sizes.append(int(x.shape[0]))
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


class ConstantSource:
    dim = 2

    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(n, self.dim, device=device)

    def metadata(self) -> dict[str, str | int]:
        return {"name": "constant_zero", "dim": self.dim}


class ConstantTarget:
    dim = 2

    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.ones(n, self.dim, device=device)

    def metadata(self) -> dict[str, str | int]:
        return {"name": "constant_one", "dim": self.dim}


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


def test_sample_and_plot_chunks_large_final_sample_batches(tmp_path) -> None:
    config = _sampling_config(seed=654)
    config["sampling"]["n_samples"] = 9
    config["sampling"]["n_trajectories"] = 2
    config["sampling"]["sample_batch_size"] = 4
    model = RecordingZeroVelocity()

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    generated = np.load(tmp_path / "samples" / "euler_nfe3.npy")
    assert generated.shape == (9, 3)
    assert summary["sample_batch_size"] == 4
    assert max(model.batch_sizes) == 4
    assert 1 in model.batch_sizes


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


def test_sample_and_plot_converts_x_predictions_to_velocity(tmp_path) -> None:
    config = _sampling_config(seed=111)
    config["objective"] = {
        "name": "flow_matching",
        "model_output": "x",
        "x_prediction": {"loss_space": "clean", "min_denom": 0.05},
    }

    sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        path=LinearPath(),
        model=TrainableConstantVelocity(dim=2, value=1.0),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    generated = np.load(tmp_path / "samples" / "euler_nfe3.npy")
    assert np.allclose(generated, 1.0)


def test_sample_and_plot_writes_umap_trajectory_when_enabled(tmp_path, monkeypatch) -> None:
    config = _sampling_config(seed=888)
    config["sampling"]["trajectory_umap"] = {
        "enabled": True,
        "max_target_points": 4,
        "n_neighbors": 3,
        "save_coordinates": True,
    }

    def fake_umap_plot(trajectory, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")
        coordinates_path = kwargs["coordinates_path"]
        np.savez_compressed(coordinates_path, trajectory=np.zeros((*trajectory.shape[:2], 3)))
        interactive_path = kwargs["interactive_path"]
        interactive_path.write_text("html", encoding="utf-8")
        return {
            "plot_path": str(output_path),
            "coordinates_path": str(coordinates_path),
            "interactive_path": str(interactive_path),
            "n_steps": int(trajectory.shape[0]),
            "n_trajectories": int(trajectory.shape[1]),
            "target_points": int(kwargs["max_target_points"]),
            "n_neighbors": int(kwargs["n_neighbors"]),
        }

    monkeypatch.setattr(
        "fm_lab.training.trainer.plot_umap_projected_trajectories",
        fake_umap_plot,
    )

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=ZeroVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert (tmp_path / "plots" / "trajectory_umap3d_euler_nfe3.png").exists()
    assert (tmp_path / "plots" / "trajectory_umap3d_euler_nfe3.html").exists()
    assert (tmp_path / "trajectories" / "euler_nfe3_umap3d.npz").exists()
    assert summary["trajectory_umap"]["euler"]["target_points"] == 4
    assert summary["trajectory_umap"]["euler"]["n_neighbors"] == 3


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


def test_train_flow_matching_restores_best_early_stopping_checkpoint(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {"name": "flow_matching"},
        "training": {
            "batch_size": 8,
            "steps": 4,
            "log_every": 1,
            "lr": 0.0,
            "early_stopping": {
                "enabled": True,
                "warmup_steps": 0,
                "patience_steps": 1,
                "min_delta": 0.0,
                "ema_alpha": 1.0,
            },
        },
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=TrainableConstantVelocity(dim=2, value=0.0),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    checkpoint = load_checkpoint(tmp_path / "checkpoint.pt")

    assert metrics["trained_steps"] == 2
    assert metrics["checkpoint_step"] == 1
    assert metrics["restored_best_checkpoint"] is True
    assert metrics["final_loss"] == metrics["checkpoint_loss"]
    assert checkpoint["step"] == 1
    assert checkpoint["metrics"]["checkpoint_step"] == 1


def test_train_flow_matching_kernel_vstar_learned_acceleration_smoke(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {
            "name": "flow_matching",
            "straightness": {"weight": 0.1, "sample_size": 4},
            "interpolant_acceleration": {"weight": 0.001},
            "learned_interpolant": {
                "mode": "kernel_vstar",
                "estimator_size": 4,
                "query_size": 2,
                "bandwidth": 10.0,
            },
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

    train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=TwoMoons(noise=0.0),
        source=GaussianSource(dim=2),
        coupling=IndependentCoupling(),
        path=LearnedAccelerationPath(dim=2, hidden_dim=8, depth=1),
        model=TrainableTimeScaledVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    checkpoint = load_checkpoint(tmp_path / "checkpoint.pt")
    history_text = (tmp_path / "diagnostics" / "training_history.csv").read_text()

    assert "path_state_dict" in checkpoint
    assert "kernel_vstar_straightness_loss" in history_text
    assert "kernel_vstar_effective_sample_size_mean" in history_text
    assert (tmp_path / "plots" / "training_loss.png").exists()
    assert (tmp_path / "samples" / "euler_nfe3.npy").exists()
    assert (tmp_path / "trajectories" / "euler_nfe3.npy").exists()


def test_train_diffusion_epsilon_skips_velocity_sampling(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {"name": "diffusion", "prediction_type": "epsilon"},
        "training": {"batch_size": 8, "steps": 1, "log_every": 1, "lr": 1.0e-3},
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=GaussianDiffusionPath(schedule="linear"),
        model=TrainableConstantVelocity(dim=2, value=0.5),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert metrics["sampling"]["skipped"] is True
    assert "epsilon" in metrics["sampling"]["reason"]
    assert (tmp_path / "checkpoint.pt").exists()
    assert not (tmp_path / "samples").exists()
    assert not (tmp_path / "trajectories").exists()


class TrainableTimeScaledVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * t[:, None] * x


class TrainableConstantVelocity(nn.Module):
    def __init__(self, dim: int, value: float) -> None:
        super().__init__()
        self.velocity = nn.Parameter(torch.full((dim,), value))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return self.velocity.expand_as(x)


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
