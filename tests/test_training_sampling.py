import numpy as np
import torch
from torch import nn

from fm_lab.data import GaussianMixture3D
from fm_lab.solvers import EulerSolver, HeunSolver
from fm_lab.sources import GaussianSource
from fm_lab.training.trainer import sample_and_plot


class ZeroVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


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
