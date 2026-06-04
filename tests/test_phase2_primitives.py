import torch

from fm_lab.data import Checkerboard, GaussianMixture2D, TwoMoons
from fm_lab.paths import LinearPath
from fm_lab.solvers import EulerSolver, HeunSolver, MidpointSolver, RK4Solver


def test_toy_distributions_sample_2d_batches() -> None:
    for distribution in [TwoMoons(), Checkerboard(), GaussianMixture2D()]:
        samples = distribution.sample(32)

        assert samples.shape == (32, 2)
        assert torch.isfinite(samples).all()


def test_linear_path_matches_endpoint_velocity() -> None:
    path = LinearPath()
    x0 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    x1 = torch.tensor([[2.0, 5.0], [4.0, 7.0]])
    t = torch.tensor([0.25, 0.75])

    xt = path.sample_xt(x0, x1, t)
    velocity = path.target_velocity(x0, x1, t)

    assert torch.allclose(xt, torch.tensor([[0.5, 2.0], [3.5, 6.0]]))
    assert torch.allclose(velocity, x1 - x0)


def test_solvers_integrate_constant_velocity() -> None:
    x0 = torch.zeros(8, 2)
    t_grid = torch.linspace(0.0, 1.0, 5)

    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    for solver in [EulerSolver(), HeunSolver(), MidpointSolver(), RK4Solver()]:
        final = solver.solve(v_fn, x0, t_grid)

        assert torch.allclose(final, torch.ones_like(x0), atol=1e-6)
