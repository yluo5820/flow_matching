import torch

from fm_lab.data import Checkerboard, GaussianMixture2D, TwoMoons
from fm_lab.paths import GaussianDiffusionPath, LinearPath, SphericalPath, TangentNormalPath
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


def test_gaussian_diffusion_path_returns_joint_targets() -> None:
    path = GaussianDiffusionPath(schedule="linear", sigma_min=1e-4)
    epsilon = torch.tensor([[2.0, -1.0], [0.5, 1.5]])
    data = torch.tensor([[10.0, 4.0], [2.0, -2.0]])
    t = torch.tensor([0.25, 0.75])

    sample = path.sample_training_tuple(epsilon, data, t)

    expected_xt = torch.tensor([[4.0, 0.25], [1.625, -1.125]])
    expected_score = torch.tensor([[-2.0 / 0.75, 1.0 / 0.75], [-0.5 / 0.25, -1.5 / 0.25]])
    assert torch.allclose(sample.xt, expected_xt)
    assert torch.allclose(sample.epsilon, epsilon)
    assert torch.allclose(sample.score_target, expected_score)
    assert torch.allclose(sample.velocity_target, data - epsilon)
    assert torch.allclose(sample.alpha_t, t)
    assert torch.allclose(sample.sigma_t, 1.0 - t)


def test_solvers_integrate_constant_velocity() -> None:
    x0 = torch.zeros(8, 2)
    t_grid = torch.linspace(0.0, 1.0, 5)

    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    for solver in [EulerSolver(), HeunSolver(), MidpointSolver(), RK4Solver()]:
        final = solver.solve(v_fn, x0, t_grid)

        assert torch.allclose(final, torch.ones_like(x0), atol=1e-6)


def test_spherical_path_stays_on_unit_sphere_for_unit_endpoints() -> None:
    path = SphericalPath(interpolate_radius=True)
    x0 = torch.tensor([[1.0, 0.0]])
    x1 = torch.tensor([[0.0, 1.0]])
    t = torch.tensor([0.5])

    xt = path.sample_xt(x0, x1, t)
    velocity = path.target_velocity(x0, x1, t)

    assert torch.allclose(xt.norm(dim=1), torch.ones(1), atol=1e-6)
    assert torch.isfinite(velocity).all()


def test_tangent_normal_path_takes_short_angular_route() -> None:
    path = TangentNormalPath()
    x0 = torch.tensor([[1.0, 0.0]])
    x1 = torch.tensor([[0.0, 1.0]])
    t = torch.tensor([0.5])

    xt = path.sample_xt(x0, x1, t)
    velocity = path.target_velocity(x0, x1, t)

    expected = torch.tensor([[2**-0.5, 2**-0.5]])
    assert torch.allclose(xt, expected, atol=1e-6)
    assert torch.isfinite(velocity).all()
