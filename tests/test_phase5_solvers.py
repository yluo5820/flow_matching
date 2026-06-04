import torch

from fm_lab.solvers import EulerSolver, ScipyDopri5Solver, make_time_grid


def test_time_grid_schedules_are_monotone() -> None:
    for schedule in ["uniform", "quadratic", "reverse_quadratic", "cosine"]:
        grid = make_time_grid(8, schedule=schedule)

        assert torch.all(grid[1:] >= grid[:-1])
        assert grid[0] == 0.0
        assert grid[-1] == 1.0


def test_scipy_dopri_matches_euler_for_constant_velocity() -> None:
    x0 = torch.zeros(4, 2)
    t_grid = make_time_grid(2)

    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    euler = EulerSolver().solve(v_fn, x0, t_grid)
    dopri = ScipyDopri5Solver().solve(v_fn, x0, t_grid)

    assert torch.allclose(dopri, euler, atol=1e-5)
