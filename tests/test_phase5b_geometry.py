import torch

from fm_lab.data import SphericalShell, SwissRoll
from fm_lab.diagnostics import radial_deviation, radial_tangent_velocity_2d
from fm_lab.sources import SphericalShellSource


def test_shell_and_swiss_roll_shapes() -> None:
    assert SphericalShell(dim=5).sample(16).shape == (16, 5)
    assert SwissRoll().sample(16).shape == (16, 3)
    assert SphericalShellSource(dim=4).sample(16).shape == (16, 4)


def test_radial_deviation_is_zero_on_configured_radius() -> None:
    x = torch.tensor([[1.0, 0.0], [0.0, -1.0]])

    result = radial_deviation(x, radii=(1.0,))

    assert result["radial_deviation_mean"] == 0.0


def test_radial_tangent_decomposition_2d() -> None:
    x = torch.tensor([[1.0, 0.0]])
    velocity = torch.tensor([[0.0, 2.0]])

    result = radial_tangent_velocity_2d(x, velocity)

    assert result["radial_velocity_abs_mean"] == 0.0
    assert result["tangent_velocity_abs_mean"] == 2.0
