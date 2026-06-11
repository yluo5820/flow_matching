import torch

from fm_lab.data import (
    GaussianMixture3D,
    HelixMixture,
    MultiSwissRoll,
    MultiTorus,
    NestedSphericalShells,
    SphericalShell,
    SwissRoll,
    Torus,
)
from fm_lab.diagnostics import radial_deviation, radial_tangent_velocity_2d
from fm_lab.sources import SphericalShellSource


def test_shell_and_swiss_roll_shapes() -> None:
    assert SphericalShell(dim=5).sample(16).shape == (16, 5)
    assert SwissRoll().sample(16).shape == (16, 3)
    assert SphericalShellSource(dim=4).sample(16).shape == (16, 4)


def test_harder_3d_toy_shapes() -> None:
    distributions = [
        GaussianMixture3D(n_modes=6),
        MultiSwissRoll(n_rolls=2),
        Torus(),
        MultiTorus(n_tori=2),
        HelixMixture(n_helixes=3),
        NestedSphericalShells(radii=(0.5, 1.0), noise=0.0),
    ]

    for distribution in distributions:
        samples = distribution.sample(32)

        assert samples.shape == (32, 3)
        assert torch.isfinite(samples).all()
        assert distribution.metadata()["dim"] == 3


def test_gaussian_mixture_3d_log_prob_shape() -> None:
    distribution = GaussianMixture3D(n_modes=4)
    samples = distribution.sample(8)

    log_prob = distribution.log_prob(samples)

    assert log_prob is not None
    assert log_prob.shape == (8,)


def test_nested_shell_samples_land_on_configured_radii_without_noise() -> None:
    distribution = NestedSphericalShells(radii=(0.5, 1.0), noise=0.0)

    samples = distribution.sample(64)
    distances = (samples.norm(dim=1)[:, None] - torch.tensor([0.5, 1.0])[None, :]).abs()

    assert distances.min(dim=1).values.max() < 1e-5


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
