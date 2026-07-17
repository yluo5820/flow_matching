import pytest
import torch

from fm_lab.diagnostics.probes.subspaces import (
    PrincipalDirection,
    deterministic_random_unit_direction,
    projected_descent_direction,
    top_centered_covariance_direction,
)


def test_top_centered_direction_returns_public_result_type() -> None:
    result = top_centered_covariance_direction(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))

    assert isinstance(result, PrincipalDirection)
    assert result.vector.shape == (2,)
    assert result.explained_fraction == pytest.approx(1.0)


def test_top_centered_direction_matches_dense_svd_up_to_sign() -> None:
    rows = torch.tensor(
        [
            [3.0, 0.0, 0.2, 1.0],
            [2.0, 1.0, 0.1, 1.0],
            [-2.0, 0.0, -0.2, 1.0],
            [-3.0, -1.0, -0.1, 1.0],
        ]
    )

    result = top_centered_covariance_direction(rows)
    _, singular_values, vh = torch.linalg.svd(
        rows.double() - rows.double().mean(dim=0),
        full_matrices=False,
    )

    assert result.vector.dtype == torch.float32
    assert result.vector.device.type == "cpu"
    assert torch.linalg.vector_norm(result.vector).item() == pytest.approx(1.0)
    assert abs(torch.dot(result.vector.double(), vh[0]).item()) == pytest.approx(1.0, abs=1e-6)
    assert result.eigenvalue == pytest.approx(float(singular_values[0].square()))
    assert result.explained_fraction == pytest.approx(
        float(singular_values[0].square() / singular_values.square().sum())
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (torch.ones(1, 3), "two rows"),
        (torch.tensor([[1.0, float("nan")], [2.0, 3.0]]), "finite"),
        (torch.ones(3, 2), "zero centered rank"),
        (torch.ones(2, 3, 1), "matrix"),
    ],
)
def test_top_centered_direction_rejects_invalid_rows(
    rows: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        top_centered_covariance_direction(rows)


def test_projected_descent_direction_is_oriented_and_reports_fraction() -> None:
    subspace = torch.tensor([3.0, 4.0, 0.0])
    mean_gradient = torch.tensor([2.0, -1.0, 8.0])

    result = projected_descent_direction(
        subspace,
        mean_gradient,
        minimum_projection_fraction=1e-8,
    )

    unit = subspace / torch.linalg.vector_norm(subspace)
    expected_projection = unit * torch.dot(unit, mean_gradient)
    assert torch.allclose(
        result.vector,
        -expected_projection / torch.linalg.vector_norm(expected_projection),
    )
    assert torch.dot(result.vector, mean_gradient).item() < 0
    assert result.projection_fraction == pytest.approx(
        float(
            torch.linalg.vector_norm(expected_projection) / torch.linalg.vector_norm(mean_gradient)
        )
    )


def test_projected_descent_direction_rejects_invalid_or_negligible_projection() -> None:
    with pytest.raises(ValueError, match="same shape"):
        projected_descent_direction(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="negligible"):
        projected_descent_direction(
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            minimum_projection_fraction=1e-6,
        )


def test_random_unit_direction_is_deterministic_and_keyed() -> None:
    first = deterministic_random_unit_direction(
        100,
        base_seed=19,
        key=(0, 20_000, 2, "middle.conv2.weight", 7),
    )
    repeated = deterministic_random_unit_direction(
        100,
        base_seed=19,
        key=(0, 20_000, 2, "middle.conv2.weight", 7),
    )
    changed = deterministic_random_unit_direction(
        100,
        base_seed=19,
        key=(0, 20_000, 2, "middle.conv2.weight", 8),
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert torch.linalg.vector_norm(first).item() == pytest.approx(1.0, abs=1e-6)
