import numpy as np
import pytest
import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.functional_calibration import (
    cell_microbatch_rows,
    deterministic_random_unit_direction,
    projected_descent_direction,
    top_centered_covariance_direction,
    virtual_layer_update,
)
from fm_lab.diagnostics.long_tail_geometry.manifest import build_probe_manifest


def _manifest():
    return build_probe_manifest(
        np.arange(12, dtype=np.int64),
        np.repeat(np.arange(2, dtype=np.int64), 6),
        split="a",
        rows_per_class_per_stratum=4,
        batch_size=1,
        time_strata=((0.1, 0.4), (0.6, 0.9)),
        seed=17,
    )


def test_cell_microbatch_rows_returns_stable_within_cell_positions() -> None:
    manifest = _manifest()

    rows = cell_microbatch_rows(manifest, class_id=1, stratum_id=0)

    assert len(rows) == 4
    assert [int(manifest.microbatch_ids[value].item()) for value in rows] == [8, 9, 10, 11]
    assert all(int(manifest.labels[value].item()) == 1 for value in rows)
    assert all(int(manifest.stratum_ids[value].item()) == 0 for value in rows)
    assert [int(value.item()) for value in rows] != [0, 1, 2, 3]


def test_cell_microbatch_rows_rejects_missing_or_mixed_cells() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="no microbatches"):
        cell_microbatch_rows(manifest, class_id=9, stratum_id=0)


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
    assert abs(torch.dot(result.vector.double(), vh[0]).item()) == pytest.approx(
        1.0, abs=1e-6
    )
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
            torch.linalg.vector_norm(expected_projection)
            / torch.linalg.vector_norm(mean_gradient)
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


def test_virtual_layer_update_restores_exact_parameter_after_success_and_error() -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    original = model[0].weight.detach().clone()
    direction = torch.arange(6, dtype=torch.float32) + 1
    direction /= torch.linalg.vector_norm(direction)

    with virtual_layer_update(
        model,
        layer_name="0.weight",
        direction=direction,
        relative_step=1e-3,
    ) as applied_norm:
        assert not torch.equal(model[0].weight, original)
        assert applied_norm == pytest.approx(
            1e-3 * float(torch.linalg.vector_norm(original))
        )
    assert torch.equal(model[0].weight, original)

    with pytest.raises(RuntimeError, match="inside"):
        with virtual_layer_update(
            model,
            layer_name="0.weight",
            direction=direction,
            relative_step=1e-3,
        ):
            raise RuntimeError("inside")
    assert torch.equal(model[0].weight, original)


@pytest.mark.parametrize(
    ("direction", "relative_step", "message"),
    [
        (torch.ones(5), 1e-3, "shape"),
        (torch.ones(6), 0.0, "positive"),
        (torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.float32), 1e-3, "unit"),
    ],
)
def test_virtual_layer_update_rejects_invalid_update(
    direction: torch.Tensor,
    relative_step: float,
    message: str,
) -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))

    with pytest.raises(ValueError, match=message):
        with virtual_layer_update(
            model,
            layer_name="0.weight",
            direction=direction,
            relative_step=relative_step,
        ):
            pass
