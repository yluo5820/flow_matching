import numpy as np
import pytest
import torch

from fm_lab.diagnostics.probes.controls import (
    permutation_null,
    planted_low_rank_control,
    projection_overlap,
)


def test_projection_overlap_is_one_for_equal_and_zero_for_orthogonal_basis() -> None:
    identity = torch.eye(6)

    assert projection_overlap(identity[:, :2], identity[:, :2]) == pytest.approx(1.0)
    assert projection_overlap(identity[:, :2], identity[:, 2:4]) == pytest.approx(0.0)


def test_permuted_labels_return_null_on_exchangeable_rows() -> None:
    values = torch.ones(20, 3)
    labels = np.repeat(np.arange(2), 10)

    def group_gap(rows: torch.Tensor, groups: np.ndarray) -> float:
        first = rows[torch.from_numpy(groups == 0)].mean()
        second = rows[torch.from_numpy(groups == 1)].mean()
        return float((first - second).abs())

    result = permutation_null(
        values,
        labels,
        statistic=group_gap,
        permutations=499,
        seed=7,
    )

    assert result.observed == 0.0
    assert result.p_value == 1.0
    assert np.count_nonzero(result.null_values) == 0


def test_planted_low_rank_control_recovers_dimension_and_subspace() -> None:
    result = planted_low_rank_control(
        ambient_dim=256,
        rank=4,
        rows=128,
        noise_std=0.02,
        seed=11,
    )

    assert result.recovered_rank == 4
    assert result.subspace_overlap > 0.95


@pytest.mark.parametrize(
    ("ambient_dim", "rank", "rows"),
    [(4, 5, 16), (32, 0, 16), (32, 4, 3)],
)
def test_planted_control_rejects_impossible_dimensions(
    ambient_dim: int,
    rank: int,
    rows: int,
) -> None:
    with pytest.raises(ValueError, match="rank"):
        planted_low_rank_control(
            ambient_dim=ambient_dim,
            rank=rank,
            rows=rows,
            noise_std=0.02,
            seed=11,
        )
