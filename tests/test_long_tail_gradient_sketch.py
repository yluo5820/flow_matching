import pytest
import torch

from fm_lab.diagnostics.long_tail_geometry.sketch import (
    CountSketchSpec,
    validate_sketch,
)


def test_countsketch_is_deterministic_and_has_linear_storage() -> None:
    first = CountSketchSpec.build(input_dim=10_000, output_dim=4_096, seed=23)
    second = CountSketchSpec.build(input_dim=10_000, output_dim=4_096, seed=23)

    assert torch.equal(first.buckets, second.buckets)
    assert torch.equal(first.signs, second.signs)
    assert first.buckets.numel() + first.signs.numel() == 20_000


def test_countsketch_approximates_cosines_without_dense_projection(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(23)
    basis, _ = torch.linalg.qr(torch.randn(10_000, 4, generator=generator))
    coefficients = torch.randn(16, 4, generator=generator) * torch.tensor(
        [4.0, 3.0, 2.0, 1.0]
    )
    rows = coefficients @ basis.T
    rows = rows + 1.0e-3 * torch.randn(16, 10_000, generator=generator)
    spec = CountSketchSpec.build(input_dim=10_000, output_dim=4_096, seed=23)
    real_zeros = torch.zeros

    def checked_zeros(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], tuple) else size
        assert shape not in {(10_000, 10_000), (10_000, 4_096)}
        return real_zeros(*size, **kwargs)

    monkeypatch.setattr(torch, "zeros", checked_zeros)

    sketched = spec.apply(rows)
    result = validate_sketch(rows, sketched, rank=4)

    assert sketched.shape == (16, 4_096)
    assert result.max_absolute_cosine_error < 0.04
    assert result.normalized_subspace_overlap_error < 0.06


def test_countsketch_rejects_input_dimension_mismatch() -> None:
    spec = CountSketchSpec.build(input_dim=12, output_dim=4, seed=1)
    with pytest.raises(ValueError, match="input dimension"):
        spec.apply(torch.randn(2, 11))
