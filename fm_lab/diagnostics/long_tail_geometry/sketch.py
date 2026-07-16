"""Sparse fixed CountSketch projections and fidelity measurements."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CountSketchSpec:
    input_dim: int
    output_dim: int
    seed: int
    buckets: torch.Tensor
    signs: torch.Tensor

    @classmethod
    def build(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        seed: int,
    ) -> CountSketchSpec:
        if input_dim < 1:
            raise ValueError("CountSketch input_dim must be positive.")
        if output_dim < 2:
            raise ValueError("CountSketch output_dim must be at least two.")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        buckets = torch.randint(
            output_dim,
            (input_dim,),
            generator=generator,
            dtype=torch.int64,
        )
        signs = (
            2
            * torch.randint(
                2,
                (input_dim,),
                generator=generator,
                dtype=torch.int64,
            )
            - 1
        ).float()
        return cls(
            input_dim=int(input_dim),
            output_dim=int(output_dim),
            seed=int(seed),
            buckets=buckets,
            signs=signs,
        )

    def apply(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 2 or rows.shape[1] != self.input_dim:
            raise ValueError(
                f"CountSketch input dimension must be {self.input_dim}, got {tuple(rows.shape)}."
            )
        buckets = self.buckets.to(device=rows.device)
        signs = self.signs.to(device=rows.device, dtype=rows.dtype)
        output = torch.zeros(
            (rows.shape[0], self.output_dim),
            device=rows.device,
            dtype=rows.dtype,
        )
        output.scatter_add_(
            1,
            buckets.expand(rows.shape[0], -1),
            rows * signs,
        )
        return output


@dataclass(frozen=True)
class SketchValidation:
    max_absolute_cosine_error: float
    normalized_subspace_overlap_error: float


def validate_sketch(
    exact_rows: torch.Tensor,
    sketched_rows: torch.Tensor,
    *,
    rank: int,
) -> SketchValidation:
    """Compare row cosines and sample-space principal subspaces."""

    if exact_rows.ndim != 2 or sketched_rows.ndim != 2:
        raise ValueError("Sketch validation inputs must be matrices.")
    if exact_rows.shape[0] != sketched_rows.shape[0]:
        raise ValueError("Exact and sketched rows must have the same row count.")
    if rank < 1 or rank > exact_rows.shape[0]:
        raise ValueError("Sketch validation rank is out of range.")
    exact = _normalize_rows(exact_rows.double())
    sketched = _normalize_rows(sketched_rows.double())
    exact_gram = exact @ exact.T
    sketched_gram = sketched @ sketched.T
    cosine_error = float((exact_gram - sketched_gram).abs().max())
    exact_basis = _top_eigenvectors(exact_gram, rank)
    sketched_basis = _top_eigenvectors(sketched_gram, rank)
    overlap = torch.linalg.matrix_norm(
        exact_basis.T @ sketched_basis,
        ord="fro",
    ).square() / rank
    subspace_error = float((1.0 - overlap).clamp(0.0, 1.0))
    return SketchValidation(
        max_absolute_cosine_error=cosine_error,
        normalized_subspace_overlap_error=subspace_error,
    )


def _normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(rows, dim=1)
    if not torch.isfinite(rows).all() or torch.any(norms == 0):
        raise ValueError("Sketch validation rows must be finite and nonzero.")
    return rows / norms[:, None]


def _top_eigenvectors(gram: torch.Tensor, rank: int) -> torch.Tensor:
    _, eigenvectors = torch.linalg.eigh(gram)
    return eigenvectors[:, -rank:]
