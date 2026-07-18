from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fm_lab.geometry_explorer.synthetic_long_tail_geometry import (
    evaluate_local_geometry,
    evaluate_memorization,
    fixed_class_velocity,
    tangent_projection_scores,
)


class ProjectionFlowSolver:
    def __init__(self, tangent_dim: int) -> None:
        self.tangent_dim = tangent_dim

    def solve(
        self,
        v_fn,
        x0: torch.Tensor,
        t_grid: torch.Tensor,
        return_trajectory: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del v_fn, kwargs
        final = x0.clone()
        if float(t_grid[-1]) > float(t_grid[0]):
            final[:, self.tangent_dim :] = 0.0
        return torch.stack([x0, final]) if return_trajectory else final


class ContextRecordingVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_labels: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del t
        self.last_labels = context["class_labels"].detach().cpu()
        return torch.zeros_like(x)


def test_fixed_class_velocity_supplies_conditioning_context() -> None:
    model = ContextRecordingVelocity()
    wrapped = fixed_class_velocity(model, class_id=2)

    result = wrapped(torch.zeros(4, 3), torch.full((4,), 0.5))

    assert torch.equal(model.last_labels, torch.full((4,), 2, dtype=torch.long))
    assert torch.equal(result, torch.zeros(4, 3))


def test_tangent_projection_detects_preserved_and_missing_direction() -> None:
    pushforward = torch.diag(torch.tensor([2.0, 1.0, 0.001]))
    tangents = torch.eye(3)

    scores = tangent_projection_scores(pushforward, tangents, rank=2)

    assert torch.allclose(scores, torch.tensor([1.0, 1.0, 0.0]), atol=1e-5)


def test_tangent_projection_rejects_zero_tangent() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        tangent_projection_scores(torch.eye(2), torch.zeros(1, 2), rank=1)


def test_local_geometry_writes_only_compact_spectra_and_scores(tmp_path: Path) -> None:
    result = evaluate_local_geometry(
        model=ContextRecordingVelocity(),
        ode_solver=ProjectionFlowSolver(tangent_dim=2),
        queries=np.array([[0.2, -0.1, 0.0]], dtype=np.float32),
        class_ids=np.array([1]),
        renderer_tangents=np.array([np.eye(3)], dtype=np.float32),
        tangent_names=("tx", "ty", "tz"),
        renderer_rank=2,
        output_dir=tmp_path / "geometry",
        t_values=(0.5,),
        num_directions=8,
        nfe=4,
        num_trace_samples=None,
        device="cpu",
        seed=3,
        source_revision="test",
    )

    assert result["query_count"] == 1
    assert len(result["class_summary"]) == 1
    with np.load(tmp_path / "geometry" / "spectra.npz") as spectra:
        assert spectra["singular_values"].shape == (1, 3)
        assert set(spectra.files) == {"singular_values"}
    assert not (tmp_path / "geometry" / "pushforwards.npz").exists()


def test_memorization_keeps_copy_and_geometry_metrics_separate(tmp_path: Path) -> None:
    train_images = np.array([[[[0, 1]]], [[[2, 3]]]], dtype=np.uint8)
    heldout_images = np.array([[[[4, 5]]], [[[6, 7]]], [[[8, 9]]]], dtype=np.uint8)
    generated_images = np.array([[[[0, 1]]], [[[5, 6]]]], dtype=np.uint8)
    train_factors = np.array([[0.0, 0.0], [1.0, 1.0]])
    heldout_factors = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    generated_factors = np.array([[0.0, 0.0], [0.8, 0.9]])
    train_features = np.array([[0.0, 0.0], [1.0, 1.0]])
    heldout_features = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    generated_features = np.array([[0.0, 0.0], [0.9, 0.9]])

    result = evaluate_memorization(
        generated_images=generated_images,
        generated_factors=generated_factors,
        generated_features=generated_features,
        training_images=train_images,
        training_factors=train_factors,
        training_features=train_features,
        heldout_images=heldout_images,
        heldout_factors=heldout_factors,
        heldout_features=heldout_features,
        output_dir=tmp_path / "memorization",
        source_revision="test",
        context={"condition_id": "bounded_tail", "training_unique_count": 2},
    )

    assert result["summary"]["exact_copy_rate"] == 0.5
    assert result["summary"]["generated_count"] == 2
    assert "geometric_deficit" not in result["summary"]
    assert result["summary"]["near_duplicate_threshold"] > 0.0
    destination = tmp_path / "memorization"
    assert destination.is_symlink()
    payload = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert payload == result
    assert payload["provenance"]["context"] == {
        "condition_id": "bounded_tail",
        "training_unique_count": 2,
    }


def test_memorization_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "memorization"
    destination.symlink_to(tmp_path / "missing", target_is_directory=True)
    values = np.array([[0.0], [1.0]])
    images = np.arange(2, dtype=np.uint8).reshape(2, 1, 1, 1)

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_memorization(
            generated_images=images,
            generated_factors=values,
            generated_features=values,
            training_images=images,
            training_factors=values,
            training_features=values,
            heldout_images=images,
            heldout_factors=values,
            heldout_features=values,
            output_dir=destination,
        )
