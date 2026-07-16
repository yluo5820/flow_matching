import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.manifest import ProbeBatch
from fm_lab.diagnostics.long_tail_geometry.measurements import (
    CheckpointMeasurements,
    collect_checkpoint_measurements,
)
from fm_lab.paths import LinearPath
from fm_lab.training.losses import build_objective


class TinyConditionalVelocity(nn.Module):
    is_class_conditional = True

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del context
        return self.output(torch.tanh(self.hidden(x + t[:, None])))


def _probe_batches(*, source_shift: float = 0.0) -> tuple[ProbeBatch, ...]:
    batches = []
    for batch_id in range(3):
        x0 = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
        x0 = x0 + batch_id * 0.1 + source_shift
        x1 = torch.tensor([[0.8, 0.2], [-0.1, 0.7]]) - batch_id * 0.05
        class_id = batch_id % 2
        stratum_id = batch_id % 2
        batches.append(
            ProbeBatch(
                x0=x0,
                x1=x1,
                t=torch.full((2,), 0.2 + 0.2 * stratum_id),
                labels=torch.full((2,), class_id, dtype=torch.int64),
                original_indices=np.array([2 * batch_id, 2 * batch_id + 1]),
                stratum_ids=np.full(2, stratum_id, dtype=np.int64),
                microbatch_ids=np.full(2, batch_id, dtype=np.int64),
            )
        )
    return tuple(batches)


def _ordinary_objective():
    return build_objective(
        {
            "name": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
            "modifiers": [],
        }
    )


def _collect(model: nn.Module | None = None) -> CheckpointMeasurements:
    if model is None:
        torch.manual_seed(7)
        model = TinyConditionalVelocity()
    return collect_checkpoint_measurements(
        model=model,
        objective=_ordinary_objective(),
        path=LinearPath(),
        batches_by_view={
            "a": _probe_batches(),
            "a_source_1": _probe_batches(source_shift=0.03),
            "b": _probe_batches(source_shift=0.01),
            "b_source_1": _probe_batches(source_shift=0.04),
        },
        layer_names=("hidden.weight", "output.weight"),
        sketch_dim=8,
        sketch_seed=23,
        checkpoint_step=3,
        checkpoint_sha256="a" * 64,
        preregistration_sha256="b" * 64,
        manifest_digests={
            "a": "c" * 64,
            "a_source_1": "d" * 64,
            "b": "e" * 64,
            "b_source_1": "f" * 64,
        },
    )


def test_collection_streams_fixed_sketches_without_model_mutation(monkeypatch) -> None:
    torch.manual_seed(7)
    model = TinyConditionalVelocity()
    model.train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    real_zeros = torch.zeros

    def checked_zeros(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], tuple) else size
        assert shape not in {(16, 16), (16, 8)}
        return real_zeros(*size, **kwargs)

    monkeypatch.setattr(torch, "zeros", checked_zeros)

    measurements = _collect(model)

    assert len(measurements.metadata) == 12
    assert set(measurements.metadata["probe_view"]) == {
        "a",
        "a_source_1",
        "b",
        "b_source_1",
    }
    assert set(measurements.metadata["target_split"]) == {"a", "b"}
    assert measurements.sketches["hidden.weight"].shape == (12, 8)
    assert measurements.sketches["output.weight"].shape == (12, 8)
    assert measurements.exact_norms["hidden.weight"].shape == (12,)
    assert torch.all(measurements.exact_norms["hidden.weight"] > 0)
    assert torch.allclose(
        torch.linalg.vector_norm(measurements.sketches["hidden.weight"], dim=1),
        torch.ones(12),
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[name], model.state_dict()[name]) for name in before)
    assert model.training is True


def test_collection_rejects_mixed_cell_microbatch() -> None:
    batches = list(_probe_batches())
    mixed = dataclasses.replace(batches[0], labels=torch.tensor([0, 1]))

    with pytest.raises(ValueError, match="one class"):
        collect_checkpoint_measurements(
            model=TinyConditionalVelocity(),
            objective=_ordinary_objective(),
            path=LinearPath(),
            batches_by_view={"a": (mixed,)},
            layer_names=("hidden.weight",),
            sketch_dim=8,
            sketch_seed=23,
            checkpoint_step=3,
            checkpoint_sha256="a" * 64,
            preregistration_sha256="b" * 64,
            manifest_digests={"a": "c" * 64},
        )


def test_checkpoint_measurements_round_trip_parquet_and_npz(tmp_path: Path) -> None:
    original = _collect()

    original.save(tmp_path)
    restored = CheckpointMeasurements.load(tmp_path)

    pd.testing.assert_frame_equal(restored.metadata, original.metadata)
    assert restored.layer_shapes == original.layer_shapes
    assert restored.manifest_digests == original.manifest_digests
    for layer_name in original.sketches:
        assert torch.equal(restored.sketches[layer_name], original.sketches[layer_name])
        assert torch.equal(restored.exact_norms[layer_name], original.exact_norms[layer_name])
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["passed"] is True
    assert complete["row_count"] == 12
    assert complete["checkpoint_sha256"] == "a" * 64


def test_measurement_save_is_idempotent_but_refuses_different_provenance(
    tmp_path: Path,
) -> None:
    original = _collect()
    original.save(tmp_path)
    original.save(tmp_path)
    changed = dataclasses.replace(original, checkpoint_sha256="0" * 64)

    with pytest.raises(ValueError, match="completed measurement"):
        changed.save(tmp_path)


def test_load_rejects_tampered_measurement_file(tmp_path: Path) -> None:
    original = _collect()
    original.save(tmp_path)
    parquet_path = tmp_path / "gradient_rows.parquet"
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        CheckpointMeasurements.load(tmp_path)
