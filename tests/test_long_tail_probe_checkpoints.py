from pathlib import Path

import torch
from long_tail_geometry_helpers import build_probe_fixture

from fm_lab.diagnostics.long_tail_geometry.checkpoints import (
    evaluate_probe_loss,
    restore_probe_model,
)
from fm_lab.utils.checkpoints import save_checkpoint


def test_restored_checkpoint_reproduces_probe_loss_bitwise(tmp_path: Path) -> None:
    config, target, source, path, objective, model, manifest = build_probe_fixture(
        tmp_path
    )
    before = evaluate_probe_loss(
        model=model,
        objective=objective,
        path=path,
        manifest=manifest,
        target=target,
        source=source,
        device=torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        step=3,
        config=config,
        prediction_contract={
            "path": "linear",
            "objective": "flow_matching",
            "model_output": "target",
            "loss_space": "velocity",
        },
        metrics={"probe_loss": before.mean_loss},
    )

    restored, restored_config = restore_probe_model(
        checkpoint_path,
        device=torch.device("cpu"),
    )
    after = evaluate_probe_loss(
        model=restored,
        objective=objective,
        path=path,
        manifest=manifest,
        target=target,
        source=source,
        device=torch.device("cpu"),
    )

    assert restored_config == config
    assert before.mean_loss == after.mean_loss
    assert before.row_losses_sha256 == after.row_losses_sha256
    assert torch.equal(before.row_losses, after.row_losses)
