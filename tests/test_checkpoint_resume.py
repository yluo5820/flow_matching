import random

import numpy as np
import torch
from torch import nn

from fm_lab.utils.checkpoints import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


def test_checkpoint_round_trips_extended_training_state(tmp_path) -> None:
    model = nn.Linear(2, 2)
    ema_model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=7,
        config={"training": {"steps": 10}},
        metrics={"loss": 1.0},
        prediction_contract={
            "path": "linear",
            "objective": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
        },
        training_contract={
            "version": 2,
            "payload": {"objective": {}, "path": {}, "data": {}},
            "sha256": "digest",
        },
        resume_state={
            "version": 1,
            "early_stopping": {"enabled": False},
            "best_training_state": None,
        },
        history=[{"step": 1, "loss": 2.0}],
        rng_state=capture_rng_state(),
    )
    payload = load_checkpoint(path)

    assert payload["step"] == 7
    assert "model_state_dict" in payload
    assert "ema_model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert "scheduler_state_dict" in payload
    assert payload["history"] == [{"step": 1, "loss": 2.0}]
    assert payload["prediction_contract"] == {
        "path": "linear",
        "objective": "flow_matching",
        "model_output": "velocity",
        "loss_space": "velocity",
    }
    assert payload["training_contract"] == {
        "version": 2,
        "payload": {"objective": {}, "path": {}, "data": {}},
        "sha256": "digest",
    }
    assert payload["resume_state"] == {
        "version": 1,
        "early_stopping": {"enabled": False},
        "best_training_state": None,
    }
    assert set(payload["rng_state_dict"]) >= {"python", "numpy", "torch"}


def test_rng_state_restore_reproduces_all_cpu_generators() -> None:
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(()))

    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(()))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
