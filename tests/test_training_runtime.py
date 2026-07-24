import torch
from torch import nn

from fm_lab.training.runtime import (
    build_optimizer,
    build_warmup_scheduler,
    create_ema_model,
    update_ema_model,
)


def test_reference_optimizer_is_adam_without_weight_decay() -> None:
    model = nn.Linear(2, 1)
    optimizer = build_optimizer(model, {"optimizer": "adam", "lr": 2e-4})

    assert type(optimizer) is torch.optim.Adam
    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_linear_warmup_reaches_base_learning_rate() -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = build_warmup_scheduler(optimizer, warmup_steps=2)

    assert optimizer.param_groups[0]["lr"] == 0.5
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 1.0


def test_zero_start_warmup_begins_at_zero() -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = build_warmup_scheduler(
        optimizer,
        warmup_steps=2,
        convention="zero_start",
    )

    assert optimizer.param_groups[0]["lr"] == 0.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_ema_update_tracks_parameters_and_copies_buffers() -> None:
    model = nn.BatchNorm1d(2)
    ema = create_ema_model(model)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.running_mean.fill_(4.0)

    update_ema_model(ema, model, decay=0.5)

    assert torch.equal(ema.weight, torch.full_like(ema.weight, 2.0))
    assert torch.equal(ema.running_mean, torch.full_like(ema.running_mean, 4.0))
    assert not any(parameter.requires_grad for parameter in ema.parameters())
