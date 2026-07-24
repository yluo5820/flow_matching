"""Reusable optimizer, warmup, and EMA machinery for long training runs."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0 if name == "adam" else 0.01))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError("training.optimizer must be 'adam' or 'adamw'.")


def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    convention: str = "standard",
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if warmup_steps < 0:
        raise ValueError("training.warmup_steps must be non-negative.")
    if warmup_steps == 0:
        return None
    normalized = str(convention).lower()
    if normalized not in {"standard", "zero_start"}:
        raise ValueError(
            "training warmup convention must be 'standard' or 'zero_start'."
        )
    if normalized == "zero_start":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(step, warmup_steps) / warmup_steps,
        )
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0),
    )


def create_ema_model(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, *, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("training.ema_decay must be in [0, 1).")
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer.detach())
