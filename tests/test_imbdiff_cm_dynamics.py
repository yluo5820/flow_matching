from __future__ import annotations

import copy
import csv
import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from fm_lab.diagnostics.imbdiff_cm_dynamics import ImbDiffCMDynamicsObserver
from fm_lab.integrations.official_imbdiff_cm import OfficialImbDiffCMTerms
from fm_lab.training.runtime import update_ema_model


class Conv2d_LoRA(nn.Module):
    def __init__(self, *, rank: int = 2) -> None:
        super().__init__()
        self.r = rank
        self.weight = nn.Parameter(torch.randn(3, 3, 3, 3) * 0.05)
        self.lora_A = nn.Parameter(torch.randn(rank, 27) * 0.05)
        self.lora_B = nn.Parameter(torch.randn(3, rank) * 0.01)

    def forward(self, inputs: torch.Tensor, *, use_cm: bool) -> torch.Tensor:
        weight = self.weight
        if use_cm:
            weight = weight + (self.lora_B @ self.lora_A).reshape_as(weight)
        return F.conv2d(inputs, weight, padding=1)


class TinyCMModel(nn.Module):
    is_official_imbdiff_cm = True

    def __init__(self) -> None:
        super().__init__()
        self.block = Conv2d_LoRA()

    def forward(
        self,
        inputs: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        y: torch.Tensor | None,
        augm,
        use_cm: bool,
    ) -> torch.Tensor:
        del timesteps, y, augm
        return self.block(inputs, use_cm=use_cm)


def _terms(
    model: TinyCMModel,
    *,
    noisy: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    timesteps: torch.Tensor,
) -> OfficialImbDiffCMTerms:
    full = model(noisy, timesteps, y=labels, augm=None, use_cm=True)
    general = model(noisy, timesteps, y=labels, augm=None, use_cm=False)
    base = (full - target).square().flatten(1).mean(1)
    distance = (full - general).square().flatten(1).mean(1)
    consistency_scale = torch.tensor([0.8, 0.15, 0.05])[labels]
    diversity_scale = torch.tensor([0.05, 0.15, 0.8])[labels]
    consistency = 3.0 * consistency_scale * distance
    diversity = -3.0 * diversity_scale * distance
    coefficient = 3.0 * (consistency_scale - 0.2 * diversity_scale)
    total = base + coefficient * distance
    return OfficialImbDiffCMTerms(
        noisy=noisy,
        target=target,
        timesteps=timesteps,
        conditioned_labels=labels,
        capacity_on=full,
        capacity_off=general,
        base_per_sample=base,
        distance_per_sample=distance,
        coefficient_per_sample=coefficient,
        consistency_per_sample=consistency,
        diversity_per_sample=diversity,
        total_per_sample=total,
        loss=total.mean(),
        dropout_mode="independent",
        capacity_on_enabled=True,
        capacity_off_enabled=False,
        unconditional_batch=False,
    )


def test_dynamics_observer_preserves_adam_step_and_writes_measurements(tmp_path) -> None:
    torch.manual_seed(7)
    observed_model = TinyCMModel()
    plain_model = copy.deepcopy(observed_model)
    ema_model = copy.deepcopy(observed_model)
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    noisy = torch.randn(6, 3, 8, 8)
    target = torch.randn_like(noisy)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    timesteps = torch.tensor([100, 200, 400, 500, 700, 900])
    objective = SimpleNamespace(
        class_counts=(100, 10, 1),
        timesteps=1000,
        consistency_weight=1.0,
        diversity_weight=0.2,
    )
    observer = ImbDiffCMDynamicsObserver(
        config={
            "steps": [1],
            "max_layers": 1,
            "component_gradients": True,
            "conditioned_gradients": True,
            "functional_updates": True,
        },
        run_dir=tmp_path,
        model=observed_model,
        objective=objective,
        ema_model=ema_model,
    )
    observed_optimizer = torch.optim.Adam(observed_model.parameters(), lr=1e-3)
    plain_optimizer = torch.optim.Adam(plain_model.parameters(), lr=1e-3)
    observed_terms = _terms(
        observed_model,
        noisy=noisy,
        target=target,
        labels=labels,
        timesteps=timesteps,
    )
    plain_terms = _terms(
        plain_model,
        noisy=noisy,
        target=target,
        labels=labels,
        timesteps=timesteps,
    )

    state = observer.before_backward(
        step=1,
        model=observed_model,
        ema_model=ema_model,
        terms=observed_terms,
        labels=labels,
    )
    observed_terms.loss.backward()
    plain_terms.loss.backward()
    observed_optimizer.step()
    plain_optimizer.step()
    update_ema_model(ema_model, observed_model, decay=0.9)
    observer.after_optimizer_step(
        state,
        model=observed_model,
        ema_model=ema_model,
    )
    summary = observer.finalize(final_step=1)

    for observed, plain in zip(
        observed_model.parameters(),
        plain_model.parameters(),
        strict=True,
    ):
        assert torch.equal(observed, plain)
    assert summary["observed_steps"] == [1]
    output = tmp_path / "cm_dynamics"
    assert json.loads((output / "summary.json").read_text())["complete"] is True

    with (output / "gradient_components.csv").open(newline="") as handle:
        component_rows = list(csv.DictReader(handle))
    assert len(component_rows) == 5 * 4
    total_a = next(
        row
        for row in component_rows
        if row["stratum"] == "total" and row["capacity_group"] == "expert_a"
    )
    assert float(total_a["gradient_norm"]) > 0.0
    assert 0.0 < float(total_a["expert_gradient_energy_fraction"]) < 1.0

    with (output / "layer_gradient_components.csv").open(newline="") as handle:
        layer_component_rows = list(csv.DictReader(handle))
    assert len(layer_component_rows) == 5
    assert {row["component"] for row in layer_component_rows} == {
        "base",
        "consistency",
        "diversity",
        "cm",
        "total",
    }

    with (output / "conditioned_gradients.csv").open(newline="") as handle:
        conditioned_rows = list(csv.DictReader(handle))
    assert len(conditioned_rows) == 6 * 4
    assert {row["stratum"] for row in conditioned_rows} == {
        "many",
        "medium",
        "few",
        "late_low_noise",
        "middle",
        "early_high_noise",
    }

    with (output / "layer_updates.csv").open(newline="") as handle:
        layer_row = next(csv.DictReader(handle))
    assert float(layer_row["factor_a_gradient_norm"]) > 0.0
    assert float(layer_row["factor_b_gradient_norm"]) > 0.0
    assert float(layer_row["effective_expert_update_norm"]) > 0.0
    assert float(layer_row["effective_update_reconstruction_relative_error"]) < 1e-5
    assert float(layer_row["ema_effective_expert_update_norm"]) > 0.0

    with (output / "functional_updates.csv").open(newline="") as handle:
        functional_rows = list(csv.DictReader(handle))
    assert {row["scope"] for row in functional_rows} == {
        "all",
        "many",
        "medium",
        "few",
        "late_low_noise",
        "middle",
        "early_high_noise",
    }
    all_row = next(row for row in functional_rows if row["scope"] == "all")
    assert float(all_row["full_update_rms"]) > 0.0
    assert float(all_row["general_update_rms"]) > 0.0
    assert float(all_row["expert_effect_update_rms"]) > 0.0
    for prefix in ("full", "general", "expert_effect"):
        spectral_sum = sum(
            float(all_row[f"{prefix}_spectral_{band}"])
            for band in ("low", "mid_low", "mid_high", "high")
        )
        assert abs(spectral_sum - 1.0) < 1e-5


def test_zero_initialized_b_factor_exposes_factor_gradient_asymmetry(tmp_path) -> None:
    torch.manual_seed(11)
    model = TinyCMModel()
    model.block.lora_B.data.zero_()
    noisy = torch.randn(3, 3, 8, 8)
    target = torch.randn_like(noisy)
    labels = torch.tensor([0, 1, 2])
    timesteps = torch.tensor([100, 500, 900])
    terms = _terms(
        model,
        noisy=noisy,
        target=target,
        labels=labels,
        timesteps=timesteps,
    )
    observer = ImbDiffCMDynamicsObserver(
        config={
            "steps": [1],
            "max_layers": 1,
            "conditioned_gradients": False,
            "functional_updates": False,
        },
        run_dir=tmp_path,
        model=model,
        objective=SimpleNamespace(
            class_counts=(100, 10, 1),
            timesteps=1000,
            consistency_weight=1.0,
            diversity_weight=0.2,
        ),
        ema_model=None,
    )

    state = observer.before_backward(
        step=1,
        model=model,
        ema_model=None,
        terms=terms,
        labels=labels,
    )

    grad_a = state.selected_gradients["total"]["block.lora_A"]
    grad_b = state.selected_gradients["total"]["block.lora_B"]
    assert torch.count_nonzero(grad_a) == 0
    assert torch.linalg.vector_norm(grad_b) > 0
