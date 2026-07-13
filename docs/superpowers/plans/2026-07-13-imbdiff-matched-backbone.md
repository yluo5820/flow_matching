# Matched-Backbone Local ImbDiff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all five local CIFAR-10 ImbDiff methods on the previous successful 32/64/128 image U-Net backbone with optional CM adapters and a bounded, early-stopped two-hour-scale profile.

**Architecture:** Move the DDPM U-Net's private capacity routing into reusable model utilities, then add zero-initialized up-path adapters to `ImageUNetVelocity` without changing its capacity-disabled backbone. Reconfigure only the five local ImbDiff YAML files for the shared 1.08M-parameter conditional backbone, batch 32, 8,000 maximum steps, and approximately 64-step DDIM; keep early stopping active while restoring its live and EMA snapshots from the same selected step.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, Ruff, YAML configs, `.conda/fm_lab`.

## Global Constraints

- Use `ImageUNetVelocity` with CHW shape `[3, 32, 32]`, base channels 32, time embedding 128, SiLU, and a zero-initialized head in all five local configs.
- Preserve the capacity-disabled image U-Net's existing convolution types, backbone state-dict keys, and tensor shapes.
- Install CM adapters only in `up1_block` and `up0_block`; their zero-initialized B factors must initially leave base and full outputs identical.
- Retain discrete diffusion, epsilon/x-vloss targets, classifier-free conditioning, CBDM, OC, CM, timestep diagnostics, and live/EMA comparison behavior.
- Keep early stopping enabled with warmup 4,000 and patience 2,000; do not add a wall-clock kill switch.
- Restore EMA from the same best-step snapshot as the live model without changing the early-stopping decision.
- Change only the five YAML files under `configs/imbdiff/local/`; paper-scale CIFAR-10 and CIFAR-100 configs remain byte-for-byte unchanged.
- Preserve the unrelated untracked `docs/experimental_ideas_retrospective.md` and exclude it from every commit.
- Run all verification through `.conda/fm_lab`; commit and push each verified nontrivial checkpoint.

---

### Task 1: Extract reusable capacity routing without changing DDPM behavior

**Files:**
- Modify: `fm_lab/models/capacity.py`
- Modify: `fm_lab/models/ddpm_unet.py`
- Modify: `tests/test_capacity_manipulation.py`

**Interfaces:**
- Produces: `CapacityConfig.build(...) -> CapacityConfig`, `CapacityConfig.conv(...) -> nn.Conv2d`, `apply_capacity_conv(layer, inputs, use_capacity=...) -> torch.Tensor`, and `use_capacity_from_context(context) -> bool`.
- Preserves: `DDPMUNet.capacity_metadata()` values and every existing DDPM/CM forward behavior.

- [ ] **Step 1: Add failing tests for the reusable capacity API**

Add imports and tests to `tests/test_capacity_manipulation.py`:

```python
import pytest

from fm_lab.models.capacity import (
    CapacityConfig,
    apply_capacity_conv,
    use_capacity_from_context,
)


def test_capacity_config_builds_only_selected_switchable_convolutions() -> None:
    capacity = CapacityConfig.build(
        rank=0,
        rank_ratio=0.25,
        adapter_scale=0.5,
        parts=["up"],
    )
    up = capacity.conv("up", 8, 12, 3, padding=1)
    down = capacity.conv("down", 8, 12, 3, padding=1)
    inputs = torch.randn(2, 8, 5, 5)

    assert isinstance(up, models.SwitchableLowRankConv2d)
    assert isinstance(down, torch.nn.Conv2d)
    assert not isinstance(down, models.SwitchableLowRankConv2d)
    assert torch.equal(
        apply_capacity_conv(up, inputs, use_capacity=True),
        apply_capacity_conv(up, inputs, use_capacity=False),
    )
    assert use_capacity_from_context({"use_capacity": False}) is False
    assert use_capacity_from_context({}) is True


def test_capacity_config_rejects_unknown_model_parts() -> None:
    with pytest.raises(ValueError, match="Unsupported capacity parts"):
        CapacityConfig.build(
            rank=1,
            rank_ratio=0.0,
            adapter_scale=1.0,
            parts=["unknown"],
        )
```

- [ ] **Step 2: Run the new tests and verify the public API is missing**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_capacity_manipulation.py -k "capacity_config"
```

Expected: collection fails because `CapacityConfig` is not exported by `fm_lab.models.capacity`.

- [ ] **Step 3: Move the generic routing primitives into `capacity.py`**

Keep `SwitchableLowRankConv2d` unchanged and add:

```python
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityConfig:
    rank: int
    rank_ratio: float
    adapter_scale: float
    parts: frozenset[str]

    @classmethod
    def build(
        cls,
        *,
        rank: int,
        rank_ratio: float,
        adapter_scale: float,
        parts: Sequence[str],
    ) -> "CapacityConfig":
        normalized_parts = frozenset(str(part).lower() for part in parts)
        supported = {"head", "down", "middle", "up", "tail"}
        invalid = normalized_parts - supported
        if invalid:
            raise ValueError(f"Unsupported capacity parts: {sorted(invalid)}")
        return cls(
            rank=int(rank),
            rank_ratio=float(rank_ratio),
            adapter_scale=float(adapter_scale),
            parts=normalized_parts,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.parts) and (self.rank > 0 or self.rank_ratio > 0)

    def conv(
        self,
        part: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
    ) -> nn.Conv2d:
        if part not in self.parts:
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )
        return SwitchableLowRankConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            rank=self.rank,
            rank_ratio=self.rank_ratio,
            adapter_scale=self.adapter_scale,
        )


def apply_capacity_conv(
    layer: nn.Conv2d,
    inputs: torch.Tensor,
    *,
    use_capacity: bool,
) -> torch.Tensor:
    if isinstance(layer, SwitchableLowRankConv2d):
        return layer(inputs, use_adapter=use_capacity)
    return layer(inputs)


def use_capacity_from_context(context: object) -> bool:
    if isinstance(context, dict) and "use_capacity" in context:
        return bool(context["use_capacity"])
    return True
```

In `fm_lab/models/ddpm_unet.py`, remove `from dataclasses import dataclass`, import
these names, replace `_CapacityConfig`, `_apply_conv`, and
`_use_capacity_from_context` references with the public names, and delete the
three private definitions:

```python
from fm_lab.models.capacity import (
    CapacityConfig,
    SwitchableLowRankConv2d,
    apply_capacity_conv,
    use_capacity_from_context,
)
```

- [ ] **Step 4: Verify the reusable API and all existing DDPM capacity behavior**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_capacity_manipulation.py tests/test_ddpm_unet.py
.conda/fm_lab/bin/ruff check fm_lab/models/capacity.py fm_lab/models/ddpm_unet.py tests/test_capacity_manipulation.py
git diff --check
```

Expected: all capacity and DDPM U-Net tests pass; Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit and push the refactor checkpoint**

```bash
git add fm_lab/models/capacity.py fm_lab/models/ddpm_unet.py tests/test_capacity_manipulation.py
git commit -m "Share model capacity routing"
git push
```

---

### Task 2: Add optional up-path capacity to the previous image U-Net

**Files:**
- Modify: `fm_lab/models/image.py`
- Modify: `fm_lab/experiments/factory.py`
- Modify: `tests/test_capacity_manipulation.py`
- Modify: `tests/test_class_conditioning.py`

**Interfaces:**
- Consumes: Task 1's `CapacityConfig`, `apply_capacity_conv`, and `use_capacity_from_context`.
- Produces: optional image-U-Net constructor arguments `capacity_rank`, `capacity_rank_ratio`, `capacity_adapter_scale`, and `capacity_parts`, plus `ImageUNetVelocity.capacity_metadata() -> dict[str, object]`.

- [ ] **Step 1: Add a failing image-U-Net factory and adapter-placement test**

Add this config helper and test to `tests/test_capacity_manipulation.py`:

```python
def _cm_image_model_config() -> dict:
    return {
        "model": {
            "name": "image_unet",
            "image_shape": [3, 8, 8],
            "base_channels": 32,
            "time_embedding_dim": 128,
            "activation": "silu",
            "zero_init_head": True,
            "capacity": {
                "enabled": True,
                "rank_ratio": 0.25,
                "adapter_scale": 0.5,
                "parts": ["up"],
            },
        },
        "conditioning": {"enabled": True, "num_classes": 10},
        "diffusion": {"timesteps": 1000},
    }


def test_image_unet_factory_places_cm_capacity_only_in_up_blocks() -> None:
    model = build_model(_cm_image_model_config(), dim=3 * 8 * 8)
    adapter_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, models.SwitchableLowRankConv2d)
    ]

    assert adapter_names == [
        "up1_block.conv1",
        "up1_block.conv2",
        "up0_block.conv1",
        "up0_block.conv2",
    ]
    assert model.capacity_metadata() == {
        "enabled": True,
        "rank": 0,
        "rank_ratio": 0.25,
        "adapter_scale": 0.5,
        "parts": ["up"],
        "adapter_layers": 4,
    }
```

- [ ] **Step 2: Run the placement test and verify no image adapters are built**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_capacity_manipulation.py::test_image_unet_factory_places_cm_capacity_only_in_up_blocks
```

Expected: FAIL because the factory ignores `model.capacity` for `image_unet` and the model has no `capacity_metadata` method.

- [ ] **Step 3: Add a failing context-switch and CM-objective compatibility test**

Add imports and tests:

```python
from fm_lab.training.losses import build_objective


def test_image_unet_capacity_switch_preserves_base_branch() -> None:
    torch.manual_seed(4)
    model = build_model(_cm_image_model_config(), dim=3 * 8 * 8).eval()
    with torch.no_grad():
        model.output_block[-1].weight.normal_(std=0.02)
        model.output_block[-1].bias.zero_()
    inputs = torch.randn(2, 3 * 8 * 8)
    timesteps = torch.tensor([10, 20])
    labels = torch.tensor([1, 2])
    full_context = {"class_labels": labels, "use_capacity": True}
    base_context = {"class_labels": labels, "use_capacity": False}

    initial_full = model(inputs, timesteps, context=full_context)
    initial_base = model(inputs, timesteps, context=base_context)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, models.SwitchableLowRankConv2d):
                module.adapter_b.normal_(std=0.2)
    changed_full = model(inputs, timesteps, context=full_context)
    unchanged_base = model(inputs, timesteps, context=base_context)

    assert torch.equal(initial_full, initial_base)
    assert not torch.equal(changed_full, initial_full)
    assert torch.equal(unchanged_base, initial_base)


def test_cm_objective_accepts_capacity_enabled_image_unet() -> None:
    model = build_model(_cm_image_model_config(), dim=3 * 8 * 8)
    objective = build_objective(
        {
            "name": "cm",
            "prediction_type": "epsilon",
            "oc": {"transfer_mode": "t2h", "cut_time": -1},
            "cm": {"consistency_weight": 1.0, "diversity_weight": 0.2},
        },
        diffusion_config={"timesteps": 1000},
        class_counts=[100, 50, 25, 12, 6, 3, 2, 1, 1, 1],
    )
    labels = torch.tensor([0, 9])

    loss, metrics = objective(
        model=model,
        path=None,
        x0=torch.randn(2, 3 * 8 * 8),
        x1=torch.randn(2, 3 * 8 * 8),
        t=torch.tensor([100, 900]),
        class_labels=labels,
        original_class_labels=labels,
    )

    assert torch.isfinite(loss)
    assert "cm_loss" in metrics
```

- [ ] **Step 4: Run both behavior tests and verify they fail on missing routing**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_capacity_manipulation.py -k "image_unet or cm_objective_accepts"
```

Expected: FAIL because the image U-Net does not install or switch adapters.

- [ ] **Step 5: Implement optional image-U-Net capacity without changing the base path**

Import Task 1's helpers in `fm_lab/models/image.py`, extend `ImageUNetVelocity.__init__`, validate only the designed up part, and build one capacity config:

```python
from collections.abc import Sequence

from fm_lab.models.capacity import (
    CapacityConfig,
    SwitchableLowRankConv2d,
    apply_capacity_conv,
    use_capacity_from_context,
)

# New ImageUNetVelocity.__init__ keyword arguments:
capacity_rank: int = 0,
capacity_rank_ratio: float = 0.0,
capacity_adapter_scale: float = 1.0,
capacity_parts: Sequence[str] = (),

self._capacity = CapacityConfig.build(
    rank=capacity_rank,
    rank_ratio=capacity_rank_ratio,
    adapter_scale=capacity_adapter_scale,
    parts=capacity_parts,
)
unsupported_parts = self._capacity.parts - {"up"}
if unsupported_parts:
    raise ValueError(
        f"ImageUNetVelocity supports capacity only in ['up'], got {sorted(unsupported_parts)}."
    )
```

Pass `capacity=self._capacity` and `capacity_part="up"` only to `up1_block` and `up0_block`. Extend `TimeResBlock` with optional capacity construction and routing while leaving its default ordinary-convolution behavior unchanged:

```python
capacity: CapacityConfig | None = None,
capacity_part: str = "",

conv = capacity.conv if capacity is not None else None
self.conv1 = (
    conv(capacity_part, in_channels, out_channels, 3, padding=1)
    if conv is not None
    else nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
)
self.conv2 = (
    conv(capacity_part, out_channels, out_channels, 3, padding=1)
    if conv is not None
    else nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
)
```

Add `use_capacity: bool = True` to `TimeResBlock.forward` and call both convolutions through `apply_capacity_conv`. In `ImageUNetVelocity.forward`, derive the flag once and pass it to all six residual blocks; only the two up blocks can contain adapters. Add:

```python
def capacity_metadata(self) -> dict[str, object]:
    adapter_layers = sum(
        isinstance(module, SwitchableLowRankConv2d) for module in self.modules()
    )
    return {
        "enabled": self._capacity.enabled,
        "rank": self._capacity.rank,
        "rank_ratio": self._capacity.rank_ratio,
        "adapter_scale": self._capacity.adapter_scale,
        "parts": sorted(self._capacity.parts),
        "adapter_layers": adapter_layers,
    }
```

In the `image_unet` factory branch, parse `model.capacity` and pass:

```python
capacity_config = model_config.get("capacity", {}) or {}
capacity_enabled = bool(capacity_config.get("enabled", False))
return ImageUNetVelocity(
    dim=dim,
    image_shape=image_shape,
    base_channels=int(model_config.get("base_channels", 32)),
    time_embedding_dim=int(model_config.get("time_embedding_dim", 128)),
    activation=model_config.get("activation", "silu"),
    zero_init_head=bool(model_config.get("zero_init_head", True)),
    num_classes=num_classes,
    class_embedding_dim=class_embedding_dim,
    capacity_rank=(int(capacity_config.get("rank", 0)) if capacity_enabled else 0),
    capacity_rank_ratio=(
        float(capacity_config.get("rank_ratio", 0.0)) if capacity_enabled else 0.0
    ),
    capacity_adapter_scale=float(capacity_config.get("adapter_scale", 1.0)),
    capacity_parts=(
        tuple(capacity_config.get("parts", ())) if capacity_enabled else ()
    ),
)
```

- [ ] **Step 6: Add a capacity-disabled regression assertion to class-conditioning tests**

In `tests/test_class_conditioning.py`, add `SwitchableLowRankConv2d` to the
existing `fm_lab.models` import, then extend the image U-Net case so the default
conditional CIFAR-sized model has ordinary convolutions and the expected base
parameter count:

```python
image_model = ImageUNetVelocity(
    dim=3 * 32 * 32,
    image_shape=(3, 32, 32),
    base_channels=32,
    time_embedding_dim=128,
    num_classes=10,
)
assert sum(parameter.numel() for parameter in image_model.parameters()) == 1_078_569
assert not any(
    isinstance(module, SwitchableLowRankConv2d)
    for module in image_model.modules()
)
assert image_model.capacity_metadata()["enabled"] is False
```

- [ ] **Step 7: Verify image conditioning, adapter switching, factory wiring, and CM**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_capacity_manipulation.py tests/test_class_conditioning.py tests/test_discrete_objective.py
.conda/fm_lab/bin/ruff check fm_lab/models/image.py fm_lab/experiments/factory.py tests/test_capacity_manipulation.py tests/test_class_conditioning.py
git diff --check
```

Expected: all selected tests pass and the disabled model remains exactly 1,078,569 parameters.

- [ ] **Step 8: Commit and push the matched-backbone capability**

```bash
git add fm_lab/models/image.py fm_lab/experiments/factory.py tests/test_capacity_manipulation.py tests/test_class_conditioning.py
git commit -m "Adapt image U-Net for capacity manipulation"
git push
```

---

### Task 3: Apply the bounded matched-backbone profile to all five local configs

**Files:**
- Modify: `tests/test_config_smoke.py`
- Modify: `configs/imbdiff/local/cifar10_lt_ddpm_epsilon_local.yaml`
- Modify: `configs/imbdiff/local/cifar10_lt_x_vloss_local.yaml`
- Modify: `configs/imbdiff/local/cifar10_lt_cbdm_local.yaml`
- Modify: `configs/imbdiff/local/cifar10_lt_oc_local.yaml`
- Modify: `configs/imbdiff/local/cifar10_lt_cm_local.yaml`

**Interfaces:**
- Consumes: Task 2's capacity-aware `image_unet` factory path.
- Produces: one shared local model/training/sampling profile; objective-specific sections and CM's `capacity` section remain method-specific.

- [ ] **Step 1: Change the existing local-profile test to require the approved policy**

Replace the DDPM-specific model assertions in `test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile` with:

```python
assert config["model"]["name"] == "image_unet"
assert config["model"]["image_shape"] == [3, 32, 32]
assert config["model"]["base_channels"] == 32
assert config["model"]["time_embedding_dim"] == 128
assert config["model"]["activation"] == "silu"
assert config["model"]["zero_init_head"] is True
assert config["training"]["batch_size"] == 32
assert config["training"]["steps"] == 8000
assert config["training"]["warmup_steps"] == 500
assert config["training"]["checkpoint_every"] == 2000
assert config["training"]["ema_decay"] == 0.999
assert config["training"]["early_stopping"] == {
    "enabled": True,
    "patience_steps": 2000,
    "warmup_steps": 4000,
    "min_delta": 0.0001,
    "ema_alpha": 0.3,
}
assert config["sampling"]["n_samples"] == 256
assert config["sampling"]["sample_batch_size"] == 32
assert config["sampling"]["plot_max_points"] == 64
assert config["sampling"]["ddim_skip"] == 16
assert config["sampling"]["live_ema_comparison"] == {
    "enabled": True,
    "n_samples": 64,
}
```

After building the model, require exact base capacity for the four non-CM methods and adapter-only growth for CM:

```python
if capacity_enabled:
    assert parameter_count > 1_078_569
else:
    assert parameter_count == 1_078_569
assert model.capacity_metadata()["enabled"] is capacity_enabled
```

- [ ] **Step 2: Run the focused test and verify the old DDPM profile fails**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py::test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile
```

Expected: FAIL because `model.name` is still `ddpm_unet` and batch size is still 16.

- [ ] **Step 3: Update all five local YAML files**

Replace each model section with the approved shared fields. Retain this additional section only in the CM file:

```yaml
  capacity:
    enabled: true
    rank_ratio: 0.1
    adapter_scale: 1.0
    reference_declared_scale: 0.5
    parts: [up]
```

Set the exact training and sampling values from the design:

```yaml
training:
  optimizer: adam
  lr: 0.0002
  batch_size: 32
  steps: 8000
  warmup_steps: 500
  early_stopping:
    enabled: true
    patience_steps: 2000
    warmup_steps: 4000
    min_delta: 0.0001
    ema_alpha: 0.3
  gradient_clip: 1.0
  ema_decay: 0.999
  log_every: 250
  checkpoint_every: 2000
```

```yaml
sampling:
  sampler: ddim
  n_samples: 256
  sample_batch_size: 32
  plot_max_points: 64
  classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  ddim_skip: 16
  eta: 0.0
  classifier_free_guidance:
    enabled: true
    convention: fm_lab
    scale: 1.0
    paper_omega: 0.0
  live_ema_comparison:
    enabled: true
    n_samples: 64
```

- [ ] **Step 4: Verify all config contracts and publish the profile checkpoint**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py
.conda/fm_lab/bin/ruff check tests/test_config_smoke.py
git diff --check
git add tests/test_config_smoke.py configs/imbdiff/local
git commit -m "Match local ImbDiff backbone and budget"
git push
```

Expected: all config tests pass; only the five local YAML files differ from their previous versions.

---

### Task 4: Keep live and EMA states coherent at the selected early-stop step

**Files:**
- Modify: `fm_lab/training/trainer.py`
- Modify: `tests/test_training_sampling.py`

**Interfaces:**
- Consumes: existing `create_ema_model`, `_clone_state`, `_TrainingState`, and early-stopping selection flow.
- Produces: `_TrainingState.ema_model_state`, `_capture_training_state(..., ema_model=...)`, and `_restore_training_state(..., ema_model=...)`.

- [ ] **Step 1: Make the existing end-to-end early-stop test expose stale EMA restoration**

In `test_train_flow_matching_restores_best_early_stopping_checkpoint`, change the training config and add assertions:

```python
"training": {
    "batch_size": 8,
    "steps": 4,
    "log_every": 1,
    "optimizer": "adam",
    "lr": 0.1,
    "ema_decay": 0.5,
    "early_stopping": {
        "enabled": True,
        "warmup_steps": 0,
        "patience_steps": 1,
        "min_delta": 1.0e9,
        "ema_alpha": 1.0,
    },
},
```

After loading the final checkpoint:

```python
model_velocity = checkpoint["model_state_dict"]["velocity"]
ema_velocity = checkpoint["ema_model_state_dict"]["velocity"]
assert torch.equal(model_velocity, torch.zeros_like(model_velocity))
assert torch.equal(ema_velocity, model_velocity)
```

- [ ] **Step 2: Run the regression test and verify EMA remains from the stop step**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_training_sampling.py::test_train_flow_matching_restores_best_early_stopping_checkpoint
```

Expected: FAIL on `torch.equal(ema_velocity, model_velocity)` because only the live model is restored to step 1.

- [ ] **Step 3: Capture and restore optional EMA state with every candidate**

Extend `_TrainingState`:

```python
ema_model_state: dict[str, Any] | None = None
```

Extend capture:

```python
def _capture_training_state(
    *,
    model: nn.Module,
    ema_model: nn.Module | None,
    path: FlowPath,
    theta_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer | None,
    step: int,
) -> _TrainingState:
    path_state = path.state_dict() if isinstance(path, nn.Module) else None
    return _TrainingState(
        step=step,
        model_state=_clone_state(model.state_dict()),
        ema_model_state=(
            _clone_state(ema_model.state_dict()) if ema_model is not None else None
        ),
        path_state=_clone_state(path_state) if path_state is not None else None,
        theta_optimizer_state=_clone_state(theta_optimizer.state_dict()),
        psi_optimizer_state=(
            _clone_state(psi_optimizer.state_dict())
            if psi_optimizer is not None
            else None
        ),
    )
```

Pass `ema_model=ema_model` at all three `_capture_training_state` call sites. Extend restore and its call site:

```python
def _restore_training_state(
    state: _TrainingState,
    *,
    model: nn.Module,
    ema_model: nn.Module | None,
    path: FlowPath,
    theta_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer | None,
) -> None:
    model.load_state_dict(state.model_state)
    if state.ema_model_state is not None:
        if ema_model is None:
            raise ValueError(
                "Best checkpoint state includes EMA weights, but no EMA model exists."
            )
        ema_model.load_state_dict(state.ema_model_state)
    theta_optimizer.load_state_dict(state.theta_optimizer_state)
    if state.path_state is not None:
        if not isinstance(path, nn.Module):
            raise ValueError(
                "Best checkpoint state includes a path module, but path is not a module."
            )
        path.load_state_dict(state.path_state)
    if state.psi_optimizer_state is not None:
        if psi_optimizer is None:
            raise ValueError(
                "Best checkpoint state includes a path optimizer, but no path optimizer exists."
            )
        psi_optimizer.load_state_dict(state.psi_optimizer_state)
```

- [ ] **Step 4: Verify early stopping, resume, and sampling remain correct**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_training_sampling.py tests/test_discrete_sampling.py tests/test_early_stopping.py
.conda/fm_lab/bin/ruff check fm_lab/training/trainer.py tests/test_training_sampling.py
git diff --check
```

Expected: all selected tests pass, including exact equality between selected live and EMA weights in the regression fixture.

- [ ] **Step 5: Commit and push the checkpoint-coherence fix**

```bash
git add fm_lab/training/trainer.py tests/test_training_sampling.py
git commit -m "Restore EMA with best training checkpoint"
git push
```

---

### Task 5: Full verification and matched-backbone CPU smoke

**Files:**
- Generate locally: `runs/verification/matched_backbone_ddpm/`

**Interfaces:**
- Consumes: the final local DDPM config through `fm-lab-train`.
- Produces: a capacity-disabled matched-backbone checkpoint, normal samples, paired live/EMA arrays, a comparison plot, and timestep-bucket CSV fields.

- [ ] **Step 1: Run the full Conda verification gate**

Run:

```bash
.conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check .
.conda/fm_lab/bin/python -m pip check
```

Expected: zero test failures, `All checks passed!`, and `No broken requirements found.`

- [ ] **Step 2: Run a one-step real CIFAR-10 DDPM smoke**

First verify that the diagnostic output path is unused, then run:

```bash
test ! -e runs/verification/matched_backbone_ddpm
.conda/fm_lab/bin/fm-lab-train \
  --config configs/imbdiff/local/cifar10_lt_ddpm_epsilon_local.yaml \
  --device cpu \
  --steps 1 \
  --batch-size 2 \
  --n-samples 2 \
  --sample-batch-size 2 \
  --plot-max-points 2 \
  --output-dir runs/verification/matched_backbone_ddpm
```

Expected: one training step completes and the bounded normal plus paired sampling passes run through `ImageUNetVelocity`.

- [ ] **Step 3: Validate the smoke artifacts and model identity**

Run:

```bash
.conda/fm_lab/bin/python - <<'PY'
import csv
import json
from fm_lab.experiments.factory import build_model, build_source
from fm_lab.utils.config import load_config

run = "runs/verification/matched_backbone_ddpm"
metrics = json.load(open(f"{run}/metrics.json"))
config = load_config(f"{run}/config.yaml")
source = build_source(config)
model = build_model(config, dim=source.dim)
fields = set(next(csv.DictReader(open(f"{run}/diagnostics/training_history.csv"))).keys())
expected_fields = {
    "diffusion_loss_low_noise",
    "diffusion_loss_low_noise_count",
    "diffusion_loss_mid_noise",
    "diffusion_loss_mid_noise_count",
    "diffusion_loss_high_noise",
    "diffusion_loss_high_noise_count",
}
assert config["model"]["name"] == "image_unet"
assert sum(parameter.numel() for parameter in model.parameters()) == 1_078_569
assert model.capacity_metadata()["enabled"] is False
assert metrics["sampling"]["live_ema_comparison"]["n_samples"] == 2
assert expected_fields <= fields
PY
test -f runs/verification/matched_backbone_ddpm/samples/ddim.npy
test -f runs/verification/matched_backbone_ddpm/samples/live_diagnostic.npy
test -f runs/verification/matched_backbone_ddpm/samples/ema_diagnostic.npy
test -f runs/verification/matched_backbone_ddpm/plots/live_vs_ema.png
git status --short --branch
```

Expected: every assertion and file check passes; Git status contains only the pre-existing untracked retrospective document.

- [ ] **Step 4: Hand off the full from-scratch DDPM command**

Use a new output directory so no prior checkpoint or artifact is reused:

```bash
test ! -e runs/imbdiff/local/cifar10_lt_ir100_ddpm_epsilon_matched
.conda/fm_lab/bin/fm-lab-train \
  --config configs/imbdiff/local/cifar10_lt_ddpm_epsilon_local.yaml \
  --device cpu \
  --output-dir runs/imbdiff/local/cifar10_lt_ir100_ddpm_epsilon_matched
```

Expected: at most 8,000 steps, normal early stopping no earlier than approximately step 6,000, then 256 normal DDIM samples and 64 paired live/EMA samples.
