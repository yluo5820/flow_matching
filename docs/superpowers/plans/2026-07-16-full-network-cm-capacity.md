# Full-network CM Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend continuous CM from four decoder adapters to canonical switchable low-rank capacity across the complete conditional image U-Net, then measure the effect in a 1,000-step controlled pilot.

**Architecture:** `CapacityConfig` constructs switchable convolutional and linear layers by named model section. `ImageUNetVelocity` threads the existing `use_capacity` context through conditioning, encoder, bottleneck, decoder, and output layers. Zero-output adapters preserve exact initial branch parity and the forked RNG preserves same-seed shared weights.

**Tech Stack:** Python 3.12, PyTorch, pytest, Ruff, YAML experiment configuration.

## Global Constraints

- Use `.conda/fm_lab/bin/python` for Python commands.
- Keep the continuous linear path, target prediction, velocity base loss, CM weights, optimizer, and pilot length unchanged.
- Flatten convolution weights as `[out_channels, in_channels * kernel_height * kernel_width]`.
- Keep biases, embeddings, and normalization parameters shared.
- Use test-first red-green-refactor cycles.

---

### Task 1: Canonical switchable layer primitives

**Files:**
- Modify: `fm_lab/models/capacity.py`
- Modify: `fm_lab/models/__init__.py`
- Test: `tests/test_capacity_manipulation.py`

**Interfaces:**
- Produces: `SwitchableLowRankLinear(nn.Linear)` and `apply_capacity_linear(layer, inputs, *, use_capacity)`.
- Changes: `SwitchableLowRankConv2d.adapter_a` has shape `[rank, in_channels * kh * kw]`; `adapter_b` has shape `[out_channels, rank]`.
- Extends: `CapacityConfig.linear(part, in_features, out_features, *, bias=True)`.

- [ ] **Step 1: Write failing primitive tests**

Add tests asserting canonical factor shapes and linear switching:

```python
def test_low_rank_conv_uses_canonical_flattened_kernel_factors() -> None:
    layer = models.SwitchableLowRankConv2d(4, 6, 3, rank=2)
    assert layer.adapter_a.shape == (2, 4 * 3 * 3)
    assert layer.adapter_b.shape == (6, 2)


def test_low_rank_linear_switch_applies_factorized_weight() -> None:
    layer = models.SwitchableLowRankLinear(2, 1, rank=1, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
        layer.adapter_a.fill_(2.0)
        layer.adapter_b.fill_(3.0)
    inputs = torch.ones(1, 2)
    assert torch.equal(layer(inputs, use_adapter=False), torch.zeros(1, 1))
    assert torch.equal(layer(inputs, use_adapter=True), torch.full((1, 1), 12.0))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.conda/fm_lab/bin/python -m pytest tests/test_capacity_manipulation.py -k "canonical_flattened or low_rank_linear" -q
```

Expected: failures because the convolution factors use the old spatial split and `SwitchableLowRankLinear` does not exist.

- [ ] **Step 3: Implement canonical shared adapter logic**

Use a private initializer shared by convolution and linear layers:

```python
def _initialize_adapter(module: nn.Module, *, rows: int, columns: int, rank: int) -> None:
    module.adapter_a = nn.Parameter(module.weight.new_empty(rank, columns))
    module.adapter_b = nn.Parameter(module.weight.new_zeros(rows, rank))
    with torch.random.fork_rng(devices=[]):
        nn.init.kaiming_normal_(module.adapter_a, a=math.sqrt(5))
```

For convolution forward, compute `(adapter_b @ adapter_a).reshape_as(weight)`. For linear forward, compute `F.linear(inputs, weight + scale * update, bias)`. Register `None` parameters when rank is zero and export the new class from `fm_lab.models`.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```bash
.conda/fm_lab/bin/python -m pytest tests/test_capacity_manipulation.py -q
```

Expected: all capacity tests pass.

- [ ] **Step 5: Commit**

```bash
git add fm_lab/models/capacity.py fm_lab/models/__init__.py tests/test_capacity_manipulation.py
git commit -m "Generalize switchable low-rank capacity layers"
```

### Task 2: Full image U-Net coverage

**Files:**
- Modify: `fm_lab/models/image.py`
- Test: `tests/test_capacity_manipulation.py`

**Interfaces:**
- Consumes: `CapacityConfig.conv`, `CapacityConfig.linear`, `apply_capacity_conv`, and `apply_capacity_linear`.
- Produces: capacity sections `conditioning`, `head`, `down`, `middle`, `up`, and `tail` for `ImageUNetVelocity`.
- Changes: `TimeResBlock` also makes `time_proj` and channel-changing `skip` switchable for its assigned section.

- [ ] **Step 1: Write failing coverage and parity tests**

Create an all-section config and assert representative module types:

```python
ALL_CAPACITY_PARTS = ["conditioning", "head", "down", "middle", "up", "tail"]


def test_image_unet_full_capacity_covers_every_model_section() -> None:
    config = _cm_image_model_config(parts=ALL_CAPACITY_PARTS)
    model = build_model(config, dim=3 * 8 * 8)
    names = {
        name for name, module in model.named_modules()
        if isinstance(module, (models.SwitchableLowRankConv2d, models.SwitchableLowRankLinear))
    }
    assert {"time_mlp.0", "class_projection", "input_block.conv1",
            "down1.0", "middle.conv1", "up1_block.conv1", "output_block.2"} <= names
```

Add a same-seed comparison that filters `adapter_a` and `adapter_b` and asserts every remaining state tensor equals the non-capacity model. Extend the existing branch-switch test to perturb both adapter classes and assert capacity-off output remains unchanged.

- [ ] **Step 2: Verify RED**

Run:

```bash
.conda/fm_lab/bin/python -m pytest tests/test_capacity_manipulation.py -k "full_capacity or same_seed" -q
```

Expected: failure because `ImageUNetVelocity` rejects all sections except `up`.

- [ ] **Step 3: Thread capacity through the model**

Build conditioning layers with `self._capacity.linear("conditioning", ...)`, construct every `TimeResBlock` with its section, construct downsampling/output convolutions through `self._capacity.conv`, and compute them with the corresponding apply helper:

```python
use_capacity = use_capacity_from_context(context)
time_features = apply_capacity_linear(
    self.time_mlp[0], self.time_embedding(t), use_capacity=use_capacity
)
time_features = self.time_mlp[1](time_features)
time_features = apply_capacity_linear(
    self.time_mlp[2], time_features, use_capacity=use_capacity
)
```

Pass `use_capacity` to all residual blocks and use the apply helpers for downsampling and the tail. Count both switchable layer classes in `capacity_metadata()` and report separate convolution/linear counts in addition to `adapter_layers`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.conda/fm_lab/bin/python -m pytest tests/test_capacity_manipulation.py tests/test_config_smoke.py -q
```

Expected: all model capacity and config tests pass.

- [ ] **Step 5: Commit**

```bash
git add fm_lab/models/image.py tests/test_capacity_manipulation.py
git commit -m "Apply CM capacity across image U-Net"
```

### Task 3: Reproducible full-network CM configuration

**Files:**
- Modify: `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm.yaml`
- Modify: `tests/test_config_smoke.py`
- Modify: `README.md`
- Modify: `docs/cli.md`

**Interfaces:**
- Produces: production config selecting all six capacity sections.
- Preserves: rank ratio `0.1`, adapter scale `1.0`, and all objective/training/sampling fields.

- [ ] **Step 1: Write failing config assertion**

Update the CM config expectation to:

```python
assert configs[3]["model"]["capacity"]["parts"] == [
    "conditioning", "head", "down", "middle", "up", "tail"
]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.conda/fm_lab/bin/python -m pytest tests/test_config_smoke.py -k continuous_fashion -q
```

Expected: failure showing the current value is `['up']`.

- [ ] **Step 3: Update configuration and documentation**

Set:

```yaml
capacity:
  enabled: true
  rank_ratio: 0.1
  adapter_scale: 1.0
  parts: [conditioning, head, down, middle, up, tail]
```

Describe CM as full-network canonical low-rank decomposition in the README and CLI guide.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.conda/fm_lab/bin/python -m pytest tests/test_config_smoke.py tests/test_docs_coverage.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm.yaml tests/test_config_smoke.py README.md docs/cli.md
git commit -m "Configure full-network CM capacity"
```

### Task 4: Verification and controlled pilot

**Files:**
- No tracked source files expected.
- Local output: `runs/pilots/fashion_mnist_cm_full_network/` (ignored).

- [ ] **Step 1: Run complete verification**

```bash
.conda/fm_lab/bin/python -m pytest -q
.conda/fm_lab/bin/python -m ruff check fm_lab tests
git diff --check
```

Expected: zero test failures, Ruff errors, or whitespace errors.

- [ ] **Step 2: Run the 1,000-step pilot**

Copy the ignored CM pilot configuration to a new ignored local file, change only `experiment.name`, `experiment.output_dir`, `training.steps: 1000`, `training.log_every: 100`, and `sampling.n_samples: 100`, then run:

```bash
.conda/fm_lab/bin/fm-lab-train --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm_full_network_pilot.yaml
```

Expected: training completes without non-finite loss.

- [ ] **Step 3: Compare diagnostic endpoints**

Read the last record of `diagnostics/training_history.csv` and compare `base.loss`, `cm.loss_to_base_ratio`, `cm.distance.{many,medium,few}`, `cm.distance.max`, `gradient_norm`, and elapsed time against `runs/pilots/fashion_mnist_cm_velocity/`.

- [ ] **Step 4: Commit any verification-driven fixes**

If verification required source corrections, stage only those tracked files and commit them with a message describing the corrected behavior. Do not commit pilot configs or run artifacts.
