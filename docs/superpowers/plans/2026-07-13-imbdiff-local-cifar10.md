# Local CIFAR-10 ImbDiff Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five CPU-oriented CIFAR-10 long-tail configs that preserve DDPM, x-vloss, CBDM, OC, and CM semantics while completing in approximately 30–60 minutes on the current Mac.

**Architecture:** Create a separate `configs/imbdiff/local/` matrix so paper-scale configs remain unchanged. Every local method uses the existing class-conditional `DDPMUNet` at approximately 1.12 million parameters, a shared 5000-step budget, local early stopping, and reduced DDIM sampling; method-specific objective and CM capacity sections remain explicit.

**Tech Stack:** YAML experiment configs, Python 3.11, PyTorch, pytest, Ruff, `.conda/fm_lab`, `fm-lab-train`.

## Global Constraints

- Create exactly five local CIFAR-10 configs and no CIFAR-100 local configs in this round.
- Leave all nine paper-scale `configs/imbdiff/*.yaml` files unchanged.
- Use `DDPMUNet` with `base_channels: 32`, `channel_multipliers: [1, 2, 2]`, `attention_levels: [1]`, and `num_res_blocks: 1`.
- Use batch size 16, 5000 maximum steps, 500 optimizer-warmup steps, and learning rate 0.0002.
- Use early-stopping warmup 2000, patience 1000, minimum delta 0.0001, and EMA alpha 0.05.
- Use 1000 generated samples, sample batch size 16, plot limit 100, and DDIM skip 50.
- Run all commands through `.conda/fm_lab` and commit/push each verified nontrivial change.

---

### Task 1: Add and enforce the five-config local matrix

**Files:**
- Modify: `tests/test_config_smoke.py`
- Create: `configs/imbdiff/local/cifar10_lt_ddpm_epsilon_local.yaml`
- Create: `configs/imbdiff/local/cifar10_lt_x_vloss_local.yaml`
- Create: `configs/imbdiff/local/cifar10_lt_cbdm_local.yaml`
- Create: `configs/imbdiff/local/cifar10_lt_oc_local.yaml`
- Create: `configs/imbdiff/local/cifar10_lt_cm_local.yaml`

**Interfaces:**
- Consumes: `load_config(path)`, `build_source(config)`, and `build_model(config, dim)`.
- Produces: five independently runnable class-conditional local configs with unique run directories.

- [ ] **Step 1: Write the failing local-matrix test**

Add this test to `tests/test_config_smoke.py`:

```python
def test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile() -> None:
    expected_files = {
        "cifar10_lt_ddpm_epsilon_local.yaml": ("discrete_diffusion", "epsilon", False),
        "cifar10_lt_x_vloss_local.yaml": ("discrete_diffusion", "x_vloss", False),
        "cifar10_lt_cbdm_local.yaml": ("cbdm", "epsilon", False),
        "cifar10_lt_oc_local.yaml": ("oc", "epsilon", False),
        "cifar10_lt_cm_local.yaml": ("cm", "epsilon", True),
    }
    paths = sorted(Path("configs/imbdiff/local").glob("*.yaml"))

    assert {path.name for path in paths} == set(expected_files)
    for path in paths:
        objective_name, prediction_type, capacity_enabled = expected_files[path.name]
        config = load_config(path)
        assert config["model"]["base_channels"] == 32
        assert config["model"]["channel_multipliers"] == [1, 2, 2]
        assert config["model"]["attention_levels"] == [1]
        assert config["model"]["num_res_blocks"] == 1
        assert config["objective"]["name"] == objective_name
        assert config["objective"]["prediction_type"] == prediction_type
        assert config["training"]["batch_size"] == 16
        assert config["training"]["steps"] == 5000
        assert config["training"]["warmup_steps"] == 500
        assert config["training"]["early_stopping"] == {
            "enabled": True,
            "patience_steps": 1000,
            "warmup_steps": 2000,
            "min_delta": 0.0001,
            "ema_alpha": 0.05,
        }
        assert config["sampling"]["n_samples"] == 1000
        assert config["sampling"]["sample_batch_size"] == 16
        assert config["sampling"]["plot_max_points"] == 100
        assert config["sampling"]["ddim_skip"] == 50
        assert config["experiment"]["output_dir"].startswith("runs/imbdiff/local/")

        source = build_source(config)
        model = build_model(config, dim=source.dim)
        assert model.is_class_conditional
        assert 1_000_000 < sum(parameter.numel() for parameter in model.parameters()) < 1_700_000
        assert model.capacity_metadata()["enabled"] is capacity_enabled
```

- [ ] **Step 2: Run the new test and verify the missing-directory failure**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py::test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile
```

Expected: FAIL because the local file set is empty rather than the five expected names.

- [ ] **Step 3: Create each method-specific local config**

Create the five files using their paper-profile counterpart as the semantic reference, with these exact experiment and objective mappings:

| Local file | Experiment name | Track | Objective |
|---|---|---|---|
| `cifar10_lt_ddpm_epsilon_local.yaml` | `cifar10_lt_ir100_local_ddpm_epsilon` | `local_ddpm_epsilon` | `{name: discrete_diffusion, prediction_type: epsilon}` |
| `cifar10_lt_x_vloss_local.yaml` | `cifar10_lt_ir100_local_x_vloss` | `local_x_vloss` | `{name: discrete_diffusion, prediction_type: x_vloss}` |
| `cifar10_lt_cbdm_local.yaml` | `cifar10_lt_ir100_local_cbdm` | `local_cbdm` | CBDM train distribution, tau 0.001, gamma 0.25 |
| `cifar10_lt_oc_local.yaml` | `cifar10_lt_ir100_local_oc` | `local_oc` | OC t2h transfer with cut time -1 |
| `cifar10_lt_cm_local.yaml` | `cifar10_lt_ir100_local_cm` | `local_cm` | CM consistency 1.0, diversity 0.2, OC t2h transfer |

Every file uses these data, source, coupling, path, conditioning, diffusion,
and solver sections:

```yaml
data:
  name: cifar10_lt
  root: data/cifar10
  train: true
  download: true
  imbalance_type: exp
  imbalance_factor: 0.01
  subset_seed: 0
  normalize: minus_one_one
  horizontal_flip: true

source:
  name: gaussian
  dim: 3072

coupling:
  name: independent
  shuffle_target: false

path:
  name: gaussian_diffusion

conditioning:
  enabled: true
  num_classes: 10
  dropout_probability: 0.1

diffusion:
  timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  variance: fixed_large

solvers:
  names: [euler]
```

Every file uses this model block; only CM additionally includes the shown capacity section:

```yaml
model:
  name: ddpm_unet
  image_shape: [3, 32, 32]
  base_channels: 32
  channel_multipliers: [1, 2, 2]
  attention_levels: [1]
  num_res_blocks: 1
  dropout: 0.1
```

```yaml
  capacity:
    enabled: true
    rank_ratio: 0.1
    adapter_scale: 1.0
    reference_declared_scale: 0.5
    parts: [up]
```

Every file uses this training block:

```yaml
training:
  optimizer: adam
  lr: 0.0002
  batch_size: 16
  steps: 5000
  warmup_steps: 500
  early_stopping:
    enabled: true
    patience_steps: 1000
    warmup_steps: 2000
    min_delta: 0.0001
    ema_alpha: 0.05
  gradient_clip: 1.0
  ema_decay: 0.9999
  log_every: 250
  checkpoint_every: 2500
```

Every file uses this sampling budget while retaining CIFAR-10 classes and classifier-free guidance scale 2.5:

```yaml
sampling:
  sampler: ddim
  n_samples: 1000
  sample_batch_size: 16
  plot_max_points: 100
  classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  ddim_skip: 50
  eta: 0.0
  classifier_free_guidance:
    enabled: true
    convention: fm_lab
    scale: 2.5
    paper_omega: 1.5
```

- [ ] **Step 4: Run focused verification under Conda**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py
.conda/fm_lab/bin/ruff check tests/test_config_smoke.py
git diff --check
```

Expected: all config tests pass, Ruff reports `All checks passed!`, and whitespace validation exits successfully.

- [ ] **Step 5: Commit and push the local config matrix**

```bash
git add tests/test_config_smoke.py configs/imbdiff/local
git commit -m "Add local CIFAR-10 ImbDiff profiles"
git push
```

Expected: the five configs and their regression test are pushed to `codex/imbdiff-local-cifar10`.

---

### Task 2: Verify the local CM profile end to end

**Files:**
- Generate locally: `runs/verification/local_cifar10_cm/`

**Interfaces:**
- Consumes: `configs/imbdiff/local/cifar10_lt_cm_local.yaml` through `.conda/fm_lab/bin/fm-lab-train`.
- Produces: a compact CM checkpoint, metrics, generated labels, and DDIM samples.

- [ ] **Step 1: Run the full repository verification under Conda**

Run:

```bash
.conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check .
.conda/fm_lab/bin/python -m pip check
```

Expected: zero test failures, `All checks passed!`, and `No broken requirements found.`

- [ ] **Step 2: Run a one-step local CM CPU smoke experiment**

Run:

```bash
.conda/fm_lab/bin/fm-lab-train \
  --config configs/imbdiff/local/cifar10_lt_cm_local.yaml \
  --device cpu \
  --steps 1 \
  --batch-size 2 \
  --n-samples 2 \
  --sample-batch-size 2 \
  --plot-max-points 2 \
  --output-dir runs/verification/local_cifar10_cm
```

Expected: exit code 0 and `Finished run: runs/verification/local_cifar10_cm`.

- [ ] **Step 3: Validate generated artifacts and loaded policy**

Run:

```bash
.conda/fm_lab/bin/python -c 'import json, yaml; r="runs/verification/local_cifar10_cm"; m=json.load(open(f"{r}/metrics.json")); c=yaml.safe_load(open(f"{r}/config.yaml")); assert m["device"] == "cpu"; assert m["trained_steps"] == 1; assert m["early_stopping"]["enabled"] is True; assert m["objective"]["cm"]["base_target"] == "oc"; assert c["model"]["base_channels"] == 32; assert c["model"]["num_res_blocks"] == 1'
test -f runs/verification/local_cifar10_cm/checkpoint.pt
test -f runs/verification/local_cifar10_cm/samples/ddim.npy
test -f runs/verification/local_cifar10_cm/samples/generated_labels.npy
```

Expected: every assertion and file check exits successfully.

- [ ] **Step 4: Verify final branch state**

Run:

```bash
git status --short --branch
```

Expected: a clean `codex/imbdiff-local-cifar10` branch tracking its remote.
