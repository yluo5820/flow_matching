# CM on Local X-VLoss Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local CIFAR-10-LT Capacity Manipulation experiment use the validated x-prediction plus v-loss baseline and its controlled local training schedule.

**Architecture:** Keep the existing CM-capable `ImageUNetVelocity`, up-path low-rank adapters, OC transfer, and CM consistency/diversity losses unchanged. Modify only the local CM experiment configuration and its regression expectations so CM differs from the x-vloss baseline only through CM-specific behavior.

**Tech Stack:** Python 3.11, PyTorch, YAML experiment configuration, pytest, Ruff, `fm-lab-train` CLI.

## Global Constraints

- Use CIFAR-10-LT exponential imbalance ratio 100 with subset seed 0.
- Use `ImageUNetVelocity` with base channels 32 and time embedding dimension 128.
- Use `prediction_type: x_vloss`.
- Use Adam at learning rate 0.0002 and batch size 32.
- Use a 12,000-step ceiling, 500-step optimizer warmup, 6,000-step early-stopping warmup, and 4,000-step early-stopping patience.
- Keep early stopping enabled, EMA decay 0.999, DDIM skip 16, and classifier-free guidance scale 1.0.
- Keep CM up-path capacity rank ratio 0.1, consistency weight 1.0, diversity weight 0.2, and tail-to-head OC transfer unchanged.
- Do not modify or stage `docs/experimental_ideas_retrospective.md`.

---

### Task 1: Match the local CM experiment to x-vloss

**Files:**
- Modify: `tests/test_config_smoke.py`
- Modify: `configs/imbdiff/local/cifar10_lt_cm_local.yaml`

**Interfaces:**
- Consumes: `load_config(path: str | Path) -> dict`, `build_model(config: dict, dim: int) -> nn.Module`, and the existing `DiscreteDiffusionObjective` support for `method="cm"` with `prediction_type="x_vloss"`.
- Produces: a local CM YAML whose shared objective and schedule match `cifar10_lt_x_vloss_local.yaml` while `model.capacity`, `objective.oc`, and `objective.cm` remain CM-specific.

- [ ] **Step 1: Change the configuration regression expectation first**

In `test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile`, replace the CM entry in `expected_files` with:

```python
"cifar10_lt_cm_local.yaml": ("cm", "x_vloss", True, 12000, 6000, 4000),
```

Retain the existing assertions that the model has enabled capacity and more than 1,078,569 parameters. Add explicit CM assertions after model construction:

```python
if path.name == "cifar10_lt_cm_local.yaml":
    assert config["model"]["capacity"] == {
        "enabled": True,
        "rank_ratio": 0.1,
        "adapter_scale": 1.0,
        "reference_declared_scale": 0.5,
        "parts": ["up"],
    }
    assert config["objective"]["cm"] == {
        "consistency_weight": 1.0,
        "diversity_weight": 0.2,
    }
```

- [ ] **Step 2: Run the focused test and confirm the intended red state**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py::test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile
```

Expected: FAIL because the current CM YAML still reports `prediction_type: epsilon` and the 8,000/4,000/2,000 schedule.

- [ ] **Step 3: Apply the minimal CM YAML change**

In `configs/imbdiff/local/cifar10_lt_cm_local.yaml`, change exactly these values:

```yaml
objective:
  name: cm
  prediction_type: x_vloss

training:
  steps: 12000
  early_stopping:
    enabled: true
    patience_steps: 4000
    warmup_steps: 6000
```

Leave all other model, objective, optimization, sampling, and CM fields unchanged.

- [ ] **Step 4: Run configuration and objective regression tests**

Run:

```bash
.conda/fm_lab/bin/pytest -q \
  tests/test_config_smoke.py \
  tests/test_capacity_manipulation.py \
  tests/test_discrete_objective.py
.conda/fm_lab/bin/ruff check tests/test_config_smoke.py
git diff --check
```

Expected: all tests pass, Ruff reports `All checks passed!`, and `git diff --check` emits no output.

- [ ] **Step 5: Run a one-step CPU smoke through the actual CM path**

Run:

```bash
test ! -e runs/verification/cm_x_vloss_local
.conda/fm_lab/bin/fm-lab-train \
  --config configs/imbdiff/local/cifar10_lt_cm_local.yaml \
  --device cpu \
  --steps 1 \
  --batch-size 2 \
  --n-samples 2 \
  --sample-batch-size 2 \
  --plot-max-points 2 \
  --output-dir runs/verification/cm_x_vloss_local
```

Expected: exit code 0, `metrics.json` reports objective method `cm` and prediction type `x_vloss`, and the run writes `samples/ddim.npy`, `samples/live_diagnostic.npy`, `samples/ema_diagnostic.npy`, `plots/generated_samples.png`, and `plots/live_vs_ema.png`.

- [ ] **Step 6: Commit and push the verified experiment**

```bash
git add configs/imbdiff/local/cifar10_lt_cm_local.yaml tests/test_config_smoke.py
git diff --cached --check
git commit -m "Run local CM on x-vloss baseline"
git push origin main
```

Expected: the commit and push succeed; `docs/experimental_ideas_retrospective.md` remains untracked and unstaged.
