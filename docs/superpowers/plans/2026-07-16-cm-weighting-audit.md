# CM Weighting Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe CM diagnostics and run matched short pilots that isolate the effect of restoring the paper's unattenuated diversity weighting.

**Architecture:** Extend `CMModifier` logging with per-class distances and observed counts. Add a generic metric-based diagnostic stop policy at the trainer boundary so an unsafe objective can stop before backward or optimizer mutation, then compare an abort-guarded unbounded CM pilot with a bounded CM pilot while holding every other training variable fixed.

**Tech Stack:** Python 3.11, PyTorch, YAML experiment configuration, pytest, Ruff, `.conda/fm_lab/bin` executables.

## Global Constraints

- Keep continuous linear interpolation, target prediction, velocity supervision, decoder-only rank-0.1 capacity, optimizer, time sampling, seed, and Fashion-MNIST IR100 data unchanged.
- Use `consistency_weight: 1.0` and `diversity_weight: 1.0` for both new pilots.
- The unbounded pilot is diagnostic-only and must stop before an unsafe step updates parameters.
- The bounded pilot starts with `diversity_margin: 0.001`; replace it with `0.01` only if saturation exceeds 80% before step 200.
- Do not launch a full Fashion-MNIST or CIFAR-10 experiment.
- Use `.conda/fm_lab/bin/python`, `.conda/fm_lab/bin/pytest`, `.conda/fm_lab/bin/ruff`, and `.conda/fm_lab/bin/fm-lab-train`.

---

### Task 1: Per-class CM capacity-distance diagnostics

**Files:**
- Modify: `fm_lab/training/long_tail.py:252-332`
- Test: `tests/test_continuous_long_tail.py`

**Interfaces:**
- Consumes: `CMModifier.group_distance_metrics(distance: Tensor, labels: Tensor) -> dict[str, float]`.
- Produces: stable `cm.distance.class_<id>` and `cm.count.class_<id>` metrics for every configured class, alongside existing group metrics.

- [ ] **Step 1: Write the failing per-class metrics test**

Add a test that calls `group_distance_metrics` with `class_counts=[100, 10, 1]`, distances `[1.0, 3.0, 5.0]`, and labels `[0, 0, 1]`. Assert:

```python
assert metrics["cm.distance.class_0"] == pytest.approx(2.0)
assert metrics["cm.count.class_0"] == 2.0
assert metrics["cm.distance.class_1"] == pytest.approx(5.0)
assert metrics["cm.count.class_1"] == 1.0
assert metrics["cm.distance.class_2"] == 0.0
assert metrics["cm.count.class_2"] == 0.0
```

- [ ] **Step 2: Verify the test fails for the missing metrics**

Run:

```bash
.conda/fm_lab/bin/pytest tests/test_continuous_long_tail.py -k per_class_distance -q
```

Expected: FAIL with `KeyError: 'cm.distance.class_0'`.

- [ ] **Step 3: Implement per-class distance and count logging**

In `group_distance_metrics`, loop over `range(len(self.class_counts))`, compute the masked mean when observed, use zero when absent, and log both value and count:

```python
for class_id in range(len(self.class_counts)):
    class_distance = distance[labels == class_id]
    value = class_distance.mean() if len(class_distance) else distance.new_tensor(0.0)
    metrics[f"cm.distance.class_{class_id}"] = float(value.detach().cpu())
    metrics[f"cm.count.class_{class_id}"] = float(len(class_distance))
```

- [ ] **Step 4: Verify the focused test passes**

Run:

```bash
.conda/fm_lab/bin/pytest tests/test_continuous_long_tail.py -k 'per_class_distance or cm_' -q
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the diagnostic metrics**

```bash
git add fm_lab/training/long_tail.py tests/test_continuous_long_tail.py
git commit -m "Log per-class CM capacity distances"
```

---

### Task 2: Pre-update metric safety stop

**Files:**
- Modify: `fm_lab/training/trainer.py:80-410`
- Test: `tests/test_training_sampling.py`

**Interfaces:**
- Consumes: `training.diagnostic_stop` with `enabled`, `finite_metrics`, `min_metrics`, and `max_metrics`.
- Produces: `_MetricStopPolicy.update(record, step) -> bool`, `summary() -> dict[str, Any]`, and top-level `metrics["diagnostic_stop"]`.
- Invariant: a triggering step performs no backward pass, optimizer step, scheduler step, or EMA update.

- [ ] **Step 1: Write failing validation and default-disable tests**

Test `_build_metric_stop_policy({})` and assert its summary is:

```python
{
    "enabled": False,
    "stopped": False,
    "stop_step": None,
    "reason": None,
}
```

Add parameterized invalid configurations for an empty metric name, non-finite threshold, and a metric present in both `min_metrics` and `max_metrics`. Each must raise `ValueError` mentioning `training.diagnostic_stop`.

- [ ] **Step 2: Run the policy tests and verify RED**

```bash
.conda/fm_lab/bin/pytest tests/test_training_sampling.py -k metric_stop_policy -q
```

Expected: FAIL because `_build_metric_stop_policy` does not exist.

- [ ] **Step 3: Implement `_MetricStopPolicy` and its builder**

Add a dataclass with these fields:

```python
enabled: bool = False
finite_metrics: tuple[str, ...] = ()
min_metrics: dict[str, float] = field(default_factory=dict)
max_metrics: dict[str, float] = field(default_factory=dict)
stopped: bool = False
stop_step: int | None = None
reason: str | None = None
```

`update` must check configured finite metrics first, then minimum thresholds,
then maximum thresholds. A missing configured metric raises `ValueError`. On the
first violation it records a deterministic reason such as
`"cm.loss_to_base_ratio=-1.2 < -1.0"` and returns `True`.

- [ ] **Step 4: Verify the policy unit tests pass**

```bash
.conda/fm_lab/bin/pytest tests/test_training_sampling.py -k metric_stop_policy -q
```

Expected: all selected tests PASS.

- [ ] **Step 5: Write a failing trainer integration test**

Configure a two-step tiny run with:

```python
"diagnostic_stop": {
    "enabled": True,
    "max_metrics": {"loss": -1.0},
}
```

The threshold intentionally triggers on step 1. Snapshot model parameters before
training and assert afterward that parameters are unchanged,
`trained_steps == 0`, `diagnostic_stop.stopped is True`, and
`diagnostic_stop.stop_step == 1`.

- [ ] **Step 6: Run the integration test and verify RED**

```bash
.conda/fm_lab/bin/pytest tests/test_training_sampling.py -k diagnostic_stop_prevents_update -q
```

Expected: FAIL because training ignores `diagnostic_stop`.

- [ ] **Step 7: Integrate the stop before backward**

Build the policy beside early stopping. Include `metric_stop.enabled` in
`should_record`. After the objective returns and the record is assembled, call
`metric_stop.update(record, step)`. If it triggers, append the trigger record,
set `final_step = step - 1`, show `stopped="diagnostic"`, and break before
`zero_grad`, `backward`, or any optimizer mutation. Persist
`metric_stop.summary()` in final metrics.

- [ ] **Step 8: Verify focused trainer behavior and regression tests**

```bash
.conda/fm_lab/bin/pytest tests/test_training_sampling.py -k 'diagnostic_stop or early_stopping' -q
```

Expected: all selected tests PASS.

- [ ] **Step 9: Commit the safety stop**

```bash
git add fm_lab/training/trainer.py tests/test_training_sampling.py
git commit -m "Add metric safety stop for diagnostic runs"
```

---

### Task 3: Matched CM weighting pilot configurations

**Files:**
- Create, ignored: `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm_pilot_paper_unbounded.yaml`
- Create, ignored: `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm_pilot_paper_bounded.yaml`

**Interfaces:**
- Consumes: production `fashion_mnist_lt_ir100_x_vloss_cm.yaml` fields and Task 2's `training.diagnostic_stop`.
- Produces: two locally reproducible 1,000-step pilot configs differing only in diversity mode, margin, stop policy, experiment name, and output directory.

- [ ] **Step 1: Create the two local pilot configs**

Copy the canonical decoder-only CM pilot controls: 1,000 steps, batch size 256,
learning rate `1e-4`, warmup 500, gradient clip 1.0, logit-normal time sampling
`(-0.8, 0.8)`, seed 0, and 100 diagnostic samples. Configure unbounded safety:

```yaml
diagnostic_stop:
  enabled: true
  finite_metrics: [loss, base.loss, cm.loss, cm.distance.max]
  min_metrics:
    cm.loss_to_base_ratio: -1.0
  max_metrics:
    cm.distance.max: 1.0
```

Configure bounded CM with `diversity_mode: bounded` and
`diversity_margin: 0.001`; leave diagnostic stopping disabled.

- [ ] **Step 2: Validate that only intended pilot fields differ**

Pilot configs are intentionally ignored temporal artifacts, so validate them
with a local assertion script rather than adding a tracked test that depends on
ignored files. Load both configs with `load_config`, compare their data, source,
coupling, path, model, conditioning, solvers, and sampling sections, then compare
training after removing `diagnostic_stop`. Assert both CM modifiers use weights
`1.0`; assert only the bounded modifier adds mode `bounded` and margin `0.001`.

```bash
.conda/fm_lab/bin/python - <<'PY'
from pathlib import Path
from fm_lab.config import load_config

root = Path("configs/fashion_mnist_lt")
u = load_config(root / "fashion_mnist_lt_ir100_x_vloss_cm_pilot_paper_unbounded.yaml")
b = load_config(root / "fashion_mnist_lt_ir100_x_vloss_cm_pilot_paper_bounded.yaml")
for section in ("data", "source", "coupling", "path", "model", "conditioning", "solvers", "sampling"):
    assert u[section] == b[section]
ut, bt = dict(u["training"]), dict(b["training"])
ut.pop("diagnostic_stop")
assert ut == bt
um = u["objective"]["modifiers"][0]
bm = b["objective"]["modifiers"][0]
assert um["consistency_weight"] == bm["consistency_weight"] == 1.0
assert um["diversity_weight"] == bm["diversity_weight"] == 1.0
assert bm["diversity_mode"] == "bounded"
assert bm["diversity_margin"] == 0.001
PY
```

Expected: exit 0 with no assertion failure. The pilot YAML files remain ignored
and are not committed.

---

### Task 4: Verification and pilot execution

**Files:**
- Read outputs: `runs/pilots/fashion_mnist_cm_paper_unbounded/`
- Read outputs: `runs/pilots/fashion_mnist_cm_paper_bounded/`

**Interfaces:**
- Consumes: Tasks 1--3.
- Produces: a decision table against the thresholds in the approved design.

- [ ] **Step 1: Run complete pre-pilot verification**

```bash
.conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check .
git diff --check
```

Expected: all tests pass, Ruff reports `All checks passed!`, and diff check is silent.

- [ ] **Step 2: Run the unbounded diagnostic pilot**

```bash
.conda/fm_lab/bin/fm-lab-train \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm_pilot_paper_unbounded.yaml
```

Expected: clean completion at 1,000 steps or a clean diagnostic stop with its
reason recorded in `metrics.json`; never a non-finite checkpoint.

- [ ] **Step 3: Run the bounded pilot**

```bash
.conda/fm_lab/bin/fm-lab-train \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm_pilot_paper_bounded.yaml
```

Expected: 1,000 finite steps. If saturation is above 80% before step 200,
create one replacement config with margin `0.01` and a distinct `_margin01`
output directory, then rerun once without deleting the first diagnostic record.

- [ ] **Step 4: Analyze the latter-half trajectories**

Read both `diagnostics/training_history.csv` files. For steps 500--1,000,
compute count-weighted per-class mean distance using
`cm.distance.class_<id>` and `cm.count.class_<id>`. Report final and
latter-half many/medium/few distances, CM/base ratio, saturation, base loss,
runtime, stop status, and whether every decision criterion passed.

- [ ] **Step 5: Run fresh post-pilot verification**

```bash
.conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check .
git diff --check
git status --short --branch
```

Expected: all tracked changes committed, all checks pass, and only ignored pilot artifacts remain.
