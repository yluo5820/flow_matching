# ImbDiff Conda Environment and Early-Stopping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable one early-stopping policy in all nine ImbDiff experiments, prove the documented Conda environment runs the project and a CIFAR-10 CM experiment, and remove the redundant `.venv`.

**Architecture:** Keep the Python early-stopping default opt-in so unrelated experiments do not change. Encode the same policy explicitly in every ImbDiff YAML, enforce the complete nine-file matrix with a config regression test, and delete `.venv` only after Conda dependency, test, and end-to-end smoke checks succeed.

**Tech Stack:** Python 3.11 Conda environment, PyTorch, pytest, Ruff, YAML experiment configs, `fm-lab-train`.

## Global Constraints

- Use `.conda/fm_lab` as the canonical local environment.
- Apply early stopping only to all nine files under `configs/imbdiff/`.
- Use `enabled: true`, `patience_steps: 10000`, `warmup_steps: 20000`, `min_delta: 0.0001`, and `ema_alpha: 0.01` verbatim.
- Do not change the Python-level early-stopping default or unrelated experiment configs.
- Do not remove `.venv` until every Conda verification command passes.
- Commit and push each verified, nontrivial repository change.

---

### Task 1: Enforce and enable the shared ImbDiff early-stopping policy

**Files:**
- Modify: `tests/test_config_smoke.py`
- Modify: `configs/imbdiff/cifar10_lt_ddpm_epsilon.yaml`
- Modify: `configs/imbdiff/cifar10_lt_x_vloss.yaml`
- Modify: `configs/imbdiff/cifar10_lt_cbdm.yaml`
- Modify: `configs/imbdiff/cifar10_lt_oc.yaml`
- Modify: `configs/imbdiff/cifar10_lt_cm.yaml`
- Modify: `configs/imbdiff/cifar100_lt_ddpm_epsilon.yaml`
- Modify: `configs/imbdiff/cifar100_lt_cbdm.yaml`
- Modify: `configs/imbdiff/cifar100_lt_oc.yaml`
- Modify: `configs/imbdiff/cifar100_lt_cm.yaml`

**Interfaces:**
- Consumes: `load_config(path: str | Path) -> dict[str, Any]`.
- Produces: an identical `training.early_stopping` mapping in every ImbDiff config.

- [ ] **Step 1: Write the failing matrix test**

Add this test to `tests/test_config_smoke.py`:

```python
def test_all_imbdiff_configs_enable_shared_early_stopping() -> None:
    paths = sorted(Path("configs/imbdiff").glob("*.yaml"))
    expected = {
        "enabled": True,
        "patience_steps": 10000,
        "warmup_steps": 20000,
        "min_delta": 0.0001,
        "ema_alpha": 0.01,
    }

    assert len(paths) == 9
    for path in paths:
        assert load_config(path)["training"]["early_stopping"] == expected
```

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py::test_all_imbdiff_configs_enable_shared_early_stopping
```

Expected: FAIL with `KeyError: 'early_stopping'` because the policy is not yet present.

- [ ] **Step 3: Add the policy to all nine configs**

Under each file's `training:` section, immediately after optimizer `warmup_steps`, add:

```yaml
  early_stopping:
    enabled: true
    patience_steps: 10000
    warmup_steps: 20000
    min_delta: 0.0001
    ema_alpha: 0.01
```

- [ ] **Step 4: Run focused verification under Conda**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py tests/test_early_stopping.py
.conda/fm_lab/bin/ruff check tests/test_config_smoke.py
git diff --check
```

Expected: all tests pass, Ruff reports `All checks passed!`, and `git diff --check` exits with no output.

- [ ] **Step 5: Commit and push the verified config policy**

```bash
git add tests/test_config_smoke.py configs/imbdiff
git commit -m "Enable early stopping for ImbDiff experiments"
git push
```

Expected: one commit containing the regression test and nine config updates is pushed to `codex/imbdiff-cm-round6`.

---

### Task 2: Prove Conda parity and remove the redundant virtual environment

**Files:**
- Remove locally: `.venv/` (git-ignored generated environment)
- Generate locally: `runs/verification/conda_cifar10_cm/` (git-ignored smoke artifacts)

**Interfaces:**
- Consumes: `.conda/fm_lab/bin/python`, `.conda/fm_lab/bin/pytest`, `.conda/fm_lab/bin/ruff`, and `.conda/fm_lab/bin/fm-lab-train`.
- Produces: a Conda-generated CIFAR-10 CM checkpoint, metrics, and samples; no second Python environment.

- [ ] **Step 1: Verify Conda dependency consistency**

Run:

```bash
.conda/fm_lab/bin/python -m pip check
.conda/fm_lab/bin/python -c 'import fm_lab, torch, torchvision, yaml; print(fm_lab.__file__); print(torch.__version__); print(torchvision.__version__)'
```

Expected: `No broken requirements found.` followed by the local editable `fm_lab` path and installed PyTorch/torchvision versions.

- [ ] **Step 2: Run the full repository verification under Conda**

Run:

```bash
.conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check .
```

Expected: the full suite passes with zero failures and Ruff reports `All checks passed!`.

- [ ] **Step 3: Run a real CIFAR-10 CM CPU smoke experiment under Conda**

Run:

```bash
.conda/fm_lab/bin/fm-lab-train \
  --config configs/imbdiff/cifar10_lt_cm.yaml \
  --device cpu \
  --steps 1 \
  --batch-size 2 \
  --n-samples 2 \
  --sample-batch-size 2 \
  --plot-max-points 2 \
  --output-dir runs/verification/conda_cifar10_cm
```

Expected: exit code 0, `Finished run: runs/verification/conda_cifar10_cm`, and files `checkpoint.pt`, `metrics.json`, and `samples/ddim.npy` exist.

- [ ] **Step 4: Inspect smoke evidence before deletion**

Run:

```bash
test -f runs/verification/conda_cifar10_cm/checkpoint.pt
test -f runs/verification/conda_cifar10_cm/metrics.json
test -f runs/verification/conda_cifar10_cm/samples/ddim.npy
.conda/fm_lab/bin/python -c 'import json; p="runs/verification/conda_cifar10_cm/metrics.json"; m=json.load(open(p)); assert m["device"] == "cpu"; assert m["trained_steps"] == 1; assert m["objective"]["cm"]["base_target"] == "oc"'
```

Expected: every command exits with code 0.

- [ ] **Step 5: Remove only the verified redundant environment**

Run:

```bash
rm -rf .venv
test ! -e .venv
.conda/fm_lab/bin/fm-lab-train --help >/dev/null
```

Expected: `.venv` no longer exists and the Conda training entry point still exits successfully.

- [ ] **Step 6: Verify final repository state**

Run:

```bash
git status --short --branch
```

Expected: the branch tracks `origin/codex/imbdiff-cm-round6` with no uncommitted tracked or untracked changes.
