# Long-Tail Experimental Strand Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the falsified long-tail gradient/functional-geometry framework while retaining a hypothesis-neutral probe toolkit and the paper-derived Capacity Manipulation baseline.

**Architecture:** Relocate reusable probe primitives before deleting their old package. Then reduce CM to its paper-derived continuous-flow adaptation, remove the audit-only training stop, delete all geometry protocols and working artifacts, and preserve the scientific result in a postmortem.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, Ruff, YAML, setuptools.

## Global Constraints

- Retain CM adapters, routing, `CMModifier`, checkpoint compatibility, and `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm.yaml`.
- Describe CM as a **continuous-flow adaptation**, never an exact reproduction.
- Preserve the user's loss-row precision order as `.cpu().double()` in the migrated checkpoint probe.
- Do not touch ignored `runs/` artifacts.
- Leave no alias for `fm_lab.diagnostics.long_tail_geometry`, its CLIs, or `diagnostic_stop`.
- Retained probes compute measurements only; no head/tail conclusions, gates, stage state, or prescribed actions.
- Retain independent `fm_lab.geometry_explorer` and `fm_lab.image_diagnostics` code.
- Use `apply_patch` for edits and inspect the staged file list before every commit.

## Resulting File Map

Create `fm_lab/diagnostics/probes/{__init__,manifest,checkpoints,gradients,sketch,controls,subspaces,perturbations}.py`. Create focused `tests/test_probe_*.py`, `docs/capacity_manipulation.md`, and `docs/long_tail_geometry_postmortem.md`. Modify `fm_lab/training/{long_tail,trainer}.py`, their tests, `docs/diagnostics.md`, and `pyproject.toml`. Delete `fm_lab/diagnostics/long_tail_geometry/`, both `run_long_tail_geometry_*` CLIs, protocol configs/tests, CM pilot configs, and superseded geometry/CM-audit implementation diaries.

---

### Task 1: Extract the Generic Deterministic Probe Core

**Files:**
- Create: `fm_lab/diagnostics/probes/__init__.py`
- Create from retained code: `fm_lab/diagnostics/probes/{manifest,checkpoints,gradients,sketch,controls}.py`
- Create by relocating tests: `tests/test_probe_{manifest,checkpoints,gradients,sketch,controls}.py`

**Interfaces:**
- Consumes: target `diagnostic_samples`, source `sample`, experiment factories, objective/path prediction APIs.
- Produces: `ProbeBatch`, `ProbeManifest`, `ProbeLossResult`, `ProbeLayer`, `GradientRows`, `CountSketchSpec`, `SketchValidation`, permutation/control records and functions.

- [ ] **Step 1: Point retained tests at the future package**

Relocate these tests without weakening assertions:

```text
test_long_tail_probe_manifest.py     -> test_probe_manifest.py
test_long_tail_probe_checkpoints.py  -> test_probe_checkpoints.py
test_long_tail_gradient_probe.py     -> test_probe_gradients.py
test_long_tail_gradient_sketch.py    -> test_probe_sketch.py
test_long_tail_geometry_controls.py  -> test_probe_controls.py
```

Replace every import root with `fm_lab.diagnostics.probes`. Move only the checkpoint fixture needed by `test_probe_checkpoints.py` into that file or `tests/probe_helpers.py`; it must not import `long_tail_geometry_helpers`.

- [ ] **Step 2: Verify the tests fail before extraction**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_probe_*.py
```

Expected: collection fails with `ModuleNotFoundError: fm_lab.diagnostics.probes`.

- [ ] **Step 3: Relocate the retained modules**

Copy the validated bodies of `manifest.py`, `checkpoints.py`, `gradients.py`, `sketch.py`, and `controls.py`; change internal imports to the new package. Replace `Stage-0` in replay messages with `Deterministic probe`, but retain the ordinary flow-matching/capacity-disabled safety checks. Preserve:

```python
losses.append(
    (prediction - expected).square().flatten(1).mean(1).cpu().double()
)
```

Create `__init__.py` with explicit imports for all public classes/functions from those five modules and:

```python
__all__ = [name for name in globals() if not name.startswith("_")]
```

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_probe_*.py
.conda/fm_lab/bin/ruff check fm_lab/diagnostics/probes tests/test_probe_*.py
git add fm_lab/diagnostics/probes tests/test_probe_*.py
git commit -m "Extract generic diagnostic probe core"
```

Expected: tests pass, Ruff is clean, and the staged list excludes the old unstaged checkpoint path.

---

### Task 2: Extract Subspace and Reversible Perturbation Primitives

**Files:**
- Create: `fm_lab/diagnostics/probes/subspaces.py`
- Create: `fm_lab/diagnostics/probes/perturbations.py`
- Modify: `fm_lab/diagnostics/probes/__init__.py`
- Create: `tests/test_probe_subspaces.py`, `tests/test_probe_perturbations.py`

**Interfaces:**
- Consumes: `resolve_probe_layers(model, names)` and CPU tensors.
- Produces: `PrincipalDirection`, `ProjectedDirection`, three direction functions, `projection_overlap`, and `virtual_layer_update`.

- [ ] **Step 1: Move only mechanism-free tests**

Move the tests for top centered covariance direction, projected descent, deterministic keyed random direction, and exact parameter restoration from `test_long_tail_geometry_functional_calibration.py`. Add:

```python
direction = top_centered_covariance_direction(
    torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
)
assert isinstance(direction, PrincipalDirection)
assert direction.explained_fraction == pytest.approx(1.0)
```

- [ ] **Step 2: Verify missing modules**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_probe_subspaces.py tests/test_probe_perturbations.py
```

Expected: collection fails because the modules do not exist.

- [ ] **Step 3: Implement generic subspace operations**

Use the validated calculation bodies with these public result types:

```python
@dataclass(frozen=True)
class PrincipalDirection:
    vector: torch.Tensor
    eigenvalue: float
    explained_fraction: float

@dataclass(frozen=True)
class ProjectedDirection:
    vector: torch.Tensor
    projection_fraction: float

```

Retain the existing signatures of `top_centered_covariance_direction`,
`projected_descent_direction`, and `deterministic_random_unit_direction`,
changing only the first two return annotations to the generic result types
above. The complete function bodies are the already-tested bodies at
`functional_calibration.py:148-233`; copy them without modifying validation,
CPU conversion, eigendecomposition, orientation, hashing, seeding, or
normalization.

Re-export `projection_overlap` from `controls.py`. Do not migrate paired-audit directions, relative-benefit slopes, scale selection, class-response metrics, or decisions.

- [ ] **Step 4: Implement exact reversible perturbation**

Move the complete `virtual_layer_update` definition from
`functional_calibration.py:236-289`, retaining its signature and every input
validation. Its `finally` block must restore with `parameter.copy_(original)`
under `torch.no_grad()` and verify `torch.equal`. Import
`resolve_probe_layers` from the new gradients module and export the new API
from `probes/__init__.py`.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_probe_subspaces.py tests/test_probe_perturbations.py
.conda/fm_lab/bin/ruff check fm_lab/diagnostics/probes tests/test_probe_subspaces.py tests/test_probe_perturbations.py
git add fm_lab/diagnostics/probes tests/test_probe_subspaces.py tests/test_probe_perturbations.py
git commit -m "Extract generic subspace and perturbation probes"
```

---

### Task 3: Retain the Paper-Derived CM Adaptation

**Files:**
- Modify: `fm_lab/training/long_tail.py`, `tests/test_continuous_long_tail.py`, `tests/test_config_smoke.py`
- Retain unchanged: `fm_lab/models/capacity.py`, CM model/factory/prediction wiring, `tests/test_capacity_manipulation.py`, canonical CM config
- Create: `docs/capacity_manipulation.md`
- Delete: `configs/fashion_mnist_lt/*cm_bounded*.yaml`, `configs/fashion_mnist_lt/*cm_pilot*.yaml`
- Delete: 2026-07-16 CM-weighting/full-network plans and specs

**Interfaces:**
- Produces: unbounded `CMModifier` with paper defaults and a documented continuous-flow adaptation.

- [ ] **Step 1: Write the reduced CM contract test**

Delete bounded-diversity tests and add:

```python
def test_cm_defaults_match_released_core_weighting() -> None:
    modifier = CMModifier(class_counts=[100, 10])
    assert modifier.consistency_weight == 1.0
    assert modifier.diversity_weight == 0.2
    assert modifier.comparison_space == "target"
    assert "diversity_mode" not in modifier.metadata()
    assert "diversity_margin" not in modifier.metadata()
```

Delete the bounded-config test in `test_config_smoke.py`; retain assertions for `parts: [up]`, weights `1.0/0.2`, and target comparison space.

- [ ] **Step 2: Verify the new contract fails, then simplify CM**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_continuous_long_tail.py::test_cm_defaults_match_released_core_weighting
```

Expected initially: FAIL because old metadata exposes bounded fields.

Make `CMModifier` fields exactly `class_counts`, `consistency_weight=1.0`, `diversity_weight=0.2`, `comparison_space="target"`, and `name="cm"`. Compute:

```python
diversity = -num_classes * (
    inverse_probabilities[labels] * self.diversity_weight * distance
).mean()
```

Remove bounded parsing, saturation metrics, and bounded metadata. Do not change frequency normalization or capacity-on/off prediction routing.

- [ ] **Step 3: Document correspondence and delete audit variants**

`docs/capacity_manipulation.md` must state that retained pieces are zero-initialized switchable low-rank capacity, on/off distance, frequency weighting, `rank_ratio=0.1`, up-block placement, and `1.0/0.2`; differences are continuous FM, flattened-kernel factors, and absence of endpoint transfer. Link the OpenReview paper and authors' repository.

Delete all pilot/bounded configs and:

```text
docs/superpowers/plans/2026-07-16-cm-weighting-audit.md
docs/superpowers/plans/2026-07-16-full-network-cm-capacity.md
docs/superpowers/specs/2026-07-16-cm-weighting-audit-design.md
docs/superpowers/specs/2026-07-16-full-network-cm-capacity-design.md
```

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_capacity_manipulation.py tests/test_continuous_long_tail.py tests/test_config_smoke.py -k 'cm or continuous_fashion'
.conda/fm_lab/bin/ruff check fm_lab/models/capacity.py fm_lab/training/long_tail.py tests/test_capacity_manipulation.py tests/test_continuous_long_tail.py tests/test_config_smoke.py
git add fm_lab/training/long_tail.py tests/test_continuous_long_tail.py tests/test_config_smoke.py configs/fashion_mnist_lt docs
git commit -m "Retain paper-derived CM adaptation baseline"
```

Confirm the staged diff does not delete the canonical CM config or capacity model.

---

### Task 4: Remove the Audit-Only Training Stop

**Files:**
- Modify: `fm_lab/training/trainer.py`, `tests/test_training_sampling.py`

**Interfaces:**
- Produces: unchanged ordinary training/early stopping without a `diagnostic_stop` API or metrics key.

- [ ] **Step 1: Add a failing absence assertion**

In an existing short test that receives training metrics, add:

```python
assert "diagnostic_stop" not in metrics
```

Delete `test_metric_stop_policy_*` and `test_diagnostic_stop_prevents_update_that_triggers_it`.

- [ ] **Step 2: Verify failure and remove the feature**

Run the selected short training test; expect failure because the key exists. Then remove `_MetricStopPolicy`, `_build_metric_stop_policy`, config parsing, trainable-path validation, `metric_stop.enabled` record forcing, `metric_stop.update`, the stop-before-update branch, and the metrics entry. Restore:

```python
should_record = should_log or early_stopping.enabled
```

Preserve `_EarlyStopping` and checkpoint schedules.

- [ ] **Step 3: Verify and commit**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_training_sampling.py
.conda/fm_lab/bin/ruff check fm_lab/training/trainer.py tests/test_training_sampling.py
git add fm_lab/training/trainer.py tests/test_training_sampling.py
git commit -m "Remove geometry audit training stop"
```

---

### Task 5: Delete Geometry Protocols and Write the Postmortem

**Files:**
- Delete: `fm_lab/diagnostics/long_tail_geometry/`, both long-tail geometry CLIs, all protocol configs/tests/helper
- Modify: `pyproject.toml`, `docs/diagnostics.md`
- Create: `docs/long_tail_geometry_postmortem.md`
- Delete: superseded geometry plans/specs listed below

**Interfaces:**
- Consumes: Tasks 1–4 with no retained import of the old package.
- Produces: no executable geometry protocol; one historical record.

- [ ] **Step 1: Prove retained code no longer needs the old package**

```bash
rg -n 'fm_lab\.diagnostics\.long_tail_geometry|run_long_tail_geometry|diagnostic_stop' fm_lab tests pyproject.toml docs/diagnostics.md configs
```

Expected: references occur only in the old package, protocol tests/CLIs/configs/docs; no new probe file imports it.

- [ ] **Step 2: Delete protocol code and data contracts**

Use explicit `apply_patch` deletions for all files under the old package, both `run_long_tail_geometry_*` files, all `configs/**/long_tail_geometry_*.yaml`, `fashion_mnist_lt_geometry_stage0.yaml`, `cifar10_lt_geometry_falsification.yaml`, all `tests/test_long_tail_geometry_*.py`, and `tests/long_tail_geometry_helpers.py`.

Before deleting old `checkpoints.py`, verify `.cpu().double()` exists in the new file and `tests/test_probe_checkpoints.py` passes.

- [ ] **Step 3: Remove packaging surface**

Delete from `pyproject.toml`:

```toml
long-tail-geometry = ["pyarrow>=16.0"]
fm-lab-long-tail-geometry-stage0 = "fm_lab.experiments.run_long_tail_geometry_stage0:main"
fm-lab-long-tail-geometry-observation0 = "fm_lab.experiments.run_long_tail_geometry_observation0:main"
```

Retain `fm-lab-geometry`.

- [ ] **Step 4: Replace protocol docs with a generic API boundary**

Rewrite the affected `docs/diagnostics.md` section to say:

```markdown
## Deterministic model probes

`fm_lab.diagnostics.probes` provides deterministic sample manifests, exact
checkpoint loss replay, selected-layer gradient rows, CountSketch validation,
permutation/planted controls, subspace directions, and reversible parameter
perturbations. Callers own hypotheses, selection rules, and decisions.
```

Remove all Stage 0, Observation 0, calibration, audit, and natural-image commands.

- [ ] **Step 5: Record the evidence and delete diaries**

The postmortem must record: Fashion-MNIST positive-control failure; mixed/class-heterogeneous audit slopes (normalized `1.82596`, `1.86796`; raw `1.66326`, `1.08182`); CIFAR shared rank-1 layer counts `0/5/6/2/4/1` at steps `0/2500/10000/25000/50000/100000`; final shared layers `1/2/2/0` for ranks `1/2/4/8`, below minimum `5`; and the decision not to promote this operational hypothesis. State that ignored run artifacts remain untouched.

Delete these geometry diaries:

```text
docs/superpowers/plans/2026-07-16-cifar10-lt-transport-falsification.md
docs/superpowers/plans/2026-07-16-long-tail-functional-calibration.md
docs/superpowers/plans/2026-07-16-long-tail-functional-geometry-representation-audit.md
docs/superpowers/plans/2026-07-16-long-tail-geometry-observation0.md
docs/superpowers/plans/2026-07-16-long-tail-geometry-stage0.md
docs/superpowers/plans/2026-07-16-long-tail-gradient-geometry-observation-experiments.md
docs/superpowers/specs/2026-07-16-cifar10-lt-transport-falsification-design.md
docs/superpowers/specs/2026-07-16-long-tail-functional-calibration-design.md
docs/superpowers/specs/2026-07-16-long-tail-functional-geometry-representation-audit-design.md
```

- [ ] **Step 6: Verify absence and commit**

```bash
test -z "$(rg -l 'fm_lab\.diagnostics\.long_tail_geometry|run_long_tail_geometry|diagnostic_stop' fm_lab tests pyproject.toml configs docs/diagnostics.md)"
test -z "$(rg -l 'pyarrow' fm_lab pyproject.toml)"
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_probe_*.py
.conda/fm_lab/bin/ruff check fm_lab/diagnostics/probes
git add pyproject.toml fm_lab configs tests docs
git commit -m "Remove falsified long-tail geometry protocols"
```

Confirm no `runs/` artifact is staged.

---

### Task 6: Full Regression Verification

**Files:** Modify only for discovered stale imports or formatting failures.

- [ ] **Step 1: Verify file boundaries**

```bash
test ! -d fm_lab/diagnostics/long_tail_geometry
test -f fm_lab/diagnostics/probes/checkpoints.py
test -f configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm.yaml
test -z "$(find configs/fashion_mnist_lt -maxdepth 1 -type f -name '*cm_pilot*' -print)"
test -z "$(find configs/fashion_mnist_lt -maxdepth 1 -type f -name '*cm_bounded*' -print)"
test -z "$(rg -l 'long_tail_geometry|diagnostic_stop' fm_lab tests pyproject.toml configs docs/diagnostics.md)"
```

- [ ] **Step 2: Run focused and full suites**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_probe_*.py tests/test_capacity_manipulation.py tests/test_continuous_long_tail.py tests/test_config_smoke.py tests/test_training_sampling.py
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check fm_lab tests
```

Expected: all tests pass and Ruff reports `All checks passed!`; report environmental skips exactly.

- [ ] **Step 3: Smoke import and audit the final diff**

```bash
PYTHONPATH=. .conda/fm_lab/bin/python -c 'from fm_lab.diagnostics import probes; from fm_lab.training.long_tail import CMModifier; print(probes.ProbeManifest.__name__, CMModifier.__name__)'
git status --short
git diff --check
rg -n '\.cpu\(\)\.double\(\)' fm_lab/diagnostics/probes/checkpoints.py
git diff --stat 08e8491..HEAD
```

Expected: Python prints `ProbeManifest CMModifier`; no unstaged edit remains at the deleted old checkpoint path because its behavior is migrated and tested; no whitespace errors or run artifacts.

- [ ] **Step 4: Commit verification-only fixes if needed**

If verification required edits, stage only those files and commit with `git commit -m "Finish long-tail cleanup verification"`. Do not create an empty commit.
