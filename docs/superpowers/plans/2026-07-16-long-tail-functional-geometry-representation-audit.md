# Representation-Matched Functional Geometry Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable Probe-A audit that compares raw and row-normalized exact rank-1 gradient bases through cross-fitted zero-step slopes and a locked local finite-step grid without changing the failed functional lock.

**Architecture:** A new immutable audit preregistration binds the already-blocked functional calibration. A separate audit module reuses its validated context and exact replay primitives, builds four paired cross-fits per cell, writes digest-bound per-seed/checkpoint chunks, and derives a non-unlocking interpretation summary. The existing Observation-0 CLI gains a distinct `audit-functional-geometry` command.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, PyYAML, pytest.

## Global Constraints

- Probe A only; never load `probe_b.npz`.
- Preserve every file beneath `STUDY/aggregate/functional_calibration/` byte-for-byte.
- Require the original functional decision to remain blocked with next action `stop_stage1_and_revise_functional_geometry`.
- Use checkpoints 500 and 20,000; layers `down2_block.conv2.weight` and `middle.conv2.weight`; classes `0, 2, 3, 4, 6, 9`; stratum 0; rank 1.
- Use four circular offsets `(0, 4, 8, 12)` with 8/4/4 fit/scale/evaluation positions.
- Compare `raw` and `row_normalized` bases while orienting both with the identical raw scale mean gradient.
- Use finite relative steps exactly `(1e-4, 3e-4, 1e-3)` and never reinterpret them as satisfying the original 1% gate.
- No status or artifact from this audit may unlock Stage 1.
- Preserve and do not stage the pre-existing edit in `fm_lab/diagnostics/long_tail_geometry/checkpoints.py`.

---

### Task 1: Lock the audit preregistration

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/functional_audit_preregistration.py`
- Create: `configs/fashion_mnist_lt/long_tail_geometry_functional_audit.yaml`
- Create: `tests/test_long_tail_geometry_functional_audit_preregistration.py`

**Interfaces:**
- Consumes: the canonical Observation-0 digest and functional-calibration preregistration digest.
- Produces: `FunctionalGeometryAuditPreregistration.load(path)`, `.to_dict()`, `.digest`, `.lock(path)`, `.fold_positions`.

- [ ] **Step 1: Write failing schema and fold tests**

```python
def test_canonical_audit_preregistration_locks_representation_comparison() -> None:
    prereg = FunctionalGeometryAuditPreregistration.load(AUDIT_CONFIG)
    assert prereg.functional_preregistration_sha256 == (
        "f40251443426eab2f24d89cdf359f07615ccda788341d16719c1f0ec40836bdf"
    )
    assert prereg.basis_kinds == ("raw", "row_normalized")
    assert prereg.fold_offsets == (0, 4, 8, 12)
    assert prereg.relative_step_grid == (1e-4, 3e-4, 1e-3)
    folds = prereg.fold_positions
    assert folds[0] == {
        "fit": tuple(range(8)),
        "scale": tuple(range(8, 12)),
        "evaluation": tuple(range(12, 16)),
    }
    assert folds[3] == {
        "fit": (12, 13, 14, 15, 0, 1, 2, 3),
        "scale": (4, 5, 6, 7),
        "evaluation": (8, 9, 10, 11),
    }
```

Also test immutable round trips, unknown/missing fields, Probe B, a changed digest, non-blocked required state, duplicate bases/offsets, unbalanced fold coverage, a grid other than the three locked values, rank other than one, and any class/layer/checkpoint scope drift.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_functional_audit_preregistration.py
```

Expected: import failure for the missing preregistration module.

- [ ] **Step 3: Implement the frozen schema**

Use a frozen dataclass with exact nested-key validation. The fold property must construct positions as:

```python
@property
def fold_positions(self) -> tuple[dict[str, tuple[int, ...]], ...]:
    folds = []
    for offset in self.fold_offsets:
        order = tuple((offset + index) % self.microbatches_per_cell for index in range(16))
        folds.append({
            "fit": order[:8],
            "scale": order[8:12],
            "evaluation": order[12:16],
        })
    return tuple(folds)
```

Validation must count each position exactly twice in fit, once in scale, and once in evaluation across folds. `lock()` follows the existing functional-preregistration immutable-file pattern.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all tests in the file pass.

- [ ] **Step 5: Commit the contract**

```bash
git add fm_lab/diagnostics/long_tail_geometry/functional_audit_preregistration.py configs/fashion_mnist_lt/long_tail_geometry_functional_audit.yaml tests/test_long_tail_geometry_functional_audit_preregistration.py
git commit -m "Lock representation-matched audit contract"
```

### Task 2: Add paired basis, slope, finite-error, and analysis primitives

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/functional_audit.py`
- Create: `tests/test_long_tail_geometry_functional_audit.py`

**Interfaces:**
- Consumes: `GradientRows`, `top_centered_covariance_direction`, `projected_descent_direction`.
- Produces: `paired_projected_directions(...)`, `relative_benefit_slope(...)`, `analyze_functional_geometry_audit(...)`, `FunctionalGeometryAuditDecision`.

- [ ] **Step 1: Write a failing paired-representation test**

```python
def test_paired_directions_change_only_fit_representation() -> None:
    raw = torch.tensor([
        [100.0, 0.0],
        [-100.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.0, 1.2],
        [0.0, -0.8],
        [1.0, 1.0],
        [2.0, 1.0],
    ])
    rows = GradientRows(
        raw=raw,
        norms=torch.linalg.vector_norm(raw, dim=1),
        normalized=raw / torch.linalg.vector_norm(raw, dim=1)[:, None],
    )
    result = paired_projected_directions(
        rows,
        fit_positions=(0, 1, 2, 3, 4, 5),
        scale_positions=(6, 7),
        minimum_projection_fraction=1e-8,
    )
    assert set(result) == {"raw", "row_normalized"}
    assert result["raw"].basis_kind == "raw"
    assert result["row_normalized"].basis_kind == "row_normalized"
    assert result["raw"].orientation_gradient_sha256 == (
        result["row_normalized"].orientation_gradient_sha256
    )
```

The scale fixture is non-cancelling so both projections are valid. Assert a
nontrivial raw/normalized basis cosine and exact vector digests.

- [ ] **Step 2: Write a failing analytic-slope test**

```python
def test_relative_benefit_slope_matches_definition() -> None:
    direction = torch.tensor([3.0, 4.0]) / 5.0
    mean_gradient = torch.tensor([2.0, -1.0])
    slope = relative_benefit_slope(
        direction=direction,
        evaluation_mean_gradient=mean_gradient,
        parameter_norm=10.0,
        base_loss=4.0,
    )
    assert slope == pytest.approx(-10.0 * torch.dot(mean_gradient, direction).item() / 4.0)
```

Test shape mismatch, non-unit direction, non-finite values, nonpositive parameter norm/base loss, and agreement with a tiny linear model's central finite difference.

- [ ] **Step 3: Write failing block-analysis tests**

Construct complete synthetic slope and finite tables for two layers, three seeds, six classes, four folds, two bases, two checkpoints, and all six evaluation classes. Verify:

```python
decision = analyze_functional_geometry_audit(slopes, finite, prereg)
assert decision.stage1_unlocked is False
assert decision.probe_b_opened is False
assert decision.status == "normalized_representation_rescue"
assert decision.next_action == "review_separate_small_local_step_preregistration"
```

Add fixtures for `representation_independent_local_transport`, `no_transferable_local_descent`, and `mixed_or_class_heterogeneous_transport`. Assert folds collapse inside each seed/class block before bootstrap, paired intervals use normalized-minus-raw block values, selectivity uses the worst off-class slope, seed repeats use seed medians, local concordance compares finite benefit with `epsilon * slope`, and missing/duplicate/non-finite cells fail closed.

- [ ] **Step 4: Run the pure tests and verify RED**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_functional_audit.py
```

Expected: import failure for the missing audit module.

- [ ] **Step 5: Implement minimal pure primitives**

The slope implementation is exactly:

```python
def relative_benefit_slope(*, direction, evaluation_mean_gradient, parameter_norm, base_loss):
    return -float(parameter_norm) * float(
        torch.dot(evaluation_mean_gradient.float().cpu(), direction.float().cpu())
    ) / float(base_loss)
```

`paired_projected_directions` fits `top_centered_covariance_direction` once on raw fit rows and once on normalized fit rows, but passes the same `rows.raw[scale_positions].mean(0)` to both calls of `projected_descent_direction`.

For analysis, group each direction-class target and selectivity slope by checkpoint/layer/basis/seed/class and take the median over folds. Bootstrap the resulting 18 values. Compute paired deltas by merging normalized and raw block tables on checkpoint/layer/seed/class before subtraction. Always return `stage1_unlocked=False`.

- [ ] **Step 6: Run the pure tests and verify GREEN**

Run the command from Step 4. Expected: all tests in the file pass.

- [ ] **Step 7: Commit the pure audit layer**

```bash
git add fm_lab/diagnostics/long_tail_geometry/functional_audit.py tests/test_long_tail_geometry_functional_audit.py
git commit -m "Add paired functional audit primitives"
```

### Task 3: Build digest-bound collection and resumability

**Files:**
- Modify: `fm_lab/diagnostics/long_tail_geometry/functional_audit.py`
- Create: `tests/test_long_tail_geometry_functional_audit_service.py`

**Interfaces:**
- Consumes: `prepare_calibration_context`, the completed blocked functional lock, `_load_probe_components`, `cell_microbatch_rows`, `materialize_probe_batch`, `collect_gradient_rows`, `_mean_batch_loss`, `virtual_layer_update`.
- Produces: `prepare_functional_geometry_audit_context(...)`, `collect_audit_chunk(...)`, `run_functional_geometry_audit(...)`, `FunctionalGeometryAuditResult`.

- [ ] **Step 1: Write failing context and safety tests**

Use a toy `CalibrationContext` and temporary completed functional artifact. Verify the audit rejects:

```python
changed_lock = {**lock, "stage1_unlocked": True}
with pytest.raises(ValueError, match="blocked functional calibration"):
    validate_blocked_functional_lock(changed_lock, prereg)
```

Also reject a changed next action, `probe_b_opened=True`, functional-preregistration digest mismatch, missing/tampered `complete.json`, incomplete checkpoints/classes/layers, or an attempt to use any manifest other than Probe A. Monkeypatch `ProbeManifest.load` to raise if its path contains `probe_b.npz`.

- [ ] **Step 2: Write a failing real collector test on a tiny model**

Create fixed `ProbeBatch` rows for two classes and a two-layer toy model. Assert one chunk contains:

- `2 layers * 2 bases * 4 folds * 2 direction classes * 2 evaluation classes` slope rows;
- `2 layers * 2 bases * 4 folds * 2 direction classes * 2 partitions * 3 steps` finite rows;
- one basis-comparison row per layer/fold/direction class;
- identical raw orientation-gradient digests for paired bases;
- finite evaluation benefits whose sign approaches the analytic slope prediction as epsilon decreases;
- exact model parameters before and after collection.

- [ ] **Step 3: Write failing artifact/resume tests**

Monkeypatch `collect_audit_chunk` to return deterministic complete tables. Require files:

```python
for name in (
    "preregistration.yaml",
    "slopes.csv",
    "finite_steps.csv",
    "basis_comparison.csv",
    "audit_summary.json",
    "complete.json",
):
    assert (artifact_dir / name).is_file()
```

Assert the summary contains `stage1_unlocked: false`, `probe_b_opened: false`, the original functional-lock digest, all upstream digests, and a closed-vocabulary next action. A second invocation must make no collector calls. Tampering with a chunk, aggregate CSV, summary, original lock, checkpoint, or preregistration must raise. Snapshot hashes of every original functional-calibration file before and after the service and assert equality. Assert no path named `stage1` is created.

- [ ] **Step 4: Run the service tests and verify RED**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_functional_audit_service.py
```

Expected: missing service interfaces.

- [ ] **Step 5: Implement context validation and one-pass collection**

Load the locked functional preregistration from
`STUDY/aggregate/functional_calibration/preregistration.yaml`, pass it through
`prepare_calibration_context`, validate the completed failed decision and file
digests, then wrap it in an audit context whose artifact directory is
`STUDY/aggregate/functional_geometry_audit`.

For each seed/checkpoint, load one model and materialize all 16 batches per
selected class. Collect all raw/normalized gradient rows once per class. For
each fold, fit paired directions from the direction class, compute all
cross-class analytic slopes using evaluation-class raw means and base losses,
then evaluate the target class only on scale/evaluation partitions at the three
finite steps under `virtual_layer_update`.

- [ ] **Step 6: Implement immutable chunks and aggregate completion**

Each seed/checkpoint chunk writes three CSVs plus one JSON identity sidecar that
hashes all three. Concatenate only complete chunks. Write aggregate files
atomically, analyze them, then write `audit_summary.json` and `complete.json`.
A completed run validates every aggregate and input digest before returning.

- [ ] **Step 7: Run service and adjacent calibration tests**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_functional_audit_service.py tests/test_long_tail_geometry_functional_service.py tests/test_long_tail_geometry_functional_calibration.py
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit the service**

```bash
git add fm_lab/diagnostics/long_tail_geometry/functional_audit.py tests/test_long_tail_geometry_functional_audit_service.py
git commit -m "Add resumable representation audit service"
```

### Task 4: Expose the audit command and production handoff

**Files:**
- Modify: `fm_lab/experiments/run_long_tail_geometry_observation0.py`
- Modify: `tests/test_long_tail_geometry_observation0.py`
- Modify: `docs/diagnostics.md`

**Interfaces:**
- Consumes: `run_functional_geometry_audit(...)`.
- Produces: `audit-functional-geometry --study-dir STUDY --audit-preregistration FILE --device auto`.

- [ ] **Step 1: Write failing CLI tests**

```python
args = parse_args([
    "audit-functional-geometry",
    "--study-dir", "runs/study",
    "--audit-preregistration", "configs/audit.yaml",
    "--device", "cpu",
])
assert args.command == "audit-functional-geometry"
```

Monkeypatch the service result and require output lines for audit status, both
layer summaries, original lock remaining blocked, Probe B remaining sealed, and
the only audit next action. Existing prepare/collect/analyze/calibrate parsing
must remain unchanged, and `stage1` must remain invalid.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_observation0.py
```

Expected: parser rejects `audit-functional-geometry`.

- [ ] **Step 3: Add CLI wiring and documentation**

The command dispatch is:

```python
if args.command == "audit-functional-geometry":
    result = run_functional_geometry_audit(
        study_dir=study_dir,
        audit_preregistration_path=args.audit_preregistration,
        device=resolve_device(args.device),
    )
```

Document the exact production command, artifact directory, non-unlocking
interpretation, and the fact that the failed functional lock is immutable.

- [ ] **Step 4: Run CLI and audit tests**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_observation0.py tests/test_long_tail_geometry_functional_audit_preregistration.py tests/test_long_tail_geometry_functional_audit.py tests/test_long_tail_geometry_functional_audit_service.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the handoff**

```bash
git add fm_lab/experiments/run_long_tail_geometry_observation0.py tests/test_long_tail_geometry_observation0.py docs/diagnostics.md
git commit -m "Expose functional geometry audit command"
```

### Task 5: Verify scientific and repository integrity

**Files:**
- Verify only; do not modify the original functional-calibration artifacts or the pre-existing `checkpoints.py` edit.

**Interfaces:**
- Consumes: the complete branch.
- Produces: fresh evidence for correctness and the production command for the user.

- [ ] **Step 1: Check diff scope and formatting**

```bash
git diff --check main...HEAD
git status --short
```

Confirm the only unrelated working-tree entry is the pre-existing unstaged
`fm_lab/diagnostics/long_tail_geometry/checkpoints.py` edit.

- [ ] **Step 2: Run static checks**

```bash
PYTHONPATH=. .conda/fm_lab/bin/python -m compileall -q fm_lab
.conda/fm_lab/bin/ruff check fm_lab tests
```

Expected: both commands exit zero.

- [ ] **Step 3: Run the complete test suite**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q
```

Expected: zero failures.

- [ ] **Step 4: Smoke-test the command contract without production recomputation**

```bash
PYTHONPATH=. .conda/fm_lab/bin/python -m fm_lab.experiments.run_long_tail_geometry_observation0 audit-functional-geometry --help
PYTHONPATH=. .conda/fm_lab/bin/python -c "from fm_lab.diagnostics.long_tail_geometry.functional_audit_preregistration import FunctionalGeometryAuditPreregistration; print(FunctionalGeometryAuditPreregistration.load('configs/fashion_mnist_lt/long_tail_geometry_functional_audit.yaml').digest)"
```

Expected: help lists the three required arguments and the canonical audit config
loads with a stable 64-character digest.

- [ ] **Step 5: Inspect non-unlocking invariants**

Search the branch diff and confirm no audit code writes beneath
`functional_calibration`, loads Probe B, sets `stage1_unlocked=True`, creates a
Stage-1 config, or trains a model. Confirm all raw/normalized comparisons share
the orientation-gradient digest and fold identity.

- [ ] **Step 6: Commit any verification-only documentation correction, then rerun affected checks**

If no correction is required, create no commit. If documentation alone needs a
correction, stage only that document, commit it, and repeat Steps 1-4.
