# Probe-A Functional Calibration Implementation Plan

> **For Codex:** Execute this plan task by task with test-first development. Do not open Probe-B, create a Stage-1 run config, or train any model.

**Goal:** Add a resumable functional calibration that recomputes exact rank-1 directions from the reliable Observation-0 cells, selects a shared layerwise perturbation producing a 1% Probe-A loss effect, measures held-out class selectivity against matched random controls, and writes the only artifact allowed to unlock Stage 1.

**Architecture:** A frozen calibration-preregistration module owns the scientific contract. A functional-calibration module contains small exact linear-algebra, perturbation, scale-selection, bootstrap, artifact, and orchestration units. The existing checkpoint replay gains a batch-level entry point so selected Probe-A partitions can be evaluated without fabricating manifests. The Observation-0 CLI gains one `calibrate` command which validates the completed pilot before calling the new service.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, PyYAML, pytest.

**Approved design:** `docs/superpowers/specs/2026-07-16-long-tail-functional-calibration-design.md`

---

## Task 1: Lock the calibration preregistration

**Files:**

- Create: `fm_lab/diagnostics/long_tail_geometry/functional_preregistration.py`
- Create: `configs/fashion_mnist_lt/long_tail_geometry_functional_calibration.yaml`
- Create: `tests/test_long_tail_geometry_functional_preregistration.py`

### Step 1: Write failing contract tests

Test that the canonical file loads into immutable tuples and contains exactly:

- Observation-0 digest `6cd1bcf...78e24`;
- checkpoints `(500, 20000)`, primary `20000`, positive control `500`;
- stratum ID `0`, bounds `(0.02, 0.10)`, rank `1`;
- the two approved layers and six common classes;
- partition sizes `8/4/4` totaling 16;
- the nine approved scale-grid values, target `0.01`, tolerance `[0.0075, 0.0125]`, local-linearity limit `0.10`, and max selected step `0.01`;
- 99 random controls, 10,000 bootstrap resamples, and fixed RNG seeds.

Test digest stability, locked-file round trip, rejection of unknown keys, overlapping/invalid partitions, wrong scope, Probe-B, a target other than 1%, too few random controls, and a changed Observation-0 digest.

### Step 2: Run the tests and observe the missing module

Run:

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_functional_preregistration.py
```

Expected: FAIL because the module does not exist.

### Step 3: Implement the minimal frozen contract

Use a frozen dataclass. `load()` accepts an exact nested YAML schema, normalizes sequences to tuples, validates every scientific invariant, and exposes a canonical SHA-256 digest. `lock()` writes an immutable copy and rejects any existing file whose parsed digest differs.

### Step 4: Run the focused tests

Expected: PASS.

### Step 5: Commit

```bash
git add fm_lab/diagnostics/long_tail_geometry/functional_preregistration.py \
  configs/fashion_mnist_lt/long_tail_geometry_functional_calibration.yaml \
  tests/test_long_tail_geometry_functional_preregistration.py
git commit -m "Lock Probe-A functional calibration contract"
```

## Task 2: Add exact cross-fitted direction primitives

**Files:**

- Create: `fm_lab/diagnostics/long_tail_geometry/functional_calibration.py`
- Create: `tests/test_long_tail_geometry_functional_calibration.py`

### Step 1: Write failing mathematical tests

Cover:

- selecting stable within-cell microbatch positions without using global IDs;
- exact top centered covariance direction via the sample Gram matrix;
- agreement with `torch.linalg.svd` up to sign;
- projected descent orientation and projection fraction;
- rejection of fewer than two rows, non-finite rows, zero centered rank, shape mismatch, and negligible projected mean gradient;
- deterministic random unit directions keyed by seed/checkpoint/class/layer/control;
- exact model-parameter restoration after a virtual update, including exception paths.

### Step 2: Run and observe the failure

Run the focused file and confirm missing imports/functions.

### Step 3: Implement minimal primitives

Use the `B x B` centered Gram eigendecomposition rather than a full `D x D` covariance or dense right-SVD. Return CPU float tensors with explicit metadata. The virtual-update context mutates one resolved weight under `torch.no_grad()`, then copies back an exact clone and verifies equality.

### Step 4: Run the focused tests

Expected: PASS.

### Step 5: Commit

```bash
git add fm_lab/diagnostics/long_tail_geometry/functional_calibration.py \
  tests/test_long_tail_geometry_functional_calibration.py
git commit -m "Add exact functional direction primitives"
```

## Task 3: Evaluate selected Probe-A batches exactly

**Files:**

- Modify: `fm_lab/diagnostics/long_tail_geometry/checkpoints.py`
- Modify: `tests/test_long_tail_probe_checkpoints.py`
- Modify: `tests/test_long_tail_geometry_functional_calibration.py`

### Step 1: Write failing replay tests

Add tests that a new `evaluate_probe_batches()` entry point:

- returns the same row losses as `evaluate_probe_loss()` for all manifest batches;
- accepts a selected tuple of materialized batches;
- preserves model mode;
- rejects an empty iterable and every nonordinary objective/model condition already guarded by manifest replay.

Add a functional test showing that a one-layer virtual update changes only the expected finite loss and fully restores the baseline result afterward.

### Step 2: Run and observe missing API failure

### Step 3: Refactor replay without changing semantics

Move the existing exact per-row replay loop behind `evaluate_probe_batches()`. Keep `evaluate_probe_loss()` as a materialization wrapper so existing callers and hashes remain unchanged.

### Step 4: Run both focused test files

Expected: PASS.

### Step 5: Commit

```bash
git add fm_lab/diagnostics/long_tail_geometry/checkpoints.py \
  tests/test_long_tail_probe_checkpoints.py \
  tests/test_long_tail_geometry_functional_calibration.py
git commit -m "Replay selected probe batches exactly"
```

## Task 4: Implement scale selection and the statistical gate

**Files:**

- Modify: `fm_lab/diagnostics/long_tail_geometry/functional_calibration.py`
- Modify: `tests/test_long_tail_geometry_functional_calibration.py`

### Step 1: Write failing pure-analysis tests

Use synthetic scale and response tables to test:

- one shared epsilon per layer chosen by median scale-partition benefit, with smaller-step tie breaking;
- target interval and relative-step validity;
- doubled-step local-linearity error;
- target benefit, worst non-target harm, and selectivity margin from a complete response matrix;
- deterministic seed/class block bootstrap confidence bounds;
- matched-random 99th-percentile comparison;
- the two-layer final-checkpoint pass rule, two-of-three seed repeat rule, and positive-control reporting;
- fail-closed behavior for missing/duplicate cells, NaNs, incomplete random controls, or an invalid scale.

### Step 2: Run and observe missing analysis functions

### Step 3: Implement the pure analysis layer

Keep table construction and statistical decisions independent of model execution. Emit a frozen `FunctionalCalibrationDecision` with `stage1_unlocked`, per-layer diagnostics, positive-control status, and an explicit `next_action` from a closed vocabulary.

### Step 4: Run focused tests

Expected: PASS.

### Step 5: Commit

```bash
git add fm_lab/diagnostics/long_tail_geometry/functional_calibration.py \
  tests/test_long_tail_geometry_functional_calibration.py
git commit -m "Add functional scale and selectivity gate"
```

## Task 5: Build the resumable study service and artifact contract

**Files:**

- Modify: `fm_lab/diagnostics/long_tail_geometry/functional_calibration.py`
- Create: `tests/test_long_tail_geometry_functional_service.py`

### Step 1: Write failing service tests with the tiny Observation-0 fixture

Test that the service:

- rejects anything except `network_wide_measurable` with the expected next action;
- validates study/preregistration digest, primary Probe-A manifest digest, registry seeds, checkpoint hashes, layer/class measurability, checkpoint steps, stratum, and 16 microbatches;
- never loads `probe_b.npz` (monkeypatch its loader path to fail if touched);
- recomputes fit/scale gradients from the exact checkpoint and ordinary objective;
- selects the final-checkpoint layerwise scale before applying it unchanged to step 500;
- streams matched random controls without saving random tensors;
- writes exact direction artifacts with tensor and provenance digests;
- writes `scale_grid.csv`, `responses.csv`, `functional_lock.json`, and a final digest-bearing `complete.json` atomically;
- returns a completed artifact unchanged on a second invocation;
- rejects a changed direction, checkpoint, response table, or preregistration instead of silently resuming it;
- never writes Stage-1 configs or directories.

Use a tiny explicit calibration preregistration with two classes, one or two toy layers, small partitions, a short scale grid, and at least three random controls. Do not weaken production validation through hidden environment flags.

### Step 2: Run and observe missing service failure

### Step 3: Implement the service

Implement `calibrate_observation0_functional_overlap(...)`. Reuse the locked run configs, `restore_probe_model`, `collect_gradient_rows`, Probe-A manifest, target/source/path/objective factories, and registry. Keep one model/checkpoint active at a time. Cache materialized cell batches within that checkpoint only. Save exact primary directions before finite evaluations; generate random controls one at a time.

`functional_lock.json` must contain every upstream digest and all selected epsilons. `complete.json` hashes every aggregate file and direction index. Only a complete, digest-valid result is resumable.

### Step 4: Run focused service tests

Expected: PASS.

### Step 5: Commit

```bash
git add fm_lab/diagnostics/long_tail_geometry/functional_calibration.py \
  tests/test_long_tail_geometry_functional_service.py
git commit -m "Add resumable Probe-A functional calibration"
```

## Task 6: Expose the command and document the handoff

**Files:**

- Modify: `fm_lab/experiments/run_long_tail_geometry_observation0.py`
- Modify: `tests/test_long_tail_geometry_observation0.py`
- Modify: `docs/diagnostics.md`

### Step 1: Write failing CLI tests

Require:

```text
calibrate --study-dir STUDY --calibration-preregistration FILE --device auto
```

The command must print the lock status, selected layerwise epsilons, positive-control status, and only allowed next action. Existing `prepare`, `collect`, and `analyze` behavior must remain unchanged. `stage1` must still be an invalid command.

### Step 2: Run and observe parser failure

### Step 3: Implement CLI wiring and documentation

Document the exact real-study command and artifacts. State explicitly that Probe-B remains unused and that Stage 1 is unlocked only by `functional_lock.json`.

### Step 4: Run focused long-tail geometry tests

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_functional_preregistration.py \
  tests/test_long_tail_geometry_functional_calibration.py \
  tests/test_long_tail_geometry_functional_service.py \
  tests/test_long_tail_probe_checkpoints.py \
  tests/test_long_tail_geometry_observation0.py
```

Expected: PASS.

### Step 5: Commit

```bash
git add fm_lab/experiments/run_long_tail_geometry_observation0.py \
  tests/test_long_tail_geometry_observation0.py docs/diagnostics.md
git commit -m "Expose functional calibration command"
```

## Task 7: Verify the full branch

### Step 1: Static repository checks

```bash
git diff --check main...HEAD
PYTHONPATH=. ../../.conda/fm_lab/bin/python -m compileall -q fm_lab
```

### Step 2: Full suite

Ensure the ignored Three.js vendor asset is available to the worktree test runtime, then run:

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q
```

Expected: all tests pass. Do not attribute the known offline `cdn.jsdelivr.net` test failure to the branch if the vendor fixture is absent; supply the same ignored local vendor asset used by the main workspace and rerun.

### Step 3: CLI smoke and artifact dry run

Run `--help`, parse/load the production calibration preregistration, and execute the tiny service end to end. Confirm no path named `stage1` exists.

### Step 4: Inspect the final diff and scientific scope

Confirm:

- only Probe-A is opened;
- the selected scope is exactly 500/20,000, stratum 0, rank 1, two layers, six classes;
- fit/scale/evaluation rows are disjoint;
- step 500 reuses final-checkpoint epsilon;
- random controls are matched and deterministic;
- model parameters restore exactly;
- every artifact is digest-bound;
- no training or Stage-1 launch path was added.

### Step 5: Request code review, address findings, and finish the branch

Use the requesting-code-review and verification-before-completion workflows. Re-run affected focused tests after any review change, then re-run the full suite before merging.
