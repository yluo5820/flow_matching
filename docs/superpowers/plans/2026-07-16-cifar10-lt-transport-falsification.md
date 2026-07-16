# CIFAR-10-LT Transport Falsification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a digest-locked CIFAR-10-LT IR100 experiment that decides whether Fashion-MNIST's low-dimensional gradient geometry and cross-fitted sign transport survive on natural images.

**Architecture:** Extend the existing CIFAR target with deterministic paired diagnostic pools, reuse the dataset-independent Observation-0 sketch/reliability pipeline, and add a separate natural-image transport service that consumes the completed CIFAR reliability artifacts. The transport service reuses the existing exact-gradient cross-fit collector but owns its preregistration, baseline-learning guard, geometry prerequisite, interference summaries, terminal decisions, resumable artifacts, and CLI surface.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, SciPy-free Spearman correlations through pandas, YAML, pytest, Ruff, existing `fm_lab` checkpoint and long-tail geometry infrastructure.

## Global Constraints

- Use ordinary class-conditional linear-path flow matching only; no CM, capacity adapters, objective modifiers, EMA, or early stopping.
- Use CIFAR-10-LT with imbalance factor `0.01`, mapping multiplier `3`, mapping offset `0`, diagnostic pool `256` per class, and training seeds `0`, `1`, and `2`.
- Train 100,000 steps and lock checkpoints `0`, `2500`, `10000`, `25000`, `50000`, and `100000`.
- Preserve the existing Fashion-MNIST preregistration digest and completed artifact semantics.
- Functional transport uses Probe A only with four balanced `8/4/4` cross-fit folds; Probe B remains confined to the geometry reliability prerequisite.
- Test all ten CIFAR classes in the two locked layers `down2_block.conv2.weight` and `middle.conv2.weight`.
- The primary endpoint is the zero-step relative-benefit slope at step `100000`; finite steps are descriptive and never target a benefit magnitude.
- No terminal status opens Stage 1, CM, capacity allocation, or a parameter intervention.
- All new behavioral production code follows a witnessed red-green TDD cycle.
- Preserve the user's unrelated edit in `fm_lab/diagnostics/long_tail_geometry/checkpoints.py`.

---

### Task 1: Deterministic CIFAR diagnostic pools

**Files:**
- Modify: `tests/test_cifar_lt_data.py`
- Modify: `fm_lab/data/cifar_lt.py`
- Modify: `fm_lab/experiments/factory.py`

**Interfaces:**
- Extends: `ImbalancedCIFARImages(..., dequantize: bool = False, frequency_mapping_offset: int | None = None, frequency_mapping_multiplier: int = 3, diagnostic_pool_per_class: int = 0)`.
- Produces: `diagnostic_indices(split: str) -> np.ndarray`.
- Produces: `diagnostic_samples(split: str, *, original_indices: np.ndarray | None = None, dequantization_seeds: np.ndarray | None = None, device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]`.
- Preserves: existing CIFAR-LT count and sampling behavior when no frequency mapping is configured.

- [ ] **Step 1: Add failing pool, restoration, and factory tests**

Add tests using `_write_balanced_binary_cifar` that construct 10 classes with 40 rows each, reserve a diagnostic pool of 8, and assert:

```python
target = ImbalancedCIFARImages(
    dataset="cifar10",
    root=tmp_path,
    imbalance_factor=0.1,
    subset_seed=7,
    horizontal_flip=True,
    dequantize=True,
    frequency_mapping_offset=0,
    frequency_mapping_multiplier=3,
    diagnostic_pool_per_class=8,
)
probe_a = target.diagnostic_indices("a")
probe_b = target.diagnostic_indices("b")
assert len(probe_a) == len(probe_b) == 40
assert set(probe_a).isdisjoint(probe_b)
assert set(probe_a).isdisjoint(target.selected_indices)
assert set(probe_b).isdisjoint(target.selected_indices)

requested = probe_a[[3, 0, 7]]
seeds = np.asarray([11, 12, 13], dtype=np.int64)
first, labels, returned = target.diagnostic_samples(
    "a", original_indices=requested, dequantization_seeds=seeds
)
second, _, _ = target.diagnostic_samples(
    "a", original_indices=requested, dequantization_seeds=seeds
)
assert np.array_equal(returned.astype(np.int64), requested)
assert torch.equal(first, second)
assert first.shape == (3, 3072)
```

Also assert that changing `horizontal_flip` does not change diagnostic rows, that a missing diagnostic ID is rejected, and that `build_target` passes all new YAML fields.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_cifar_lt_data.py
```

Expected: failures because the constructor and diagnostic methods do not exist.

- [ ] **Step 3: Implement the minimal CIFAR target behavior**

Use `nested_frequency_split` exactly as Fashion-MNIST does. Store raw Probe-A and Probe-B tensors, labels, and original indices. Apply stochastic uniform dequantization during training, but deterministic per-row uniform dequantization in `diagnostic_samples`; never call horizontal flipping in the diagnostic path. Record `frequency_mapping`, `dequantize`, and SHA-256 digests of both probe index vectors in metadata.

- [ ] **Step 4: Run focused verification**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_cifar_lt_data.py tests/test_cifar_background.py
PYTHONPATH=. .conda/fm_lab/bin/ruff check fm_lab/data/cifar_lt.py fm_lab/experiments/factory.py tests/test_cifar_lt_data.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the dataset unit**

```bash
git add fm_lab/data/cifar_lt.py fm_lab/experiments/factory.py tests/test_cifar_lt_data.py
git commit -m "Add deterministic CIFAR geometry probes"
```

---

### Task 2: Dataset-generic Observation-0 and canonical CIFAR study

**Files:**
- Modify: `tests/test_long_tail_geometry_preregistration.py`
- Modify: `tests/test_long_tail_geometry_observation0.py`
- Modify: `fm_lab/diagnostics/long_tail_geometry/preregistration.py`
- Create: `configs/cifar10_lt/cifar10_lt_geometry_falsification.yaml`
- Create: `configs/cifar10_lt/long_tail_geometry_observation0_preregistration.yaml`

**Interfaces:**
- Extends: `Observation0Preregistration` to accept exactly `fashion_mnist_lt` and `cifar10_lt`.
- Produces: three prepared CIFAR seed configs through the unchanged `prepare_observation0_study` API.
- Preserves: the canonical Fashion-MNIST object, serialization, digest, and validation.

- [ ] **Step 1: Add failing generic-validation and CIFAR config tests**

Load both canonical preregistrations and assert the CIFAR contract:

```python
assert cifar.dataset == "cifar10_lt"
assert cifar.training_seeds == (0, 1, 2)
assert cifar.checkpoint_steps == (0, 2500, 10000, 25000, 50000, 100000)
assert cifar.layers == fashion.layers
assert cifar.primary_microbatches_per_cell == 16
assert cifar.minimum_common_classes == 5
```

Prepare the CIFAR study in a temporary directory and assert each seed config has `source.dim == 3072`, `model.image_shape == [3, 32, 32]`, 100,000 steps, a 256-row diagnostic pool, no capacity, no EMA, no early stopping, and no objective modifiers. Snapshot the Fashion preregistration digest before and after the code change.

- [ ] **Step 2: Run the tests and verify RED**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_preregistration.py \
  tests/test_long_tail_geometry_observation0.py
```

Expected: failure because CIFAR is rejected as a Fashion-MNIST-only study and the canonical files are absent.

- [ ] **Step 3: Generalize only the dataset guard and add the locked YAMLs**

Change the preregistration guard to accept a frozen supported set `{"fashion_mnist_lt", "cifar10_lt"}` while retaining all other protocol checks. The CIFAR base config uses independent coupling, a linear path, `ImageUNetVelocity` base width 32, batch size 64, learning rate `2e-4`, logit-normal time sampling, random training flips, and deterministic dequantization through the diagnostic interface.

- [ ] **Step 4: Verify both protocols**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_preregistration.py \
  tests/test_long_tail_geometry_observation0.py \
  tests/test_long_tail_geometry_stage0.py
PYTHONPATH=. .conda/fm_lab/bin/ruff check fm_lab/diagnostics/long_tail_geometry/preregistration.py tests/test_long_tail_geometry_preregistration.py tests/test_long_tail_geometry_observation0.py
```

Expected: both Fashion-MNIST and CIFAR tests pass.

- [ ] **Step 5: Commit the CIFAR Observation-0 unit**

```bash
git add fm_lab/diagnostics/long_tail_geometry/preregistration.py tests/test_long_tail_geometry_preregistration.py tests/test_long_tail_geometry_observation0.py configs/cifar10_lt
git commit -m "Lock CIFAR-10-LT geometry observation"
```

---

### Task 3: Immutable natural-image transport preregistration

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/natural_image_preregistration.py`
- Create: `tests/test_long_tail_geometry_natural_image_preregistration.py`
- Create after computing the Observation-0 digest: `configs/cifar10_lt/long_tail_geometry_natural_image_transport.yaml`

**Interfaces:**
- Produces: `NaturalImageTransportPreregistration.load(path)`, `.from_dict(values)`, `.to_dict()`, `.digest`, `.fold_positions`, and `.lock(path)`.
- Locks: Observation-0 digest, three checkpoints, all ten classes, two layers, stratum 0, rank 1, four balanced folds, paired bases, finite grid, bootstrap settings, learning ratio, and common-class threshold.

- [ ] **Step 1: Write failing schema, fold-balance, and lock tests**

Assert the canonical values:

```python
assert prereg.checkpoint_steps == (0, 10_000, 100_000)
assert prereg.baseline_checkpoint_step == 0
assert prereg.early_checkpoint_step == 10_000
assert prereg.primary_checkpoint_step == 100_000
assert prereg.layers == ("down2_block.conv2.weight", "middle.conv2.weight")
assert prereg.classes == tuple(range(10))
assert prereg.fold_offsets == (0, 4, 8, 12)
assert prereg.basis_kinds == ("raw", "row_normalized")
assert prereg.relative_step_grid == (3e-5, 1e-4, 3e-4, 1e-3)
assert prereg.maximum_final_to_baseline_loss_ratio == 0.70
assert prereg.minimum_reliable_common_classes == 5
```

For every position 0 through 15, assert two fit appearances, one scale appearance, and one evaluation appearance. Test missing/unknown keys, changed Observation-0 digest, changed checkpoints/layers/classes, Probe B, unbalanced folds, changed finite grid, and immutable-lock rejection.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_natural_image_preregistration.py
```

Expected: import failure for the new preregistration class.

- [ ] **Step 3: Implement the frozen exact-key schema**

Use the same canonical JSON SHA-256 and immutable YAML lock pattern as the existing functional audit preregistration. Compute the canonical CIFAR Observation-0 digest with:

```bash
PYTHONPATH=. .conda/fm_lab/bin/python -c 'from fm_lab.diagnostics.long_tail_geometry.preregistration import Observation0Preregistration as P; print(P.load("configs/cifar10_lt/long_tail_geometry_observation0_preregistration.yaml").digest)'
```

Insert that exact output into the canonical transport YAML and the test's identity assertion.

- [ ] **Step 4: Verify and commit**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_natural_image_preregistration.py
PYTHONPATH=. .conda/fm_lab/bin/ruff check fm_lab/diagnostics/long_tail_geometry/natural_image_preregistration.py tests/test_long_tail_geometry_natural_image_preregistration.py
git add fm_lab/diagnostics/long_tail_geometry/natural_image_preregistration.py tests/test_long_tail_geometry_natural_image_preregistration.py configs/cifar10_lt/long_tail_geometry_natural_image_transport.yaml
git commit -m "Lock natural-image transport contract"
```

Expected: tests and Ruff pass; the commit contains the schema and its canonical YAML.

---

### Task 4: Pure natural-image analysis and terminal decisions

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/natural_image.py`
- Create: `tests/test_long_tail_geometry_natural_image.py`

**Interfaces:**
- Produces: `NaturalImageTransportDecision` and `NaturalImageTransportAnalysis` frozen dataclasses.
- Produces: `analyze_natural_image_transport(slopes: pd.DataFrame, finite_steps: pd.DataFrame, basis_comparison: pd.DataFrame, reliability: pd.DataFrame, *, class_counts: tuple[int, ...], class_ranks: tuple[int, ...], preregistration: NaturalImageTransportPreregistration) -> NaturalImageTransportAnalysis`.
- Produces: complete `class_transport` table and named 10-by-10 NumPy interference matrices.

- [ ] **Step 1: Add failing synthetic-analysis tests**

Build complete synthetic tables across three checkpoints, two layers, three seeds, four folds, two bases, and ten classes. Add separate tests for:

- `baseline_not_learned` when final/base median loss ratio is `0.8`;
- `no_reliable_cifar_geometry` when fewer than five common classes repeat measurably;
- `natural_image_transport_confirmed` when both row-normalized lower bounds and selectivity are positive;
- `geometry_without_transport` when both row-normalized upper bounds are non-positive;
- `heterogeneous_natural_image_transport` when only one layer passes;
- exact fold collapse before bootstrap;
- complete class transport and every 10-by-10 matrix;
- locked frequency counts/ranks and finite Spearman outputs;
- finite-step local-linearity summaries without any benefit threshold.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_natural_image.py
```

Expected: import failure for `analyze_natural_image_transport`.

- [ ] **Step 3: Implement analysis using existing cross-fit table semantics**

Reuse the existing fold-block, bootstrap, paired-difference, and finite-error helpers from `functional_audit.py`; do not duplicate the mathematical definitions. Geometry passes only when at least five classes are measurable in both locked layers at the primary checkpoint, stratum, representation, and rank in at least two seeds. Evaluate terminal conditions in the exact order: baseline learning, geometry, confirmed transport, geometry without transport, heterogeneous.

Build matrix keys as `checkpoint_<step>__<sanitized_layer>__<basis_kind>` and ensure every matrix is ordered by `preregistration.classes` on both axes.

- [ ] **Step 4: Run focused verification**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_natural_image.py \
  tests/test_long_tail_geometry_functional_audit.py
PYTHONPATH=. .conda/fm_lab/bin/ruff check fm_lab/diagnostics/long_tail_geometry/natural_image.py tests/test_long_tail_geometry_natural_image.py
```

Expected: new and existing audit mathematics tests pass.

- [ ] **Step 5: Commit the analysis unit**

```bash
git add fm_lab/diagnostics/long_tail_geometry/natural_image.py tests/test_long_tail_geometry_natural_image.py
git commit -m "Analyze natural-image gradient transport"
```

---

### Task 5: Fail-closed, resumable transport service

**Files:**
- Modify: `fm_lab/diagnostics/long_tail_geometry/natural_image.py`
- Create: `tests/test_long_tail_geometry_natural_image_service.py`

**Interfaces:**
- Produces: `prepare_natural_image_transport_context(*, study_dir, preregistration_path) -> NaturalImageTransportContext`.
- Produces: `collect_natural_image_transport_chunk(*, context, seed, checkpoint_step, device) -> FunctionalGeometryAuditChunk`.
- Produces: `run_natural_image_transport_falsification(*, study_dir, preregistration_path, device) -> NaturalImageTransportResult`.
- Writes: immutable aggregate artifacts below `aggregate/natural_image_transport_falsification`.

- [ ] **Step 1: Add failing upstream-validation and service tests**

Construct a tiny completed Observation-0 fixture and monkeypatch only the expensive chunk collector. Assert rejection of the wrong dataset, wrong Observation-0 digest, incomplete registry, changed run config, missing or changed checkpoint, incomplete measurement identity, changed reliability digest, wrong Probe-A cell size, and any transport layer absent from the measurements.

For a valid fixture, assert nine chunks in seed-major/checkpoint-minor order, aggregate sorting, preregistration locking, all seven aggregate artifacts, and a `complete.json` that binds every file and upstream digest. Repeat the command and assert no collector call. Corrupt one chunk and one aggregate in separate tests and assert fail-closed rejection.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_natural_image_service.py
```

Expected: missing service functions.

- [ ] **Step 3: Implement context, collection, atomic writes, and reload validation**

Validate the completed primary Observation-0 reliability outputs and digests, but do not require the global Observation-0 status to have passed; the new locked-cell geometry prerequisite owns that decision. Materialize only Probe-A stratum-0 batches. Load checkpoints through the same compatibility path as functional calibration. Call the existing `collect_audit_metrics` for exact gradients and finite responses. Write per-chunk `slopes.csv`, `finite_steps.csv`, `basis_comparison.csv`, and `complete.json`, then aggregate through the pure analyzer.

- [ ] **Step 4: Run focused service verification**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_natural_image_service.py \
  tests/test_long_tail_geometry_functional_audit_service.py \
  tests/test_long_tail_geometry_functional_service.py
PYTHONPATH=. .conda/fm_lab/bin/ruff check fm_lab/diagnostics/long_tail_geometry/natural_image.py tests/test_long_tail_geometry_natural_image_service.py
```

Expected: all service and regression tests pass.

- [ ] **Step 5: Commit the service unit**

```bash
git add fm_lab/diagnostics/long_tail_geometry/natural_image.py tests/test_long_tail_geometry_natural_image_service.py
git commit -m "Add resumable CIFAR transport falsification"
```

---

### Task 6: CLI, documentation, and final verification

**Files:**
- Modify: `fm_lab/experiments/run_long_tail_geometry_observation0.py`
- Modify: `tests/test_long_tail_geometry_observation0.py`
- Modify: `docs/diagnostics.md`
- Modify: `fm_lab/diagnostics/long_tail_geometry/__init__.py`

**Interfaces:**
- Adds: `falsify-natural-image-transport --study-dir PATH --transport-preregistration PATH --device auto`.
- Prints: terminal status, baseline loss ratio, common reliable classes, per-layer normalized slope/selectivity, and the only allowed next action.

- [ ] **Step 1: Add failing CLI parsing and output tests**

Monkeypatch the service result and assert exact lines beginning with:

```text
Natural-image falsification: <status>
Baseline learned: yes|no
Reliable common classes: <count>
<layer>: normalized_slope=<value>, selectivity=<value>
Only allowed next action: <action>
```

Also assert the command parser requires both paths and accepts `--device`.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=. .conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_observation0.py
```

Expected: parser rejection because the subcommand is absent.

- [ ] **Step 3: Implement CLI and document the exact workflow**

Document these commands with the canonical paths and study directory:

```bash
PYTHONPATH=. .conda/fm_lab/bin/python -m fm_lab.experiments.run_long_tail_geometry_observation0 prepare \
  --preregistration configs/cifar10_lt/long_tail_geometry_observation0_preregistration.yaml \
  --study-dir runs/long_tail_geometry/cifar10_lt/natural_image_falsification

PYTHONPATH=. .conda/fm_lab/bin/fm-lab-train --config runs/long_tail_geometry/cifar10_lt/natural_image_falsification/configs/seed_0.yaml --device auto
PYTHONPATH=. .conda/fm_lab/bin/fm-lab-train --config runs/long_tail_geometry/cifar10_lt/natural_image_falsification/configs/seed_1.yaml --device auto
PYTHONPATH=. .conda/fm_lab/bin/fm-lab-train --config runs/long_tail_geometry/cifar10_lt/natural_image_falsification/configs/seed_2.yaml --device auto
```

Then document one `collect` command per seed, `analyze`, and the final falsification command. State explicitly that calibration and functional-audit commands are not part of the CIFAR workflow.

- [ ] **Step 4: Run complete verification**

```bash
PYTHONPATH=.:tests .conda/fm_lab/bin/pytest -q
PYTHONPATH=. .conda/fm_lab/bin/ruff check .
PYTHONPATH=. .conda/fm_lab/bin/python -m fm_lab.experiments.run_long_tail_geometry_observation0 falsify-natural-image-transport --help
git diff --check
```

Expected: zero pytest failures, Ruff clean, CLI help exits zero, and no whitespace errors.

- [ ] **Step 5: Review the implementation against the design**

Confirm every design status has a test, the canonical study has no CM/capacity/modifier/EMA path, all ten classes enter the complete response matrix, Probe B is absent from transport materialization, and the Fashion-MNIST canonical digest is unchanged.

- [ ] **Step 6: Commit the integration**

```bash
git add fm_lab/experiments/run_long_tail_geometry_observation0.py fm_lab/diagnostics/long_tail_geometry/__init__.py tests/test_long_tail_geometry_observation0.py docs/diagnostics.md
git commit -m "Expose CIFAR transport falsification workflow"
```

Expected: the final commit contains only CLI/export/documentation integration.
