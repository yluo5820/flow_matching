# Long-Tail Geometry Observation 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the immutable preregistration, streaming checkpoint measurement, and cross-seed reliability analysis needed to run the three-seed Fashion-MNIST Observation-0 noise-ceiling pilot without inspecting or tuning on Probe-B.

**Architecture:** Add three focused modules beneath `fm_lab.diagnostics.long_tail_geometry`: one validates and locks the scientific protocol, one streams exact gradient norms and fixed sketches into resumable artifacts, and one computes Probe-A/Probe-B reliability and its max-statistic class-permutation null. A small experiment service prepares the three seed configs, collects one completed training run at a time, and analyzes the study only after the registry contains all preregistered seeds.

**Tech Stack:** Python 3.11, PyTorch 2.2+, NumPy, pandas, PyArrow/Parquet, pytest, Ruff, YAML, existing `fm_lab` checkpoints and Stage-0 probes.

## Global Constraints

- Observation 0 uses ordinary flow matching only: no CM, objective modifiers, capacity adapters, EMA substitution, or early stopping.
- The pilot uses frequency mapping offset 0 under paired training seeds 0, 1, and 2; it cannot estimate a frequency effect.
- The pilot checkpoints are steps 0, 500, 1,000, 3,000, 7,000, 13,000, and 20,000; step 0 is a structural control and is excluded from the primary gate.
- Probe-A and Probe-B use one manifest seed shared across all training seeds, mappings, and checkpoints.
- A second source-noise replica preserves target IDs, dequantization, timesteps, strata, and microbatch membership exactly; only its source seeds change.
- The primary allocation is 16 microbatches of 8 examples per class/timestep/split; the only permitted escalation is 32 microbatches.
- The centered-covariance reliability gate uses ranks 1, 2, 4, and 8. Ranks 16 and 32 remain preregistered descriptive ranks but are reported only when supported by sample rank.
- Probe-B evaluates only fixed candidates. It never selects a layer, checkpoint, timestep, representation, rank, threshold, or escalation.
- A cell is measurable only when its Probe-A/Probe-B overlap exceeds a 99th-percentile max-statistic null formed by permuting Probe-B class identities within checkpoint/timestep blocks, and this repeats in at least two of three seeds.
- The network-wide gate requires at least five common measurable classes in two adjacent non-output layers at the same checkpoint, timestep stratum, and rank.
- Output-layer-only reliability is reportable but cannot pass the network-wide gate.
- Stage 1 remains locked after this implementation. A separate Stage-1 lock must record the Probe-A functional calibration of the smallest meaningful overlap before any mapping other than offset 0 is trained.
- No full parameter covariance, dense projection matrix, or collection of all full gradient rows may be allocated.
- Failed, missing, and excluded runs remain visible in the registry or exclusion log.

---

### Task 1: Immutable Observation-0 preregistration and study registry

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/preregistration.py`
- Create: `fm_lab/diagnostics/long_tail_geometry/registry.py`
- Modify: `fm_lab/diagnostics/long_tail_geometry/__init__.py`
- Create: `configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml`
- Create: `tests/test_long_tail_geometry_preregistration.py`

**Interfaces:**
- Produces: `Observation0Preregistration.load(path)`, `.digest`, `.lock(path)`, and validated immutable fields.
- Produces: `Observation0Run(seed, mapping_offset, condition, run_dir, status, exclusion_reason)`.
- Produces: `prepare_observation0_registry(preregistration: Observation0Preregistration, study_dir: str | Path) -> pd.DataFrame`.
- Produces: `update_observation0_run(study_dir: str | Path, *, seed: int, status: str, run_dir: str | Path | None = None, measurement_digest: str = "", exclusion_reason: str = "") -> pd.DataFrame`.
- Consumes later: every collector and analyzer takes the preregistration object and checks its digest.

- [ ] **Step 1: Write failing preregistration-schema and lock tests**

```python
def test_canonical_observation0_preregistration_is_fully_locked() -> None:
    prereg = Observation0Preregistration.load(
        "configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml"
    )
    assert prereg.training_seeds == (0, 1, 2)
    assert prereg.checkpoint_steps == (0, 500, 1000, 3000, 7000, 13000, 20000)
    assert prereg.observation0_mapping_offsets == (0,)
    assert prereg.stage1_mapping_offsets == tuple(range(10))
    assert prereg.gate_ranks == (1, 2, 4, 8)
    assert prereg.descriptive_ranks == (1, 2, 4, 8, 16, 32)
    assert prereg.required_seed_repeats == 2
    assert prereg.minimum_common_classes == 5
    assert prereg.stage1_requires_functional_lock is True


def test_lock_refuses_to_replace_a_different_preregistration(tmp_path: Path) -> None:
    prereg = canonical_test_preregistration()
    path = prereg.lock(tmp_path / "preregistration.yaml")
    assert Observation0Preregistration.load(path).digest == prereg.digest
    changed = dataclasses.replace(prereg, null_permutations=prereg.null_permutations + 1)
    with pytest.raises(ValueError, match="locked preregistration"):
        changed.lock(path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_preregistration.py
```

Expected: import failure for `Observation0Preregistration`.

- [ ] **Step 3: Implement the immutable schema**

Use a frozen dataclass whose constructor copies all sequences into tuples. `load` accepts the exact nested YAML sections `study`, `training`, `frequency`, `probe`, `gradient`, `reliability`, and `stage1_lock`. Validation must enforce all Global Constraints, unique seeds/checkpoints/layers, five valid timestep intervals, offset 0 for the pilot, the complete Latin offsets for Stage 1, a 0.99 null quantile, and a primary row count equal to `microbatch_size * primary_microbatches_per_cell`.

The digest is SHA-256 of a canonical sorted JSON representation. `lock(path)` behaves as follows:

```python
def lock(self, path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        existing = Observation0Preregistration.load(output)
        if existing.digest != self.digest:
            raise ValueError("Refusing to replace a locked preregistration.")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    save_config(self.to_dict(), output)
    return output
```

The canonical YAML fixes:

```yaml
schema_version: 1
study:
  name: fashion_mnist_lt_ir100_observation0
  dataset: fashion_mnist_lt
  base_config: configs/fashion_mnist_lt/fashion_mnist_lt_geometry_stage0.yaml
training:
  seeds: [0, 1, 2]
  checkpoint_steps: [0, 500, 1000, 3000, 7000, 13000, 20000]
frequency:
  multiplier: 3
  observation0_mapping_offsets: [0]
  stage1_mapping_offsets: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  balanced_control: true
probe:
  manifest_seed: 20260716
  splits: [a, b]
  source_noise_replicas: 2
  microbatch_size: 8
  primary_microbatches_per_cell: 16
  escalation_microbatches_per_cell: 32
  time_strata:
    - [0.02, 0.10]
    - [0.10, 0.30]
    - [0.30, 0.70]
    - [0.70, 0.90]
    - [0.90, 0.98]
gradient:
  layers:
    - input_block.conv2.weight
    - down2_block.conv2.weight
    - middle.conv2.weight
    - up1_block.conv2.weight
    - up0_block.conv2.weight
    - output_block.2.weight
  sketch_dim: 4096
  max_sketch_dim: 16384
  sketch_seed: 20260716
  storage_dtype: float32
reliability:
  representation: centered_covariance
  gate_ranks: [1, 2, 4, 8]
  descriptive_ranks: [1, 2, 4, 8, 16, 32]
  null_generator: class_label_permutation
  null_permutations: 999
  null_quantile: 0.99
  required_seed_repeats: 2
  minimum_common_classes: 5
  exclude_checkpoint_zero_from_gate: true
stage1_lock:
  required: true
  functional_loss_change_fraction: 0.01
```

- [ ] **Step 4: Write failing registry tests**

```python
def test_prepare_registry_contains_all_three_pilot_runs_and_exclusion_header(tmp_path):
    prereg = canonical_test_preregistration()
    registry = prepare_observation0_registry(prereg, tmp_path)
    assert list(registry["seed"]) == [0, 1, 2]
    assert set(registry["mapping_offset"]) == {0}
    assert set(registry["status"]) == {"planned"}
    assert (tmp_path / "aggregate/exclusion_log.csv").exists()


def test_registry_rejects_unregistered_seed_and_invalid_status(tmp_path):
    prereg = canonical_test_preregistration()
    prepare_observation0_registry(prereg, tmp_path)
    with pytest.raises(ValueError, match="not preregistered"):
        update_observation0_run(tmp_path, seed=9, status="measured")
    with pytest.raises(ValueError, match="status"):
        update_observation0_run(tmp_path, seed=0, status="silently_dropped")
```

- [ ] **Step 5: Implement registry creation and atomic updates**

Use CSV columns:

```python
REGISTRY_COLUMNS = (
    "study_digest",
    "condition",
    "mapping_offset",
    "seed",
    "run_dir",
    "status",
    "measurement_digest",
    "exclusion_reason",
)
VALID_STATUSES = frozenset({"planned", "trained", "measured", "excluded"})
```

Write updates to `run_registry.csv.tmp`, then replace `run_registry.csv`. An excluded row requires a non-empty reason and is appended to `exclusion_log.csv`; no other status may carry an exclusion reason.

- [ ] **Step 6: Run focused tests and Ruff**

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_preregistration.py
PYTHONPATH=. ../../.conda/fm_lab/bin/ruff check \
  fm_lab/diagnostics/long_tail_geometry/preregistration.py \
  fm_lab/diagnostics/long_tail_geometry/registry.py \
  tests/test_long_tail_geometry_preregistration.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add fm_lab/diagnostics/long_tail_geometry configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml tests/test_long_tail_geometry_preregistration.py
git commit -m "Add Observation-0 preregistration contracts"
```

---

### Task 2: Exact step-zero checkpoint control

**Files:**
- Modify: `fm_lab/training/trainer.py`
- Modify: `tests/test_training_sampling.py`

**Interfaces:**
- Extends: `training.checkpoint_steps` to accept zero.
- Produces: `checkpoints/step_000000.pt` before any optimizer update or training-batch draw.
- Preserves: every existing positive checkpoint schedule and resume contract.

- [ ] **Step 1: Add failing step-zero checkpoint tests**

```python
def _train_checkpoint_fixture(
    *,
    config: dict,
    model: nn.Module,
    run_dir: Path,
) -> None:
    train_flow_matching(
        config=config,
        run_dir=run_dir,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )


def test_training_saves_exact_initial_state_when_checkpoint_zero_is_requested(tmp_path):
    model = TinyVelocity()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    config = checkpoint_schedule_config([0, 1, 3])
    _train_checkpoint_fixture(config=config, model=model, run_dir=tmp_path)
    zero = load_checkpoint(tmp_path / "checkpoints/step_000000.pt")
    assert zero["step"] == 0
    assert zero["history"] == []
    assert all(torch.equal(zero["model_state_dict"][name], value) for name, value in initial.items())
    assert sorted(path.name for path in (tmp_path / "checkpoints").glob("*.pt")) == [
        "step_000000.pt", "step_000001.pt", "step_000003.pt"
    ]


def test_training_does_not_resave_step_zero_when_resuming(tmp_path):
    first_run = tmp_path / "first"
    first_config = checkpoint_schedule_config([0, 1])
    _train_checkpoint_fixture(
        config=first_config,
        model=TinyVelocity(),
        run_dir=first_run,
    )
    zero_hash = file_sha256(first_run / "checkpoints/step_000000.pt")
    resumed = checkpoint_schedule_config([0, 2])
    resumed["training"]["resume_from"] = str(
        first_run / "checkpoints/step_000001.pt"
    )
    _train_checkpoint_fixture(
        config=resumed,
        model=TinyVelocity(),
        run_dir=tmp_path / "resumed",
    )
    assert not (tmp_path / "resumed/checkpoints/step_000000.pt").exists()
    assert file_sha256(first_run / "checkpoints/step_000000.pt") == zero_hash
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_training_sampling.py -k 'checkpoint_zero or initial_state'
```

Expected: configuration rejection because zero is outside the current allowed range.

- [ ] **Step 3: Allow zero and save before the training loop**

Change `_explicit_checkpoint_steps` to allow `0 <= value <= total_steps`. After the model, optimizer, scheduler, objective, and checkpoint contracts exist—but before the first target/source sample—write step zero only when `0 in checkpoint_steps and resume_checkpoint is None`:

```python
if 0 in checkpoint_steps and resume_checkpoint is None:
    save_checkpoint(
        run_dir / "checkpoints" / "step_000000.pt",
        model=model,
        ema_model=ema_model,
        optimizer=theta_optimizer,
        scheduler=theta_scheduler,
        step=0,
        config=checkpoint_config,
        prediction_contract=prediction_contract,
        training_contract=training_contract,
        resume_state=_checkpoint_resume_state(early_stopping, best_state),
        metrics={"latest_loss": float("nan"), "initial_control": True},
        history=[],
        rng_state=capture_rng_state(),
    )
```

Do not include zero in the in-loop checkpoint condition.

- [ ] **Step 4: Run checkpoint and resume tests**

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_training_sampling.py \
  tests/test_checkpoint_resume.py \
  tests/test_long_tail_probe_checkpoints.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add fm_lab/training/trainer.py tests/test_training_sampling.py
git commit -m "Save exact step-zero training controls"
```

---

### Task 3: Streaming checkpoint-gradient measurement artifacts

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/measurements.py`
- Modify: `fm_lab/diagnostics/long_tail_geometry/manifest.py`
- Modify: `fm_lab/diagnostics/long_tail_geometry/__init__.py`
- Create: `tests/test_long_tail_geometry_measurements.py`
- Modify: `tests/test_long_tail_probe_manifest.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `CheckpointMeasurements(metadata, sketches, exact_norms, layer_shapes, manifest_digests, checkpoint_sha256, preregistration_sha256)`.
- Produces: `build_source_noise_replica(manifest: ProbeManifest, *, seed: int) -> ProbeManifest`.
- Produces: `collect_checkpoint_measurements(*, model, objective, path, batches_by_view, layer_names, sketch_dim, sketch_seed, checkpoint_step, checkpoint_sha256, preregistration_sha256, manifest_digests) -> CheckpointMeasurements`.
- Produces: `.save(directory)` and `CheckpointMeasurements.load(directory)`.
- Produces files: `gradient_rows.parquet`, `gradient_sketches.npz`, and `complete.json`.

- [ ] **Step 1: Add PyArrow to a dedicated optional extra**

```toml
[project.optional-dependencies]
long-tail-geometry = [
  "pyarrow>=16.0",
]
```

The core import of `fm_lab` must not import PyArrow. Only artifact save/load paths import pandas' Parquet backend.

- [ ] **Step 2: Write failing streaming-collection tests**

First add a manifest-replica test:

```python
def test_source_noise_replica_changes_only_source_seeds():
    manifest = balanced_probe_manifest()
    replica = build_source_noise_replica(manifest, seed=31)
    assert replica.digest != manifest.digest
    assert not np.array_equal(replica.source_seeds, manifest.source_seeds)
    for field in (
        "original_indices",
        "labels",
        "dequantization_seeds",
        "timesteps",
        "stratum_ids",
        "microbatch_ids",
    ):
        assert np.array_equal(getattr(replica, field), getattr(manifest, field))
```

The implementation uses a local `np.random.RandomState(seed)` and constructs a new immutable `ProbeManifest`; it never mutates either manifest or global NumPy state.

Use the existing tiny conditional model and three deterministic `ProbeBatch` objects. The primary test must assert:

```python
measurements = collect_checkpoint_measurements(
    model=model,
    objective=ordinary_objective,
    path=LinearPath(),
    batches_by_view={
        "a": probe_batches_a,
        "a_source_1": probe_batches_a_replica,
        "b": probe_batches_b,
        "b_source_1": probe_batches_b_replica,
    },
    layer_names=("hidden.weight", "output.weight"),
    sketch_dim=8,
    sketch_seed=23,
    checkpoint_step=3,
    checkpoint_sha256="a" * 64,
    preregistration_sha256="b" * 64,
    manifest_digests={"a": "c" * 64, "b": "d" * 64},
)
assert len(measurements.metadata) == 12
assert set(measurements.metadata["probe_view"]) == {
    "a", "a_source_1", "b", "b_source_1"
}
assert measurements.sketches["hidden.weight"].shape == (12, 8)
assert measurements.exact_norms["hidden.weight"].shape == (12,)
assert all(parameter.grad is None for parameter in model.parameters())
```

Patch `torch.zeros` to reject `(P, P)` and `(P, sketch_dim)` allocations, and assert model weights and mode are restored.

- [ ] **Step 3: Verify RED**

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_measurements.py -k collect
```

Expected: import failure for `collect_checkpoint_measurements`.

- [ ] **Step 4: Implement one-pass collection**

For each batch, compute the ordinary scalar objective once, call `torch.autograd.grad` for every requested layer in the same backward pass, record the exact norm, and immediately apply a fixed `CountSketchSpec`. If a layer has no more parameters than `sketch_dim`, retain its exact flattened row without allocating a projection. Normalize stored sketches row-wise; never retain full rows after the current batch.

Metadata has one row per microbatch with exact columns:

```python
(
    "checkpoint_step",
    "probe_view",
    "target_split",
    "class_id",
    "stratum_id",
    "microbatch_id",
    "batch_size",
    "loss",
    "original_indices_sha256",
)
```

Reject a batch containing multiple classes, strata, or microbatch IDs. Reject non-finite/zero gradients and any objective modifier or capacity-enabled model.

- [ ] **Step 5: Write failing round-trip and resume-fence tests**

```python
def test_checkpoint_measurements_round_trip_parquet_and_npz(tmp_path):
    original = tiny_measurements()
    original.save(tmp_path)
    restored = CheckpointMeasurements.load(tmp_path)
    pd.testing.assert_frame_equal(restored.metadata, original.metadata)
    assert torch.equal(restored.sketches["hidden.weight"], original.sketches["hidden.weight"])
    assert json.loads((tmp_path / "complete.json").read_text())["passed"] is True


def test_measurement_save_is_idempotent_but_refuses_different_provenance(tmp_path):
    original = tiny_measurements()
    original.save(tmp_path)
    original.save(tmp_path)
    changed = dataclasses.replace(original, checkpoint_sha256="e" * 64)
    with pytest.raises(ValueError, match="completed measurement"):
        changed.save(tmp_path)
```

- [ ] **Step 6: Implement deterministic artifact save/load**

Write scalar metadata and exact norms to `gradient_rows.parquet`, sketches and layer-shape arrays to `gradient_sketches.npz`, then atomically write `complete.json` last. `complete.json` includes schema version, row count, split counts, layer names/shapes/sketch dimensions, checkpoint/preregistration/manifest hashes, SHA-256 for both data files, and `passed: true`. A matching complete artifact is a no-op; any provenance mismatch fails closed.

- [ ] **Step 7: Run focused tests and Ruff**

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_measurements.py \
  tests/test_long_tail_gradient_probe.py \
  tests/test_long_tail_gradient_sketch.py
PYTHONPATH=. ../../.conda/fm_lab/bin/ruff check \
  fm_lab/diagnostics/long_tail_geometry/measurements.py \
  tests/test_long_tail_geometry_measurements.py
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml fm_lab/diagnostics/long_tail_geometry tests/test_long_tail_geometry_measurements.py
git commit -m "Add streaming long-tail gradient measurements"
```

---

### Task 4: Observation-0 reliability, max-statistic null, and gate

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/reliability.py`
- Modify: `fm_lab/diagnostics/long_tail_geometry/__init__.py`
- Create: `tests/test_long_tail_geometry_reliability.py`

**Interfaces:**
- Produces: `analyze_seed_reliability(measurements, preregistration) -> pd.DataFrame`.
- Produces: `aggregate_observation0_reliability(seed_tables, preregistration) -> Observation0Decision`.
- Produces: `Observation0Decision.save(directory)` writing `reliability.csv`, `noise_ceiling.json`, and `gram_matrices.npz`.

- [ ] **Step 1: Write failing basis/statistic tests**

```python
def test_centered_cell_statistics_report_overlap_angles_rank_and_concentration():
    a, b = planted_cell_rows(rows=16, dim=64, rank=4, noise=0.01, seed=7)
    result = centered_cell_statistics(a, b, ranks=(1, 2, 4, 8, 16))
    rank4 = result[result["rank"] == 4].iloc[0]
    assert rank4["available"]
    assert rank4["projection_overlap"] > 0.95
    assert rank4["largest_principal_angle_degrees"] < 15
    assert not result[result["rank"] == 16].iloc[0]["available"]
    assert 0 <= rank4["directional_concentration_a"] <= 1
    assert rank4["effective_rank_a"] >= 1
```

The rank-16 centered basis is unavailable because 16 centered rows have rank at most 15.

- [ ] **Step 2: Verify RED, then implement cell statistics using sample matrices**

Center rows only for covariance bases. Use thin SVD of the `B x D` sketch matrix and never form `D x D`. Principal angles are singular values of `U_A.T @ U_B`; projection overlap is their squared sum divided by rank. Effective rank is `exp(-sum(p log p))` over nonzero centered singular-value energy. Mean-direction cosine and concentration use the uncentered normalized rows.

- [ ] **Step 3: Write failing max-statistic class-permutation tests**

```python
def test_class_permutation_null_controls_the_maximum_across_cells():
    cells = synthetic_seed_measurements(
        shared_by_class=True,
        checkpoints=(500, 1000),
        strata=(0, 1),
        classes=range(10),
        seed=3,
    )
    table = analyze_seed_reliability(cells, test_prereg(null_permutations=199))
    assert table["null_threshold"].nunique() == 1
    assert table["projection_overlap"].max() > table["null_threshold"].iloc[0]


def test_random_probe_b_subspaces_do_not_pass_the_reliability_null():
    cells = synthetic_seed_measurements(shared_by_class=False, seed=4)
    table = analyze_seed_reliability(cells, test_prereg(null_permutations=199))
    assert not table["measurable"].any()
```

- [ ] **Step 4: Implement the matched null without repeated SVDs**

For each checkpoint/stratum/layer/rank block, compute the `10 x 10` matrix of all Probe-A-class versus Probe-B-class projection overlaps once. Each null permutation selects one off-label assignment per block. The null statistic is the maximum selected overlap across all checkpoint/stratum/class blocks for the same layer and rank. Its preregistered 0.99 quantile is attached to every corresponding observed diagonal overlap. Use a keyed NumPy RNG based only on the preregistration seed, layer, rank, and training seed.

- [ ] **Step 5: Write failing three-seed and adjacent-layer gate tests**

```python
def test_observation0_passes_only_for_repeated_adjacent_nonoutput_layers():
    seed_tables = planted_seed_tables(
        measurable_layers=("down2_block.conv2.weight", "middle.conv2.weight"),
        measurable_classes=range(6),
        passing_seeds=(0, 1),
    )
    decision = aggregate_observation0_reliability(seed_tables, canonical_test_preregistration())
    assert decision.status == "network_wide_measurable"
    assert decision.network_wide_gate_passed
    assert decision.required_escalation is False


def test_output_only_reliability_is_reported_but_does_not_pass():
    seed_tables = planted_seed_tables(
        measurable_layers=("output_block.2.weight",),
        measurable_classes=range(10),
        passing_seeds=(0, 1, 2),
    )
    decision = aggregate_observation0_reliability(seed_tables, canonical_test_preregistration())
    assert decision.status == "output_layer_only"
    assert not decision.network_wide_gate_passed
    assert decision.required_escalation is True


def test_incomplete_seed_set_is_not_interpreted_as_a_null():
    with pytest.raises(ValueError, match="all preregistered training seeds"):
        aggregate_observation0_reliability({0: seed_table()}, canonical_test_preregistration())
```

- [ ] **Step 6: Implement the aggregate decision**

For each fixed checkpoint/stratum/rank, count seeds in which each class/layer cell is measurable. A class repeats when count is at least two. For each adjacent pair among the first five layers, intersect repeated class IDs. Pass only if an intersection contains at least five classes. Exclude checkpoint zero. If no pair passes but output has five repeated classes, status is `output_layer_only`; otherwise status is `escalate_probe_rows`. The analyzer never returns Outcome D directly: only the second analysis at 32 microbatches may return `network_wide_practical_null`.

- [ ] **Step 7: Implement deterministic artifacts and run tests**

`noise_ceiling.json` includes preregistration digest, seed measurement digests, complete/incomplete seed list, pass status, passing cells/pairs, output-only cells, and the only allowed next action. `reliability.csv` contains every available and unavailable rank row for both centered covariance and uncentered second moment, plus Probe-A split-half overlap, Probe-A/Probe-B overlap, source-noise-replica overlap, mean-direction cosine, directional concentration, and effective rank. Only centered-covariance Probe-A/Probe-B overlap at gate ranks enters the primary gate. `gram_matrices.npz` stores only sample-space Gram matrices keyed by seed/checkpoint/probe-view/class/stratum/layer.

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_reliability.py
PYTHONPATH=. ../../.conda/fm_lab/bin/ruff check \
  fm_lab/diagnostics/long_tail_geometry/reliability.py \
  tests/test_long_tail_geometry_reliability.py
```

- [ ] **Step 8: Commit**

```bash
git add fm_lab/diagnostics/long_tail_geometry tests/test_long_tail_geometry_reliability.py
git commit -m "Add Observation-0 reliability gate"
```

---

### Task 5: Resumable study preparation, collection service, and CLI

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/observation0.py`
- Create: `fm_lab/experiments/run_long_tail_geometry_observation0.py`
- Modify: `pyproject.toml`
- Modify: `docs/cli.md`
- Modify: `docs/diagnostics.md`
- Create: `tests/test_long_tail_geometry_observation0.py`
- Modify: `tests/test_docs_coverage.py` only if its existing generic console-script check requires no focused assertion.

**Interfaces:**
- Produces CLI: `fm-lab-long-tail-geometry-observation0 prepare|collect|analyze`.
- Produces: three seed configs and `aggregate/run_registry.csv` during `prepare`.
- Produces: checkpoint measurement directories and registry transition to `measured` during `collect`.
- Produces: final Observation-0 reliability artifacts during `analyze`.

- [ ] **Step 1: Write failing study-preparation tests**

```python
def test_prepare_writes_three_ordinary_fm_seed_configs(tmp_path):
    result = prepare_observation0_study(prereg_path, tmp_path)
    assert len(result.run_configs) == 3
    for seed, config_path in enumerate(result.run_configs):
        config = load_config(config_path)
        assert config["experiment"]["seed"] == seed
        assert config["data"]["frequency_mapping"]["offset"] == 0
        assert config["objective"]["modifiers"] == []
        assert not config["model"]["capacity"]["enabled"]
        assert config["training"]["checkpoint_steps"] == [0, 500, 1000, 3000, 7000, 13000, 20000]
        assert not config["training"]["early_stopping"]["enabled"]
```

Preparation locks a copy of the preregistration under `aggregate/`, writes seed configs under `configs/seed_<seed>.yaml`, and refuses any base config that fails the Stage-0 ordinary-FM fence.

- [ ] **Step 2: Write failing checkpoint-collection/resume tests**

Use the Fashion-MNIST IDX fixture, a one-step config with checkpoints 0 and 1, one stratum, one row per class, two layers, and an untrained/tiny trained checkpoint pair. Assert:

```python
summary = collect_observation0_run(
    preregistration=test_prereg,
    study_dir=study_dir,
    run_dir=run_dir,
    device=torch.device("cpu"),
)
assert summary.completed_steps == (0, 1)
assert summary.skipped_steps == ()
assert registry_row(study_dir, seed=0)["status"] == "measured"

repeated = collect_observation0_run(
    preregistration=test_prereg,
    study_dir=study_dir,
    run_dir=run_dir,
    device=torch.device("cpu"),
)
assert repeated.completed_steps == ()
assert repeated.skipped_steps == (0, 1)
```

The collector builds or loads study-level Probe-A/B manifests using `manifest_seed`, plus their source-noise replicas using fixed derived seeds `manifest_seed + 10_001` and `manifest_seed + 20_001`. It verifies the same four digests for every training seed, restores each raw checkpoint, and calls `collect_checkpoint_measurements`. It updates the registry only after every required checkpoint has a valid `complete.json`.

- [ ] **Step 3: Write failing analyze completeness test**

```python
def test_analyze_refuses_registry_without_all_three_measured_seeds(tmp_path):
    prepare_observation0_study(prereg_path, tmp_path)
    update_observation0_run(tmp_path, seed=0, status="measured", measurement_digest="a" * 64)
    with pytest.raises(ValueError, match="all preregistered training seeds"):
        analyze_observation0_study(preregistration=prereg, study_dir=tmp_path)
```

- [ ] **Step 4: Implement service boundaries and fail-closed provenance**

`collect` validates checkpoint config compatibility against the prepared seed config and records checkpoint hashes. It must not accept EMA weights, missing checkpoint steps, changed manifest digests, or measurements with a different preregistration digest. `analyze` loads only registry-listed artifacts and never globs arbitrary run directories.

- [ ] **Step 5: Register and test the CLI**

```toml
fm-lab-long-tail-geometry-observation0 = "fm_lab.experiments.run_long_tail_geometry_observation0:main"
```

Commands:

```bash
fm-lab-long-tail-geometry-observation0 prepare \
  --preregistration configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml \
  --study-dir runs/long_tail_geometry/fashion_mnist/observation0

fm-lab-long-tail-geometry-observation0 collect \
  --study-dir runs/long_tail_geometry/fashion_mnist/observation0 \
  --run-dir runs/long_tail_geometry/fashion_mnist/observation0/mapping_0/seed_0 \
  --device auto

fm-lab-long-tail-geometry-observation0 analyze \
  --study-dir runs/long_tail_geometry/fashion_mnist/observation0
```

Update the CLI table and diagnostic artifact guide with the gate semantics and the explicit statement that `escalate_probe_rows` is not Outcome D.

- [ ] **Step 6: Run all Observation-0 tests**

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q \
  tests/test_long_tail_geometry_preregistration.py \
  tests/test_long_tail_geometry_measurements.py \
  tests/test_long_tail_geometry_reliability.py \
  tests/test_long_tail_geometry_observation0.py \
  tests/test_long_tail_geometry_stage0.py \
  tests/test_training_sampling.py \
  tests/test_docs_coverage.py
PYTHONPATH=. ../../.conda/fm_lab/bin/ruff check fm_lab tests docs
```

- [ ] **Step 7: Run a toy end-to-end execution smoke**

Prepare a test preregistration with seeds `[0, 1, 2]`, checkpoints `[0, 1]`, one timestep stratum, one microbatch per class, and the two toy U-Net layers. Generate all three toy runs, collect both checkpoints, then analyze synthetic planted reliability tables through the same artifact loader. The smoke must produce a complete registry, two manifests, six checkpoint measurement directories, `reliability.csv`, and `noise_ceiling.json` without launching any mapping other than offset 0.

- [ ] **Step 8: Run full verification**

```bash
PYTHONPATH=.:tests ../../.conda/fm_lab/bin/pytest -q
PYTHONPATH=. ../../.conda/fm_lab/bin/ruff check .
git diff --check
```

- [ ] **Step 9: Commit**

```bash
git add fm_lab configs pyproject.toml docs tests
git commit -m "Add resumable Observation-0 study runner"
```

## Exit Criterion

This implementation is complete only when:

- the canonical preregistration loads and locks without ambiguity;
- step zero is proven bitwise equal to the pre-training model state;
- measurement collection streams all layers from one backward per microbatch and round-trips its Parquet/NPZ provenance;
- the matched null rejects synthetic random structure and accepts a planted repeated structure;
- incomplete or output-only evidence cannot pass the network-wide gate;
- a repeated collector invocation skips only matching complete artifacts;
- Probe-A split halves, Probe-A/Probe-B, and fixed-target source-noise replicas are all represented in the noise-ceiling artifacts;
- the toy execution smoke and complete repository suite pass;
- no Stage-1 mapping run is scheduled by this implementation.
