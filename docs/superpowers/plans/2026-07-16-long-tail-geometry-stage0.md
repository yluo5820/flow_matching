# Long-Tail Gradient Geometry Stage 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the deterministic data, probe, checkpoint, gradient-sketch, and synthetic-control pipeline required before the counterfactual long-tail geometry study begins.

**Architecture:** Extend the existing Fashion-MNIST long-tail target with an opt-in counterfactual frequency mapping and a permanently held-out diagnostic pool, leaving every existing config unchanged. A new `fm_lab.diagnostics.long_tail_geometry` package will own immutable probe manifests, deterministic tuple materialization, layer-gradient collection, CountSketch validation, and null/positive controls. A Stage-0 CLI will consume an ordinary-flow-matching checkpoint and fail closed unless every preregistered validation passes.

**Tech Stack:** Python 3.11, PyTorch 2.2+, NumPy, SciPy, pytest, Ruff, YAML, existing `fm_lab` factories/checkpoints/objectives.

## Global Constraints

- Do not enable CM or any capacity adapter in Stage 0.
- Do not use CM checkpoints, CM distances, or CM-selected layers.
- Preserve existing Fashion-MNIST and CIFAR-10 long-tail config behavior when `data.frequency_mapping` is absent.
- Reserve the diagnostic pool before constructing any long-tail training subset.
- Use class-local deterministic orders shared by all frequency mappings; a higher-count subset must contain every lower-count subset for the same class.
- The primary ten-class mapping is `rank_m(c) = (3 * c + m) mod 10`, for `m = 0,...,9`.
- Probe-A and Probe-B are disjoint and never available through the target's training sampler.
- Probe tuples are paired across mappings by original dataset index, dequantization seed, source-noise seed, timestep, and microbatch row.
- Gradient probes use the ordinary flow-matching loss and `torch.autograd.grad`; they must not mutate model parameters or leave `.grad` tensors behind.
- No `P x P` covariance or projection matrix may be allocated.
- Stage 0 is a fail-closed validator: any failed invariant produces a nonzero CLI exit and a report with `passed: false`.

---

### Task 1: Counterfactual frequency-map and nested-split primitives

**Files:**
- Modify: `fm_lab/data/long_tail.py`
- Modify: `tests/test_fashion_mnist_lt_data.py`

**Interfaces:**
- Produces: `FrequencySplit(train_indices, probe_a_indices, probe_b_indices, class_counts, class_ranks)`
- Produces: `frequency_rank_mapping(num_classes: int, multiplier: int, offset: int) -> np.ndarray`
- Produces: `nested_frequency_split(labels: np.ndarray, *, num_classes: int, imbalance_factor: float, seed: int, diagnostic_pool_per_class: int, multiplier: int, offset: int) -> FrequencySplit`

- [ ] **Step 1: Add failing balance and Latin-mapping tests**

```python
def test_frequency_rank_mappings_balance_class_and_rank() -> None:
    mappings = np.stack(
        [frequency_rank_mapping(10, multiplier=3, offset=m) for m in range(10)]
    )
    assert all(np.array_equal(np.sort(row), np.arange(10)) for row in mappings)
    assert all(np.array_equal(np.sort(mappings[:, c]), np.arange(10)) for c in range(10))


def test_frequency_rank_mapping_requires_coprime_multiplier() -> None:
    with pytest.raises(ValueError, match="coprime"):
        frequency_rank_mapping(10, multiplier=2, offset=0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_fashion_mnist_lt_data.py -k frequency_rank_mapping
```

Expected: import failure for `frequency_rank_mapping`.

- [ ] **Step 3: Implement the immutable split value and mapping function**

```python
@dataclass(frozen=True)
class FrequencySplit:
    train_indices: np.ndarray
    probe_a_indices: np.ndarray
    probe_b_indices: np.ndarray
    class_counts: tuple[int, ...]
    class_ranks: tuple[int, ...]


def frequency_rank_mapping(num_classes: int, *, multiplier: int, offset: int) -> np.ndarray:
    if num_classes < 2:
        raise ValueError("Frequency mapping requires at least two classes.")
    if math.gcd(multiplier, num_classes) != 1:
        raise ValueError("Frequency mapping multiplier must be coprime with num_classes.")
    classes = np.arange(num_classes, dtype=np.int64)
    return (multiplier * classes + int(offset)) % num_classes
```

- [ ] **Step 4: Add failing nestedness, exclusion, and count tests**

```python
def test_nested_frequency_split_reserves_probe_pool_before_training() -> None:
    labels = np.repeat(np.arange(10), 100)
    split = nested_frequency_split(
        labels,
        num_classes=10,
        imbalance_factor=0.01,
        seed=17,
        diagnostic_pool_per_class=20,
        multiplier=3,
        offset=0,
    )
    assert len(split.probe_a_indices) == 100
    assert len(split.probe_b_indices) == 100
    assert not set(split.train_indices) & set(split.probe_a_indices)
    assert not set(split.train_indices) & set(split.probe_b_indices)
    assert not set(split.probe_a_indices) & set(split.probe_b_indices)
    assert max(split.class_counts) == 80
    assert min(split.class_counts) == 1


def test_same_class_frequency_subsets_are_nested_across_mappings() -> None:
    labels = np.repeat(np.arange(10), 100)
    splits = [
        nested_frequency_split(
            labels,
            num_classes=10,
            imbalance_factor=0.01,
            seed=17,
            diagnostic_pool_per_class=20,
            multiplier=3,
            offset=m,
        )
        for m in range(10)
    ]
    for class_id in range(10):
        class_sets = [
            set(s.train_indices[labels[s.train_indices] == class_id]) for s in splits
        ]
        ordered = sorted(class_sets, key=len)
        assert all(left <= right for left, right in zip(ordered, ordered[1:], strict=True))
```

- [ ] **Step 5: Run the nested-split tests and verify RED**

Expected: import failure for `nested_frequency_split`.

- [ ] **Step 6: Implement nested splitting**

```python
def nested_frequency_split(
    labels: np.ndarray,
    *,
    num_classes: int,
    imbalance_factor: float,
    seed: int,
    diagnostic_pool_per_class: int,
    multiplier: int,
    offset: int,
) -> FrequencySplit:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("Long-tail labels must be a vector.")
    if diagnostic_pool_per_class <= 0 or diagnostic_pool_per_class % 2:
        raise ValueError("diagnostic_pool_per_class must be positive and even.")
    if not 0.0 < imbalance_factor <= 1.0:
        raise ValueError("imbalance_factor must be in (0, 1].")
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts != counts[0]):
        raise ValueError("Source split must be class-balanced before splitting.")
    if diagnostic_pool_per_class >= int(counts[0]):
        raise ValueError("Diagnostic pool must leave at least one training example.")

    ranks = frequency_rank_mapping(
        num_classes, multiplier=multiplier, offset=offset
    )
    probe_half = diagnostic_pool_per_class // 2
    n_max = int(counts[0]) - diagnostic_pool_per_class
    rng = np.random.RandomState(seed)
    train_parts: list[np.ndarray] = []
    probe_a_parts: list[np.ndarray] = []
    probe_b_parts: list[np.ndarray] = []
    retained_counts: list[int] = []
    for class_id in range(num_classes):
        ordered = np.flatnonzero(labels == class_id)
        rng.shuffle(ordered)
        probe_a_parts.append(ordered[:probe_half])
        probe_b_parts.append(ordered[probe_half:diagnostic_pool_per_class])
        candidates = ordered[diagnostic_pool_per_class:]
        exponent = float(ranks[class_id]) / (num_classes - 1.0)
        keep = max(1, int(n_max * imbalance_factor**exponent))
        train_parts.append(candidates[:keep])
        retained_counts.append(keep)
    return FrequencySplit(
        train_indices=np.concatenate(train_parts).astype(np.int64, copy=False),
        probe_a_indices=np.concatenate(probe_a_parts).astype(np.int64, copy=False),
        probe_b_indices=np.concatenate(probe_b_parts).astype(np.int64, copy=False),
        class_counts=tuple(retained_counts),
        class_ranks=tuple(int(rank) for rank in ranks),
    )
```

The RNG order must not depend on `offset`; only the retained prefix length may change.

- [ ] **Step 7: Run focused and legacy data tests**

```bash
.conda/fm_lab/bin/pytest -q tests/test_fashion_mnist_lt_data.py tests/test_cifar_lt_data.py
.conda/fm_lab/bin/ruff check fm_lab/data/long_tail.py tests/test_fashion_mnist_lt_data.py
```

- [ ] **Step 8: Commit**

```bash
git add fm_lab/data/long_tail.py tests/test_fashion_mnist_lt_data.py
git commit -m "Add counterfactual long-tail frequency splits"
```

---

### Task 2: Fashion-MNIST target integration and held-out diagnostic access

**Files:**
- Modify: `fm_lab/data/fashion_mnist.py`
- Modify: `fm_lab/experiments/factory.py`
- Modify: `tests/test_fashion_mnist_lt_data.py`
- Create: `tests/long_tail_geometry_helpers.py`

**Interfaces:**
- Consumes: the full `nested_frequency_split` signature defined in Task 1
- Produces: opt-in `data.frequency_mapping` with keys `offset`, `multiplier`, and `diagnostic_pool_per_class`
- Produces: `LongTailedFashionMNIST.diagnostic_samples(split: str, *, original_indices: np.ndarray | None = None, dequantization_seeds: np.ndarray | None = None, device=None) -> tuple[Tensor, Tensor, np.ndarray]`
- Produces test helpers: `write_balanced_fashion_mnist(root, examples_per_class)` and `geometry_toy_config(root, output_dir)`

- [ ] **Step 1: Add a failing factory integration test**

```python
def test_fashion_mnist_factory_builds_counterfactual_mapping_with_held_out_probes(tmp_path):
    _write_balanced_fashion_mnist(tmp_path, examples_per_class=100)
    target = build_target({"data": {
        "name": "fashion_mnist_lt",
        "root": str(tmp_path),
        "imbalance_factor": 0.01,
        "subset_seed": 7,
        "normalize": "minus_one_one",
        "frequency_mapping": {
            "offset": 4,
            "multiplier": 3,
            "diagnostic_pool_per_class": 20,
        },
    }})
    assert target.metadata()["frequency_mapping"]["offset"] == 4
    assert len(target.diagnostic_indices("a")) == 100
    assert len(target.diagnostic_indices("b")) == 100
    assert not set(target.selected_indices) & set(target.diagnostic_indices("a"))
```

- [ ] **Step 2: Verify RED**

Expected: the target ignores `frequency_mapping` and lacks `diagnostic_indices`.

- [ ] **Step 3: Integrate the opt-in split without changing legacy behavior**

Add nullable dataclass fields for offset/multiplier/pool size. In `_load`, call the existing `long_tail_indices` when no offset is configured; otherwise call `nested_frequency_split`. Retain only the reserved raw images and labels needed by Probe-A/B. Include mapping, ranks, probe hashes, and probe counts in `metadata()`.

- [ ] **Step 4: Add failing deterministic dequantization tests**

```python
def test_diagnostic_samples_are_addressed_by_original_id_and_seed(tmp_path):
    _write_balanced_fashion_mnist(tmp_path, examples_per_class=100)
    target = LongTailedFashionMNIST(
        root=tmp_path,
        imbalance_factor=0.01,
        subset_seed=7,
        normalize="minus_one_one",
        dequantize=True,
        frequency_mapping_offset=0,
        frequency_mapping_multiplier=3,
        diagnostic_pool_per_class=20,
    )
    ids = target.diagnostic_indices("a")[:3]
    seeds = np.array([11, 12, 13], dtype=np.int64)
    first, labels, returned = target.diagnostic_samples(
        "a", original_indices=ids, dequantization_seeds=seeds
    )
    second, _, _ = target.diagnostic_samples(
        "a", original_indices=ids, dequantization_seeds=seeds
    )
    changed, _, _ = target.diagnostic_samples(
        "a", original_indices=ids, dequantization_seeds=seeds + 1
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    assert np.array_equal(returned.astype(np.int64), ids)
```

- [ ] **Step 5: Implement diagnostic sample materialization**

Use a private `_seeded_dequantization(raw_images, seeds)` helper that creates one CPU `torch.Generator` per example, draws exactly one image-shaped uniform tensor, applies the existing 256-level normalization, and never mutates global RNG state. Reject duplicate requested IDs, IDs outside the selected probe split, seed-length mismatches, and unknown split names.

- [ ] **Step 6: Verify target and factory behavior**

```bash
.conda/fm_lab/bin/pytest -q tests/test_fashion_mnist_lt_data.py tests/test_config_smoke.py
.conda/fm_lab/bin/ruff check fm_lab/data/fashion_mnist.py fm_lab/experiments/factory.py tests/test_fashion_mnist_lt_data.py
```

- [ ] **Step 7: Extract the reusable test fixture**

Move the existing IDX gzip writers from `tests/test_fashion_mnist_lt_data.py` into `tests/long_tail_geometry_helpers.py` under the public test-only name `write_balanced_fashion_mnist`. Add:

```python
def geometry_toy_config(root: Path, output_dir: Path) -> dict[str, object]:
    return {
        "experiment": {"name": "geometry_stage0_toy", "seed": 0, "output_dir": str(output_dir)},
        "data": {
            "name": "fashion_mnist_lt",
            "root": str(root),
            "train": True,
            "download": False,
            "normalize": "minus_one_one",
            "dequantize": True,
            "imbalance_type": "exp",
            "imbalance_factor": 0.1,
            "subset_seed": 7,
            "frequency_mapping": {
                "offset": 0,
                "multiplier": 3,
                "diagnostic_pool_per_class": 2,
            },
        },
        "source": {"name": "gaussian", "dim": 784},
        "coupling": {"name": "independent"},
        "path": {"name": "linear"},
        "model": {
            "name": "image_unet",
            "image_shape": [1, 28, 28],
            "base_channels": 8,
            "time_embedding_dim": 16,
            "zero_init_head": False,
            "capacity": {"enabled": False},
        },
        "conditioning": {
            "enabled": True,
            "num_classes": 10,
            "embedding_dim": 16,
            "dropout_probability": 0.0,
        },
        "objective": {
            "name": "flow_matching",
            "model_output": "target",
            "loss_space": "velocity",
            "min_denom": 0.05,
            "modifiers": [],
        },
        "training": {"batch_size": 10, "steps": 1, "lr": 1.0e-4, "early_stopping": {"enabled": False}},
        "sampling": {"n_samples": 10, "n_trajectories": 2, "nfe": 2},
    }
```

- [ ] **Step 8: Commit**

```bash
git add fm_lab/data/fashion_mnist.py fm_lab/experiments/factory.py tests/test_fashion_mnist_lt_data.py tests/long_tail_geometry_helpers.py
git commit -m "Expose held-out Fashion-MNIST probe pools"
```

---

### Task 3: Immutable paired probe manifests and tuple materialization

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/__init__.py`
- Create: `fm_lab/diagnostics/long_tail_geometry/manifest.py`
- Create: `tests/test_long_tail_probe_manifest.py`

**Interfaces:**
- Produces: `ProbeManifest`
- Produces: `build_probe_manifest(sample_ids, labels, *, split, rows_per_class_per_stratum, batch_size, time_strata, seed) -> ProbeManifest`
- Produces: `materialize_probe_batch(target, source, manifest, row_indices, *, device) -> ProbeBatch`
- Produces: `ProbeBatch(x0, x1, t, labels, original_indices, stratum_ids, microbatch_ids)`

- [ ] **Step 1: Add failing manifest balance and serialization tests**

```python
def test_probe_manifest_is_balanced_over_class_and_timestep(tmp_path):
    ids = np.arange(40)
    labels = np.repeat(np.arange(4), 10)
    manifest = build_probe_manifest(
        ids,
        labels,
        split="a",
        rows_per_class_per_stratum=8,
        batch_size=4,
        time_strata=((0.02, 0.10), (0.10, 0.30)),
        seed=19,
    )
    assert manifest.num_rows == 4 * 2 * 8
    assert manifest.digest == ProbeManifest.load(manifest.save(tmp_path / "probe.npz")).digest
    for class_id in range(4):
        for stratum in range(2):
            assert np.sum((manifest.labels == class_id) & (manifest.stratum_ids == stratum)) == 8
```

- [ ] **Step 2: Verify RED**

Expected: module import failure.

- [ ] **Step 3: Implement `ProbeManifest`**

Store copied, read-only arrays for original index, class label, dequantization seed, source seed, exact float64 timestep, stratum ID, and microbatch ID. Validate equal lengths; finite timesteps strictly inside the registered stratum; class/stratum balance; exactly `batch_size` rows per microbatch; unique `(split, class, stratum, row)` provenance; and SHA-256 over canonical array bytes plus schema version. Save with `np.savez_compressed` and reject schema/digest mismatches on load.

- [ ] **Step 4: Add failing cross-mapping tuple-pairing test**

```python
def test_same_manifest_materializes_identical_tuples_across_frequency_mappings(tmp_path):
    _write_balanced_fashion_mnist(tmp_path, examples_per_class=100)
    common = dict(
        root=tmp_path,
        imbalance_factor=0.01,
        subset_seed=7,
        normalize="minus_one_one",
        dequantize=True,
        frequency_mapping_multiplier=3,
        diagnostic_pool_per_class=20,
    )
    target0 = LongTailedFashionMNIST(**common, frequency_mapping_offset=0)
    target7 = LongTailedFashionMNIST(**common, frequency_mapping_offset=7)
    _, labels, ids = target0.diagnostic_samples("a")
    manifest = build_probe_manifest(
        ids.astype(np.int64),
        labels.numpy(),
        split="a",
        rows_per_class_per_stratum=4,
        batch_size=4,
        time_strata=((0.02, 0.10), (0.10, 0.30)),
        seed=19,
    )
    rows = np.arange(16)
    batch0 = materialize_probe_batch(target0, GaussianSource(dim=784), manifest, rows, device="cpu")
    batch7 = materialize_probe_batch(target7, GaussianSource(dim=784), manifest, rows, device="cpu")
    assert torch.equal(batch0.x0, batch7.x0)
    assert torch.equal(batch0.x1, batch7.x1)
    assert torch.equal(batch0.t, batch7.t)
    assert torch.equal(batch0.labels, batch7.labels)
```

- [ ] **Step 5: Implement deterministic tuple materialization**

Request `x1` by original probe IDs and dequantization seeds. For each source seed, use a forked CPU generator to draw one source row without touching global RNG state. Use stored timesteps directly. Require the target's returned IDs and labels to match the manifest exactly.

- [ ] **Step 6: Verify manifest behavior**

```bash
.conda/fm_lab/bin/pytest -q tests/test_long_tail_probe_manifest.py
.conda/fm_lab/bin/ruff check fm_lab/diagnostics/long_tail_geometry tests/test_long_tail_probe_manifest.py
```

- [ ] **Step 7: Commit**

```bash
git add fm_lab/diagnostics/long_tail_geometry tests/test_long_tail_probe_manifest.py
git commit -m "Add paired long-tail probe manifests"
```

---

### Task 4: Explicit checkpoint schedule and deterministic probe-loss replay

**Files:**
- Modify: `fm_lab/training/trainer.py`
- Create: `fm_lab/diagnostics/long_tail_geometry/checkpoints.py`
- Modify: `tests/test_training_sampling.py`
- Create: `tests/test_long_tail_probe_checkpoints.py`

**Interfaces:**
- Produces: `training.checkpoint_steps: list[int]`, mutually exclusive with positive `checkpoint_every`
- Produces: `evaluate_probe_loss(*, model, objective, path, manifest, target, source, device) -> ProbeLossResult`
- Produces: `restore_probe_model(checkpoint_path, *, device) -> tuple[nn.Module, dict]`

- [ ] **Step 1: Add failing explicit-schedule training test**

```python
def test_training_saves_only_requested_checkpoint_steps(tmp_path):
    config = _tiny_training_config(steps=5)
    config["training"]["checkpoint_steps"] = [1, 3, 5]
    train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=TinyVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    assert sorted(p.name for p in (tmp_path / "checkpoints").glob("*.pt")) == [
        "step_000001.pt", "step_000003.pt", "step_000005.pt"
    ]
```

Define the test-local helper without implicit defaults:

```python
def _tiny_training_config(*, steps: int) -> dict[str, object]:
    return {
        "experiment": {"seed": 0},
        "path": {"name": "linear"},
        "objective": {"name": "flow_matching"},
        "training": {
            "batch_size": 2,
            "steps": steps,
            "log_every": 1,
            "optimizer": "adam",
            "lr": 1.0e-3,
            "early_stopping": {"enabled": False},
        },
        "sampling": {"n_samples": 2, "n_trajectories": 1, "nfe": 1},
    }
```

- [ ] **Step 2: Verify RED**

Expected: no intermediate checkpoint files.

- [ ] **Step 3: Implement schedule validation and saving**

Reject non-integers, duplicates, nonpositive values, values above `training.steps`, and simultaneous positive `checkpoint_every`. Add `checkpoint_steps` to `_RUNTIME_TRAINING_FIELDS` so resume compatibility ignores the output schedule. Save when either the validated explicit set contains `step` or the legacy interval fires.

- [ ] **Step 4: Add failing checkpoint replay test**

```python
def test_restored_checkpoint_reproduces_probe_loss_bitwise(tmp_path):
    config, target, source, path, objective, model, manifest = build_probe_fixture(tmp_path)
    before = evaluate_probe_loss(
        model=model,
        objective=objective,
        path=path,
        manifest=manifest,
        target=target,
        source=source,
        device=torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        step=3,
        config=config,
        metrics={"probe_loss": before.mean_loss},
    )
    restored, saved = restore_probe_model(checkpoint_path, device=torch.device("cpu"))
    after = evaluate_probe_loss(
        model=restored,
        objective=objective,
        path=path,
        manifest=manifest,
        target=target,
        source=source,
        device=torch.device("cpu"),
    )
    assert before.mean_loss == after.mean_loss
    assert before.row_losses_sha256 == after.row_losses_sha256
```

Define `build_probe_fixture` in `tests/long_tail_geometry_helpers.py`. It must call `write_balanced_fashion_mnist(root, examples_per_class=10)`, obtain `config = geometry_toy_config(root, output_dir)`, build target/source/path/model through the production factories, build the objective with the target's class counts, read Probe-A IDs and labels, and construct a one-stratum manifest with one row per class and `batch_size=10`. Return those seven objects in the order used above.

- [ ] **Step 5: Implement replay helpers**

Build the ordinary objective from checkpoint config, validate that `objective.modifiers` is empty and capacity metadata is disabled, iterate manifest microbatches in stored order under `model.eval()`, compute per-row ordinary FM losses without gradients, and hash the contiguous float64 row-loss vector. `restore_probe_model` must use existing factories and `validate_checkpoint_compatibility`; it must not load optimizer state.

- [ ] **Step 6: Verify checkpoint behavior**

```bash
.conda/fm_lab/bin/pytest -q tests/test_training_sampling.py -k checkpoint
.conda/fm_lab/bin/pytest -q tests/test_long_tail_probe_checkpoints.py
.conda/fm_lab/bin/ruff check fm_lab/training/trainer.py fm_lab/diagnostics/long_tail_geometry/checkpoints.py tests/test_long_tail_probe_checkpoints.py
```

- [ ] **Step 7: Commit**

```bash
git add fm_lab/training/trainer.py fm_lab/diagnostics/long_tail_geometry/checkpoints.py tests/test_training_sampling.py tests/test_long_tail_probe_checkpoints.py
git commit -m "Add geometry checkpoint replay validation"
```

---

### Task 5: Layer gradients, exact Gram statistics, and fixed CountSketch

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/gradients.py`
- Create: `fm_lab/diagnostics/long_tail_geometry/sketch.py`
- Create: `tests/test_long_tail_gradient_probe.py`
- Create: `tests/test_long_tail_gradient_sketch.py`

**Interfaces:**
- Produces: `resolve_probe_layers(model, names) -> tuple[ProbeLayer, ...]`
- Produces: `collect_gradient_rows(*, model, objective, path, batches, layer_names) -> dict[str, GradientRows]`
- Produces: `CountSketchSpec.build(input_dim, output_dim, seed)` and `CountSketchSpec.apply(rows)`
- Produces: `validate_sketch(exact_rows, sketched_rows, *, rank) -> SketchValidation`

- [ ] **Step 1: Add failing layer-resolution and no-mutation tests**

```python
def test_gradient_probe_collects_requested_layers_without_mutation():
    model = _tiny_conditional_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rows = collect_gradient_rows(
        model=model,
        objective=objective,
        path=LinearPath(),
        batches=batches,
        layer_names=("hidden.weight", "output.weight"),
    )
    assert rows["hidden.weight"].raw.shape[0] == len(batches)
    assert torch.all(rows["hidden.weight"].norms > 0)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[k], model.state_dict()[k]) for k in before)
```

- [ ] **Step 2: Verify RED**

Expected: module import failure.

- [ ] **Step 3: Implement exact gradient collection**

Resolve exact `.weight` parameter names, reject duplicates/missing/frozen parameters, preserve model training mode, call `torch.autograd.grad` once per stored microbatch, flatten each layer independently, move rows to float32 CPU immediately, and store raw norms plus normalized rows. Zero or nonfinite norms fail with the layer and manifest row in the message.

- [ ] **Step 4: Add failing CountSketch fidelity and allocation tests**

```python
def test_countsketch_approximates_cosines_and_never_allocates_projection_matrix(monkeypatch):
    generator = torch.Generator().manual_seed(23)
    rows = torch.randn(16, 10_000, generator=generator)
    spec = CountSketchSpec.build(input_dim=10_000, output_dim=4_096, seed=23)
    sketched = spec.apply(rows)
    result = validate_sketch(rows, sketched, rank=4)
    assert result.max_absolute_cosine_error < 0.04
    assert result.normalized_subspace_overlap_error < 0.06
```

The test also wraps tensor allocation and fails on any allocation of shape `(10_000, 10_000)` or `(10_000, 4_096)`.

- [ ] **Step 5: Implement sparse CountSketch and validation**

`CountSketchSpec` stores one bucket and one sign per input coordinate. Apply with `scatter_add_` over feature columns; do not materialize a projection matrix. Compute exact and sketched row-cosine matrices and top-rank sample-space subspaces. Return max absolute cosine error and normalized projection-overlap error. Reject output dimensions below two and ranks above the nonzero sample rank. The loose unit-test thresholds above only prevent a flaky randomized unit test; the Stage-0 scientific thresholds remain exactly `0.02` and `0.03`.

- [ ] **Step 6: Verify gradient and sketch modules**

```bash
.conda/fm_lab/bin/pytest -q tests/test_long_tail_gradient_probe.py tests/test_long_tail_gradient_sketch.py
.conda/fm_lab/bin/ruff check fm_lab/diagnostics/long_tail_geometry/gradients.py fm_lab/diagnostics/long_tail_geometry/sketch.py tests/test_long_tail_gradient_probe.py tests/test_long_tail_gradient_sketch.py
```

- [ ] **Step 7: Commit**

```bash
git add fm_lab/diagnostics/long_tail_geometry tests/test_long_tail_gradient_probe.py tests/test_long_tail_gradient_sketch.py
git commit -m "Add deterministic long-tail gradient sketches"
```

---

### Task 6: Null controls and planted low-rank recovery

**Files:**
- Create: `fm_lab/diagnostics/long_tail_geometry/controls.py`
- Create: `tests/test_long_tail_geometry_controls.py`

**Interfaces:**
- Produces: `projection_overlap(a, b) -> float`
- Produces: `permutation_null(values, labels, *, statistic, permutations, seed) -> PermutationResult`
- Produces: `planted_low_rank_control(*, ambient_dim, rank, rows, noise_std, seed) -> PlantedControlResult`

- [ ] **Step 1: Add failing projection-overlap tests**

```python
def test_projection_overlap_is_one_for_equal_basis_and_zero_for_orthogonal_basis():
    eye = torch.eye(6)
    assert projection_overlap(eye[:, :2], eye[:, :2]) == pytest.approx(1.0)
    assert projection_overlap(eye[:, :2], eye[:, 2:4]) == pytest.approx(0.0)
```

- [ ] **Step 2: Verify RED, then implement normalized overlap**

QR-orthonormalize each input and return `||U.T @ V||_F^2 / min(k_u, k_v)`. Reject empty, nonfinite, or row-dimension-mismatched bases.

- [ ] **Step 3: Add failing null and positive-control tests**

```python
def test_permuted_labels_return_null_on_exchangeable_rows():
    result = permutation_null(_exchangeable_rows(), _balanced_labels(), statistic=_group_gap, permutations=499, seed=7)
    assert result.p_value > 0.05


def test_planted_low_rank_control_recovers_dimension_and_subspace():
    result = planted_low_rank_control(ambient_dim=256, rank=4, rows=128, noise_std=0.02, seed=11)
    assert result.recovered_rank == 4
    assert result.subspace_overlap > 0.95
```

- [ ] **Step 4: Implement deterministic permutation and planted controls**

Use the plus-one permutation p-value `(1 + exceedances) / (1 + permutations)`. Preserve label multiplicities exactly. The planted control creates an orthonormal rank-`r` basis, Gaussian coefficients, and isotropic noise; recover rank using the largest eigengap of the sample-space Gram spectrum restricted to ranks 1–32; map the recovered basis to parameter space without forming `P x P`.

- [ ] **Step 5: Verify controls**

```bash
.conda/fm_lab/bin/pytest -q tests/test_long_tail_geometry_controls.py
.conda/fm_lab/bin/ruff check fm_lab/diagnostics/long_tail_geometry/controls.py tests/test_long_tail_geometry_controls.py
```

- [ ] **Step 6: Commit**

```bash
git add fm_lab/diagnostics/long_tail_geometry/controls.py tests/test_long_tail_geometry_controls.py
git commit -m "Add long-tail geometry pipeline controls"
```

---

### Task 7: Fail-closed Stage-0 validator and smoke configuration

**Files:**
- Create: `fm_lab/experiments/run_long_tail_geometry_stage0.py`
- Create: `configs/fashion_mnist_lt/fashion_mnist_lt_geometry_stage0.yaml`
- Modify: `pyproject.toml`
- Modify: `tests/test_config_smoke.py`
- Create: `tests/test_long_tail_geometry_stage0.py`

**Interfaces:**
- Produces CLI: `fm-lab-long-tail-geometry-stage0 --config <yaml> --checkpoint <pt> --output-dir <dir> --device <device>`
- Produces: `diagnostics/long_tail_geometry/stage0_report.json`
- Produces: `diagnostics/long_tail_geometry/probe_a.npz` and `probe_b.npz`
- Produces: per-layer `gradient_rows_<layer>.npz` only when every pre-gradient invariant passes

- [ ] **Step 1: Add a failing config/CM-fence test**

```python
def test_stage0_config_uses_ordinary_flow_matching_without_capacity():
    config = load_config("configs/fashion_mnist_lt/fashion_mnist_lt_geometry_stage0.yaml")
    assert config["objective"].get("modifiers", []) == []
    assert not config["model"].get("capacity", {}).get("enabled", False)
    assert config["training"]["checkpoint_steps"] == [500, 1000, 3000, 7000, 13000, 20000]
```

- [ ] **Step 2: Verify RED, then add the canonical config**

Copy the ordinary Fashion-MNIST target-prediction/velocity-loss settings, set frequency mapping offset 0, multiplier 3, diagnostic pool 500, disable early stopping, configure the six exact checkpoint steps, and add:

```yaml
diagnostics:
  long_tail_geometry:
    pairing_check_offsets: [0, 1, 7]
    probe_splits: [a, b]
    rows_per_class_per_stratum: 128
    microbatch_size: 8
    time_strata:
      - [0.02, 0.10]
      - [0.10, 0.30]
      - [0.30, 0.70]
      - [0.70, 0.90]
      - [0.90, 0.98]
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
    max_cosine_error: 0.02
    max_subspace_error: 0.03
    permutation_count: 999
```

- [ ] **Step 3: Add a failing end-to-end validator test using toy data/model**

```python
def test_stage0_validator_writes_passing_report_for_valid_pipeline(tmp_path):
    config, checkpoint_path = write_geometry_toy_checkpoint(tmp_path)
    report = run_stage0_validation(config=config, checkpoint_path=checkpoint_path, output_dir=tmp_path / "out", device=torch.device("cpu"))
    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"].values())
    assert (tmp_path / "out/diagnostics/long_tail_geometry/probe_a.npz").exists()


def test_stage0_validator_fails_before_gradients_when_pairing_breaks(tmp_path, monkeypatch):
    config, checkpoint_path = write_geometry_toy_checkpoint(tmp_path)
    output_dir = tmp_path / "out"
    real_materialize = stage0_module.materialize_probe_batch

    def corrupt_second_mapping(*args, **kwargs):
        batch = real_materialize(*args, **kwargs)
        if getattr(args[0], "frequency_mapping_offset", None) == 7:
            batch = dataclasses.replace(batch, x1=batch.x1 + 0.01)
        return batch

    monkeypatch.setattr(stage0_module, "materialize_probe_batch", corrupt_second_mapping)
    with pytest.raises(Stage0ValidationError, match="paired probe tuples"):
        run_stage0_validation(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=torch.device("cpu"),
        )
    report_path = output_dir / "diagnostics/long_tail_geometry/stage0_report.json"
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert not list(report_path.parent.glob("gradient_rows_*.npz"))
```

Add `write_geometry_toy_checkpoint(tmp_path)` to `tests/long_tail_geometry_helpers.py`. It must extend `geometry_toy_config` with a minimal `diagnostics.long_tail_geometry` section using offsets `[0, 1, 7]`, one timestep stratum, one row per class, microbatch size 10, layers `input_block.conv2.weight` and `output_block.2.weight`, sketch dimension 64, maximum sketch dimension 256, 99 permutations, and the scientific error thresholds. Build the model through `build_model`, save its untrained state with `save_checkpoint`, and return `(config, checkpoint_path)`.

- [ ] **Step 4: Implement the validator in ordered gates**

Run and record these gates in order:

1. config fence: ordinary FM, no modifiers, no capacity, no early stopping;
2. ten-map Latin balance, nestedness, and identical Probe-A/B IDs;
3. manifest round-trip digest and tuple equality for configured offsets;
4. checkpoint reload and exact row-loss hash reproduction;
5. exact/sketch fidelity on a preregistered 5% row subset for every layer; if 4,096 coordinates fail, double only the sketch dimension with the same seed until the thresholds pass or `max_sketch_dim` is reached, recording every attempt;
6. class-label and rank-label permutation nulls on exchangeable synthetic rows;
7. planted-rank recovery and overlap thresholds.

Write the report after every gate so failures are inspectable. Include git commit, config SHA-256, checkpoint SHA-256, target/probe hashes, manifest hashes, device, PyTorch/NumPy versions, thresholds, observed values, and failure messages. Do not catch `KeyboardInterrupt` or `SystemExit`.

- [ ] **Step 5: Register the CLI and verify all Stage-0 tests**

```bash
.conda/fm_lab/bin/pytest -q \
  tests/test_fashion_mnist_lt_data.py \
  tests/test_long_tail_probe_manifest.py \
  tests/test_long_tail_probe_checkpoints.py \
  tests/test_long_tail_gradient_probe.py \
  tests/test_long_tail_gradient_sketch.py \
  tests/test_long_tail_geometry_controls.py \
  tests/test_long_tail_geometry_stage0.py \
  tests/test_config_smoke.py
.conda/fm_lab/bin/ruff check fm_lab tests/test_long_tail_*.py tests/test_fashion_mnist_lt_data.py
```

- [ ] **Step 6: Run the full verification suite**

```bash
.conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check .
git diff --check
```

- [ ] **Step 7: Run the real-data Stage-0 smoke sequence**

Use a temporary override with `training.steps: 5`, `checkpoint_steps: [1, 3, 5]`, `rows_per_class_per_stratum: 8`, and the output layer plus one upstream layer. Train the ordinary model, then run the validator on `step_000005.pt`. The report must pass before scheduling any 20,000-step mapping run.

- [ ] **Step 8: Commit**

```bash
git add fm_lab/experiments/run_long_tail_geometry_stage0.py configs/fashion_mnist_lt/fashion_mnist_lt_geometry_stage0.yaml pyproject.toml tests/test_config_smoke.py tests/test_long_tail_geometry_stage0.py
git commit -m "Add long-tail geometry Stage 0 validator"
```

## Stage-0 Exit Criterion

Do not begin the ten-mapping causal discovery runs unless:

- the full suite and Ruff pass;
- the real-data smoke report has `passed: true`;
- all configured mappings share identical Probe-A/B IDs and tuple hashes;
- checkpoint replay matches the exact row-loss hash;
- every validated layer meets both sketch tolerances;
- permutation controls are null at `p > 0.05`;
- the planted control recovers the exact rank with overlap above `0.95`.

If any condition fails, retain the report and stop. Do not loosen a threshold in the same run.
