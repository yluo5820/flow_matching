# JiT-Style X-Prediction/V-Loss Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the four continuous Fashion-MNIST target-output/velocity-loss experiments with JiT-style logit-normal time sampling and symmetric denominator-clamped supervision.

**Architecture:** Add a focused immutable training-time sampler that the trainer builds once from `training.time_sampling`. Keep path prediction conversion generic; change `FlowMatchingObjective` to derive supervision in the model-output space and convert it through the same state as the model prediction, including OC-transferred endpoints. Migrate the controlled config matrix to JiT parameters and verify with deterministic unit tests plus a short real CPU run.

**Tech Stack:** Python 3.11, PyTorch, PyYAML, pytest, Ruff, `.conda/fm_lab/bin`.

## Global Constraints

- Use `.conda/fm_lab/bin/python`, `.conda/fm_lab/bin/pytest`, and `.conda/fm_lab/bin/ruff` with `PYTHONPATH=$PWD` inside the worktree.
- Preserve the linear conditional interpolant and the path-aware prediction value objects.
- Preserve Euler/NFE-64 as the controlled generation and evaluation protocol.
- Default and legacy string `training.time_sampling: uniform` behavior must remain unchanged.
- JiT configs use logit-normal `mean: -0.8`, `std: 0.8`, and `objective.min_denom: 0.05`.
- Denominator flooring must be symmetric between prediction and supervision.
- Do not add gradient clipping; add top-level `training.warmup_steps: 500` and `training.ema_decay: 0.9999`.
- Do not alter unrelated Gaussian-diffusion objectives or discrete-array evaluators.

---

### Task 1: Configurable Training-Time Sampling

**Files:**
- Create: `fm_lab/training/time_sampling.py`
- Modify: `fm_lab/training/__init__.py`
- Modify: `fm_lab/training/losses.py:23-30`
- Modify: `fm_lab/training/trainer.py:62-130,454-510,919-949`
- Test: `tests/test_training_sampling.py`

**Interfaces:**
- Produces: `TrainingTimeSampler.sample(batch_size: int, device: torch.device, *, generator: torch.Generator | None = None) -> torch.Tensor`.
- Produces: `build_training_time_sampler(config: str | Mapping[str, Any] | None) -> TrainingTimeSampler`.
- Preserves: `sample_uniform_time(batch_size, device, eps=1e-5)` as a compatibility wrapper.
- Consumes: `training.time_sampling` from the resolved training configuration.

- [ ] **Step 1: Write failing sampler tests**

Add tests that specify exact seeded behavior and validation:

```python
def test_logit_normal_time_sampler_matches_seeded_definition() -> None:
    sampler = build_training_time_sampler(
        {"name": "logit_normal", "mean": -0.8, "std": 0.8}
    )
    actual_generator = torch.Generator().manual_seed(123)
    expected_generator = torch.Generator().manual_seed(123)
    actual = sampler.sample(4096, torch.device("cpu"), generator=actual_generator)
    expected = torch.sigmoid(
        torch.randn(4096, generator=expected_generator) * 0.8 - 0.8
    )
    assert torch.equal(actual, expected)
    assert bool(((actual > 0) & (actual < 1)).all())
    assert float(actual.mean()) == pytest.approx(0.33, abs=0.02)


@pytest.mark.parametrize(
    "config",
    [
        {"name": "unknown"},
        {"name": "logit_normal", "mean": float("nan"), "std": 0.8},
        {"name": "logit_normal", "mean": -0.8, "std": 0.0},
        {"name": "logit_normal", "mean": -0.8, "std": float("inf")},
    ],
)
def test_training_time_sampler_rejects_invalid_config(config) -> None:
    with pytest.raises(ValueError, match="training.time_sampling"):
        build_training_time_sampler(config)
```

Also assert `None`, `"uniform"`, and `{"name": "uniform"}` produce the existing seeded uniform formula.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_training_sampling.py -k 'time_sampler' -q
```

Expected: collection or assertion failures because the new module and builder do not exist.

- [ ] **Step 3: Implement the immutable sampler and compatibility wrapper**

Implement a frozen dataclass with validated `name`, `mean`, `std`, and uniform `eps`. Use `torch.rand(..., generator=generator)` for uniform and `torch.randn(..., generator=generator).mul(std).add(mean).sigmoid()` for logit-normal. Reject unknown keys so mistyped experiment settings cannot be ignored.

- [ ] **Step 4: Integrate one built sampler into both trainer paths**

Build the sampler once near the other `training_config` settings in `train_flow_matching`. Pass it into `_sample_training_batch` and `_train_learned_acceleration_step`; replace the hard-coded `sample_uniform_time` call with `time_sampler.sample(...)`. Keep the default uniform behavior when the key is absent.

- [ ] **Step 5: Add a trainer integration test**

Use a recording objective or path to capture `t` from a one-step run configured with `{"name": "logit_normal", "mean": -8.0, "std": 0.01}` and assert every observed time is below `0.01`. Run the same helper without the setting to prove the legacy path still runs.

- [ ] **Step 6: Run focused tests and Ruff**

```bash
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_training_sampling.py tests/test_training_runtime.py -q
PYTHONPATH=$PWD .conda/fm_lab/bin/ruff check fm_lab/training/time_sampling.py fm_lab/training/losses.py fm_lab/training/trainer.py tests/test_training_sampling.py
```

Expected: all selected tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 7: Commit Task 1**

```bash
git add fm_lab/training/time_sampling.py fm_lab/training/__init__.py fm_lab/training/losses.py fm_lab/training/trainer.py tests/test_training_sampling.py
git commit -m "Add configurable training time sampling"
```

---

### Task 2: Symmetric Endpoint Supervision Conversion

**Files:**
- Modify: `fm_lab/training/losses.py:218-336`
- Test: `tests/test_objectives.py`
- Test: `tests/test_continuous_long_tail.py`
- Test: `tests/test_path_prediction.py`

**Interfaces:**
- Consumes: existing `PathPredictionState.prediction(...).convert(...)`.
- Produces: a focused private helper that selects ground truth in model-output space: target→`x1`, source→`x0`, velocity→`target_velocity`.
- Preserves: exact fast path for velocity-output/velocity-loss without modifiers.

- [ ] **Step 1: Write the near-endpoint RED test**

Add a model that returns the supplied clean target and evaluate at `t=0.9999`:

```python
def test_target_output_velocity_loss_clamps_prediction_and_supervision_symmetrically() -> None:
    path = LinearPath()
    x0 = torch.tensor([[1.0, -1.0]])
    x1 = torch.tensor([[3.0, 1.0]])
    t = torch.tensor([0.9999])
    objective = FlowMatchingObjective(
        model_output="target", loss_space="velocity", min_denom=0.05
    )
    loss, _ = objective(model=ReturnsFixedTensor(x1), path=path, x0=x0, x1=x1, t=t)
    assert torch.equal(loss, torch.tensor(0.0))
```

Add an imperfect prediction test showing the velocity error is exactly
`(prediction - x1) / 0.05`, rather than division by `0.0001` or comparison with
the unclamped path velocity.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_objectives.py -k 'symmetrically or near_endpoint' -q
```

Expected: the perfect-prediction assertion fails because current supervision is the unclamped `x1 - x0` velocity.

- [ ] **Step 3: Implement symmetric supervision selection**

After OC target transfer is resolved, select the native ground-truth tensor using `self.model_output`, bind it with `state.prediction(native_target, self.model_output)`, and convert it to `self.loss_space`. Keep direct `target_velocity` only in the existing velocity/velocity/no-modifier fast path. Do not mutate `LinearPredictionState` or globally redefine path velocity.

- [ ] **Step 4: Add source and velocity preservation tests**

Assert source-output conversion is symmetric near `t=0`, and that a fixed velocity model obtains the identical velocity/velocity loss and gradient it had before this change.

- [ ] **Step 5: Add OC-transferred supervision coverage**

Construct a deterministic OC transfer at `t=0.9999` with `min_denom=0.05`. Assert a model returning the transferred target has zero base loss, while transfer metrics and modifier order remain unchanged.

- [ ] **Step 6: Run focused objective/modifier tests**

```bash
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_objectives.py tests/test_path_prediction.py tests/test_continuous_long_tail.py -q
PYTHONPATH=$PWD .conda/fm_lab/bin/ruff check fm_lab/training/losses.py tests/test_objectives.py tests/test_continuous_long_tail.py
```

Expected: all tests pass and Ruff is clean.

- [ ] **Step 7: Commit Task 2**

```bash
git add fm_lab/training/losses.py tests/test_objectives.py tests/test_path_prediction.py tests/test_continuous_long_tail.py
git commit -m "Clamp endpoint supervision symmetrically"
```

---

### Task 3: Migrate the Controlled Fashion-MNIST Matrix

**Files:**
- Modify: `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss.yaml`
- Modify: `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cbdm.yaml`
- Modify: `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_oc.yaml`
- Modify: `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm.yaml`
- Modify: `tests/test_config_smoke.py`
- Modify: `README.md`
- Modify: `docs/cli.md`
- Modify: `tests/test_docs_coverage.py`

**Interfaces:**
- Consumes: Task 1 `training.time_sampling` mapping and Task 2 symmetric loss.
- Produces: four otherwise-controlled JiT-style experiment configurations.

- [ ] **Step 1: Update config tests first**

Change the common-field assertions to require:

```python
assert config["objective"]["min_denom"] == 0.05
assert config["training"]["time_sampling"] == {
    "name": "logit_normal",
    "mean": -0.8,
    "std": 0.8,
}
assert config["training"]["warmup_steps"] == 500
assert config["training"]["ema_decay"] == 0.9999
```

Require every OC modifier to use `min_denom: 0.05`. Preserve exact modifier order and all controlled common fields.

- [ ] **Step 2: Run config tests and verify RED**

```bash
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_config_smoke.py -k 'fashion_mnist_lt' -q
```

Expected: failures show the old `0.001` floor and missing sampler/warmup/EMA settings.

- [ ] **Step 3: Migrate all four YAML files**

Apply identical objective/training fields to the four configs. Change only OC modifier floors where applicable; do not change CBDM/CM weights, capacity adapters, data, seed, model, solver, or sampling counts.

- [ ] **Step 4: Document the JiT-derived stabilization**

Add a concise note near the four commands explaining that the model predicts the clean target, trains in velocity space, samples logit-normal time with `(-0.8, 0.8)`, and symmetrically floors the denominator at `0.05`. Explicitly retain Euler/NFE-64 evaluation provenance.

- [ ] **Step 5: Run config and documentation tests**

```bash
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_config_smoke.py tests/test_docs_coverage.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Run a real short CPU baseline regression**

Use a new temporary directory and the real CLI:

```bash
PYTHONPATH=$PWD .conda/fm_lab/bin/python -m fm_lab.experiments.run_train \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss.yaml \
  --output-dir /tmp/fm_lab_jit_x_vloss_smoke \
  --device cpu --steps 8 --batch-size 16 --n-samples 20 \
  --n-trajectories 2 --nfe 2 --sample-batch-size 10 --plot-max-points 20
```

Assert `checkpoint.pt`, `samples/euler_nfe2.npy`, and balanced labels exist. Parse `diagnostics/training_history.csv`; require all losses finite and `max(loss) < 50`, which separates the stabilized run from the reproduced hundreds-to-thousands regression without asserting convergence in eight steps.

- [ ] **Step 7: Run complete verification**

```bash
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest -q
PYTHONPATH=$PWD .conda/fm_lab/bin/ruff check .
git diff --check
```

Expected: all tests pass, Ruff reports `All checks passed!`, and the diff check is empty.

- [ ] **Step 8: Commit Task 3**

```bash
git add configs/fashion_mnist_lt README.md docs/cli.md tests/test_config_smoke.py tests/test_docs_coverage.py
git commit -m "Stabilize Fashion-MNIST x-vloss experiments"
```

---

## Final Review Checklist

- [ ] Compare the implementation against `docs/superpowers/specs/2026-07-15-jit-x-vloss-stabilization-design.md`.
- [ ] Confirm the original failing baseline mechanism is covered by a RED/GREEN test.
- [ ] Confirm all four configs remain controlled except for modifier/capacity differences.
- [ ] Confirm exact-resume training contracts include the new sampler and optimizer settings.
- [ ] Confirm no generated smoke artifacts or `.npz`/`.npy` files are staged.
- [ ] Request whole-change code review before integration.

