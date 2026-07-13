# Local ImbDiff Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct short-run EMA and sampling defaults, expose timestep-specific undertraining without extra forwards, and save paired live-versus-EMA diffusion diagnostics.

**Architecture:** Keep paper-scale configs and the scalar optimization objective unchanged. Local configs opt into safer 10000-step settings and a bounded 100-sample comparison; the discrete objective derives bucket metrics from its existing per-sample error tensor, while the trainer uses identical initial noise to sample live and EMA weights through the existing DDIM implementation.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, Ruff, YAML configs, `.conda/fm_lab`.

## Global Constraints

- Change only the five configs under `configs/imbdiff/local/`; paper-scale configs remain byte-for-byte unchanged.
- Preserve scalar DDPM, x-vloss, CBDM, OC, and CM loss values and gradients.
- Add no model forward passes to timestep-resolved training diagnostics.
- Normal sampling continues to use EMA when EMA exists.
- Paired comparison is opt-in, capped by the normal `n_samples`, and uses identical labels and initial noise.
- Preserve the unrelated untracked `docs/experimental_ideas_retrospective.md` and exclude it from every commit.
- Run all verification through `.conda/fm_lab`; commit and push each verified nontrivial checkpoint.

---

### Task 1: Correct all five local experiment defaults

**Files:**
- Modify: `tests/test_config_smoke.py`
- Modify: `configs/imbdiff/local/cifar10_lt_ddpm_epsilon_local.yaml`
- Modify: `configs/imbdiff/local/cifar10_lt_x_vloss_local.yaml`
- Modify: `configs/imbdiff/local/cifar10_lt_cbdm_local.yaml`
- Modify: `configs/imbdiff/local/cifar10_lt_oc_local.yaml`
- Modify: `configs/imbdiff/local/cifar10_lt_cm_local.yaml`

**Interfaces:**
- Consumes: `load_config(path) -> dict`.
- Produces: one shared reliability policy across the five local methods.

- [ ] **Step 1: Change the existing local config test to require the new policy**

In `test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile`, replace the old assertions with:

```python
assert config["training"]["steps"] == 10000
assert config["training"]["ema_decay"] == 0.999
assert config["training"]["early_stopping"] == {
    "enabled": True,
    "patience_steps": 2000,
    "warmup_steps": 5000,
    "min_delta": 0.0001,
    "ema_alpha": 0.3,
}
assert config["sampling"]["ddim_skip"] == 20
assert config["sampling"]["classifier_free_guidance"]["scale"] == 1.0
assert config["sampling"]["classifier_free_guidance"]["paper_omega"] == 0.0
assert config["sampling"]["live_ema_comparison"] == {
    "enabled": True,
    "n_samples": 100,
}
```

- [ ] **Step 2: Run the focused test and verify the old values fail**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py::test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile
```

Expected: FAIL because `training.steps` is still 5000.

- [ ] **Step 3: Apply the exact local policy to all five YAML files**

Use:

```yaml
training:
  steps: 10000
  ema_decay: 0.999
  early_stopping:
    enabled: true
    patience_steps: 2000
    warmup_steps: 5000
    min_delta: 0.0001
    ema_alpha: 0.3
```

Keep optimizer warmup at 500. Replace the sampling values and add:

```yaml
sampling:
  ddim_skip: 20
  classifier_free_guidance:
    enabled: true
    convention: fm_lab
    scale: 1.0
    paper_omega: 0.0
  live_ema_comparison:
    enabled: true
    n_samples: 100
```

- [ ] **Step 4: Verify and publish the config checkpoint**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_config_smoke.py
.conda/fm_lab/bin/ruff check tests/test_config_smoke.py
git diff --check
git add tests/test_config_smoke.py configs/imbdiff/local
git commit -m "Correct local diffusion training defaults"
git push
```

Expected: config tests and Ruff pass, then one six-file commit is pushed.

---

### Task 2: Add zero-extra-forward timestep loss diagnostics

**Files:**
- Modify: `tests/test_discrete_objective.py`
- Modify: `fm_lab/training/discrete_objective.py`

**Interfaces:**
- Consumes: prediction and target tensors already computed by `DiscreteDiffusionObjective.__call__`.
- Produces: unchanged scalar loss plus six diagnostic metrics when `compute_diagnostics=True`.

- [ ] **Step 1: Write a failing exact bucket test**

Add:

```python
def test_discrete_objective_logs_timestep_resolved_loss_without_changing_total() -> None:
    objective = build_objective(
        {"name": "discrete_diffusion", "prediction_type": "epsilon"},
        diffusion_config={"timesteps": 1000},
    )
    model = FixedConditionalPrediction(torch.zeros(1, 2))
    noise_levels = torch.arange(1.0, 7.0)
    noise = noise_levels[:, None].expand(6, 2)

    loss, metrics = objective(
        model=model,
        path=None,
        x0=noise,
        x1=torch.zeros_like(noise),
        t=torch.tensor([0, 100, 334, 500, 667, 999]),
        class_labels=torch.zeros(6, dtype=torch.long),
    )

    assert torch.allclose(loss, noise_levels.square().mean())
    assert metrics["diffusion_loss_low_noise"] == pytest.approx(2.5)
    assert metrics["diffusion_loss_low_noise_count"] == 2.0
    assert metrics["diffusion_loss_mid_noise"] == pytest.approx(12.5)
    assert metrics["diffusion_loss_mid_noise_count"] == 2.0
    assert metrics["diffusion_loss_high_noise"] == pytest.approx(30.5)
    assert metrics["diffusion_loss_high_noise_count"] == 2.0
```

Add a second assertion to an existing objective test that diagnostics are absent
when disabled:

```python
_, metrics = objective(
    model=model,
    path=None,
    x0=x0,
    x1=x1,
    t=t,
    class_labels=class_labels,
    compute_diagnostics=False,
)
assert not any(key.startswith("diffusion_loss_") for key in metrics)
```

- [ ] **Step 2: Run the tests and verify missing metric keys fail**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_discrete_objective.py -k "timestep_resolved or samples_integer"
```

Expected: the new test fails with `KeyError: 'diffusion_loss_low_noise'`.

- [ ] **Step 3: Preserve per-sample diffusion error and derive bucket metrics**

Replace scalar-only `_diffusion_loss` with `_diffusion_loss_per_sample`:

```python
def _diffusion_loss_per_sample(
    self,
    *,
    prediction: torch.Tensor,
    predicted_epsilon: torch.Tensor,
    target_clean: torch.Tensor,
    target_noise: torch.Tensor,
    discrete_t: torch.Tensor,
) -> torch.Tensor:
    if self.prediction_type == "epsilon":
        squared_error = (prediction - target_noise).square()
    else:
        predicted_velocity = self.diffusion.velocity_target(
            prediction, predicted_epsilon, discrete_t
        )
        target_velocity = self.diffusion.velocity_target(
            target_clean, target_noise, discrete_t
        )
        squared_error = (predicted_velocity - target_velocity).square()
    return squared_error.flatten(1).mean(1)
```

In `__call__`, compute `diffusion_loss_per_sample` once and set
`diffusion_loss = diffusion_loss_per_sample.mean()`. When diagnostics are enabled,
call:

```python
def _timestep_loss_metrics(
    self,
    per_sample_loss: torch.Tensor,
    discrete_t: torch.Tensor,
) -> dict[str, float]:
    first = self.diffusion.timesteps // 3
    second = 2 * self.diffusion.timesteps // 3
    masks = {
        "low_noise": discrete_t < first,
        "mid_noise": (discrete_t >= first) & (discrete_t < second),
        "high_noise": discrete_t >= second,
    }
    metrics = {}
    for name, mask in masks.items():
        count = int(mask.sum().item())
        metrics[f"diffusion_loss_{name}"] = (
            float(per_sample_loss[mask].detach().mean()) if count else 0.0
        )
        metrics[f"diffusion_loss_{name}_count"] = float(count)
    return metrics
```

- [ ] **Step 4: Verify all discrete objectives and publish**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_discrete_objective.py
.conda/fm_lab/bin/ruff check fm_lab/training/discrete_objective.py tests/test_discrete_objective.py
git diff --check
git add fm_lab/training/discrete_objective.py tests/test_discrete_objective.py
git commit -m "Log diffusion loss by timestep range"
git push
```

Expected: all discrete-objective tests pass with unchanged exact scalar-loss tests.

---

### Task 3: Save paired live-versus-EMA diagnostic samples

**Files:**
- Modify: `tests/test_discrete_sampling.py`
- Modify: `fm_lab/training/trainer.py`

**Interfaces:**
- Consumes: live model, optional EMA model, the normal discrete sampler settings, and `sampling.live_ema_comparison`.
- Produces: normal EMA samples plus paired diagnostic arrays, a comparison plot, and nested metrics.

- [ ] **Step 1: Write failing end-to-end comparison tests**

Extend `test_discrete_training_smoke_writes_generated_samples` with:

```python
"live_ema_comparison": {"enabled": True, "n_samples": 2},
```

and assert:

```python
comparison = metrics["sampling"]["live_ema_comparison"]
assert comparison["n_samples"] == 2
assert (tmp_path / "samples" / "live_diagnostic.npy").exists()
assert (tmp_path / "samples" / "ema_diagnostic.npy").exists()
assert (tmp_path / "plots" / "live_vs_ema.png").exists()
```

Add a separate test with training learning rate `0.0`, identical live/EMA
weights, DDIM eta `0.0`, and comparison count 2. Load both arrays with NumPy and
assert `np.array_equal(live, ema)`, proving identical labels and initial noise.

Add a third test that enables comparison without `training.ema_decay` and
expects `ValueError` matching `live_ema_comparison requires EMA`.

- [ ] **Step 2: Run focused sampling tests and verify missing artifacts fail**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_discrete_sampling.py -k "training_smoke or live_ema"
```

Expected: FAIL because `live_ema_comparison` is absent from metrics and no
diagnostic files exist.

- [ ] **Step 3: Pass both model states into the discrete sampling branch**

Change the discrete call in `train_flow_matching` from a preselected EMA model
to:

```python
sample_artifacts = sample_discrete_and_plot(
    config=config,
    run_dir=run_dir,
    target=target,
    source=source,
    model=model,
    ema_model=ema_model,
    objective=objective,
    device=device,
)
```

Add `ema_model: nn.Module | None = None` to `sample_discrete_and_plot` and use
`sampling_model = ema_model if ema_model is not None else model` for normal
generation.

- [ ] **Step 4: Implement bounded paired diagnostic generation**

Parse the optional mapping, validate positive count, cap it with
`min(configured_count, n_samples)`, and require EMA when enabled. For every
comparison chunk, create one `initial_noise = torch.randn(...)`, then call
`sample_discrete_diffusion` for live and EMA with that same tensor, labels,
sampler, prediction type, guidance, skip, and eta.

Concatenate and save:

```python
np.save(samples_dir / "live_diagnostic.npy", live_generated.numpy())
np.save(samples_dir / "ema_diagnostic.npy", ema_generated.numpy())
plot_generated_samples(
    target_samples.cpu(),
    {"live": live_generated, "ema": ema_generated},
    run_dir / "plots" / "live_vs_ema.png",
    max_points=comparison_n_samples,
    image_shape=target_metadata.get("image_shape"),
    image_value_range=target_metadata.get("image_value_range", (-1.0, 1.0)),
)
```

Return:

```python
result["live_ema_comparison"] = {
    "n_samples": comparison_n_samples,
    "live_samples_path": str(samples_dir / "live_diagnostic.npy"),
    "ema_samples_path": str(samples_dir / "ema_diagnostic.npy"),
    "plot_path": str(run_dir / "plots" / "live_vs_ema.png"),
}
```

- [ ] **Step 5: Verify sampling behavior and publish**

Run:

```bash
.conda/fm_lab/bin/pytest -q tests/test_discrete_sampling.py tests/test_training_sampling.py
.conda/fm_lab/bin/ruff check fm_lab/training/trainer.py tests/test_discrete_sampling.py
git diff --check
git add fm_lab/training/trainer.py tests/test_discrete_sampling.py
git commit -m "Compare live and EMA diffusion samples"
git push
```

Expected: all sampling tests pass, including byte-identical arrays for equal
weights and the missing-EMA validation.

---

### Task 4: Full verification and real local DDPM smoke

**Files:**
- Generate locally: `runs/verification/local_ddpm_reliability/`

**Interfaces:**
- Consumes: the corrected local DDPM config through `fm-lab-train`.
- Produces: a checkpoint, normal samples, timestep diagnostics, and paired live/EMA artifacts.

- [ ] **Step 1: Run full Conda verification**

Run:

```bash
.conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check .
.conda/fm_lab/bin/python -m pip check
```

Expected: zero failures, `All checks passed!`, and no broken requirements.

- [ ] **Step 2: Run the bounded CPU smoke**

Run:

```bash
.conda/fm_lab/bin/fm-lab-train \
  --config configs/imbdiff/local/cifar10_lt_ddpm_epsilon_local.yaml \
  --device cpu \
  --steps 1 \
  --batch-size 2 \
  --n-samples 2 \
  --sample-batch-size 2 \
  --plot-max-points 2 \
  --output-dir runs/verification/local_ddpm_reliability
```

Expected: successful training and normal plus two-sample comparison generation.

- [ ] **Step 3: Validate smoke evidence and final branch state**

Run:

```bash
.conda/fm_lab/bin/python -c 'import csv, json; r="runs/verification/local_ddpm_reliability"; m=json.load(open(f"{r}/metrics.json")); assert m["sampling"]["live_ema_comparison"]["n_samples"] == 2; fields=set(next(csv.DictReader(open(f"{r}/diagnostics/training_history.csv"))).keys()); expected={"diffusion_loss_low_noise", "diffusion_loss_low_noise_count", "diffusion_loss_mid_noise", "diffusion_loss_mid_noise_count", "diffusion_loss_high_noise", "diffusion_loss_high_noise_count"}; assert expected <= fields'
test -f runs/verification/local_ddpm_reliability/samples/ddim.npy
test -f runs/verification/local_ddpm_reliability/samples/live_diagnostic.npy
test -f runs/verification/local_ddpm_reliability/samples/ema_diagnostic.npy
test -f runs/verification/local_ddpm_reliability/plots/live_vs_ema.png
git status --short --branch
```

Expected: all artifact checks pass; only the pre-existing untracked retrospective document appears in status.
