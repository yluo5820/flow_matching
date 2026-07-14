# Continuous Long-Tail Objectives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace discrete CBDM/OC/CM training and DDPM/DDIM sampling with path-aware prediction objects, continuous linear-interpolant objectives, ODE sampling, and four controlled Fashion-MNIST IR100 experiments.

**Architecture:** An immutable `PathPrediction` delegates source/target/velocity conversions to a `PathPredictionState` supplied by a conversion-capable path. `FlowMatchingObjective` independently declares model-output and loss representations, then composes continuous CBDM, OC, and CM modifiers around the same per-sample loss. The existing ODE sampler converts any supported model output to flow velocity; the discrete process, sampler, configs, and CLI flags are removed after migration.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, PyYAML, pytest, Ruff, existing `fm_lab` path/model/trainer abstractions, `.conda/fm_lab/bin` runtime.

## Global Constraints

- Use continuous time `t in [0, 1]`; do not introduce integer training timesteps.
- Initially support conversion only for the existing linear conditional interpolant.
- Canonical prediction kinds are exactly `source`, `target`, and `velocity`; aliases are accepted only at config/CLI boundaries.
- `velocity` always means flow-matching velocity; do not add diffusion-v prediction.
- Keep `FlowPath.sample_xt` and `FlowPath.target_velocity` sufficient for existing velocity-only custom paths.
- Do not retain a discrete training or sampling compatibility branch.
- Do not silently load or reinterpret discrete checkpoints as continuous checkpoints.
- Use `.conda/fm_lab/bin/python`, `.conda/fm_lab/bin/pytest`, and `.conda/fm_lab/bin/ruff` for every command.
- Follow test-driven development: observe each named test fail before implementing its production change.

---

## File Structure

### New files

- `fm_lab/paths/prediction.py` — canonical prediction kinds, protocols, immutable prediction value object, and shared validation.
- `fm_lab/training/long_tail.py` — continuous modifier configs and CBDM/OC/CM implementations; no path algebra.
- `tests/test_path_prediction.py` — conversion identities, broadcasting, endpoint behavior, aliases, gradients, and unsupported-path failures.
- `tests/test_continuous_long_tail.py` — modifier validation, losses, gradients, time conventions, and composition tests.
- `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss.yaml` — controlled continuous target-output/velocity-loss baseline.
- `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cbdm.yaml` — baseline plus CBDM.
- `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_oc.yaml` — baseline plus OC.
- `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss_cm.yaml` — baseline plus OC and CM.

### Modified files

- `fm_lab/paths/base.py` — expose a type-safe optional conversion-capable path protocol without changing `FlowPath` requirements.
- `fm_lab/paths/linear.py` — construct `LinearPredictionState` objects.
- `fm_lab/paths/__init__.py` — export prediction types and protocols.
- `fm_lab/training/prediction.py` — replace x-only helpers with generic output normalization and a velocity-view wrapper.
- `fm_lab/training/losses.py` — make output/loss spaces independent, build modifiers, and remove the discrete factory branch.
- `fm_lab/training/trainer.py` — use generic ODE sampling for all continuous outputs and delete discrete sampling orchestration.
- `fm_lab/training/__init__.py` — export the supported continuous APIs only.
- `fm_lab/experiments/run_train.py` — rename prediction CLI flags and reject discrete-only arguments.
- `fm_lab/experiments/factory.py` — remove configuration dependence on finite diffusion timesteps where no longer needed.
- `tests/test_objectives.py` — migrate x-prediction tests to canonical target/velocity terminology.
- `tests/test_training_sampling.py` — verify output conversion and balanced conditional ODE artifacts.
- `tests/test_run_train_cli.py` — test the new CLI override schema and absence of discrete flags.
- `tests/test_capacity_manipulation.py` — exercise CM through the continuous objective.
- `tests/test_config_smoke.py` — validate the four Fashion-MNIST configs and reject discrete-only shipped keys.
- `README.md` and `docs/cli.md` — document the four-run training/evaluation workflow.
- `pyproject.toml` — only if console-script or packaging coverage changes are required by file deletion.

### Deleted files

- `fm_lab/training/discrete_objective.py`
- `fm_lab/diffusion/discrete.py`
- `fm_lab/diffusion/sampling.py`
- `fm_lab/diffusion/__init__.py` if the package has no remaining callers
- `tests/test_discrete_objective.py`
- `tests/test_discrete_sampling.py`
- `configs/imbdiff/cifar10_lt_x_vloss.yaml`
- `configs/imbdiff/cifar10_lt_cbdm.yaml`
- `configs/imbdiff/cifar10_lt_oc.yaml`
- `configs/imbdiff/cifar10_lt_cm.yaml`
- `configs/imbdiff/cifar100_lt_cbdm.yaml`
- `configs/imbdiff/cifar100_lt_oc.yaml`
- `configs/imbdiff/cifar100_lt_cm.yaml`
- `configs/imbdiff/local/cifar10_lt_x_vloss_local.yaml`
- `configs/imbdiff/local/cifar10_lt_cbdm_local.yaml`
- `configs/imbdiff/local/cifar10_lt_oc_local.yaml`
- `configs/imbdiff/local/cifar10_lt_cm_local.yaml`
- `docs/imbdiff_cm_reproduction.md`

Historical design and plan documents remain historical records; they are not runtime compatibility promises.

---

### Task 1: Path-Aware Prediction Value Objects

**Files:**
- Create: `fm_lab/paths/prediction.py`
- Create: `tests/test_path_prediction.py`
- Modify: `fm_lab/paths/base.py`
- Modify: `fm_lab/paths/linear.py`
- Modify: `fm_lab/paths/__init__.py`

**Interfaces:**
- Produces: `PredictionKind`, `normalize_prediction_kind(value: str | PredictionKind) -> PredictionKind`, `PathPredictionState`, `PathPrediction`, `ConvertibleFlowPath`, and `LinearPath.prediction_state(xt, t, *, min_denom=1e-3)`.
- Consumes: existing `expand_time`, `FlowPath`, and `LinearPath` conventions.

- [ ] **Step 1: Write failing normalization and conversion tests**

Add tests that define the public behavior, including aliases and all direct linear identities:

```python
import pytest
import torch

from fm_lab.paths import LinearPath, PredictionKind, normalize_prediction_kind


def test_prediction_kind_aliases_normalize_to_canonical_values() -> None:
    assert normalize_prediction_kind("epsilon") is PredictionKind.SOURCE
    assert normalize_prediction_kind("x") is PredictionKind.TARGET
    assert normalize_prediction_kind("v") is PredictionKind.VELOCITY
    with pytest.raises(ValueError, match="source, target, or velocity"):
        normalize_prediction_kind("score")


def test_linear_prediction_converts_target_to_source_and_velocity() -> None:
    path = LinearPath()
    source = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    target = torch.tensor([[3.0, 1.0], [-0.5, 1.25]])
    t = torch.tensor([0.25, 0.75])
    xt = path.sample_xt(source, target, t)
    prediction = path.prediction_state(xt, t).prediction(
        target, PredictionKind.TARGET
    )

    assert torch.allclose(prediction.as_source(), source)
    assert torch.allclose(prediction.as_target(), target)
    assert torch.allclose(prediction.as_velocity(), target - source)
```

- [ ] **Step 2: Run the tests and verify the missing API failure**

Run:

```bash
.conda/fm_lab/bin/pytest tests/test_path_prediction.py -q
```

Expected: collection fails because `PredictionKind` and `LinearPath.prediction_state` do not exist.

- [ ] **Step 3: Implement canonical kinds and immutable prediction delegation**

Create the core public types in `fm_lab/paths/prediction.py`:

```python
class PredictionKind(str, Enum):
    SOURCE = "source"
    TARGET = "target"
    VELOCITY = "velocity"


def normalize_prediction_kind(value: str | PredictionKind) -> PredictionKind:
    if isinstance(value, PredictionKind):
        return value
    aliases = {
        "source": PredictionKind.SOURCE,
        "epsilon": PredictionKind.SOURCE,
        "noise": PredictionKind.SOURCE,
        "target": PredictionKind.TARGET,
        "x": PredictionKind.TARGET,
        "x1": PredictionKind.TARGET,
        "clean": PredictionKind.TARGET,
        "velocity": PredictionKind.VELOCITY,
        "v": PredictionKind.VELOCITY,
        "field": PredictionKind.VELOCITY,
    }
    try:
        return aliases[str(value).lower()]
    except KeyError as exc:
        raise ValueError("prediction kind must be source, target, or velocity") from exc


@dataclass(frozen=True)
class PathPrediction:
    value: torch.Tensor
    kind: PredictionKind
    state: PathPredictionState

    def convert(self, kind: str | PredictionKind) -> torch.Tensor:
        return self.state.convert(self.value, self.kind, normalize_prediction_kind(kind))

    def as_source(self) -> torch.Tensor:
        return self.convert(PredictionKind.SOURCE)

    def as_target(self) -> torch.Tensor:
        return self.convert(PredictionKind.TARGET)

    def as_velocity(self) -> torch.Tensor:
        return self.convert(PredictionKind.VELOCITY)
```

Define `PathPredictionState` and `ConvertibleFlowPath` as runtime-checkable protocols. Do not add conversion methods to the minimal `FlowPath` protocol.

- [ ] **Step 4: Implement `LinearPredictionState` using direct formulas**

In `fm_lab/paths/linear.py`, add a frozen state whose `convert` handles each supported source kind directly. Validate shapes and positive `min_denom`; use `expand_time` for broadcasting:

```python
@dataclass(frozen=True)
class LinearPredictionState:
    xt: torch.Tensor
    t: torch.Tensor
    min_denom: float = 1e-3

    @property
    def supported_kinds(self) -> frozenset[PredictionKind]:
        return frozenset(PredictionKind)

    def prediction(self, value, kind) -> PathPrediction:
        normalized = normalize_prediction_kind(kind)
        if value.shape != self.xt.shape:
            raise ValueError("prediction value must match xt shape")
        return PathPrediction(value=value, kind=normalized, state=self)

    def convert(self, value, source_kind, target_kind) -> torch.Tensor:
        if source_kind is target_kind:
            return value
        t = expand_time(self.t, self.xt)
        if source_kind is PredictionKind.VELOCITY:
            return self.xt - t * value if target_kind is PredictionKind.SOURCE else self.xt + (1 - t) * value
        if source_kind is PredictionKind.TARGET:
            velocity = (value - self.xt) / (1 - t).clamp_min(self.min_denom)
        else:
            velocity = (self.xt - value) / t.clamp_min(self.min_denom)
        if target_kind is PredictionKind.VELOCITY:
            return velocity
        return self.xt - t * velocity if target_kind is PredictionKind.SOURCE else self.xt + (1 - t) * velocity
```

Add `LinearPath.prediction_state`. Export all public types from `fm_lab/paths/__init__.py`.

- [ ] **Step 5: Add broadcasting, endpoint, gradient, and unsupported-path tests**

Add tests that use image-shaped tensors, `t=[0, 1]`, and `requires_grad=True`. Assert finite endpoint results, exact velocity-to-endpoint conversions, non-null gradients, and a clear failure when a velocity-only custom path is asked for `prediction_state`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.conda/fm_lab/bin/pytest tests/test_path_prediction.py tests/test_objectives.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit the prediction abstraction**

```bash
git add fm_lab/paths tests/test_path_prediction.py tests/test_objectives.py
git commit -m "Add path-aware prediction conversions"
```

---

### Task 2: Independent Model-Output and Loss Spaces

**Files:**
- Modify: `fm_lab/training/prediction.py`
- Modify: `fm_lab/training/losses.py`
- Modify: `tests/test_objectives.py`
- Modify: `tests/test_training_sampling.py`

**Interfaces:**
- Consumes: `PredictionKind`, `PathPrediction`, and `ConvertibleFlowPath` from Task 1.
- Produces: `FlowMatchingObjective(model_output, loss_space, min_denom)`, `VelocityFromPrediction`, and `velocity_model_for_objective` supporting all canonical output kinds. Task 3 adds modifier composition after defining its protocol.

- [ ] **Step 1: Write failing objective-space tests**

Add tests proving output and loss spaces are independent:

```python
def test_target_output_velocity_loss_matches_analytical_linear_velocity() -> None:
    source = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[3.0, 1.0]])
    t = torch.tensor([0.25])
    model = FixedPrediction(target)
    objective = build_objective({
        "name": "flow_matching",
        "model_output": "target",
        "loss_space": "velocity",
        "min_denom": 1e-3,
    })

    loss, metrics = objective(
        model=model,
        path=LinearPath(),
        x0=source,
        x1=target,
        t=t,
    )

    assert torch.allclose(loss, torch.zeros_like(loss))
    assert metrics["model_output"] == "target"
    assert metrics["loss_space"] == "velocity"
```

Also test source-output/target-loss, alias normalization, unsupported conversion-capable paths, and metadata.

- [ ] **Step 2: Run the focused objective tests and verify failure**

```bash
.conda/fm_lab/bin/pytest tests/test_objectives.py -q
```

Expected: failures show that `loss_space` and canonical `target` output are not yet supported.

- [ ] **Step 3: Refactor `FlowMatchingObjective` around prediction objects**

Replace `x_prediction_loss_space` and `x_prediction_min_denom` with canonical fields:

```python
@dataclass
class FlowMatchingObjective:
    loss: str = "mse"
    model_output: str = "velocity"
    loss_space: str = "velocity"
    min_denom: float = 1e-3

    def __post_init__(self) -> None:
        self.model_output = normalize_prediction_kind(self.model_output).value
        self.loss_space = normalize_prediction_kind(self.loss_space).value
        if self.min_denom <= 0:
            raise ValueError("objective.min_denom must be positive")
```

In `_loss`, obtain `xt`, create `state = path.prediction_state(xt, t, min_denom=...)` when conversion is needed, wrap model output and the ground-truth representation, convert both to `loss_space`, and compute a per-sample MSE before reduction. Velocity-output/velocity-loss must continue to work for any minimal `FlowPath` without `prediction_state`.

- [ ] **Step 4: Replace the x-only sampling wrapper**

Refactor `VelocityFromXPrediction` into `VelocityFromPrediction`. Its `forward` evaluates the wrapped model, asks the path for a prediction state at the current `(x, t)`, wraps the result using `objective.model_output`, and returns `.as_velocity()`. Preserve `requires_source_label`, `is_class_conditional`, and capacity context forwarding.

- [ ] **Step 5: Run objective and sampling tests**

```bash
.conda/fm_lab/bin/pytest tests/test_objectives.py tests/test_training_sampling.py tests/test_class_conditioning.py -q
```

Expected: all tests pass, including existing velocity-only path behavior.

- [ ] **Step 6: Commit independent prediction/loss spaces**

```bash
git add fm_lab/training/prediction.py fm_lab/training/losses.py tests/test_objectives.py tests/test_training_sampling.py
git commit -m "Separate model output and loss spaces"
```

---

### Task 3: Continuous Modifier Framework and CBDM

**Files:**
- Create: `fm_lab/training/long_tail.py`
- Create: `tests/test_continuous_long_tail.py`
- Modify: `fm_lab/training/losses.py`
- Modify: `fm_lab/training/__init__.py`

**Interfaces:**
- Consumes: per-sample base losses and `PathPrediction` from Task 2.
- Produces: `ContinuousObjectiveContext`, `ContinuousObjectiveModifier`, `CBDMModifier`, and `build_continuous_modifiers(configs, class_counts)`.

- [ ] **Step 1: Write failing CBDM construction and gradient tests**

Cover auxiliary distributions, continuous high-noise weighting, both stop-gradient directions, and missing class counts:

```python
def test_cbdm_weight_is_one_minus_continuous_time() -> None:
    modifier = CBDMModifier(
        class_counts=[1, 9],
        target_distribution="uniform",
        tau=2.0,
        gamma=0.25,
        comparison_space="velocity",
    )
    assert torch.equal(
        modifier.time_weight(torch.tensor([0.0, 0.25, 1.0])),
        torch.tensor([1.0, 0.75, 0.0]),
    )
```

Add an end-to-end objective test with `modifiers: [{name: cbdm, ...}]` and a deterministic label-table model.

- [ ] **Step 2: Run the new tests and verify missing modifier APIs**

```bash
.conda/fm_lab/bin/pytest tests/test_continuous_long_tail.py -q
```

Expected: collection or construction fails because the continuous modifier module does not exist.

- [ ] **Step 3: Implement the modifier context and builder**

Define a context carrying `model`, `path`, `state`, `xt`, `t`, observed and original labels, base prediction, source, target, and per-sample base loss. Define a modifier protocol returning `(loss, metrics)` without changing path algebra.

The builder accepts a list of mappings, rejects duplicate names, validates class counts, and constructs modifiers in declared order. Unknown names list `cbdm`, `oc`, and `cm` as supported values.

- [ ] **Step 4: Implement continuous CBDM**

Port auxiliary-label sampling and stop-gradient behavior from the discrete objective, but:

- evaluate the auxiliary model at the original continuous `t`;
- wrap both model outputs through the shared path state;
- convert them to `comparison_space`;
- use `tau * (1 - t)` as the per-sample continuous weight;
- preserve `train`, `sqrt`, and `uniform` distributions;
- emit `cbdm.regularizer`, `cbdm.commitment`, and `cbdm.auxiliary_distribution` metadata.

- [ ] **Step 5: Integrate modifiers into the base objective**

`FlowMatchingObjective` reduces `base_loss_per_sample.mean()` and then adds each modifier loss. Metrics use `base.*`, `cbdm.*`, `oc.*`, and `cm.*` namespaces. `build_objective` passes target class counts to `build_continuous_modifiers`.

- [ ] **Step 6: Run CBDM and base-objective regression tests**

```bash
.conda/fm_lab/bin/pytest tests/test_continuous_long_tail.py tests/test_objectives.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit continuous CBDM**

```bash
git add fm_lab/training/long_tail.py fm_lab/training/losses.py fm_lab/training/__init__.py tests/test_continuous_long_tail.py
git commit -m "Add continuous CBDM modifier"
```

---

### Task 4: Continuous OC Target Transfer

**Files:**
- Modify: `fm_lab/training/long_tail.py`
- Modify: `fm_lab/training/losses.py`
- Modify: `tests/test_continuous_long_tail.py`

**Interfaces:**
- Consumes: `ContinuousObjectiveContext` and linear path state.
- Produces: `OCModifier.transfer_targets(context) -> TransferredTargets` and OC diagnostic metrics.

- [ ] **Step 1: Write failing OC tests**

Test each transfer mode, continuous cutoff direction, and finite endpoint weighting:

```python
def test_oc_cut_t_uses_noisy_source_to_clean_target_direction() -> None:
    modifier = OCModifier(
        class_counts=[100, 10],
        transfer_mode="t2h",
        cut_t=0.6,
    )
    accepted = modifier.apply_cutoff(
        torch.tensor([True, True, True]),
        torch.tensor([0.2, 0.6, 0.9]),
    )
    assert torch.equal(accepted, torch.tensor([False, True, True]))
```

Add deterministic tests showing `t2h` never transfers a head example to a rarer class, `h2t` does the inverse, and `full` accepts any selected reference.

- [ ] **Step 2: Run OC tests and verify failure**

```bash
.conda/fm_lab/bin/pytest tests/test_continuous_long_tail.py -k oc -q
```

Expected: failures show `OCModifier` and `cut_t` behavior are absent.

- [ ] **Step 3: Port reference selection to continuous linear geometry**

Implement reference logits from linear conditional noise-to-signal geometry:

```python
t_safe = t.clamp_min(min_denom)
noise_to_signal_sq = ((1.0 - t) / t_safe).square()
logits = -squared_distance / (2.0 * noise_to_signal_sq[:, None].clamp_min(min_denom))
```

Keep the operation under `torch.no_grad()`, stabilize logits by subtracting the row maximum, and preserve identity indices when transfer is rejected.

- [ ] **Step 4: Apply transferred endpoints before base-target conversion**

Extend objective preparation so OC may replace the clean target and corresponding source target used for supervision while leaving the actual `xt` input unchanged. Construct the ground-truth tensor directly from the transferred pair: source loss uses transferred source, target loss uses transferred target, and velocity loss uses `transferred_target - transferred_source`. Convert only the model output through the prediction state. This avoids falsely assuming that an OC-transferred endpoint pair reconstructs the original `xt`.

Emit transfer rate overall and in continuous thirds: `oc.transfer_rate`, `oc.transfer_rate.noisy`, `oc.transfer_rate.middle`, and `oc.transfer_rate.clean`.

- [ ] **Step 5: Run OC, CBDM, and objective tests**

```bash
.conda/fm_lab/bin/pytest tests/test_continuous_long_tail.py tests/test_objectives.py -q
```

Expected: all tests pass without changing CBDM behavior.

- [ ] **Step 6: Commit continuous OC**

```bash
git add fm_lab/training/long_tail.py fm_lab/training/losses.py tests/test_continuous_long_tail.py
git commit -m "Add continuous OC target transfer"
```

---

### Task 5: Continuous Capacity Manipulation

**Files:**
- Modify: `fm_lab/training/long_tail.py`
- Modify: `tests/test_continuous_long_tail.py`
- Modify: `tests/test_capacity_manipulation.py`

**Interfaces:**
- Consumes: OC composition, path prediction state, class counts, and capacity-aware `model_prediction(..., use_capacity=...)`.
- Produces: `CMModifier` comparing capacity-on/off predictions in a configured canonical space.

- [ ] **Step 1: Write failing CM validation and loss tests**

Add tests that reject CM without OC, reject a model without enabled capacity metadata, and verify comparison-space conversion:

```python
def test_cm_requires_oc_modifier_before_training() -> None:
    with pytest.raises(ValueError, match="CM requires OC"):
        build_objective(
            {
                "name": "flow_matching",
                "model_output": "target",
                "loss_space": "velocity",
                "modifiers": [{"name": "cm"}],
            },
            class_counts=[100, 10],
        )
```

Use a capacity-table model to assert consistency and diversity gradients reach the intended full/base branches.

- [ ] **Step 2: Run CM tests and verify failure**

```bash
.conda/fm_lab/bin/pytest tests/test_continuous_long_tail.py tests/test_capacity_manipulation.py -k cm -q
```

Expected: failures show the old discrete CM path or missing continuous modifier.

- [ ] **Step 3: Implement CM on prediction objects**

Evaluate capacity-on and capacity-off branches at identical `(xt, t, labels)`. Wrap both results using the same path state, convert both to `comparison_space`, and calculate class-weighted consistency/diversity distances. Preserve the existing normalized class and inverse-class probability definitions.

Validate modifier ordering during construction: exactly one OC must precede CM. Validate capacity metadata on the first objective call so the actual model is available. Emit `cm.consistency`, `cm.diversity`, and `cm.loss`.

- [ ] **Step 4: Migrate capacity tests to continuous times**

Replace integer tensors such as `torch.tensor([100, 900])` with floats in `[0,1]`, use `LinearPath`, and construct the flow-matching objective with OC and CM modifiers. Remove `diffusion.timesteps` from model-test configs where it is no longer consumed.

- [ ] **Step 5: Run modifier and capacity suites**

```bash
.conda/fm_lab/bin/pytest tests/test_continuous_long_tail.py tests/test_capacity_manipulation.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit continuous CM**

```bash
git add fm_lab/training/long_tail.py tests/test_continuous_long_tail.py tests/test_capacity_manipulation.py
git commit -m "Add continuous capacity manipulation"
```

---

### Task 6: Unified ODE Sampling and Discrete-Code Removal

**Files:**
- Modify: `fm_lab/training/trainer.py`
- Modify: `fm_lab/training/prediction.py`
- Modify: `fm_lab/training/losses.py`
- Modify: `fm_lab/training/__init__.py`
- Modify: `tests/test_training_sampling.py`
- Delete: `fm_lab/training/discrete_objective.py`
- Delete: `fm_lab/diffusion/discrete.py`
- Delete: `fm_lab/diffusion/sampling.py`
- Delete: `fm_lab/diffusion/__init__.py` if empty
- Delete: `tests/test_discrete_objective.py`
- Delete: `tests/test_discrete_sampling.py`

**Interfaces:**
- Consumes: `VelocityFromPrediction` and continuous modifiers.
- Produces: one `sample_and_plot` ODE path for velocity, source, and target model outputs.

- [ ] **Step 1: Write failing generic ODE sampling tests**

Add a target-output model whose analytical endpoint is known. Verify every solver evaluation receives converted velocity, generated labels are balanced over requested classes, and sampling metadata records output kind, path, `min_denom`, solver, NFE, guidance, and seed.

Add a checkpoint metadata test that rejects a checkpoint declaring `objective.name: discrete_diffusion` with an error containing `discrete checkpoints are incompatible`.

- [ ] **Step 2: Run sampling tests and verify failure**

```bash
.conda/fm_lab/bin/pytest tests/test_training_sampling.py -q
```

Expected: metadata assertions or the explicit checkpoint-compatibility assertion fail.

- [ ] **Step 3: Route every supported output through `sample_and_plot`**

Remove the objective-type branch that invokes `sample_discrete_and_plot`. Always build the velocity view with `velocity_model_for_objective` when the objective output is not velocity. Preserve EMA, classifier-free guidance, batched final integration, trajectories, balanced class labels, and existing artifact filenames such as `euler_nfe64.npy`.

- [ ] **Step 4: Add explicit checkpoint schema validation**

Before loading model weights for resume or sample-checkpoint workflows, compare serialized path name, objective name, model output, and loss space to the active config. Reject discrete objective names and missing continuous prediction metadata rather than guessing.

- [ ] **Step 5: Delete discrete production and test files**

Remove the finite-step diffusion package and `DiscreteDiffusionObjective`. Remove imports and factory branches. Keep `fm-lab-imbdiff-eval`; it evaluates external/generated CIFAR arrays and does not require discrete training code.

- [ ] **Step 6: Prove no production callers remain**

Run:

```bash
rg -n "DiscreteDiffusion|DiscreteDiffusionObjective|sample_discrete_diffusion|sample_discrete_and_plot" fm_lab tests configs
```

Expected: no matches and exit status 1.

- [ ] **Step 7: Run the training/sampling regression suites**

```bash
.conda/fm_lab/bin/pytest tests/test_training_sampling.py tests/test_class_conditioning.py tests/test_checkpointing.py tests/test_objectives.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit continuous-only training and sampling**

```bash
git add -A fm_lab tests
git commit -m "Remove discrete diffusion training and sampling"
```

---

### Task 7: CLI and Configuration Migration

**Files:**
- Modify: `fm_lab/experiments/run_train.py`
- Modify: `fm_lab/experiments/factory.py`
- Modify: `tests/test_run_train_cli.py`
- Modify: `tests/test_config_smoke.py`
- Create: four `configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss*.yaml` files listed above
- Delete: active `configs/imbdiff/**/*.yaml` training configs listed in File Structure
- Delete: `docs/imbdiff_cm_reproduction.md`

**Interfaces:**
- Consumes: canonical objective schema from Tasks 2–5.
- Produces: CLI flags `--model-output`, `--loss-space`, and `--prediction-min-denom`; four canonical Fashion-MNIST configs.

- [ ] **Step 1: Write failing CLI-schema tests**

Update `tests/test_run_train_cli.py` to expect:

```python
args = Namespace(
    objective="flow_matching",
    objective_loss="mse",
    model_output="target",
    loss_space="velocity",
    prediction_min_denom=0.01,
    straightness_weight=None,
    straightness_sample_size=None,
    direction_weight=None,
    speed_weight=None,
)
assert _objective_overrides(args) == {
    "name": "flow_matching",
    "loss": "mse",
    "model_output": "target",
    "loss_space": "velocity",
    "min_denom": 0.01,
}
```

Assert parser help contains no `--diffusion-prediction-type`, `--x-prediction-loss-space`, or `--x-prediction-min-denom`.

- [ ] **Step 2: Run CLI tests and verify failure**

```bash
.conda/fm_lab/bin/pytest tests/test_run_train_cli.py -q
```

Expected: tests fail against the old x/diffusion-specific flags.

- [ ] **Step 3: Implement canonical CLI overrides**

Allow `--model-output` and `--loss-space` choices `source`, `target`, and `velocity`; add `--prediction-min-denom`. Remove discrete-only parser arguments and override logic. Normalize aliases in the objective builder, not in serialized config output.

- [ ] **Step 4: Create the four controlled Fashion-MNIST configs**

All four configs must share data IR100 settings, Gaussian source dimension 784, independent coupling, `path.name: linear`, Image U-Net shape `[1, 28, 28]`, conditioning, training budget, Euler solver, NFE 64, seed 0, and exactly 10,000 balanced conditional samples.

The baseline objective is:

```yaml
objective:
  name: flow_matching
  model_output: target
  loss_space: velocity
  min_denom: 0.001
  modifiers: []
```

The CBDM config adds:

```yaml
  modifiers:
    - name: cbdm
      target_distribution: train
      tau: 0.001
      gamma: 0.25
      comparison_space: velocity
```

The OC config adds `name: oc`, `transfer_mode: t2h`, `cut_t: null`, and `min_denom: 0.001`. The CM config declares OC first, then CM with `consistency_weight: 1.0`, `diversity_weight: 0.2`, and `comparison_space: velocity`; it also enables the existing capacity adapter on Image U-Net up blocks.

- [ ] **Step 5: Delete unsupported discrete training configs and reproduction guide**

Remove the active ImbDiff training YAMLs and the discrete reproduction guide. Do not delete the CIFAR feature evaluator or its tests.

- [ ] **Step 6: Add config smoke and forbidden-key tests**

Load all shipped YAMLs and assert the four new files have identical controlled fields. Within `configs/fashion_mnist_lt` and any remaining active long-tail training configs, recursively reject keys `diffusion`, `prediction_type`, `ddim_skip`, `eta`, and `cut_time`. Do not reject unrelated continuous Gaussian-path research configs. Assert each new config builds target, source, path, model, solver, and objective in a dry run.

- [ ] **Step 7: Run CLI and config suites**

```bash
.conda/fm_lab/bin/pytest tests/test_run_train_cli.py tests/test_config_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit the continuous experiment matrix**

```bash
git add -A fm_lab/experiments configs tests/test_run_train_cli.py tests/test_config_smoke.py docs/imbdiff_cm_reproduction.md
git commit -m "Add continuous Fashion-MNIST long-tail suite"
```

---

### Task 8: Documentation, Benchmark Smoke Test, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/cli.md`
- Modify: `tests/test_docs_coverage.py`
- Modify: `tests/test_fashion_mnist_lt_eval.py` only if artifact naming assertions need continuous sampler metadata

**Interfaces:**
- Consumes: four configs and continuous ODE artifacts from Task 7.
- Produces: exact train/evaluate commands and a verified continuous benchmark workflow.

- [ ] **Step 1: Write failing documentation-coverage tests**

Require README and CLI docs to name all four configs, use `samples/euler_nfe64.npy`, pass `--generation-method` values `x_vloss`, `x_vloss_cbdm`, `x_vloss_oc`, and `x_vloss_cm`, and avoid DDPM/DDIM training commands.

- [ ] **Step 2: Run documentation tests and verify failure**

```bash
.conda/fm_lab/bin/pytest tests/test_docs_coverage.py -q
```

Expected: failures identify missing continuous-suite documentation.

- [ ] **Step 3: Document exact training commands**

Document installation refresh and one command per config using the required environment:

```bash
.conda/fm_lab/bin/python -m pip install -e .
.conda/fm_lab/bin/fm-lab-train \
  --config configs/fashion_mnist_lt/fashion_mnist_lt_ir100_x_vloss.yaml \
  --output-dir runs/fashion_mnist_lt_ir100/x_vloss \
  --device auto
```

Repeat with the CBDM, OC, and CM configs and distinct output directories. Warn that explicit output directories must be empty for a fresh controlled run.

- [ ] **Step 4: Document exact canonical evaluation commands**

For each method, evaluate `samples/euler_nfe64.npy`, `samples/generated_labels.npy`, and `checkpoint.pt` with `fm-lab-fashion-mnist-lt-eval`, `--sampler euler`, `--nfe 64`, the configured guidance scale and seed, and a method-specific evaluation directory.

- [ ] **Step 5: Run short four-variant smoke training**

Use a temporary directory and override each config to two training steps, 20 generated samples, two samples per class, Euler NFE 2, and a small sample batch. Run on CPU through `.conda/fm_lab/bin/python`; assert each run emits a checkpoint, `euler_nfe2.npy`, and labels with equal class counts. Do not run the canonical 10,000-sample metric evaluator in this smoke test.

Expected: four commands exit 0 and each artifact set is balanced.

- [ ] **Step 6: Run focused evaluator compatibility tests**

```bash
.conda/fm_lab/bin/pytest tests/test_fashion_mnist_lt_eval.py tests/test_fashion_mnist_metric_calibration.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Run repository-wide dead-code and config scans**

```bash
rg -n "DiscreteDiffusion|DiscreteDiffusionObjective|sample_discrete_diffusion|ddim_skip|prediction_type: x_vloss|cut_time:" fm_lab configs README.md docs/cli.md
```

Expected: no matches and exit status 1.

- [ ] **Step 8: Run the complete verification suite**

```bash
.conda/fm_lab/bin/pytest -q
.conda/fm_lab/bin/ruff check .
git diff --check
```

Expected: all tests pass, Ruff reports `All checks passed!`, and `git diff --check` exits 0.

- [ ] **Step 9: Commit documentation and final verification metadata**

```bash
git add README.md docs/cli.md tests/test_docs_coverage.py tests/test_fashion_mnist_lt_eval.py
git commit -m "Document continuous long-tail experiments"
```

- [ ] **Step 10: Review final history and worktree**

```bash
git status --short --branch
git log --oneline --decorate -10
```

Expected: clean feature branch with one focused commit per completed task and no untracked generated feature caches.

---

## Final Requirement Traceability

- Prediction object encapsulation: Tasks 1–2.
- Linear conditional interpolant conversions: Task 1.
- Independent target output and velocity loss: Task 2.
- Continuous CBDM: Task 3.
- Continuous OC and `cut_t`: Task 4.
- Continuous CM and capacity validation: Task 5.
- ODE sampling and balanced labels: Task 6.
- Discrete implementation removal: Tasks 6–7.
- Four Fashion-MNIST experiments: Task 7.
- Canonical evaluator compatibility and commands: Task 8.
- Custom velocity-only `FlowPath` preservation: Tasks 1–2 regression tests.
- Endpoint and unsupported-conversion handling: Task 1.
- Checkpoint incompatibility behavior: Task 6.
