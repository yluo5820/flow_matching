# Continuous Prediction Conversion Design

Date: 2026-07-14

## Goal

Replace the repository's separate discrete diffusion implementation with one
continuous-time training system based initially on the existing linear conditional
interpolant. Encapsulate source, target, and velocity prediction conversions in value
objects so objectives, long-tail methods, and samplers can consume a declared
representation without duplicating schedule algebra.

The first benchmark enabled by this work compares a target-predicting,
velocity-loss baseline with the same baseline augmented by CBDM, OC, and CM on
long-tailed Fashion-MNIST.

## Terminology

For the initial linear path,

\[
x_t = (1-t)x_0 + t x_1,
\qquad
u_t = x_1 - x_0.
\]

- `source` is \(x_0\), the Gaussian source sample. `epsilon` is accepted as a
  user-facing alias where useful, but serialized metadata uses `source`.
- `target` is \(x_1\), the clean data sample. `clean` and `x` are aliases, but
  serialized metadata uses `target`.
- `velocity` is the flow-matching velocity \(u_t\).

This design does not introduce diffusion-v prediction. On the linear conditional
interpolant, `velocity` always means flow velocity. This avoids conflating two
different quantities that are only schedule-dependently related.

"Continuous" describes the learned time-conditioned field. Sampling remains a
finite numerical integration over a chosen time grid.

## Scope

### Included

- Continuous time \(t \in [0,1]\) throughout training.
- The existing linear conditional interpolant as the only initially supported
  conversion-capable path.
- Model output and loss spaces independently configurable as `source`, `target`,
  or `velocity`.
- Prediction value objects with path-specific conversion methods.
- Continuous CBDM, OC, and CM objective modifiers.
- ODE sampling through converted flow velocity.
- Migration of long-tail Fashion-MNIST and existing ImbDiff configurations to the
  continuous system.
- Removal of the discrete objective, discrete sampler, and discrete-only tests and
  configuration fields after migration.
- Preservation of the extensible `FlowPath` protocol for custom interpolants.

### Excluded

- DDPM or DDIM parity with the original long-tail papers.
- Integer timestep schedules, learned reverse variances, and discrete posterior
  formulas.
- Reflow, rectified-flow distillation, or iterative rectification.
- Diffusion-v prediction.
- A stochastic SDE sampler in the first implementation.
- Conversion support for every existing custom interpolant.

## Design Principles

1. Prediction representation, loss representation, and sampler representation are
   independent choices.
2. Conversion algebra belongs to a path-aware value object, not to objectives or
   samplers.
3. Unsupported conversions fail explicitly rather than falling back to a guessed
   formula.
4. Endpoint singularities are handled centrally and are visible in configuration
   and metadata.
5. Existing custom paths remain valid velocity-training paths without being forced
   to support endpoint recovery.
6. The migration has one continuous end state; the discrete implementation is not
   retained as a compatibility branch.

## Architecture

### Prediction kinds

Introduce a normalized enum:

```python
class PredictionKind(str, Enum):
    SOURCE = "source"
    TARGET = "target"
    VELOCITY = "velocity"
```

Configuration parsing may accept documented aliases, but configs, checkpoints,
metrics, and provenance store only canonical values.

### Path prediction state

Add a conversion-capable state protocol separate from the minimal `FlowPath`
protocol:

```python
class PathPredictionState(Protocol):
    xt: torch.Tensor
    t: torch.Tensor

    @property
    def supported_kinds(self) -> frozenset[PredictionKind]: ...

    def prediction(
        self,
        value: torch.Tensor,
        kind: PredictionKind,
    ) -> PathPrediction: ...
```

A path opts into conversion support through:

```python
class ConvertibleFlowPath(FlowPath, Protocol):
    def prediction_state(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> PathPredictionState: ...
```

`FlowPath` itself continues to require only `sample_xt` and `target_velocity`.
Consequently, custom interpolants that only train velocity models remain valid.

### Prediction value object

`PathPrediction` is immutable and owns a raw tensor, its declared kind, and the
path state that defines conversions:

```python
@dataclass(frozen=True)
class PathPrediction:
    value: torch.Tensor
    kind: PredictionKind
    state: PathPredictionState

    def as_source(self) -> torch.Tensor: ...
    def as_target(self) -> torch.Tensor: ...
    def as_velocity(self) -> torch.Tensor: ...
    def convert(self, kind: PredictionKind) -> torch.Tensor: ...
```

The public methods delegate to the concrete state. This keeps case-specific
conversion code next to the path geometry and lets two model evaluations at the
same \((x_t,t)\) share the same state.

### Linear implementation

`LinearPredictionState` supports all three initial kinds using direct formulas:

\[
x_1 = x_t + (1-t)u,
\qquad
x_0 = x_t - tu,
\]

\[
u = \frac{x_1-x_t}{1-t},
\qquad
u = \frac{x_t-x_0}{t}.
\]

Conversions use expanded time tensors with the same broadcasting convention as
the existing paths. Direct formulas are preferred over chained conversions.

The state receives a positive `min_denom`. A conversion that divides by `t` or
`1-t` clamps only the denominator used for the calculation. Training time sampling
continues to avoid exact endpoints where possible. Metadata records `min_denom`.
Tests must cover values close to both endpoints and verify finite outputs.

## Objective Composition

Replace method-specific monolithic objective names with a base continuous objective
and zero or more long-tail modifiers.

```yaml
objective:
  name: flow_matching
  model_output: target
  loss_space: velocity
  min_denom: 0.001
  modifiers:
    - name: cbdm
      target_distribution: train
      tau: 0.001
      gamma: 0.25
```

The base objective performs this data flow:

1. Ask the path for \(x_t\), target velocity, and a prediction state.
2. Evaluate the model once in the declared `model_output` space.
3. Wrap the tensor as a `PathPrediction`.
4. Convert the prediction and ground truth to `loss_space`.
5. Compute per-sample loss before reduction so modifiers can apply weights or
   auxiliary terms.
6. Add modifier losses and return stable, namespaced metrics.

The canonical x-prediction/v-loss baseline is:

```yaml
objective:
  name: flow_matching
  model_output: target
  loss_space: velocity
```

## Continuous Long-Tail Modifiers

### CBDM

CBDM evaluates the same \((x_t,t)\) under the observed class and an auxiliary
class. Both outputs are wrapped as predictions and converted to the configured
CBDM comparison space, initially `velocity` so it matches the base loss geometry.

The old integer factor `tau * discrete_t` becomes a documented continuous weight
`tau * time_weight(t)`. The initial `time_weight(t) = 1 - t`, because the
repository's convention runs from noisy source at zero to clean target at one.
This preserves stronger weighting at higher noise without retaining integer time.
Alternative weights are future configuration additions, not implicit
schedule-dependent behavior.

CBDM retains its auxiliary target distributions (`train`, `sqrt`, and `uniform`)
and commitment stop-gradient directions.

### OC

OC changes the paired source/target used to construct the conditional training
target before prediction conversion. Reference selection uses continuous path
geometry rather than discrete alpha-bar tables.

For the linear path, the effective conditional noise-to-signal ratio is derived
from \((1-t)^2/t^2\), with numerical protection near endpoints. Transfer modes
remain `t2h`, `h2t`, and `full`.

Replace integer `cut_time` with optional `cut_t` in \([0,1]\). When set, transfer
is enabled only for samples with `t >= cut_t`; the cutoff therefore directly uses
the repository's noisy-source-to-clean-target convention. The value and comparison
rule are stored in metadata. `null` means no cutoff.

### CM

CM retains capacity-on and capacity-off model evaluations. Both outputs become
prediction objects and are converted to the configured comparison space before
consistency and diversity distances are calculated.

CM continues to require capacity-enabled model metadata and the OC target-transfer
modifier. Invalid composition fails during configuration validation, before
training begins.

## Sampling

ODE samplers consume flow velocity only. The sampler wraps each model output with
the current path prediction state and calls `as_velocity()` before integration.
Classifier-free guidance is applied to model outputs in their declared output
space before conversion; the affine linear-path conversions preserve the usual
guided combination.

Sampling metadata records:

- model output kind;
- path name;
- conversion `min_denom`;
- solver and NFE;
- guidance convention and scale;
- random seed.

No DDPM/DDIM sampling mode remains after migration.

## Configuration and Migration

Create four Fashion-MNIST IR100 configurations with identical data, model,
conditioning, optimizer, training budget, solver, NFE, guidance, and seed:

1. target-prediction/velocity-loss baseline;
2. baseline plus CBDM;
3. baseline plus OC;
4. baseline plus OC and CM.

Use the existing linear path and balanced conditional sampling of 10,000 generated
examples, exactly 1,000 per class, for the canonical evaluator.

Migrate existing long-tail CIFAR configurations to the same continuous schema or
remove configurations that no longer describe a supported experiment. Remove
discrete-only keys such as `diffusion.timesteps`, beta schedules, reverse variance,
`prediction_type: x_vloss`, `ddim_skip`, and `eta`.

Checkpoint loading does not silently reinterpret discrete checkpoints as
continuous models. A discrete checkpoint produces a clear incompatibility error.
There is no checkpoint weight converter in scope because the time semantics and
training objective changed.

## Removal Plan

After continuous parity tests pass:

- delete `DiscreteDiffusionObjective` and its factory branch;
- delete the finite-step diffusion process and discrete sampler if they have no
  remaining callers;
- remove discrete-only CLI overrides;
- migrate or delete discrete configs and documentation;
- replace discrete tests with continuous modifier, conversion, and sampling tests;
- retain no hidden legacy fallback.

Removal happens in the same feature branch as migration so the main branch never
contains two advertised long-tail systems.

## Validation and Error Handling

Configuration validation must reject:

- a requested output or loss kind unsupported by the selected path;
- source/target loss conversion without a conversion-capable path;
- non-positive `min_denom`;
- `cut_t` outside \([0,1]\);
- CBDM, OC, or CM without class counts and conditional labels;
- CM without both OC and a capacity-enabled model;
- discrete-only fields in continuous configs;
- loading a checkpoint whose prediction/path metadata differs from the run config.

Errors identify the incompatible field and list supported values.

## Testing Strategy

### Conversion unit tests

- Round-trip source, target, and velocity conversions for random linear-path
  states.
- Direct formulas match analytical values.
- Batched and image-shaped broadcasting works.
- Near-endpoint conversions remain finite and use the configured denominator.
- Unsupported kinds and paths fail explicitly.
- Gradients propagate through every training conversion.

### Objective tests

- Target-output/velocity-loss equals direct velocity loss after analytical
  conversion.
- Output and loss spaces are independently configurable.
- CBDM auxiliary and commitment gradients retain their intended stop-gradient
  behavior.
- OC transfer respects class frequency direction and continuous `cut_t`.
- CM requires OC/capacity and computes in the configured comparison space.
- Modifier metrics are namespaced and deterministic under a fixed seed.

### Sampling tests

- A target-predicting model is converted to velocity before every solver call.
- Conditional labels and classifier-free guidance survive conversion.
- A known linear field produces the expected endpoint.
- Sample artifacts contain the metadata required by the Fashion-MNIST evaluator.

### Migration tests

- All shipped configurations parse and dry-run.
- No shipped config contains discrete-only keys.
- No production caller imports the removed discrete objective or sampler.
- The full test suite and linter pass after deletion.

### Benchmark smoke test

Run short, fixed-seed versions of the four Fashion-MNIST variants and verify that
each emits a balanced generated sample set accepted by the canonical evaluator.
Metric superiority is not a correctness requirement; comparable protocol and
valid artifacts are.

## Acceptance Criteria

The design is complete when:

1. The repository exposes one continuous long-tail training path.
2. Source, target, and velocity conversions are owned by immutable path-aware
   prediction objects.
3. The linear conditional interpolant supports target-prediction/velocity-loss
   without objective-specific conversion helpers.
4. CBDM, OC, and CM compose with the same continuous base objective.
5. The four Fashion-MNIST IR100 runs share a controlled protocol and pass the
   canonical evaluator's balanced-label validation.
6. Custom `FlowPath` implementations remain valid for velocity prediction even if
   they do not implement endpoint conversions.
7. Discrete training, sampling, configuration, and compatibility fallbacks are
   absent from the supported code path.
8. Tests cover conversion identities, endpoint behavior, modifier composition,
   ODE sampling, configuration migration, and benchmark artifact compatibility.
