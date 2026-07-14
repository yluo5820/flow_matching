# JiT-Style Continuous X-Prediction/V-Loss Stabilization

## Goal

Make the four Fashion-MNIST long-tail experiments preserve the intended
JiT-style combination—target-image model output with velocity-space loss—without
the endpoint-driven loss explosions produced by uniform continuous-time
sampling and asymmetric denominator clamping.

## Root Cause

For the linear conditional interpolant

\[
x_t=(1-t)x_0+t x_1,
\]

the target-output conversion is

\[
\hat v=(\hat x_1-x_t)/(1-t).
\]

The current baseline samples `t` uniformly and clamps only the prediction-side
denominator at `0.001`, while its supervision remains the unclamped path
velocity `x1 - x0`. This produces an effective target-image error weight of up
to `1,000,000`, frequently samples the singular neighborhood, and makes exact
target predictions disagree with supervision inside the clamped region.

JiT instead samples time from a shifted logit-normal distribution and uses a
`0.05` denominator floor for both the target and predicted velocities.

## Selected Approach

Implement JiT behavior as an explicit training policy while preserving the
general path-aware prediction objects.

1. Add a config-driven training-time sampler with two supported forms:
   - existing/default `uniform` behavior;
   - `logit_normal` with finite `mean` and positive finite `std`.
2. For endpoint-output models (`source` or `target`), construct the ground-truth
   endpoint in the model-output space and pass it through the same
   `PathPredictionState` conversion used for the model prediction. For the
   Fashion-MNIST target-output/velocity-loss experiments, this means both sides
   divide by `max(1-t, min_denom)`.
3. Keep native velocity-output/velocity-loss behavior unchanged.
4. Apply the same symmetric rule to OC-transferred endpoint supervision.
5. Configure all four experiments with JiT defaults:
   - `training.time_sampling.name: logit_normal`;
   - `mean: -0.8`;
   - `std: 0.8`;
   - `objective.min_denom: 0.05`;
   - OC `min_denom: 0.05` where present;
   - LR warmup and EMA enabled using the repository's step-based trainer.
6. Preserve the controlled Euler/NFE-64 generation and evaluation protocol.
   Solver choice is orthogonal to the training-loss regression and must remain
   identical across baseline, CBDM, OC, and CM.

## Alternatives Rejected

### Train in target space

This is numerically stable but changes the experiment from x-output/v-loss to
x-output/x-loss. It does not reproduce the intended JiT objective.

### Increase `min_denom` only

This reduces the largest gradients but leaves uniform sampling concentrated too
heavily near the clean endpoint and, without symmetric supervision, retains an
irreducible loss in the clamped region.

### Internally predict velocity and reconstruct x

This is stable but makes the raw network a velocity predictor. It defeats the
purpose of testing whether an x-predicting network is advantageous.

## Interfaces

### Time sampling

Create a focused training-time sampler module exposing a validated immutable
configuration and a sampling function. It must accept both the legacy string
form and the new mapping form:

```yaml
training:
  time_sampling:
    name: logit_normal
    mean: -0.8
    std: 0.8
```

Absent configuration and `time_sampling: uniform` retain the current uniform
sampler. Invalid names, non-finite parameters, and non-positive standard
deviation fail during configuration/build, before training starts.

### Symmetric supervision conversion

`FlowMatchingObjective` retains independent `model_output` and `loss_space`
metadata. When conversion is required, it must derive the ground-truth value in
the model-output space and convert that value through the same state:

- target output: ground truth starts as `x1`;
- source output: ground truth starts as `x0`;
- velocity output: ground truth starts as `path.target_velocity(...)`.

Both prediction and supervision then use `state.prediction(...).convert(...)`.
This makes clamping symmetric without changing `LinearPredictionState`'s public
conversion API. OC replaces `x0`/`x1` with its transferred source/target before
the same rule is applied.

## Training Configuration

The four Fashion-MNIST configs remain a controlled matrix. They share all data,
model, coupling, path, time-sampling, optimizer, and sampling fields; only the
modifier list and CM capacity adapter differ.

Use a 500-step LR warmup and EMA decay `0.9999`. Do not add gradient clipping as
part of JiT parity because the public JiT recipe does not rely on it. Existing
trainer validation and exact-resume contracts must include the new sampler and
training settings automatically through the semantic configuration payload.

## Diagnostics and Tests

Test-first coverage must demonstrate:

1. A seeded logit-normal sample matches `sigmoid(N(mean, std))`, remains in
   `(0, 1)`, and has the expected shifted distribution.
2. Invalid sampler configurations fail before training.
3. A perfect target prediction near `t=1` has zero velocity loss with the
   symmetric floor; the current asymmetric implementation must fail this test.
4. An imperfect near-endpoint prediction is bounded by the configured `0.05`
   floor.
5. Velocity/velocity and legacy uniform configurations are unchanged.
6. OC-transferred supervision uses the same symmetric conversion.
7. All four shipped configs share JiT sampler/floor/warmup/EMA settings and
   continue to dry-build.
8. A short CPU baseline smoke run emits finite, non-explosive early losses,
   a checkpoint, and balanced generated labels.
9. The full test suite and Ruff remain clean.

## Success Criteria

- The baseline no longer exhibits hundreds-to-thousands-scale initial loss
  caused by endpoint sampling.
- Exact x predictions produce zero base loss at every sampled time, including
  the denominator-floor region.
- Baseline, CBDM, OC, and CM use identical JiT-style base training semantics.
- Existing velocity baselines, custom paths, checkpoint sampling, and the
  Fashion-MNIST evaluator remain compatible.

