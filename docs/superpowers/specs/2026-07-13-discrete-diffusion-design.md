# Discrete Diffusion Design

## Scope

Round 3 adds paper-compatible discrete diffusion without changing the existing
continuous flow-matching path. It covers a linear 1,000-step schedule, epsilon
and clean-image parameterizations, DDPM and DDIM sampling, classifier-free
guidance, and the requested clean-image prediction with velocity-space loss.

## Architecture

`DiscreteDiffusion` is a parameter-free mathematical component. It owns schedule
tensors, forward noising, prediction conversions, the DDPM posterior, and DDIM
updates. Both training and sampling consume this component so coefficients and
boundary behavior cannot silently diverge.

A dedicated discrete objective accepts clean data, samples integer timesteps and
Gaussian noise, constructs `x_t`, calls the conditional model, and computes one
of two losses. `epsilon` is ordinary epsilon MSE and is the paper-parity track.
`x_vloss` predicts clean `x_0`, converts both the prediction and ground truth to
the velocity parameterization `v = sqrt(alpha_bar) * epsilon -
sqrt(1-alpha_bar) * x_0`, and applies MSE in velocity space.

Discrete sampling is separate from ODE solvers. DDPM uses the configured
posterior variance, including paper-compatible `fixed_large`; DDIM uses a
configurable timestep subsequence and `eta`, defaulting to deterministic
`eta=0`. Both use the same model-prediction adapter and balanced requested class
labels.

## Classifier-free guidance

Training continues to use `-1` as the null class and the existing conditioning
dropout. Sampling uses the repository convention
`unconditional + scale * (conditional - unconditional)`. Config metadata must
record this convention. The paper's `conditional + omega *
(conditional-unconditional)` at `omega=1.5` is represented as repository scale
`2.5`.

## Configuration

Discrete experiments select `objective.name: discrete_diffusion`. Diffusion
configuration includes `timesteps`, `beta_start`, `beta_end`, and `variance`.
Sampling selects `sampler: ddpm|ddim`, `ddim_skip`, `eta`, balanced class counts,
and the explicit CFG convention. CIFAR-10-LT and CIFAR-100-LT parity configs use
the Round 1 data contract and Round 2 training protocol.

## Validation

Unit tests compare schedule and posterior formulas against direct equations,
verify epsilon/x/velocity round trips, enforce a noise-free final DDPM step,
verify deterministic DDIM output at `eta=0`, and test CFG convention conversion.
Integration tests cover discrete objective backpropagation, conditional sampling,
and configuration construction. Existing continuous objectives and samplers must
retain their current behavior.

## Out of scope

CBDM, OC, CM, large-scale metric evaluation, and full 300,001-step runs remain
later rounds. Round 3 provides the shared numerical and configuration foundation
they consume.
