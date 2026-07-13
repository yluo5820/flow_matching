# Discrete Diffusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add paper-compatible discrete DDPM/DDIM training and sampling with epsilon and x-prediction velocity-loss tracks.

**Architecture:** A parameter-free `DiscreteDiffusion` owns all schedule equations and prediction conversions. A focused objective and sampler consume it, while the experiment runner reuses existing models, conditioning, EMA, checkpoints, and CIFAR-LT data.

**Tech Stack:** Python 3.11, PyTorch, pytest, YAML.

## Global Constraints

- Preserve every existing continuous flow-matching path and configuration.
- Paper parity uses 1,000 linear betas from 0.0001 through 0.02 and `fixed_large` variance.
- Repository CFG uses `unconditional + scale * (conditional - unconditional)`; paper omega 1.5 maps to scale 2.5.
- Training inputs and generated outputs remain flattened tensors at public model boundaries.

---

### Task 1: Discrete diffusion equations

**Files:**
- Create: `fm_lab/diffusion/discrete.py`
- Create: `fm_lab/diffusion/__init__.py`
- Test: `tests/test_discrete_diffusion.py`

**Interfaces:**
- Produces: `DiscreteDiffusion(config)`, `q_sample`, prediction conversions, `p_sample`, and `ddim_step`.

- [ ] Write tests for the linear schedule, forward equation, conversions, posterior coefficients, `t=0`, and deterministic DDIM.
- [ ] Run `.conda/fm_lab/bin/pytest -q tests/test_discrete_diffusion.py` and confirm missing-module failure.
- [ ] Implement the minimal tensor equations with batch-index extraction and input validation.
- [ ] Run the focused test and targeted Ruff checks.
- [ ] Commit the independently verified numerical core.

### Task 2: Discrete training objective

**Files:**
- Create: `fm_lab/training/discrete_objective.py`
- Modify: `fm_lab/training/losses.py`
- Modify: `fm_lab/training/trainer.py`
- Test: `tests/test_discrete_objective.py`

**Interfaces:**
- Consumes: `DiscreteDiffusion` and `model_prediction`.
- Produces: `DiscreteDiffusionObjective` with `epsilon` and `x_vloss` modes.

- [ ] Write failing tests for exact epsilon MSE, velocity-space x loss, integer timestep sampling, labels, and backpropagation.
- [ ] Run focused tests and confirm failures reflect the absent objective.
- [ ] Implement the objective and route discrete batches through it without a continuous `FlowPath` interpolation.
- [ ] Run focused and existing objective/trainer tests.
- [ ] Commit the verified training path.

### Task 3: DDPM/DDIM sampling and CFG

**Files:**
- Create: `fm_lab/diffusion/sampling.py`
- Modify: `fm_lab/training/trainer.py`
- Test: `tests/test_discrete_sampling.py`

**Interfaces:**
- Consumes: `DiscreteDiffusion`, conditional model, requested labels, and prediction type.
- Produces: `sample_discrete_diffusion` returning flattened generated samples.

- [ ] Write failing tests for sampler shape, class-label forwarding, balanced labels, reproducibility, and CFG scale conversion.
- [ ] Run tests and confirm missing sampler failures.
- [ ] Implement chunked DDPM/DDIM loops using EMA model selection already provided by the trainer.
- [ ] Run focused sampling and regression tests.
- [ ] Commit the verified samplers.

### Task 4: Experiment configuration and smoke coverage

**Files:**
- Create: `configs/imbdiff/cifar10_lt_ddpm_epsilon.yaml`
- Create: `configs/imbdiff/cifar100_lt_ddpm_epsilon.yaml`
- Create: `configs/imbdiff/cifar10_lt_x_vloss.yaml`
- Modify: `docs/imbdiff_cm_reproduction.md`
- Test: `tests/test_config_smoke.py`

**Interfaces:**
- Consumes: Round 1 CIFAR-LT data, Round 2 U-Net/runtime, and Tasks 1-3.
- Produces: reproducible full and smoke-run commands.

- [ ] Write failing config construction and tiny training/sampling smoke tests.
- [ ] Add explicit paper-parity and x-vloss configurations with CFG convention metadata.
- [ ] Document full and reduced smoke commands and expected artifacts.
- [ ] Run focused tests, full pytest, targeted Ruff, and `git diff --check`.
- [ ] Commit and push the completed Round 3 branch.
