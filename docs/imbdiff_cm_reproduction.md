# ImbDiff-CM Reproduction Contract

This document freezes the experimental contract for reproducing **Improving
Diffusion Models for Class-Imbalanced Training Data via Capacity Manipulation**
before method-specific implementation begins.

Primary references:

- Paper: <https://openreview.net/forum?id=wSGle6ag5I>
- Official code: <https://github.com/Feng-Hong/ImbDiff-CM>

## Experiment tracks

Results must name the objective track explicitly:

- `ddpm_epsilon_parity`: paper-compatible epsilon prediction.
- `ddpm_x_vloss`: clean-image prediction optimized through velocity-space MSE.

Only `ddpm_epsilon_parity` is eligible for direct comparison with paper tables.

## Dataset construction

For a balanced CIFAR training set with `C` classes and `n_max` examples per
class, class `c` retains

```text
int(n_max * imbalance_factor ** (c / (C - 1)))
```

examples. `imbalance_factor = 1 / IR`. Class-local source indices are shuffled
with NumPy `RandomState(subset_seed)` in ascending class order, matching the
official implementation. The retained dataset is sampled naturally per image;
class-balanced resampling is a separately named baseline.

Default reproduction datasets:

| Dataset | IR | imbalance factor | max/min count |
|---|---:|---:|---:|
| CIFAR-10-LT | 100 | 0.01 | 5000 / 50 |
| CIFAR-10-LT | 50 | 0.02 | 5000 / 100 |
| CIFAR-100-LT | 100 | 0.01 | 500 / 5 |
| CIFAR-100-LT | 50 | 0.02 | 500 / 10 |

Training uses random horizontal flips and normalization to `[-1, 1]`. Balanced
CIFAR test data remains untouched for downstream semantic evaluation.

Every run records the class counts, subset seed, and SHA-256 of retained source
indices. Runs are comparable only when these values match.

## Paper-parity model and optimization

```yaml
model:
  base_channels: 128
  channel_multipliers: [1, 2, 2, 2]
  attention_levels: [1]
  residual_blocks_per_level: 2
  dropout: 0.1

diffusion:
  timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  variance: fixed_large

training:
  optimizer: adam
  learning_rate: 0.0002
  batch_size: 64
  total_steps: 300001
  warmup_steps: 5000
  gradient_clip: 1.0
  ema_decay: 0.9999
  condition_dropout: 0.1
  checkpoint_every: 100000
```

The parity loss is ordinary epsilon MSE. EMA parameters are used for sampling
and evaluation.

## Sampling and CFG convention

Paper evaluation generates 50,000 balanced-by-request samples with DDIM, a
1,000-step schedule, skip 20, and `omega = 1.5`.

The official implementation uses

```text
pred = conditional + omega * (conditional - unconditional)
```

while `fm_lab` currently uses

```text
pred = unconditional + scale * (conditional - unconditional).
```

Thus paper `omega = 1.5` equals `fm_lab` guidance `scale = 2.5`. Configs must
state the convention; silent conversion is prohibited.

## Methods

- **DDPM:** epsilon MSE with natural per-image sampling.
- **CBDM:** DDPM plus cross-class distribution-adjustment regularization with
  explicit stop-gradient placement.
- **OC:** Oriented Calibration using frequency-oriented batchwise transfer of
  the noise target from majority knowledge toward minority examples.
- **CM:** selected convolution weights are decomposed into general and low-rank
  expert capacity. The reference uses rank ratio `0.1`, scale `0.5`, the U-Net
  up path, `w_con = 1.0`, and `w_div = 0.2`.

Each method must reduce numerically to its base objective when disabled.

## Evaluation outputs

Each evaluated checkpoint must produce:

- overall FID, KID, generative Recall, and Inception Score;
- classwise FID;
- Many, Medium, and Few aggregate FID;
- overall FID beside split scores;
- per-repeat values plus mean and standard deviation;
- cached real/generated Inception features and evaluator provenance.

Primary IR=100 reference targets:

| Dataset | Method | FID ↓ | KID ↓ | Recall ↑ | IS ↑ |
|---|---|---:|---:|---:|---:|
| CIFAR-10-LT | DDPM | 10.697 | 0.0035 | 0.47 | 9.39 |
| CIFAR-10-LT | CBDM | 8.233 | 0.0026 | 0.53 | 9.23 |
| CIFAR-10-LT | OC | 8.390 | 0.0027 | 0.52 | 9.53 |
| CIFAR-10-LT | CM | 7.727 | 0.0023 | 0.53 | 9.52 |
| CIFAR-100-LT | DDPM | 10.163 | 0.0029 | 0.46 | 13.45 |
| CIFAR-100-LT | CBDM | 10.051 | 0.0036 | 0.51 | 12.35 |
| CIFAR-100-LT | OC | 8.309 | 0.0026 | 0.52 | 13.44 |
| CIFAR-100-LT | CM | 7.519 | 0.0017 | 0.52 | 13.45 |

Initial acceptance is correct method ranking and comparable score magnitude;
exact equality is not expected across frameworks, hardware, and random seeds.
