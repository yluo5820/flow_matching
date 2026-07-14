# CM on the Local X-VLoss Baseline

## Goal

Create the first controlled long-tail method experiment by placing Capacity
Manipulation (CM) on top of the validated CIFAR-10-LT x-prediction plus v-loss
baseline. The experiment should isolate the effect of CM rather than changing
the backbone, training budget, sampler, or data distribution at the same time.

## Chosen approach

Update the existing local CM configuration in place. A separate duplicate
configuration would add an unnecessary experiment variant, while CLI-only
overrides cannot express the early-stopping schedule.

The CM experiment will match the validated x-vloss baseline in:

- CIFAR-10-LT exponential imbalance with ratio 100 and subset seed 0;
- `ImageUNetVelocity` with base channels 32 and time embedding dimension 128;
- x-prediction plus v-loss;
- Adam with learning rate 0.0002 and batch size 32;
- a 12,000-step ceiling, 500-step optimizer warmup, and default-on early
  stopping with a 6,000-step monitoring warmup and 4,000-step patience;
- EMA decay 0.999; and
- DDIM sampling with skip 16, no additional classifier-free guidance, and the
  existing live-versus-EMA diagnostic.

CM adds only the method-specific behavior already implemented in the
repository:

- switchable low-rank adapters on the image U-Net up path with rank ratio 0.1;
- original-class transfer using OC's tail-to-head reference rule;
- full-capacity and base-capacity forward passes; and
- consistency weight 1.0 plus diversity weight 0.2.

## Objective composition

No training-objective code change is required. The current discrete objective
supports CM with `prediction_type: x_vloss`. Its primary denoising loss remains
the x-prediction v-loss. For CM regularization, both the full-capacity and
base-capacity clean-image outputs are converted to epsilon space before their
weighted consistency and diversity losses are computed. This preserves the
existing CM loss definition while changing the shared denoising baseline.

## Configuration and output

Change `configs/imbdiff/local/cifar10_lt_cm_local.yaml` to use `x_vloss` and the
same 12,000/6,000/4,000 training and early-stopping schedule as
`cifar10_lt_x_vloss_local.yaml`. Keep all CM-specific model and objective fields
unchanged.

The user-facing command will select a new output directory so it cannot mix
artifacts with the previous epsilon-based CM run.

## Verification

Automated configuration tests will assert that local CM uses x-vloss and the
matched schedule while retaining enabled up-path capacity and CM loss weights.
A one-step CPU smoke test will exercise the dual-forward CM objective, capacity
routing, checkpoint writing, and sampling path. The full experiment remains a
user-run workload.

Success means the smoke run completes without changing the CM formulation, and
the full run produces generated samples and live-versus-EMA artifacts that can
be compared directly with the local x-vloss baseline. Quantitative long-tail
evaluation is a subsequent round, not part of this configuration change.
