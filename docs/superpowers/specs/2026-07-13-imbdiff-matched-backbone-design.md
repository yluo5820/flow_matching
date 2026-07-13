# Matched-Backbone Local ImbDiff Design

## Goal

Make the local CIFAR-10 long-tail ImbDiff experiments architecturally comparable
to the previously successful unconditional CIFAR-10 run while retaining the
paper-derived discrete diffusion, class conditioning, CBDM, OC, and CM
components. The first full experiment remains DDPM epsilon prediction; the
other local method configs receive the same backbone so later comparisons do
not introduce an architecture confound.

The run should normally finish in roughly 75 to 110 minutes on the CPU used for
the repaired baseline and should not be scheduled beyond approximately two
hours. This is a step-budget estimate rather than a wall-clock kill switch.
Early stopping remains enabled and may end the run sooner when its monitored
loss has converged.

## Evidence Behind the Design

The previous successful run used a 1,060,649-parameter unconditional
`ImageUNetVelocity` with feature widths `32 -> 64 -> 128`, an 8x8 bottleneck,
batch size 32, and a selected checkpoint at step 23,500. Adding ten-class
conditioning to that backbone produces 1,078,569 parameters.

The first local DDPM used a 1,115,363-parameter `DDPMUNet`, but its feature
widths were only `32 -> 64 -> 64`, its batch size was 16, and the repaired run
selected the live checkpoint at step 5,250. Although the parameter totals were
similar, the local DDPM had a narrower bottleneck, substantially fewer
optimization examples, a smaller long-tail dataset, and the harder discrete
epsilon-prediction objective.

The repaired run also inherited the old EMA and performed only 2,250 new EMA
updates before stopping. A from-scratch run is therefore required for a clean
comparison.

## Shared Backbone

All five configs in `configs/imbdiff/local/` will use `ImageUNetVelocity` with:

```yaml
model:
  name: image_unet
  image_shape: [3, 32, 32]
  base_channels: 32
  time_embedding_dim: 128
  activation: silu
  zero_init_head: true
```

The CHW image shape matches the flattened representation emitted by the
long-tail CIFAR loader. The architecture retains the previous two downsampling
stages, widths `32 -> 64 -> 128`, nearest-neighbor upsampling, residual blocks,
and zero-initialized output convolution. Existing class embeddings and CFG
dropout support remain active through the shared conditioning configuration.

No attention or dropout is added to the base backbone. This keeps DDPM,
x-vloss, CBDM, and OC as close as possible to the earlier successful model,
apart from the intentional class-conditioning and discrete-diffusion changes.

## Capacity Manipulation Adaptation

CM must use the same base architecture as the other methods. The existing
switchable low-rank convolution will therefore be made reusable by the image
U-Net instead of retaining a separate DDPM-only implementation.

`ImageUNetVelocity` will accept an optional capacity configuration. For the CM
config, low-rank adapters are installed only on the convolutional layers in the
two up-path residual blocks. The adapter output is selected through the existing
`context["use_capacity"]` convention. Adapter B matrices remain zero-initialized,
so the capacity-enabled branch initially produces exactly the base-model output.

When capacity is disabled:

- the image U-Net contains ordinary `nn.Conv2d` layers;
- its pre-existing backbone state-dict keys and tensor shapes remain unchanged;
  the conditional model still has the intentionally added class-embedding keys,
  so the old unconditional checkpoint is not loaded as a complete checkpoint;
- `capacity_metadata()` reports disabled with zero adapter layers; and
- DDPM, x-vloss, CBDM, and OC do not pay adapter compute or parameter cost.

The CM objective's existing model validation, base/full forward passes,
consistency term, and diversity term remain unchanged.

## Paper Components Retained

The backbone change does not alter these existing components:

- 1,000-step linear-beta discrete Gaussian diffusion;
- epsilon prediction for DDPM, CBDM, OC, and CM;
- clean-image prediction with velocity-space loss for x-vloss;
- classifier-free conditioning dropout during training;
- neutral conditional sampling at CFG scale 1.0;
- CBDM auxiliary-distribution regularization;
- OC head/tail target transfer;
- CM consistency and diversity losses;
- timestep-bucket loss diagnostics; and
- paired live-versus-EMA sampling diagnostics.

Paper-scale configs outside `configs/imbdiff/local/` remain unchanged.

## Local Training Budget

All five local configs will use:

```yaml
training:
  optimizer: adam
  lr: 0.0002
  batch_size: 32
  steps: 8000
  warmup_steps: 500
  gradient_clip: 1.0
  ema_decay: 0.999
  log_every: 250
  checkpoint_every: 2000
  early_stopping:
    enabled: true
    warmup_steps: 4000
    patience_steps: 2000
    min_delta: 0.0001
    ema_alpha: 0.3
```

The earliest normal early stop is around step 6,000. At batch size 32 this is
192,000 training examples, versus only 84,000 examples at the repaired run's
selected live checkpoint. The maximum 8,000 steps process 256,000 examples.

Early stopping is not disabled, bypassed, or delayed to the old 20,000-step
paper-style warmup. One checkpoint-coherence correction is required: when the
trainer restores the best live-model state, it must restore the EMA snapshot
captured at that same step. This does not change the stopping decision or add
training work; it prevents final sampling from mixing different checkpoints.

## Sampling Budget

All five local configs will use:

```yaml
sampling:
  sampler: ddim
  n_samples: 256
  sample_batch_size: 32
  plot_max_points: 64
  ddim_skip: 16
  eta: 0.0
  classifier_free_guidance:
    enabled: true
    convention: fm_lab
    scale: 1.0
    paper_omega: 0.0
  live_ema_comparison:
    enabled: true
    n_samples: 64
```

A skip of 16 gives approximately 64 deterministic DDIM evaluations, matching
the earlier run's 64 solver evaluations more closely. Reducing normal samples
from 1,000 to 256 and paired diagnostics from 100 to 64 reserves most of the
two-hour budget for training while preserving a useful visual comparison.

## Verification

Implementation will use test-first checkpoints:

1. Capacity-disabled image U-Net construction preserves the existing output
   shape, class conditioning, ordinary convolutions, and approximately 1.08M
   parameters.
2. Capacity-enabled construction installs adapters only in the up path,
   reports accurate metadata, and produces identical base/full outputs before
   an adapter update.
3. The factory forwards image-U-Net capacity configuration correctly, and the
   existing CM objective accepts the adapted model.
4. All five local YAML configs encode the shared architecture and bounded
   training/sampling policy, while paper configs remain unchanged.
5. Early-stopping state capture and restore keep live and EMA weights from the
   same selected step.
6. The complete test suite, Ruff, and Conda dependency check pass.
7. A one-step CPU smoke of the local DDPM config produces normal DDIM samples,
   live/EMA diagnostics, and timestep-bucket history fields.

The user will run the full from-scratch DDPM experiment. The first acceptance
check is qualitative: live and EMA samples should both contain recognizable
CIFAR-10 structure and should be materially closer to the earlier unconditional
run than the repaired DDPM result. Later ImbDiff evaluation metrics remain a
separate experiment round.

## Out of Scope

- Changing the long-tail subset or imbalance ratio.
- Returning to the 35.8M-parameter paper U-Net or 300,001-step paper budget.
- Disabling early stopping or adding a wall-clock termination mechanism.
- Changing CBDM, OC, or CM loss formulas.
- Running all five full experiments locally in this implementation round.
- Modifying the paper-scale CIFAR-10 or CIFAR-100 configs.
