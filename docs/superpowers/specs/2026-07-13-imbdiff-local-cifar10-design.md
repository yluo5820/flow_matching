# Local CIFAR-10 ImbDiff Baseline Design

## Objective

Provide CPU-oriented CIFAR-10 long-tail experiments for DDPM epsilon
prediction, x-prediction with velocity loss, CBDM, OC, and CM. These local
profiles prioritize iteration speed and functional comparison over exact paper
reproduction.

## Configuration boundary

Add five configs under `configs/imbdiff/local/`. Keep the existing paper-scale
configs under `configs/imbdiff/` unchanged so local results cannot be mistaken
for reproduction results. Local outputs go under `runs/imbdiff/local/`, and
experiment names and tracks include a `local` marker.

## Compact model

All five methods use `DDPMUNet`, because it already supports class conditioning
and CM's switchable low-rank capacity. The compact architecture is:

```yaml
model:
  name: ddpm_unet
  image_shape: [3, 32, 32]
  base_channels: 32
  channel_multipliers: [1, 2, 2]
  attention_levels: [1]
  num_res_blocks: 1
  dropout: 0.1
```

This model has approximately 1.12 million parameters, close to the previous
1.06-million-parameter CIFAR ImageUNet and far below the paper profile's 35.75
million parameters. CM retains its up-block rank-ratio capacity configuration.

## Local training and sampling budget

Every method uses the same local budget:

```yaml
training:
  batch_size: 16
  steps: 5000
  warmup_steps: 500
  early_stopping:
    enabled: true
    warmup_steps: 2000
    patience_steps: 1000
    min_delta: 0.0001
    ema_alpha: 0.05

sampling:
  n_samples: 1000
  sample_batch_size: 16
  plot_max_points: 100
  ddim_skip: 50
```

The learning rate remains `0.0002`, but optimizer warmup is reduced from the
paper profile's 5000 steps to 500 so the local run reaches its full learning
rate. Diffusion, conditioning, classifier-free guidance, and method semantics
remain useful references rather than strict parity requirements. The earliest
early stop is approximately step 3000. A measured
batch-16 forward/backward pass takes about 0.34 seconds on the current CPU, so
single-forward methods target roughly 25–40 minutes and dual-forward methods
roughly 40–60 minutes. Wall time remains hardware- and data-pipeline-dependent.

## Method matrix

The local matrix contains exactly:

1. DDPM epsilon prediction;
2. DDPM x-prediction with velocity-space loss;
3. CBDM;
4. OC;
5. CM with OC target transfer and up-block capacity manipulation.

## Verification

A config regression test will require exactly five local files, the shared
compact architecture and budget, unique local output directories, and the
correct method-specific objective. It will build each model without loading
the dataset and check that the model remains class-conditional. The CM model
must report capacity enabled; the other four must report capacity disabled.

After the test-first config implementation, run focused tests, Ruff, and the
full suite under `.conda/fm_lab`. Finally, run a one-step/two-sample CIFAR-10 CM
CPU smoke experiment from the local CM config and verify its checkpoint,
metrics, sample output, compact model metadata, and enabled early stopping.
