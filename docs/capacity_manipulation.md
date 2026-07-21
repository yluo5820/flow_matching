# Capacity Manipulation Reproduction

The repository now keeps two deliberately separate implementations of Capacity
Manipulation (CM):

1. `official_imbdiff_cm` is the reproduction path. It adapts only interfaces
   and delegates the model, discrete DDPM loss, endpoint transfer, CM weighting,
   and DDIM sampler to the authors' release vendored under
   `third_party/ImbDiff-CM`.
2. The older continuous-flow modifier remains an experimental reformulation. It
   must not be used as evidence that the released CM implementation was or was
   not reproduced.

## Exact reproduction boundary

The official path uses:

- the released `UNet_CM`, including its timestep table, initialization, and
  structured low-rank convolutions;
- the released `GaussianDiffusionTrainer`, including normalized reciprocal
  class weights, whole-batch classifier-free dropout, endpoint transfer, and
  raw `MSE(h2, h1)` CM distance;
- the released `GaussianDiffusionSamplerOld` imported by
  `tools/sample_images.py`, including its 50-step DDIM update and capacity-on
  default model branch;
- Adam, the released warmup convention, gradient clipping, and EMA;
- shuffled, without-replacement epochs over the retained long-tail split.

The surrounding fm_lab code supplies run directories, checkpoints, plots, and
configuration. A parity test compares loss and every trainable gradient against
the vendored trainer on the same batch.

The 30k screening configuration is:

`configs/cifar100_lt/autodl_screen/cifar100_lt_ir100_official_cm_screen30k.yaml`

It accepts either the CIFAR-100 binary archive or AutoDL's shared
`cifar-100-python` layout under `/root/autodl-tmp/data/cifar100`.

The controlled 60k follow-up expands the release reproduction into DDPM, CBDM,
OC, released CM, pure CM, and an OC-plus-capacity control. Its exact method
definitions and configurations are documented in
`docs/official_imbdiff_matrix.md`.

## Continuous adaptation

Continuous CM configurations and `CMModifier` remain available for controlled
follow-up experiments. They are not imported by the official reproduction path.
Once the discrete screen is reproduced locally, discrete DDPM can be replaced
by a matched continuous VP schedule as one isolated experimental change.

Primary references:

- [OpenReview paper](https://openreview.net/forum?id=wSGle6ag5I)
- [Authors' implementation](https://github.com/Feng-Hong/ImbDiff-CM)
- [Released CM configuration](https://github.com/Feng-Hong/ImbDiff-CM/blob/main/configs/cifar100lt_ir100/cm.yaml)
