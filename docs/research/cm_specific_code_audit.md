# CM-specific code audit

**Date:** 2026-07-24
**Decision:** retire the Capacity Manipulation implementation as an active
research direction while preserving reusable infrastructure and the historical
scientific record.

## Retention rule

Code is retained when it is useful without the ImbDiff-CM paper, its checkpoint
format, or its general/expert loss. Code is removed when its contract requires
the vendored release, CM coefficient schedule, CM branch semantics, or a CM
checkpoint. This deliberately favors retention at the boundary: infrastructure
that is even plausibly reusable stays.

## Removed

- The `third_party/ImbDiff-CM` submodule and the adapter that translated its
  models, objectives, samplers, checkpoints, dropout behavior, and endpoint
  transfer into `fm_lab`.
- The faithful CM objective registrations and serialized aliases.
- The continuous `CMModifier` consistency/diversity loss.
- Six checkpoint-specific diagnostic families: mechanism, dropout, knowledge,
  causal intervention, sampling intervention/visualization, and live training
  dynamics.
- Their console scripts, configs, tests, and active CLI documentation.
- CM-only trainer behavior: released-data batching, released DDIM/DDPM sampling,
  capacity-off default sampling, captured objective terms, and the in-loop
  dynamics observer.
- The official 60k matrix and paper-specific reproduction/screen configs.

## Retained and generalized

| Area | Reason retained |
| --- | --- |
| CIFAR-10/100 and Fashion-MNIST long-tail datasets | General frequency/support experiments |
| Long-tail evaluation, feature caches, FID/KID/Recall, classwise and frequency-group reports | Model-independent evaluation infrastructure |
| `GaussianDiffusionPath`, `DiscreteDDPMPath`, `DiffusionObjective`, and `DDPMUNet` | General diffusion research components |
| CBDM and OC continuous modifiers | Long-tail methods independent of CM capacity allocation |
| Switchable low-rank convolution/linear adapters and U-Net placement controls | Generic capacity, ablation, fine-tuning, and intervention experiments |
| Explicit base/full adapter selection during sampling | Generic causal adapter ablation |
| CUDA mixed precision, channels-last, compilation controls, checkpoint/resume, EMA, and early stopping | General training infrastructure |
| Standard and zero-start warmup conventions | Optimizer behavior independent of any paper-specific name |
| Synthetic and Fashion-MNIST geometry/frequency experiments | The broader long-tail geometry program |

The retained convolution adapter now applies `adapter_scale`, matching the
generic linear adapter contract. The previous unscaled convolution behavior
was a release-compatibility artifact and had no remaining active configuration.

## Historical artifacts

The CM research notes, result summaries, and figures under `docs/research/` are
kept as an archive. They document completed experiments and prevent the same
questions from being repeated. They are not an active API contract: commands
or source paths mentioned in those historical reports may refer to code removed
by this audit. The Git history remains the recovery path if the reproduction or
probes are ever needed again.

## Deferred naming cleanup

`fm-lab-imbdiff-eval` and the corresponding evaluation module names are
retained for compatibility even though the evaluator itself is general
long-tail CIFAR infrastructure. Renaming that public command would create churn
without removing a scientific dependency; it can be handled separately if the
new direction needs a neutral CLI name.
