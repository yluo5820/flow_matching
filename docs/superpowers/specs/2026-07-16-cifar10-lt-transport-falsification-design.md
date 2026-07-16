# CIFAR-10-LT Transport Falsification Design

## Purpose

This study is the final natural-image falsification of the gradient-geometry
hypothesis developed on Fashion-MNIST-LT. It asks whether the same
low-dimensional class-gradient geometry and cross-fitted sign transport survive
on CIFAR-10-LT IR100 under ordinary conditional flow matching.

The study is explanatory. It does not run CM, train capacity adapters, search for
an intervention, target a 1% improvement, or unlock a later method stage. Its
terminal result determines whether the project continues as a theory of
projected-gradient transport or pivots away from this subject.

## Approaches considered

Three approaches were considered.

1. **Replay the Fashion-MNIST calibration and audit chain on CIFAR-10.** This
   maximizes code reuse, but preserves a failed 1% intervention target and makes
   a natural-image observation depend on CM-era lock semantics.
2. **Run a one-off CIFAR script.** This is quick, but permits silent layer,
   checkpoint, class, and step-size selection after looking at results.
3. **Add a separate natural-image falsification contract.** This reuses the
   existing Observation-0 sketches and cross-fitted functional primitives while
   locking CIFAR-specific data, training, analysis, and terminal decisions.

The third approach is selected. It adds only the dataset support and orchestration
needed for a clean replication; it does not modify the completed Fashion-MNIST
artifacts or reinterpret their failed functional lock.

## Scientific hypotheses

The primary hypothesis is:

> At the final CIFAR-10-LT checkpoint, the row-normalized rank-1 class-gradient
> direction fitted in each of two adjacent non-output layers has positive
> cross-fitted held-out target-loss transport, positive worst-case selectivity,
> and repeats in at least two of three training seeds.

The geometry prerequisite is:

> At the same checkpoint, low-time stratum, rank, and two locked layers, at least
> five classes have Probe-A/Probe-B subspace overlap above the matched 99th
> percentile class-permutation null in at least two of three seeds.

The baseline-learning prerequisite prevents an untrained natural-image model
from being counted as evidence against the hypothesis. Median held-out Probe-A
loss at the final checkpoint must be at most 70% of its step-zero value in both
locked layers' shared evaluation data. Failure of this prerequisite is
`baseline_not_learned`, an inconclusive implementation/training result rather
than a scientific rejection.

Raw-gradient rank-1 directions are retained as a paired descriptive control.
Row normalization is not required to beat raw geometry because the
Fashion-MNIST audit did not establish a significant paired advantage.

## Locked CIFAR-10-LT experiment

The target is CIFAR-10 training data with exponential imbalance factor 0.01,
frequency mapping multiplier 3, mapping offset 0, and subset seed 0. A
deterministic pool of 256 examples per class is removed before long-tail
subsampling: 128 examples form Probe A and 128 disjoint examples form Probe B.
Training may use random horizontal flips; diagnostic samples never use
augmentation. Training and diagnostic samples use seeded uniform dequantization
and normalization to `[-1, 1]`.

The model is the existing class-conditional `ImageUNetVelocity` with image shape
`[3, 32, 32]`, base width 32, time embedding width 128, and no capacity
adapters. The objective is ordinary linear-path flow matching with target output
and velocity-space loss. Coupling is independent. EMA, early stopping, and all
objective modifiers are disabled.

Three training seeds `(0, 1, 2)` run for 100,000 optimizer steps with batch size
64. Locked checkpoints are `(0, 2500, 10000, 25000, 50000, 100000)`. The longer
budget follows the repository's existing CIFAR image-U-Net profile rather than
reusing Fashion-MNIST's 20,000-step budget.

The Observation-0 measurement retains the Fashion-MNIST dimensionless protocol:

- Probe A and Probe B plus one source-noise replica of each;
- 16 microbatches of 8 examples per class and timestep stratum;
- strata `(0.02, 0.10)`, `(0.10, 0.30)`, `(0.30, 0.70)`, `(0.70, 0.90)`, and
  `(0.90, 0.98)`;
- row-normalized gradient sketches of width 4096, with 16384 as the only
  preregistered escalation;
- ranks `(1, 2, 4, 8)` for the gate and `(1, 2, 4, 8, 16, 32)` descriptively;
- six homologous U-Net weights from input, down, middle, up, and output blocks;
- a 99th-percentile class-label-permutation null with 999 permutations.

The natural-image transport test is locked to checkpoint steps `(0, 10000,
100000)`, low-time stratum 0, rank 1, layers
`down2_block.conv2.weight` and `middle.conv2.weight`, and all ten classes. Step 0
is the baseline-learning control, step 10,000 is an early-training comparison,
and step 100,000 is the sole primary endpoint.

## Deterministic CIFAR diagnostic target

`ImbalancedCIFARImages` gains the same diagnostic interface used by the existing
paired manifests:

- `diagnostic_indices(split)` returns immutable original CIFAR row IDs;
- `diagnostic_samples(split, original_indices, dequantization_seeds, device)`
  restores rows in requested order, applies no horizontal flip, and uses one
  deterministic dequantization seed per row;
- metadata records the frequency ranks, Probe-A and Probe-B index digests,
  dequantization flag, and retained class counts.

The factory passes `data.dequantize` and `data.frequency_mapping` to CIFAR in the
same form already supported for Fashion-MNIST. Existing CIFAR configurations
without these fields keep their current behavior.

## Cross-fitted functional transport

Each class and layer supplies 16 exact microbatch-gradient rows. Four circular
folds with offsets `(0, 4, 8, 12)` divide them into 8 fit, 4 orientation, and 4
evaluation rows. Every microbatch appears twice in fit, once in orientation, and
once in evaluation.

For each fold, the code fits paired raw and row-normalized centered rank-1
bases. Both are oriented with the same disjoint raw mean gradient. For direction
class `c`, evaluation class `j`, layer parameters `theta_l`, evaluation loss
`L_j`, evaluation gradient `g_j`, and unit oriented direction `d`, the primary
quantity is the zero-step relative-benefit slope

\[
s_{c\to j} = -\frac{\lVert\theta_l\rVert_2\langle g_j,d\rangle}{L_j}.
\]

Positive target slope means held-out infinitesimal improvement. Selectivity is
target slope minus the worst off-class harm. The complete 10 by 10 response
matrix is retained for every layer, seed, fold, checkpoint, and basis.

Finite relative steps `(3e-5, 1e-4, 3e-4, 1e-3)` are diagnostic only. A step is
locally concordant when the aggregate median relative error between finite
benefit and `epsilon * slope` is at most 10%. No finite benefit threshold enters
the primary hypothesis.

Fold values are collapsed to one median per seed-by-direction-class block before
uncertainty is computed. A deterministic 95% percentile bootstrap resamples
these 30 blocks. Folds are never treated as independent observations.

## Frequency and interference interpretation

The aggregate artifacts include:

- per-class target slopes and selectivity by seed and checkpoint;
- the full median cross-class slope matrix for each layer and basis;
- target-slope sign agreement across seeds;
- Spearman associations of target slope, selectivity, and incoming off-class
  harm with locked class frequency rank and log retained count;
- raw versus row-normalized basis cosine and paired target-slope difference;
- the largest locally concordant finite step.

Frequency associations are descriptive because ten classes do not support a
stable confirmatory correlation test. They guide theory only after the primary
transport verdict is known.

## Terminal decisions

The result is exactly one of:

- `baseline_not_learned`: the 70% held-out loss-ratio prerequisite fails;
- `no_reliable_cifar_geometry`: the locked final-checkpoint geometry
  prerequisite fails;
- `natural_image_transport_confirmed`: the geometry prerequisite passes and,
  in both locked layers, row-normalized final target-slope bootstrap lower bound
  is positive, median selectivity is positive, and at least two seed medians are
  positive;
- `geometry_without_transport`: geometry passes but the row-normalized target
  slope 95% upper bound is non-positive in both layers;
- `heterogeneous_natural_image_transport`: every other complete result.

The corresponding project-level actions are:

- fix or strengthen the ordinary CIFAR baseline only for
  `baseline_not_learned`, without changing the transport analysis;
- pivot away from spectral gradient geometry for
  `no_reliable_cifar_geometry` or `geometry_without_transport`;
- continue with a theory of sign transport and interference for
  `natural_image_transport_confirmed`;
- treat `heterogeneous_natural_image_transport` as insufficient for a method;
  continue only if class frequency or interference structure explains the
  heterogeneity in the locked outputs.

No status unlocks CM, capacity allocation, or a parameter intervention.

## Components and artifact contract

The implementation adds:

- CIFAR diagnostic-pool and deterministic-dequantization support in
  `fm_lab/data/cifar_lt.py` and the target factory;
- dataset-generic validation in the existing Observation-0 preregistration,
  while preserving the Fashion-MNIST contract;
- a CIFAR base training YAML and CIFAR Observation-0 YAML;
- `natural_image_preregistration.py` for the exact transport and terminal
  decision contract;
- `natural_image.py` for upstream validation, resumable exact-gradient chunks,
  cross-fitted analysis, frequency/interference summaries, and aggregate
  integrity checks;
- a canonical natural-image transport YAML and a
  `falsify-natural-image-transport` subcommand in the existing CLI.

Artifacts live only below
`STUDY/aggregate/natural_image_transport_falsification/`:

- `preregistration.yaml`;
- one digest-bound chunk directory per seed and checkpoint;
- `slopes.csv`, `finite_steps.csv`, and `basis_comparison.csv`;
- `class_transport.csv` and `interference_matrices.npz`;
- `falsification_summary.json` with prerequisites, layer intervals, terminal
  status, and only allowed next action;
- `complete.json` binding all outputs and upstream input digests.

Repeated commands validate and reuse complete artifacts. Partial or changed
chunks, manifests, checkpoints, run configs, reliability tables, or summaries
fail closed. Probe B is used only by the already locked geometry reliability
stage; functional transport uses Probe A cross-fitting and never selects a
scope from Probe B.

## Testing and failure handling

Tests cover deterministic CIFAR pool disjointness, retained IR100 counts,
requested-ID restoration, seeded dequantization, augmentation exclusion,
factory compatibility, both dataset preregistrations, canonical CIFAR configs,
exact cross-fit balance, geometry prerequisite extraction, baseline-learning
guard, all terminal statuses, complete 10 by 10 interference outputs,
frequency summaries, chunk resumability, digest rejection, CLI output, and the
absence of any CM or Stage-1 path.

The test suite uses small synthetic CIFAR binaries and tiny models. It does not
download CIFAR or execute the 300,000 total training steps during verification.
