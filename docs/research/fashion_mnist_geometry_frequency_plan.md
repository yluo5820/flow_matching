# Fashion-MNIST geometry-by-frequency bridge

Status: preregistration draft; no classes have been selected and no outcome model has
been trained.

## Purpose

The synthetic experiment crossed known factor dimension with class frequency, but its
manifolds were deliberately artificial. This bridge asks whether class-specific
geometric complexity measured on Fashion-MNIST predicts how generative learning changes
with unique support and training exposure.

The experiment does **not** assume that Fashion-MNIST classes are single smooth
manifolds or that one estimator recovers a true dimension. Intrinsic dimension (ID) is
an ordinal, representation-dependent predictor. Other local and global properties may
explain residual difficulty.

## Hypotheses

- **F1 — support effect:** at equal class exposure, reducing a class from 5,000 to 500
  and 50 unique images worsens held-out conditional generation.
- **F2 — exposure effect:** at fixed unique support, empirical sampling is worse than
  uniform class exposure, especially at 50 images.
- **F3 — geometry-by-support interaction:** the class with a stably higher geometry
  score has a steeper equal-exposure degradation curve than the stably lower-scoring
  class.
- **F4 — scalar insufficiency:** if F3 is not consistently ordered across FID and
  recall, ID alone is insufficient; estimator disagreement, local-ID heterogeneity,
  anisotropy, density, or multimodality become candidate explanations.
- **F5 — long-tail context:** a 5,000-image class may differ between the balanced model
  and a model whose other classes have 500 and 50 images.

F1, F2, F3, and F5 are confirmatory within this discovery protocol. F4 is the planned
fallback interpretation, not an invitation to select a favorable post-hoc estimator.

## Data partition

Fashion-MNIST has 6,000 training images and 1,000 official test images per class. For
each of its ten classes, use one deterministic permutation to create:

- geometry probe A: 500 training-split images;
- geometry probe B: 500 disjoint training-split images;
- training candidate pool: the remaining 5,000 images;
- final evaluation: all 1,000 images from the official test split.

The geometry probes never enter generative training. The test split is not inspected
until the class trio, representations, estimators, frequency mappings, model budget,
and endpoints are frozen. Within every selected class, training supports are nested:
the 50-image set is a prefix of the 500-image set, which is a prefix of the 5,000-image
set.

## Fail-closed class selection

### Representations

Use two already supported and deliberately different spaces:

1. raw pixels after non-whitened PCA to 50 dimensions, Euclidean distance;
2. normalized DINOv2 CLS features, cosine distance.

The task-trained Fashion-MNIST evaluator representation is excluded from selection to
avoid choosing classes in the same space used for the primary outcome. It may be added
after selection as a sensitivity analysis and cannot change the chosen trio.

### Primary ID diagnostics

Use global TwoNN, global MLE-LID at `k=10` and `k=20`, participation ratio, and the
90%-variance PCA dimension. Correlation dimension, ball scaling, local covariance
spectra, and other available diagnostics are recorded as secondary descriptors but do
not decide class inclusion.

For every representation, probe half, estimator, and class, repeat 100 deterministic
80% subsamples without replacement. Convert each estimator's ten class estimates to
percentile ranks before aggregation; absolute values from different estimators are not
averaged.

For class `c`, define its geometry score as the median percentile rank across the two
representations, five primary diagnostics, two probe halves, and their subsamples.
Retain the full distribution rather than only the median.

### Stability gate

A class is eligible only if:

- at least 80% of its planned estimates are finite;
- the median score from probe A and probe B differs by at most 0.10;
- the interquartile range across representation-estimator cells is at most 0.25;
- neither representation reverses the class from the lower third to the upper third,
  or vice versa, between probe halves.

Choose the eligible low class with the smallest frozen score, the eligible high class
with the largest score, and an eligible middle class closest to 0.5. Require low,
middle, and high to occupy the lower, middle, and upper score thirds, respectively,
with adjacent score gaps of at least 0.15.

If no trio passes, stop. Do not relax thresholds after looking at identities. The
fallback is an all-ten-class descriptive study or a revised protocol, not a manually
chosen trio. Class names and example images are revealed only after the selection file
and its digest are written.

This gate establishes stable ordinal separation, not known dimensions analogous to the
synthetic 1D, 3D, and 5D labels.

## Experimental conditions

Remap the selected original Fashion-MNIST labels to conditional labels `L`, `M`, and
`H` according to their frozen geometry scores. Use a three-class conditional model and
the same architecture, initialization seed, data-order seed, dequantization seed,
training budget, and generation seed in every condition.

The three cyclic mappings are:

| Mapping | Low geometry | Middle geometry | High geometry |
|---|---:|---:|---:|
| R0 | 5,000 | 500 | 50 |
| R1 | 500 | 50 | 5,000 |
| R2 | 50 | 5,000 | 500 |

Train:

1. one balanced reference at `(5000, 5000, 5000)`;
2. R0, R1, and R2 with empirical example sampling;
3. R0, R1, and R2 with uniform class sampling and uniform sampling within class.

This is seven discovery models. Every frequency-mapped dataset contains 5,550 unique
images, so the three rotations have equal total cardinality. Under uniform class
sampling, frequency changes unique support while expected label exposure remains one
third per class. Under empirical sampling, support and exposure change together as in
an ordinary long-tailed dataset.

Do not add loss reweighting, classifier-free guidance sweeps, architecture changes, or
long-tail remedies to this stage.

## Budget calibration

Calibrate training duration using only the balanced reference. Save and evaluate
predeclared checkpoints at 2,000, 5,000, and 10,000 updates. Freeze the earliest of
2,000 or 5,000 for which the next checkpoint improves macro classwise FID by less than
10%, improves requested-class accuracy by less than two percentage points, and the
earlier checkpoint has at least 80% requested-class accuracy for every selected class.
If neither checkpoint passes, use 10,000 only if every class reaches 80%; otherwise
block the experiment and revise the model before reading any long-tail outcome. Once
frozen, train no condition to a different budget.

The balanced calibration run is reused as the balanced reference if its checkpoint and
config satisfy the frozen protocol. Any command projected to exceed 30 minutes in the
working terminal is handed to the user rather than run inline.

## Evaluation

Generate 1,000 samples per selected class and compare with the corresponding 1,000
official-test images. Use the production-gated Fashion-MNIST classifier and immutable
feature caches.

Primary per-class endpoints are:

- classifier-feature FID;
- class-conditional generative recall;
- requested-class accuracy and mean requested-class probability.

Secondary endpoints are macro and worst-class FID, overall KID and recall, the
conditional confusion matrix, local-ID distributions in generated versus test data,
and nearest-training proximity for 50-image equal-exposure cells. Reference split-half
calibration supplies a metric noise floor. Generated-sample resampling quantifies
evaluation uncertainty but is not presented as model-seed uncertainty.

For each class and sampling policy, fit only a descriptive three-point response curve
against `log10(unique support)`. Report:

- support degradation: equal-exposure 50 minus equal-exposure 5,000;
- exposure penalty: empirical minus equal-exposure at the same support;
- context effect: frequency-rotation 5,000 minus balanced-reference 5,000;
- high-minus-low difference in support degradation.

With only three selected classes, do not report a correlation p-value between ID and
quality. F3 is supported only if the high-versus-low ordering agrees for worse FID and
worse recall and exceeds the reference calibration noise. A nonmonotone ordering is a
scientific result against a single scalar geometry explanation.

## Replication decision

The seven runs form a one-model-seed discovery experiment. Launch a second seed only if
F3 has the same direction in both primary distributional endpoints. The economical
confirmation is the balanced reference plus the three equal-exposure rotations (four
models). Empirical rotations need not be repeated unless their exposure effect fails to
replicate the established synthetic phenomenon.

## Required implementation before any outcome run

- Add a three-class Fashion-MNIST adapter with an immutable original-to-conditional
  label map and disjoint 500/500/5,000 splits.
- Add `empirical` and `class_balanced` sampling policies to the Fashion target.
- Add the split-half/subsample stability aggregator and selection gate on top of the
  existing raw-pixel and DINO ID outputs.
- Extend the Fashion evaluator to subset official-test features by original label while
  evaluating remapped conditional labels.
- Add an orchestrator that writes immutable manifests/configs, lists all seven commands
  in dry-run mode, records backend and seeds, resumes safely, and writes the crossed
  analysis into the living research report.
- Unit-test nested support, probe exclusion, cyclic counts, label provenance, sampling
  exposure, selection failure, and evaluator label mapping before downloading data or
  training.

## Interpretation boundary

This experiment can show that a reproducible class-geometry score predicts sensitivity
to support or exposure. It cannot establish that estimated ID is the causal property.
Class identity remains bundled with multimodality, curvature, topology, nuisance
variation, and semantic ambiguity. Frequency rotation makes the frequency contrasts
within-class and therefore causal; the geometry contrast remains an observational
comparison across three naturally different classes.

If this bridge is successful, freeze the protocol and repeat it on CIFAR-10. Do not
retune the selection score or endpoints on CIFAR-10.

## Related references

- Xiao, Rasul, and Vollgraf, [Fashion-MNIST: a Novel Image Dataset for Benchmarking
  Machine Learning Algorithms](https://arxiv.org/abs/1708.07747).
- Eser et al., [Intrinsic dimensionality as a model-free measure of class
  imbalance](https://doi.org/10.1016/j.neucom.2026.132938).
