# Fashion-MNIST Long-Tail Benchmark Design

## Objective

Create a fast conditional image-generation benchmark for comparing long-tail
methods before paying the cost of a CIFAR-10 run. The benchmark uses
Fashion-MNIST at its native one-channel `28 x 28` resolution and preserves the
project's existing evaluator vocabulary: FID, KID, Inception Score, generative
recall, classwise FID, and frequency-group FID.

The headline evaluation is deliberately balanced. Training data may be
long-tailed, but evaluation requests the same number of generated samples from
each class and compares them with the untouched, balanced official test split.
This prevents head classes from dominating the global scores and makes the
benchmark measure whether a method learns every conditional distribution.

MNIST may use the same abstractions later, but Fashion-MNIST is the initial
benchmark because its greater within-class variation should distinguish methods
that would saturate ordinary MNIST.

## Dataset contract

A reusable labeled-image dataset abstraction supports balanced and exponential
class subsampling. For ten deterministically ordered classes, exponential counts
follow

```text
n_c = floor(n_max * rho ** (c / (C - 1)))
```

where `C` is the number of classes and `rho` is the configured minimum-to-maximum
count ratio. Counts must contain at least one item per class. Index selection is
seeded and deterministic. The dataset exposes images, labels, selected source
indices, per-class counts, image shape, normalization, and dequantization
metadata.

The initial configurations use Fashion-MNIST's official training split with
imbalance factors `1.0`, `0.1`, and `0.01`. The official test split is never
subsampled for the canonical evaluation. Training and evaluator preprocessing
must share an explicitly recorded image normalization contract.

The implementation should extract the existing exponential-index selection
logic from `fm_lab/data/cifar_lt.py` rather than create a second formula. CIFAR
and Fashion-MNIST adapters may retain their dataset-specific loading and
augmentation behavior while sharing the selection primitive.

## Evaluator contract

One small classifier is trained once on the balanced official Fashion-MNIST
training split and then frozen. It exposes:

- a penultimate-layer feature vector for distribution and coverage metrics;
- ten-class probabilities for Inception Score and conditional diagnostics.

The evaluator is independent of every generative-model training run. Its
checkpoint records the dataset, split, architecture, feature dimension,
normalization, class ordering, held-out accuracy, weights fingerprint, and
checkpoint format version. A configurable minimum held-out accuracy is part of
the checkpoint-validation contract; evaluation fails rather than accepting a
checkpoint below that threshold.

The benchmark uses this domain-specific evaluator instead of resizing grayscale
images for an ImageNet Inception network. This retains the existing feature-level
metric implementations in `fm_lab/evaluation/metrics.py` while making the
representation appropriate for Fashion-MNIST.

## Sampling and reference protocol

Each canonical evaluation requests exactly 1,000 generated samples per class,
for 10,000 generated samples in total. Requested conditional labels are retained
separately from evaluator-predicted labels. The real reference contains the
10,000 examples in the untouched official Fashion-MNIST test split.

The protocol fixes and records the generative checkpoint, sampler settings,
number of integration steps, guidance settings when applicable, generation
seed, evaluator checkpoint, and reference-cache fingerprint. A report is
comparable with another report only when these protocol fields match, except for
the generative checkpoint or method fields intentionally under comparison.

## Feature caches

Real test features are computed once. Generated features are cached per
generative checkpoint and complete sampling configuration. Each cache contains
features, class probabilities, labels, sample identifiers, and a
JSON-compatible provenance payload.

Cache reuse requires an exact fingerprint match across:

- dataset identity and split;
- source sample identifiers;
- image normalization and shape;
- evaluator architecture, version, and weights;
- feature layer and feature dimension;
- generation configuration for generated caches.

A changed evaluator, preprocessing contract, or generation configuration
invalidates the corresponding cache rather than producing a warning and
continuing.

## Metrics and report

The four headline metrics are:

- **Fashion-FID:** global FID over balanced real and generated evaluator
  embeddings;
- **Fashion-KID:** global KID over the same embeddings using the project's
  existing seeded subset estimator;
- **Fashion-IS:** Inception Score over evaluator class probabilities;
- **Fashion-Recall:** the project's existing generative-recall estimator in the
  evaluator feature space.

The long-tail breakdown contains per-class FID, the unweighted macro mean of
the ten class FIDs, worst-class FID, and pooled head/middle/tail FID. Frequency
groups are deterministic class thirds ranked by the exact training counts and
use the same convention as the current CIFAR evaluator. Per-class recall is
reported after generalizing the recall computation to labeled subsets; each
subset must satisfy the estimator's minimum sample requirements.

Conditional diagnostics are not headline quality metrics. They contain
requested-class accuracy, mean probability assigned to the requested class,
the predicted-class histogram, and a requested-versus-predicted confusion
matrix. These diagnostics distinguish a poor conditional model from a feature
distribution failure without replacing FID, KID, Inception Score, or recall.

The evaluator also produces a real-versus-real calibration report from a fixed,
stratified split of the test reference. It establishes the finite-sample noise
floor but is not subtracted from generated-sample scores.

JSON is the canonical output and contains all scalar metrics, per-class values,
group definitions, confusion data, sample counts, seeds, and provenance. A
concise table is a derived human-readable view.

## Components and data flow

The benchmark has four independently testable units:

1. `LongTailedImageDataset` owns deterministic class subsampling and metadata.
   Dataset-specific adapters own downloads and decoding.
2. `FashionMNISTEvaluator` owns classifier preprocessing, feature extraction,
   probability prediction, and checkpoint validation.
3. The feature-cache pipeline batches real or generated images through the
   frozen evaluator and validates cache provenance.
4. The metric-report pipeline consumes aligned caches and delegates numerical
   calculations to the existing metric functions.

The data flow is:

```text
Fashion-MNIST train -> seeded long-tail subset -> conditional model training
Fashion-MNIST test  -> frozen evaluator       -> real feature cache
balanced conditions -> model sampler          -> generated images
generated images    -> frozen evaluator       -> generated feature cache
aligned caches      -> existing metrics       -> JSON report and table
```

No metric implementation trains or mutates the classifier, dataset, or
generative model.

## Failures and validation

Evaluation rejects missing or incompatible evaluator checkpoints, checkpoints
below the required held-out accuracy, non-finite features, inconsistent feature
dimensions, stale caches, unbalanced generated class counts, label/feature
misalignment, absent classes, and insufficient per-class sample counts. Errors
identify the incompatible field and expected value.

Tests cover:

- exact exponential counts and deterministic selected indices;
- shared long-tail logic retaining existing CIFAR behavior;
- balanced conditional-label construction and the canonical sample count;
- evaluator checkpoint metadata and accuracy validation;
- cache round trips, fingerprint matches, and invalidation;
- zero or near-zero FID for identical features and worsening metrics for
  controlled feature shifts;
- correct classwise, macro, worst-class, and head/middle/tail aggregation;
- requested-label diagnostics and confusion matrices;
- a tiny end-to-end smoke evaluation using a lightweight injected evaluator;
- calibration showing that progressively corrupted Fashion-MNIST images worsen
  FID/KID and recall in aggregate.

The full test suite must remain green. The first implementation milestone is a
reproducible Fashion-MNIST benchmark; generalized MNIST support and migration of
CIFAR onto the shared wrapper are follow-up work unless required to avoid
duplicating the long-tail selection primitive.
