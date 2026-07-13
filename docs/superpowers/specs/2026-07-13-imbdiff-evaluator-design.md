# ImbDiff Evaluator Design

## Objective

Round 4 creates one frozen evaluator for all DDPM, CBDM, OC, and CM
comparisons. FID and KID reproduce the released ImbDiff-CM equations and
TensorFlow-FID Inception-v3 preprocessing. Recall, Inception Score, classwise
FID, and frequency-group FID are clearly marked extensions because the
reference repository does not release implementations for them.

## Feature contract

Images enter as RGB CIFAR tensors in `[0, 1]`. The reference-compatible
extractor bilinearly resizes them to 299 square, maps them to `[-1, 1]`, and
uses the TensorFlow-FID Inception-v3 weights. Each image produces a 2,048-D
pool feature and classifier probabilities for Inception Score. Real features
come from the balanced CIFAR training set, matching the reference code.

Feature caches are `.npz` files containing features, probabilities, labels,
sample identifiers, and a JSON-compatible provenance payload. Cache reuse
requires an exact fingerprint match across dataset, split, preprocessing,
extractor weights, and evaluator version.

## Metrics

- FID uses SciPy `sqrtm` and the same numerical fallback as the reference.
- KID uses the unbiased cubic polynomial MMD, 100 subsets, and subset size up
  to 1,000, with an explicit seeded NumPy generator.
- Generative Recall uses the improved precision/recall manifold estimator:
  each real feature's radius is its k-th-nearest-real distance and recall is
  the fraction whose manifold contains at least one generated feature.
- Inception Score computes `exp(E[KL(p(y|x) || p(y))])` over deterministic
  splits and reports split values, mean, and standard deviation.
- Classwise FID compares generated and balanced-real features for each class.
- Many/Medium/Few are frequency-ranked class thirds derived from the exact
  long-tail training counts. Group FID pools features belonging to each third.

## Repeats and outputs

Each repeat selects generated samples without replacement using a recorded
seed. Reports contain every repeat, mean, population standard deviation,
classwise/group definitions, cache paths, sample counts, and full evaluator
provenance. JSON is the canonical output; CSV is a flattened convenience view.

## Interface and failures

`fm-lab-imbdiff-eval` accepts a run directory or explicit generated samples,
generated labels, dataset name/root, repeat count, seed, device, and cache
directory. It rejects missing/misaligned labels, fewer than two samples for
FID/KID, missing weight files, stale caches, and absent classes required for a
classwise score. It never silently substitutes torchvision ImageNet weights.

## Validation

Pure metric tests use analytically simple and synthetic features. Cache tests
verify fingerprints and round trips without loading Inception. Extractor tests
inject a tiny fake model to verify preprocessing and batching. CLI smoke tests
use cached synthetic features. Full regression tests must remain green.
