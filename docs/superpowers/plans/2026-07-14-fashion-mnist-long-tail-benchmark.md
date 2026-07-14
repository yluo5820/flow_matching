# Fashion-MNIST Long-Tail Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fast Fashion-MNIST long-tail conditional-generation benchmark that reports domain-specific FID, KID, Inception Score, recall, and tail-aware diagnostics.

**Architecture:** Reuse the existing CIFAR long-tail selector, generic feature-cache format, numerical metrics, and Fashion-MNIST classifier training path. Add a dataset adapter, expose frozen classifier embeddings and probabilities through the generic extractor, extend cached-feature reporting with balanced conditional diagnostics, and provide a Fashion-MNIST-specific CLI for reproducible extraction and evaluation.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy, scikit-learn, pytest, Ruff.

## Global Constraints

- Canonical training images are Fashion-MNIST's official training split at one-channel `28 x 28` resolution.
- Canonical real references are the untouched 10,000-image official test split.
- Canonical generation requests exactly 1,000 samples per class and 10,000 samples total.
- The headline metrics are Fashion-FID, Fashion-KID, Fashion-IS, and Fashion-Recall.
- Existing unrelated workspace changes must remain untouched.
- Every production behavior is introduced by a failing test first.

---

### Task 1: Shared Long-Tail Selection and Fashion-MNIST Dataset

**Files:**
- Create: `fm_lab/data/long_tail.py`
- Create: `fm_lab/data/fashion_mnist.py`
- Modify: `fm_lab/data/cifar_lt.py`
- Modify: `fm_lab/data/__init__.py`
- Modify: `fm_lab/experiments/factory.py`
- Create: `tests/test_fashion_mnist_lt_data.py`
- Modify: `tests/test_cifar_lt_data.py`

**Interfaces:**
- Produces: `long_tail_indices(labels: np.ndarray, *, num_classes: int, imbalance_type: str, imbalance_factor: float, seed: int) -> np.ndarray`.
- Produces: `LongTailedFashionMNIST` with `sample`, `sample_with_labels`, `all_samples_with_labels`, `labels`, `class_counts`, `selected_indices`, and `metadata`.
- Consumes: `load_dataset(InputConfig(type="fashion_mnist", ...))` for downloading and decoding Fashion-MNIST.

- [ ] **Step 1: Write failing selector and dataset tests**

  Add tests that construct a tiny balanced labeled tensor fixture, assert exact exponential counts `(10, 7, 5, 4, 3, 2, 2, 1, 1, 1)` for factor `0.1`, assert deterministic indices, verify image-label alignment, and assert `build_target({"data": {"name": "fashion_mnist_lt", ...}})` returns the new adapter. Inject a dataset loader callable so tests do not access the network.

- [ ] **Step 2: Verify the tests fail for missing interfaces**

  Run: `pytest tests/test_fashion_mnist_lt_data.py tests/test_cifar_lt_data.py -q`

  Expected: collection or assertion failure because `LongTailedFashionMNIST` and `long_tail_indices` do not exist.

- [ ] **Step 3: Implement the shared selector and adapter**

  Move the exact CIFAR selection behavior into `long_tail_indices`, including balanced-source validation, `np.random.RandomState(seed)`, per-class shuffling, and `int(n_max * imbalance_factor**exponent)`. Make CIFAR call the shared function. Implement Fashion-MNIST loading, normalization, deterministic selection, flattened sampling, and metadata with a subset SHA-256 fingerprint.

- [ ] **Step 4: Verify dataset tests pass**

  Run: `pytest tests/test_fashion_mnist_lt_data.py tests/test_cifar_lt_data.py -q`

  Expected: all selected tests pass.

- [ ] **Step 5: Commit the dataset slice**

  Run: `git add fm_lab/data/long_tail.py fm_lab/data/fashion_mnist.py fm_lab/data/cifar_lt.py fm_lab/data/__init__.py fm_lab/experiments/factory.py tests/test_fashion_mnist_lt_data.py tests/test_cifar_lt_data.py && git commit -m "Add long-tailed Fashion-MNIST data"`

### Task 2: Frozen Fashion-MNIST Feature Evaluator

**Files:**
- Modify: `fm_lab/diagnostics/mnist_eval.py`
- Modify: `fm_lab/evaluation/features.py`
- Create: `tests/test_fashion_mnist_evaluator.py`
- Modify: `tests/test_imbdiff_features.py`

**Interfaces:**
- Produces: `MNISTClassifier.forward_features(x: torch.Tensor) -> torch.Tensor`.
- Produces: `FashionMNISTFeatureEvaluator(nn.Module)` returning `(features, probabilities)` and exposing evaluator provenance.
- Produces: `extract_classifier_features(...) -> FeatureCache` with explicit preprocessing provenance.
- Preserves: `extract_inception_features(...)` as a compatible wrapper for CIFAR callers.

- [ ] **Step 1: Write failing feature/evaluator tests**

  Test that `forward_features` returns a stable 2-D embedding, the evaluator returns aligned embeddings and normalized ten-class probabilities, checkpoint validation rejects dataset/normalization mismatch and low held-out accuracy, and the generic extractor records `fashion_mnist_classifier` rather than Inception preprocessing.

- [ ] **Step 2: Verify evaluator tests fail**

  Run: `pytest tests/test_fashion_mnist_evaluator.py tests/test_imbdiff_features.py -q`

  Expected: failure because the embedding evaluator and generic extractor are missing.

- [ ] **Step 3: Implement evaluator and extraction interfaces**

  Split `MNISTClassifier` into `forward_features` plus the existing logits head. Load the existing Fashion-MNIST checkpoint payload, validate dataset identity, normalization, accuracy threshold, architecture version, and weight fingerprint, then wrap it to return features and softmax probabilities. Generalize the extraction loop while retaining the Inception wrapper and its exact provenance behavior.

- [ ] **Step 4: Verify evaluator tests pass**

  Run: `pytest tests/test_fashion_mnist_evaluator.py tests/test_imbdiff_features.py tests/test_mnist_eval.py -q`

  Expected: all selected tests pass.

- [ ] **Step 5: Commit the evaluator slice**

  Run: `git add fm_lab/diagnostics/mnist_eval.py fm_lab/evaluation/features.py tests/test_fashion_mnist_evaluator.py tests/test_imbdiff_features.py && git commit -m "Add Fashion-MNIST feature evaluator"`

### Task 3: Balanced Long-Tail Metrics and Diagnostics

**Files:**
- Modify: `fm_lab/evaluation/report.py`
- Modify: `tests/test_imbdiff_eval.py`

**Interfaces:**
- Extends: `evaluate_feature_caches(..., require_balanced_generated: bool = False, per_class_recall: bool = False, conditional_diagnostics: bool = False) -> dict[str, Any]`.
- Produces report keys: `macro_classwise_fid`, `worst_class_fid`, `classwise_recall`, and `conditional` while preserving existing CIFAR keys.

- [ ] **Step 1: Write failing report tests**

  Add fixtures with requested labels and deliberately altered probabilities. Assert rejection of unequal generated counts when balancing is required; macro and worst FID aggregation; per-class recall; requested-class accuracy; mean requested-class probability; predicted histogram; and a `10 x 10` confusion matrix.

- [ ] **Step 2: Verify report tests fail**

  Run: `pytest tests/test_imbdiff_eval.py -q`

  Expected: failure because the new report options and keys are absent.

- [ ] **Step 3: Implement minimal report extensions**

  Validate exact generated counts by class before repeat subsampling. Reuse `classwise_fid`, `generative_recall`, and `frequency_ranked_groups`; compute macro/worst summaries from repeat values; and calculate diagnostics directly from cached requested labels and probability argmax values. Keep all new behavior opt-in so the established ImbDiff report remains compatible.

- [ ] **Step 4: Verify metric tests pass**

  Run: `pytest tests/test_imbdiff_eval.py tests/test_imbdiff_metrics.py -q`

  Expected: all selected tests pass.

- [ ] **Step 5: Commit the reporting slice**

  Run: `git add fm_lab/evaluation/report.py tests/test_imbdiff_eval.py && git commit -m "Add balanced long-tail evaluator diagnostics"`

### Task 4: Fashion-MNIST Benchmark CLI and Cache Provenance

**Files:**
- Create: `fm_lab/experiments/run_fashion_mnist_lt_eval.py`
- Modify: `pyproject.toml`
- Create: `tests/test_fashion_mnist_lt_eval.py`

**Interfaces:**
- Produces CLI: `fm-lab-fashion-mnist-lt-eval`.
- Consumes either `--generated-cache` plus `--real-cache`, or generated `.npy` samples/labels plus Fashion-MNIST data root and classifier checkpoint.
- Writes canonical `metrics.json`, flattened `metrics.csv`, and validated feature caches.

- [ ] **Step 1: Write failing CLI tests**

  Test argument defaults (`1_000` per class, 10 classes, balanced enforcement), cached-feature evaluation, missing paired cache rejection, classifier provenance mismatch rejection, and report creation. Use tiny injected caches so tests do not train or download models.

- [ ] **Step 2: Verify CLI tests fail**

  Run: `pytest tests/test_fashion_mnist_lt_eval.py -q`

  Expected: import failure because the CLI module does not exist.

- [ ] **Step 3: Implement the CLI**

  Follow `run_imbdiff_eval.py` parsing and device conventions. Build or load the balanced official test reference, require equal generated counts, call the Fashion evaluator extractor, save fingerprinted caches, evaluate with all Fashion extensions enabled, and print the JSON output path. Register the console script in `pyproject.toml`.

- [ ] **Step 4: Verify CLI tests pass**

  Run: `pytest tests/test_fashion_mnist_lt_eval.py tests/test_imbdiff_feature_cache.py -q`

  Expected: all selected tests pass.

- [ ] **Step 5: Commit the CLI slice**

  Run: `git add fm_lab/experiments/run_fashion_mnist_lt_eval.py pyproject.toml tests/test_fashion_mnist_lt_eval.py && git commit -m "Add Fashion-MNIST long-tail evaluation CLI"`

### Task 5: Canonical Training Configuration and User Documentation

**Files:**
- Create: `configs/fashion_mnist_lt/fashion_mnist_lt_ir100.yaml`
- Modify: `README.md`
- Modify: `tests/test_config_smoke.py`
- Modify: `tests/test_docs_coverage.py`

**Interfaces:**
- Produces a runnable class-conditional Fashion-MNIST IR100 training configuration with one-channel `28 x 28` model/source dimensions.
- Documents evaluator training/cache creation and canonical balanced evaluation commands.

- [ ] **Step 1: Write failing configuration and documentation tests**

  Assert that the canonical config builds the new target, source and class-conditional model with matching dimensions, and that the README mentions the new console command and its balanced-reference rule.

- [ ] **Step 2: Verify documentation tests fail**

  Run: `pytest tests/test_config_smoke.py tests/test_docs_coverage.py -q`

  Expected: failure because the config and command documentation are missing.

- [ ] **Step 3: Add the canonical config and concise workflow documentation**

  Adapt the existing MNIST class-conditional model settings to `fashion_mnist_lt`, `imbalance_type: exp`, `imbalance_factor: 0.01`, native image shape `[1, 28, 28]`, and ten classes. Document training, generation with balanced labels, evaluator checkpoint creation through automatic load-or-train behavior, and evaluation.

- [ ] **Step 4: Verify config and docs tests pass**

  Run: `pytest tests/test_config_smoke.py tests/test_docs_coverage.py -q`

  Expected: all selected tests pass.

- [ ] **Step 5: Commit config and docs**

  Run: `git add configs/fashion_mnist_lt/fashion_mnist_lt_ir100.yaml README.md tests/test_config_smoke.py tests/test_docs_coverage.py && git commit -m "Document Fashion-MNIST long-tail benchmark"`

### Task 6: Calibration and Full Verification

**Files:**
- Create: `tests/test_fashion_mnist_metric_calibration.py`
- Modify only implementation files implicated by a failing calibration test.

**Interfaces:**
- Verifies the end-to-end cache-to-report contract without downloading data or running a long classifier training job.

- [ ] **Step 1: Write a failing corruption-calibration test**

  Use a deterministic lightweight evaluator and synthetic Fashion-shaped tensors. Compare clean copied images with progressively noise-corrupted versions and assert aggregate FID/KID do not improve while recall decreases for the strongest corruption.

- [ ] **Step 2: Verify calibration fails for the expected missing wiring or assertion**

  Run: `pytest tests/test_fashion_mnist_metric_calibration.py -q`

  Expected: failure until the complete generic extraction/report path is connected.

- [ ] **Step 3: Make the smallest required integration corrections**

  Correct only discovered alignment, provenance, or numerical-wiring issues, retaining the established metric formulas.

- [ ] **Step 4: Run focused and full verification**

  Run: `pytest tests/test_fashion_mnist_lt_data.py tests/test_fashion_mnist_evaluator.py tests/test_fashion_mnist_lt_eval.py tests/test_fashion_mnist_metric_calibration.py tests/test_cifar_lt_data.py tests/test_imbdiff_features.py tests/test_imbdiff_eval.py tests/test_imbdiff_metrics.py -q`

  Run: `pytest -q`

  Run: `ruff check .`

  Expected: every command exits zero with no failures or lint errors.

- [ ] **Step 5: Commit calibration and integration fixes**

  Run: `git add tests/test_fashion_mnist_metric_calibration.py fm_lab tests README.md configs pyproject.toml && git commit -m "Verify Fashion-MNIST long-tail benchmark"`
