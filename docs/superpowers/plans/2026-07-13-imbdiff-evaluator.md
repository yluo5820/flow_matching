# ImbDiff Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen, reference-compatible CIFAR evaluator with FID, KID, Recall, IS, classwise FID, and Many/Medium/Few FID.

**Architecture:** Pure NumPy/SciPy metric functions consume a versioned feature-cache artifact. A separate reference Inception extractor owns image preprocessing and weights, while a CLI orchestrates real/generated loading, repeats, grouping, and reports.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch, torchvision optional extra, pytest.

## Global Constraints

- Reference-compatible FID/KID use TensorFlow-FID Inception-v3 2,048-D features.
- Real statistics use balanced CIFAR training images.
- Extended metrics and frequency-ranked class thirds are labeled as extensions.
- No silent weight-backend substitution is allowed.

---

### Task 1: Pure feature metrics

**Files:** Create `fm_lab/evaluation/metrics.py`, `fm_lab/evaluation/groups.py`, and `tests/test_imbdiff_metrics.py`.

- [ ] Write failing equation, determinism, edge-case, classwise, and group tests.
- [ ] Implement FID, KID, Recall, IS, repeat summaries, and frequency-ranked thirds.
- [ ] Run focused pytest and Ruff; commit the verified pure core.

### Task 2: Versioned feature cache

**Files:** Create `fm_lab/evaluation/cache.py` and `tests/test_imbdiff_feature_cache.py`.

- [ ] Write failing round-trip, fingerprint, corruption, and label-alignment tests.
- [ ] Implement atomic `.npz` cache writes and strict provenance matching.
- [ ] Run focused tests and commit.

### Task 3: Reference-compatible Inception extraction

**Files:** Create `fm_lab/evaluation/inception.py`, `fm_lab/evaluation/features.py`, and `tests/test_imbdiff_features.py`; modify `pyproject.toml` only if needed.

- [ ] Write fake-model tests for range conversion, resize, batching, pool features, and probabilities.
- [ ] Port the reference TensorFlow-FID Inception patches and require the exact weight artifact.
- [ ] Run focused tests and commit.

### Task 4: Report orchestration and CLI

**Files:** Create `fm_lab/evaluation/report.py`, `fm_lab/experiments/run_imbdiff_eval.py`, and `tests/test_imbdiff_eval.py`; modify `pyproject.toml` and `docs/imbdiff_cm_reproduction.md`.

- [ ] Write failing synthetic-cache CLI and report-schema tests.
- [ ] Implement repeat selection, metric aggregation, JSON/CSV output, and provenance.
- [ ] Document commands, run full pytest/Ruff/diff verification, commit, and push Round 4.
