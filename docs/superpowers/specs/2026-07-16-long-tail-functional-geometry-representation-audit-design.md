# Representation-Matched Functional Geometry Audit Design

**Date:** 2026-07-16
**Status:** approved by the user's instruction to proceed with the recommended audit

## Purpose

The completed Probe-A functional calibration correctly blocked Stage 1, but it did
not test the exact representation that produced the Observation-0 reliability
signal. Observation 0 formed its rank-1 geometry from row-normalized gradient
sketches, whereas the calibration fitted rank-1 directions from raw exact gradient
rows. Because heterogeneous gradient norms can rotate a covariance eigenvector,
the failed lock supports a narrow conclusion about the raw-gradient construction,
not yet a general conclusion about the normalized Observation-0 geometry.

This audit asks three diagnostic questions:

1. Does the row-normalized exact rank-1 subspace have a positive infinitesimal
   held-out loss effect when oriented by a disjoint raw loss gradient?
2. Does it transport more reliably than the raw-gradient rank-1 subspace under
   identical Probe-A partitions?
3. What is the largest effect that remains compatible with the measured local
   derivative, without weakening the failed calibration's preregistered 1% gate?

The audit is explanatory. It cannot unlock Stage 1, alter
`functional_lock.json`, open Probe B, train a model, or change the original
calibration thresholds.

## Approaches considered

Three implementation strategies were considered.

1. **Rewrite and rerun the original calibration with normalized rows.** This would
   erase the distinction between a preregistered failure and a post-failure audit,
   and could accidentally turn exploratory evidence into an unlock artifact.
2. **Run a one-off analysis script.** This is quick but does not give immutable
   inputs, resumability, tamper detection, or a stable interpretation contract.
3. **Add a separate, digest-bound representation audit.** This preserves the
   failed lock, compares both representations on exactly the same rows, measures
   infinitesimal transport before finite perturbations, and produces an explicit
   non-unlocking conclusion.

The third approach is selected.

## Locked upstream state

The audit requires all of the following before it performs model computation:

- the completed Observation-0 primary study and its existing digest checks;
- the completed functional calibration with
  `stage1_unlocked == false`, `probe_b_opened == false`, and next action
  `stop_stage1_and_revise_functional_geometry`;
- the original functional preregistration digest
  `f40251443426eab2f24d89cdf359f07615ccda788341d16719c1f0ec40836bdf`;
- checkpoints 500 and 20,000, with 20,000 primary and 500 retained only as an
  early-training comparison;
- Probe A, stratum 0 `(0.02, 0.10)`, rank 1;
- layers `down2_block.conv2.weight` and `middle.conv2.weight`;
- classes `0, 2, 3, 4, 6, 9` and the three Observation-0 training seeds;
- all 16 deterministic microbatches in every selected class/stratum cell.

The audit records content digests for the original functional lock and its
`complete.json`. It never loads a path containing `probe_b.npz`.

## Balanced cross-fitting

Four circular folds are fixed by offsets `(0, 4, 8, 12)`. For offset `o`, the
ordered positions are `(o + k) mod 16` for `k = 0, ..., 15`; positions 0-7 in
that order fit the basis, positions 8-11 orient it, and positions 12-15 evaluate
it. Explicitly:

| Fold | Fit positions | Scale positions | Evaluation positions |
|---|---|---|---|
| 0 | 0-7 | 8-11 | 12-15 |
| 1 | 4-11 | 12-15 | 0-3 |
| 2 | 8-15 | 0-3 | 4-7 |
| 3 | 12-15, 0-3 | 4-7 | 8-11 |

Every microbatch therefore appears in the scale partition once, in the
evaluation partition once, and in the fit partition twice. Fold results are
collapsed to a median within each seed-by-class block before uncertainty is
computed; folds are not treated as independent samples.

## Paired basis construction

For a fixed seed, checkpoint, class, layer, and fold, collect the same exact raw
microbatch-gradient matrix `G_fit` once. Construct two rank-1 bases:

- `raw`: the top centered covariance direction of `G_fit`;
- `row_normalized`: the top centered covariance direction of rows
  `G_fit[i] / ||G_fit[i]||`.

Both bases are oriented using the same disjoint **raw** mean scale gradient:

\[
d_b = -\frac{U_b U_b^\top \bar g_{\mathrm{scale}}}
             {\lVert U_b U_b^\top \bar g_{\mathrm{scale}}\rVert_2},
\qquad b \in \{\mathrm{raw},\mathrm{row\_normalized}\}.
\]

This isolates the effect of the covariance representation. Normalizing the
orientation gradient as well would change two variables and would no longer be
the derivative of the ordinary loss. The audit records gradient-norm
coefficient of variation, each basis's explained fraction and projection
fraction, and the absolute cosine between the two fitted bases.

## Zero-step functional transport

For evaluation class `j`, let `L_j` and `g_j` be the mean ordinary Probe-A loss
and raw mean gradient on that fold's evaluation rows. The model perturbation is

\[
\theta_l(\epsilon)=\theta_l+\epsilon\lVert\theta_l\rVert_2 d_b.
\]

The exact first derivative of relative benefit at zero is

\[
s_{b,c\to j}
= \left.\frac{d}{d\epsilon}
  \left[-\frac{L_j(\theta_l(\epsilon))-L_j(\theta_l)}{L_j(\theta_l)}\right]
  \right|_{\epsilon=0}
= -\frac{\lVert\theta_l\rVert_2\langle g_j,d_b\rangle}{L_j}.
\]

Positive slope means infinitesimal improvement. The full six-by-six class
response is recorded for both bases. For each direction class, slope
selectivity is target slope minus worst off-class harm:

\[
m_{b,c}=s_{b,c\to c}-\max(0,-\min_{j\ne c}s_{b,c\to j}).
\]

This derivative is the primary audit endpoint because it separates sign
transport from finite-step curvature.

## Local finite-step check

The only finite relative steps are `(1e-4, 3e-4, 1e-3)`. No step is selected to
hit 1%, and the original 1% acceptance interval remains unchanged and failed.
For each basis and target class, finite benefit is measured on both the scale
and evaluation partitions. The audit compares it with the zero-step prediction
`epsilon * slope` and reports

\[
e(\epsilon)=
\frac{|B(\epsilon)-\epsilon s|}
     {\max(|\epsilon s|,10^{-12})}.
\]

A step is described as locally concordant when its aggregate median error is at
most 10%. This label is descriptive and cannot satisfy the original calibration
gate. Checkpoint 500 receives its own slope and finite grid; it does not reuse a
step chosen at checkpoint 20,000.

## Blockwise uncertainty and interpretation

For every checkpoint, layer, and basis, fold medians are first computed inside
each of the 18 seed-by-direction-class blocks. A deterministic 95% percentile
bootstrap then resamples those 18 blocks. Paired normalized-minus-raw
differences use the same blocks and folds before bootstrapping.

At checkpoint 20,000 a basis has **positive local transport** for a layer when:

- the 95% lower bound of median target slope is positive;
- median slope selectivity is positive; and
- at least two of three seed-level target-slope medians are positive.

The audit status is one of:

- `normalized_representation_rescue`: row-normalized transport is positive in
  both layers and the paired normalized-minus-raw slope lower bound is positive
  in both layers;
- `representation_independent_local_transport`: both representations have
  positive transport in both layers, without a two-layer paired advantage for
  normalized rows;
- `no_transferable_local_descent`: the row-normalized target-slope 95% upper
  bound is non-positive in both layers;
- `mixed_or_class_heterogeneous_transport`: every other complete result.

These labels summarize evidence; none unlocks Stage 1. The corresponding next
actions are, respectively:

- review a separately preregistered small-local-step method;
- study partition/finite-step curvature rather than representation choice;
- pivot from covariance directions to the origin of gradient-sign transport
  failure;
- analyze class- and seed-conditioned transport heterogeneity before proposing
  a method.

## Components and artifact contract

The implementation adds:

- `functional_audit_preregistration.py`: exact schema, validation, digest, and
  immutable lock;
- `functional_audit.py`: cross-fold construction, paired exact directions,
  analytic slopes, finite responses, blockwise analysis, resumable chunks, and
  artifact validation;
- a canonical Fashion-MNIST audit YAML;
- an `audit-functional-geometry` subcommand in the existing Observation-0 CLI.

Artifacts are written only beneath
`STUDY/aggregate/functional_geometry_audit/`:

- `preregistration.yaml`;
- `chunks/metrics_seed_<s>_checkpoint_<step>.csv` and digest sidecars;
- `slopes.csv`, containing every cross-class zero-step slope;
- `finite_steps.csv`, containing target scale/evaluation finite responses;
- `basis_comparison.csv`, containing paired geometry diagnostics and vector
  digests but not large direction tensors;
- `audit_summary.json`, containing bootstrap intervals, status, original blocked
  state, `probe_b_opened: false`, and the only audit next action;
- `complete.json`, binding every aggregate artifact and upstream input digest.

A repeated command validates and reuses complete artifacts. A partial or
tampered chunk, upstream lock, summary, or aggregate table is rejected rather
than silently recomputed. No file beneath `functional_calibration/` is written.

## Testing and failure handling

Tests cover the exact fold balance, representation difference under
heterogeneous row norms, analytic slope against an autograd/finite-difference
example, complete table validation, blockwise fold collapse, all interpretation
statuses, Probe-B exclusion, preservation of the original lock, resumability,
tamper rejection, CLI output, and absence of any Stage-1 path.

Numerically negligible projected directions remain invalid. Missing rows,
non-finite losses or gradients, incomplete folds/classes/seeds, changed input
digests, and a functional calibration that is not in the required blocked state
all fail closed with no summary artifact.
