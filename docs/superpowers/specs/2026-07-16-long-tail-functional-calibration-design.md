# Probe-A Functional Calibration Design

**Date:** 2026-07-16  
**Status:** approved by the user's instruction to implement the recommended Observation-0 follow-up

## Purpose

Observation 0 established that exact-gradient covariance directions are repeatable in adjacent non-output network stages. It did not establish that those directions cause a non-negligible or class-specific change in the learned flow field. This calibration asks the narrower question required before Stage 1:

> Does a rank-1 direction recovered from the reliable Observation-0 cells support a local, reproducible, class-selective decrease in ordinary flow-matching loss at a perturbation whose held-out effect is approximately 1%?

The calibration is diagnostic only. It does not train an adapter, evaluate a proposed method, inspect Probe-B, or launch any nonzero frequency mapping.

## Locked scope

- Observation-0 primary artifacts only.
- Checkpoints: step 20,000 as the primary calibration and step 500 as an early-training positive control.
- Timestep stratum: ID 0, corresponding to `(0.02, 0.10)`.
- Rank: 1.
- Layers: `down2_block.conv2.weight` and `middle.conv2.weight`.
- Classes: `0, 2, 3, 4, 6, 9`, the classes measurable for the adjacent layer pair in every seed at the final checkpoint.
- Training seeds: the three seeds locked by the Observation-0 preregistration.
- Probe view: Probe-A only. Probe-B remains unopened for later confirmatory work.

The implementation validates every one of these identities against the locked Observation-0 preregistration, registry, manifest, checkpoints, and noise-ceiling decision before doing any computation.

## Why a projected descent direction

Three constructions were considered:

1. Perturb directly along both signs of the top covariance eigenvector. This is sign-invariant, but it tests sensitivity rather than whether the recovered subspace contains a useful update.
2. Use the ordinary class mean gradient. This guarantees a local descent direction but no longer tests the spectral observation.
3. Project a held-out class mean gradient into the rank-1 covariance subspace, `d = -UU^T g`, and normalize the result. This tests whether the recovered subspace contains a useful class-conditioned update while giving the update a non-arbitrary sign.

The third construction is used. For each seed, checkpoint, class, and layer, the top right singular vector `U` is fitted from centered exact microbatch gradients. A disjoint Probe-A scale partition supplies `g`. The signed unit direction is

\[
\hat d = -\frac{UU^T\bar g_{\mathrm{scale}}}
                  {\lVert UU^T\bar g_{\mathrm{scale}}\rVert_2}.
\]

The unnormalized projection fraction

\[
\rho = \frac{\lVert UU^T\bar g_{\mathrm{scale}}\rVert_2}
              {\lVert\bar g_{\mathrm{scale}}\rVert_2}
\]

is reported. A direction with a numerically negligible projection is invalid rather than silently rescued by normalization.

## Probe-A cross-fitting

Each selected class/stratum cell contains 16 deterministic microbatches. Their stable within-cell order is divided before any loss is evaluated:

- positions 0-7: fit the centered rank-1 exact-gradient direction;
- positions 8-11: orient the projected descent direction and choose the perturbation scale;
- positions 12-15: evaluate target benefit and the complete cross-class response matrix.

This 8/4/4 split prevents direction discovery, scale calibration, and the reported selectivity result from sharing rows. The split is positional within a cell, not based on global manifest microbatch IDs.

## Perturbation and scale calibration

For layer `l`, seed `s`, and direction class `c`, perturb only that layer:

\[
\theta_l' = \theta_l + \epsilon\lVert\theta_l\rVert_2\hat d_{s,c,l}.
\]

One shared dimensionless `epsilon_l` is selected per layer at step 20,000. Sharing the scale across seeds and classes avoids making a 1% effect true by separately tuning every cell. The locked geometric grid is

`[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`.

For each grid value, actual finite-update losses are evaluated on the scale partition. The chosen value minimizes the absolute distance between the median relative same-class loss decrease and the 1% target, preferring the smaller value on a tie. Step 500 uses the same layerwise epsilon chosen at step 20,000; it is not recalibrated.

The scale is valid only when:

- its median scale-partition benefit lies in `[0.75%, 1.25%]`;
- its maximum relative layer step is at most 1%; and
- doubling the selected step changes the observed aggregate loss change by no more than 10% relative to the first-order doubling prediction.

All model mutations occur under a restoring context manager and are followed by an exact equality check, so checkpoints and in-memory base models cannot be changed accidentally.

## Functional response and controls

For direction class `c` and evaluation class `j`, record

\[
R_{c,j}=-\frac{L_j(\theta+\delta_c)-L_j(\theta)}{L_j(\theta)}.
\]

Positive values are improvements. The held-out target benefit is `R[c,c]`. Non-target harm is `max(0, -min_{j != c} R[c,j])`. The class-selectivity margin is target benefit minus non-target harm.

Matched random rank-1 subspaces are generated deterministically for every seed/checkpoint/class/layer. Each random vector is oriented by the same scale-partition mean gradient, receives the same selected epsilon, and is evaluated on the same held-out rows. This is stricter than a class permutation alone because it asks whether the observed subspace is better than a generic one-dimensional descent opportunity at the same parameter-space step.

The production preregistration uses 99 random controls. Tests may use fewer controls through an explicitly separate toy preregistration.

## Decision rule

Stage 1 is unlocked only if both selected layers pass at step 20,000:

1. the shared 1% scale is valid;
2. the seed/class block-bootstrap 95% lower bound for median target benefit is positive;
3. the bootstrap 95% upper bound for median non-target harm is less than half the median target benefit;
4. the observed median selectivity margin exceeds the 99th percentile of the matched-random-control margins; and
5. at least two of three seeds have a positive median target benefit.

Step 500 is reported as a positive control and must have a positive aggregate target benefit, but it cannot compensate for a failed final-checkpoint gate.

The bootstrap resamples the 18 seed-by-direction-class blocks, preserving each block's off-class response vector. Its random seed and 10,000 resamples are locked in the calibration preregistration.

## Artifacts and resumability

The command writes beneath `STUDY/aggregate/functional_calibration/`:

- `preregistration.yaml`: immutable calibration contract plus digest;
- `directions/seed_*/checkpoint_*/<layer>/class_*.pt`: exact unit directions and provenance;
- `scale_grid.csv`: every candidate's finite target effect;
- `responses.csv`: base and perturbed loss responses for primary and random directions;
- `functional_lock.json`: decision, confidence bounds, selected scales, input digests, and only allowed next action;
- `complete.json`: content digests proving the artifact set is complete.

A repeated command validates and reuses a complete artifact set. Partial direction files may be reused only after their provenance and tensor digest validate; aggregate decision files are replaced atomically after all requested cells finish.

`functional_lock.json` is the only artifact that may unlock Stage 1. Merely producing directions or reaching a 1% training-partition change is insufficient.

## Failure interpretation

- No valid 1% local scale: the measurable covariance geometry is too functionally weak or too nonlinear at the proposed practical threshold.
- Target benefit without selectivity: the subspace contains a generic descent direction, not class-private functional structure.
- Random controls match the selected direction: the spectral observation adds no local functional information beyond dimensionality.
- Step 500 passes but step 20,000 fails: the geometry is an acquisition-stage phenomenon and should motivate dynamics work, not a fixed late-training intervention.
- Both checkpoints fail: stop before Stage 1 and revise the geometric hypothesis.

