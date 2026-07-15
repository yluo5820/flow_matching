# CM Weighting Audit Design

## Goal

Determine whether the weak continuous Capacity Manipulation (CM) signal comes
from the extra `diversity_weight: 0.2` attenuation rather than from continuous
flow matching, target-space comparison, or adapter coverage. The audit uses
short Fashion-MNIST IR100 pilots and must not promote an unstable unbounded
negative objective to a full experiment.

## Evidence and hypothesis

The implemented class weights match the CM equations:

- consistency weight: `C * N_y / sum(N_c)`;
- diversity weight: `C / (N_y * sum_c(1 / N_c))`.

The current production config then multiplies the diversity term by another
`0.2`. For Fashion-MNIST IR100 this leaves only classes 7--9 with a net
diversity incentive. The matched 1,000-step pilots showed:

- decoder-only CM/base ratio: `0.0089%`;
- full-network CM/base ratio: `0.0288%`;
- the full-network distance was larger for medium classes than for few classes.

The primary hypothesis is that the extra diversity attenuation suppresses the
minority-specific capacity gap. The alternative is that imbalanced sampling
causes head-class consistency gradients to dominate even with the original
relative weighting.

## Pilot matrix

Use the canonical decoder-only capacity configuration for all pilots. Reuse the
existing `consistency=1, diversity=0.2, unbounded` pilot as the control.

1. **Original relative weighting, unbounded diagnostic**
   - `consistency_weight: 1.0`
   - `diversity_weight: 1.0`
   - unbounded diversity
   - 1,000 steps
   - diagnostic only; abort on non-finite values, a CM loss below the negative
     base-loss magnitude, or a capacity distance above `1.0`

2. **Original relative weighting, bounded candidate**
   - `consistency_weight: 1.0`
   - `diversity_weight: 1.0`
   - bounded diversity
   - initial margin `0.001`
   - 1,000 steps

The margin is about five times the maximum distance observed in the stable
full-network pilot (`1.89e-4`). It permits a materially larger gap without
allowing unlimited reward from increasing branch distance. If it saturates
before step 200, run one replacement pilot with margin `0.01`; do not silently
change any other variable.

## Safety behavior

Add an optional CM diagnostic stop policy to training rather than relying on a
shell-side log watcher. It checks the already-computed scalar metrics after each
step and stops cleanly with the checkpoint and stop reason recorded. The policy
is disabled by default and only the unbounded diagnostic config enables it.

The bounded loss retains the existing hard cap on diversity distance. Once the
distance reaches the margin, the negative diversity gradient becomes zero while
the consistency term remains active, preventing divergence toward negative
infinity.

## Measurements and decision rule

Record existing base loss, CM loss, CM/base ratio, and many/medium/few distance,
plus:

- bounded-diversity saturation rate;
- diagnostic stop reason and step;
- per-class capacity distance, so group averages cannot hide one collapsing
  class.

The bounded candidate justifies a longer run only if all of the following hold:

1. base loss remains finite and follows the control trajectory;
2. no safety stop occurs;
3. the few-class distance is larger than the many-class distance at the final
   checkpoint and across the latter half of training;
4. CM/base reaches at least `0.1%`, over ten times the matched decoder control;
5. saturation does not remain above `80%` for the latter half of training.

If removing `0.2` strengthens the signal but the few-class criterion still
fails, the next design will use a class-balanced auxiliary CM batch while
leaving the base flow-matching batch imbalanced. Adapter coverage will not be
changed again during this audit.

## Tests

Use test-driven development for new behavior:

- per-class distance metrics include every class with stable names;
- diagnostic stopping is disabled by default;
- each configured threshold produces a clean, metadata-recorded stop;
- bounded saturation and existing CM gradients remain correct;
- pilot configs differ only in the intended CM weighting and safety fields.

Run the focused tests, the complete pytest suite, Ruff, and `git diff --check`
before starting pilots and again before branch completion.

## Non-goals

- No full Fashion-MNIST or CIFAR-10 training run.
- No balanced auxiliary sampler in this iteration.
- No change to continuous interpolation, prediction conversion, or base loss.
- No change to capacity rank, coverage, optimizer, or time sampling.

## Reference

The audit follows the CM method described by Hong et al., *Improving Diffusion
Models for Class-Imbalanced Training Data via Capacity Manipulation*, ICLR 2026:
<https://openreview.net/forum?id=wSGle6ag5I>.
