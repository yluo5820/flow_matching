# Long-Tail Experimental Strand Cleanup Design

**Date:** 2026-07-17

**Status:** Approved direction; written specification awaiting review

**Scope:** Remove the falsified gradient-spectral/functional-geometry experiment
framework while retaining reusable diagnostics and paper-derived long-tail
baselines, including Capacity Manipulation (CM).

## Decision

Use a surgical extraction rather than reverting the strand or leaving it
deprecated in place. The cleanup will:

1. retain the long-tail benchmark, evaluation, and training foundations;
2. retain CM as a supported baseline and clearly label the present
   implementation as a continuous-flow adaptation;
3. extract genuinely reusable probe primitives from
   `fm_lab.diagnostics.long_tail_geometry` into a hypothesis-neutral package;
4. delete the geometry hypothesis' protocols, decision gates, CLIs, configs,
   tests, and superseded working documents; and
5. preserve a concise postmortem containing the negative result and the
   evidence needed to avoid repeating the same investigation.

This is a source cleanup only. Existing ignored run artifacts under `runs/`
will not be deleted or rewritten.

## Why Capacity Manipulation Stays

CM is an independent literature baseline, not a consequence of the rejected
gradient-spectral hypothesis. Hong et al.'s method adds a switchable low-rank
parameter branch and trains the distance between capacity-on and capacity-off
predictions with class-frequency-dependent consistency and diversity weights.
The authors' released CIFAR-100-LT configuration uses rank ratio `0.1`, applies
the branch to upsampling blocks, and sets `w_con=1.0`, `w_div=0.2`.

The local implementation matches that core mechanism and those defaults, but
it is not an exact reproduction of the released experiment:

- it applies CM to a continuous flow-matching objective instead of the
  authors' discrete noise-prediction diffusion objective;
- the released recipe also enables endpoint target transfer, whereas the
  current canonical local CM config uses CM alone; and
- the local convolution adapter is a conventional flattened-kernel low-rank
  factorization, while the released code uses a spatially structured matrix
  factorization before reshaping the kernel.

Accordingly, retained configs and documentation must call it a
**continuous-flow CM adaptation**. They must not claim exact paper reproduction.
The implementation will keep an extension seam for a later matched paper
baseline without restoring the discarded geometry machinery.

Primary references:

- [OpenReview paper](https://openreview.net/forum?id=wSGle6ag5I)
- [Authors' implementation](https://github.com/Feng-Hong/ImbDiff-CM)
- [Authors' released CM configuration](https://github.com/Feng-Hong/ImbDiff-CM/blob/main/configs/cifar100lt_ir100/cm.yaml)

## Retain, Extract, and Remove

| Area | Action | Resulting contract |
|---|---|---|
| Fashion-MNIST-LT and CIFAR-LT data | Retain | Deterministic imbalance construction, class counts/frequency groups, and held-out diagnostic pools remain available. |
| Evaluation | Retain | FID/KID/recall, classwise and frequency-group metrics, feature caching, and provenance remain supported. |
| Continuous training foundation | Retain | Prediction conversion, `x_vloss`, CBDM, OC, EMA, exact resume, early stopping, time sampling, and explicit checkpoint schedules remain. |
| Capacity Manipulation | Retain and clarify | Keep model adapters, capacity routing, `CMModifier`, metadata, focused unit tests, and one canonical continuous-flow CM config. |
| CM audit variants | Remove | Delete bounded-diversity, lambda sweeps, full-network, scale sweeps, short pilots, and their audit-only assertions/docs. They are investigation artifacts, not baseline API. |
| Probe primitives | Extract | Create `fm_lab.diagnostics.probes` for deterministic batch manifests/replay, checkpoint restoration and loss replay, layer resolution, exact per-layer gradients, CountSketch/control projections, pure subspace statistics, and reversible parameter perturbation/loss response. |
| Geometry protocols | Remove | Delete Observation 0, functional calibration, representation audit, CIFAR transport falsification, preregistration schemas, reliability gates, registries, statuses, and allowed-next-action logic. |
| Geometry entry points/configs/tests | Remove | Delete Stage 0/Observation 0 CLIs, geometry-specific YAML, protocol/orchestration tests, and the optional `pyarrow` dependency if no retained code imports it. |
| Training `diagnostic_stop` hook | Remove | The probe package will work from checkpoints; the audit-only training stop is not part of CM or the general training contract. Any retained probe caller that still depends on it must be migrated to explicit checkpoint schedules before deletion. |
| Historical documents | Consolidate | Remove superseded executable plans/specs for this strand and replace them with one concise postmortem plus this cleanup design. Preserve unrelated long-tail and CM baseline design history only where it remains accurate. |

## Retained Probe API Boundary

The new `fm_lab.diagnostics.probes` package is deliberately mechanism-free.
It may expose operations and measurements, but never scientific conclusions.

Permitted responsibilities:

- construct and validate deterministic sample manifests;
- replay the same examples, noise, time values, labels, and objective state;
- load a checkpoint without silently changing model or objective semantics;
- resolve explicitly named parameterized layers and compute their exact
  gradients;
- create deterministic random/control sketches and compute projection or
  subspace statistics as pure functions;
- apply a reversible perturbation to chosen parameters and measure a caller-
  supplied loss response; and
- return tensors/records with provenance sufficient for downstream analysis.

Forbidden responsibilities:

- hard-coded head/tail scientific interpretations;
- positive-control pass/fail gates;
- rank or layer selection rules tied to the abandoned hypothesis;
- protocol stage locks, preregistration state machines, or prescribed next
  actions; and
- Fashion-MNIST/CIFAR-specific conclusions embedded in library code.

This boundary keeps useful experimental instrumentation without allowing the
old hypothesis to survive under generic filenames.

## CM Baseline Contract

The retained CM path must satisfy all of the following:

1. `use_capacity=True` adds the zero-initialized low-rank branch and
   `use_capacity=False` evaluates the shared base parameterization.
2. With the branch initially zero, capacity-on and capacity-off predictions are
   identical without perturbing the base model's seeded initialization stream.
3. The modifier computes per-sample squared prediction distance in an explicit
   comparison space and applies the paper-derived frequency weights:

   \[
   L_{CM}=K\,\mathbb{E}_i\left[
     (w_{con}p_{y_i}-w_{div}\tilde p^{-1}_{y_i})
     \lVert f_{on}(x_i,t_i)-f_{off}(x_i,t_i)\rVert_2^2
   \right].
   \]

4. The canonical adaptation uses `rank_ratio=0.1`, `parts=[up]`,
   `consistency_weight=1.0`, `diversity_weight=0.2`, and unbounded distance,
   matching the released core defaults while documenting the objective and
   factorization differences.
5. Bounded diversity remains removed from the public baseline unless a future
   project independently motivates it; it is not part of the released CM
   method.
6. CM tests cover switch equivalence, gradient flow into the low-rank branch,
   frequency weighting, config/model validation, metadata, and a short training
   smoke test.

## Migration and Compatibility

- Internal imports from `fm_lab.diagnostics.long_tail_geometry` will be moved
  to `fm_lab.diagnostics.probes` only for retained primitives.
- No compatibility aliases will be left for deleted protocol services. A stale
  import should fail loudly rather than imply continued support.
- Checkpoints containing capacity parameters remain loadable by the retained
  CM-capable model definitions.
- The existing local precision-order edit in
  `fm_lab/diagnostics/long_tail_geometry/checkpoints.py`—moving to CPU before
  converting to double—will be preserved when checkpoint replay is migrated to
  the new probe package.
- Run directories and historical result JSON are external artifacts and are
  not migration targets.

## Documentation Outcome

The repository will have three current documents for this area:

1. the general long-tail benchmark/baseline documentation;
2. a CM adaptation note that states what matches the paper and what does not;
3. a geometry-strand postmortem recording the observations, failed controls,
   cross-dataset falsification, and the decision not to promote the direction.

The postmortem will report the negative evidence without preserving executable
stage machinery. Old plans that are useful only as an implementation diary will
be removed from the active docs tree.

## Verification

Implementation is complete only when:

- no production import or CLI references `long_tail_geometry`;
- no geometry-specific config or protocol test remains;
- no retained code imports `pyarrow` before its optional dependency is removed;
- the canonical CM adaptation config parses and passes a short smoke test;
- focused CM tests pass, including initial on/off equality and adapter gradient
  flow;
- focused probe tests pass for deterministic replay, gradient extraction,
  sketches/subspaces, and reversible perturbation;
- the broader data, training, evaluation, and config smoke suites pass;
- a repository-wide literal search finds no stale deleted entry-point names;
  and
- `git diff` confirms the pre-existing checkpoint precision-order change was
  preserved semantically and no ignored run artifact was touched.

## Non-Goals

- Re-running CM, improving its empirical result, or claiming the adaptation is
  a faithful reproduction.
- Implementing the authors' discrete diffusion stack or structured convolution
  factorization during this cleanup.
- Deleting reusable image-geometry exploration tools that predate and are
  independent of the long-tail gradient-spectral strand.
- Choosing the next theoretical research direction.
