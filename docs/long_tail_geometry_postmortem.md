# Long-Tail Gradient/Functional Geometry Investigation: Postmortem

**Decision:** Do not promote this direction. Retain only hypothesis-neutral
instrumentation.

## What was tested

The investigation asked whether long-tail generation produced a stable,
low-rank gradient geometry that was shared across seeds, layers, checkpoints,
and classes, and whether the resulting directions predicted beneficial local
parameter transport. The sequence included deterministic gradient probes,
positive and negative controls, a held-out functional calibration, a
representation-matched audit, and a terminal CIFAR-10-LT replication.

## Decisive observations

- Fashion-MNIST functional calibration failed its positive control, so Stage 1
  remained blocked.
- The representation audit produced mixed, class-heterogeneous transport rather
  than a stable shared mechanism. The normalized slopes for the two selected
  layers were `1.82596` and `1.86796`; the raw slopes were `1.66326` and
  `1.08182`.
- CIFAR-10-LT made the effect measurable network-wide but not stable. The number
  of layers sharing rank-1 directions across seeds at steps
  `0/2500/10000/25000/50000/100000` was `0/5/6/2/4/1`.
- At the final checkpoint, the numbers of shared selected layers were
  `1/2/2/0` for ranks `1/2/4/8`, below the preregistered minimum of `5`.

These observations reject the proposed stable, low-rank, cross-seed transport
mechanism under the tested setups. They do not establish that long-tail
generation lacks useful geometry; they show that this particular operational
hypothesis did not survive its controls and natural-image test.

## Repository consequence

The protocol state machines, gates, experiment services, CLIs, configs, and
hypothesis-specific tests were removed. Deterministic manifests, exact replay,
gradient extraction, sketches, controls, subspace mathematics, and reversible
perturbations remain under `fm_lab.diagnostics.probes`.

Historical run artifacts remain in ignored `runs/` paths. They are not part of
the supported API and were not modified by the cleanup.
