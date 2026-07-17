# Synthetic Long-Tail Geometry Report

## Research question

When class examples become scarce, does a conditional generator selectively lose valid
directions of variation, especially for higher-dimensional class manifolds?

## Frozen hypotheses

- **H1:** log class frequency and calibrated data dimension interact on factor coverage.
- **H2:** factor loss is accompanied by tangent-alignment or FM-FLIPD deficit before exact copying.
- **H3:** shared directions may be protected by cross-class capacity borrowing (exploratory).
- **H4:** long-tail context may alter an otherwise well-sampled class.

## Frozen design

<!-- GENERATED:design:START -->
| Replicate | Condition | Geometry | Frequency |
|---:|---|---|---|
| 0 | g0_balanced | geometry_0 | balanced |
| 0 | g0_f0 | geometry_0 | frequency_0 |
| 0 | g0_f1 | geometry_0 | frequency_1 |
| 0 | g0_f2 | geometry_0 | frequency_2 |
| 0 | g1_balanced | geometry_1 | balanced |
| 0 | g1_f0 | geometry_1 | frequency_0 |
| 0 | g1_f1 | geometry_1 | frequency_1 |
| 0 | g1_f2 | geometry_1 | frequency_2 |
| 0 | g2_balanced | geometry_2 | balanced |
| 0 | g2_f0 | geometry_2 | frequency_0 |
| 0 | g2_f1 | geometry_2 | frequency_1 |
| 0 | g2_f2 | geometry_2 | frequency_2 |
<!-- GENERATED:design:END -->

## Calibration record

<!-- GENERATED:calibration:START -->
```json
{
  "metric": {
    "path": "/Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/outputs/synthetic_long_tail_geometry/calibration/metric_gate.json",
    "status": "not_run"
  },
  "oracle": {
    "path": "/Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/outputs/synthetic_long_tail_geometry/calibration/oracle/oracle_gate.json",
    "status": "not_run"
  },
  "pilot": {
    "path": "/Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/outputs/synthetic_long_tail_geometry/calibration/pilot_gate.json",
    "status": "not_run"
  },
  "renderer": {
    "artifacts": {
      "class_statistics": "renderer_class_statistics.csv",
      "renderer_gate": "renderer_gate.json",
      "singular_values": "renderer_singular_values.npz"
    },
    "checks": {
      "factor_visibility": true,
      "nuisance_matching": false,
      "object_separability": true,
      "renderer_rank": false
    },
    "full_rank_fraction": 0.9045138888888888,
    "max_nuisance_standardized_difference": 2.8170312549008902,
    "median_pullback_norms": {
      "azimuth": 141.53069305419922,
      "elevation": 40.65658760070801,
      "tx": 63.047760009765625,
      "ty": 48.98526954650879,
      "tz": 37.757707595825195
    },
    "object_accuracy": 0.9991319444444444,
    "passed": false,
    "pullback_norm_ratio": 3.748392104976416,
    "relative_singular_threshold": 0.02,
    "renderer_points_per_cell": 256
  }
}
```
<!-- GENERATED:calibration:END -->

## Run ledger

<!-- GENERATED:ledger:START -->
| Stage | Condition | Replicate | Status | Output |
|---|---|---:|---|---|
| build-pools |  | 0 | complete |  |
| calibrate-renderer |  |  | complete |  |
<!-- GENERATED:ledger:END -->

## Observations

The first production-scale renderer calibration failed its preregistered gate. Object
separability passed at 0.9991 and the pullback-norm ratio passed at 3.748, but only
0.9045 of calibration points had nominal renderer rank (required: at least 0.95) and
the worst cross-object nuisance mismatch was 2.817 standardized deviations (required:
at most 0.25).

Diagnostic calibration attempts changed only the permitted scale, fixed material,
common lighting, and finite-difference settings. A 0.02 finite-difference step raised
held-out full-rank coverage above 0.98. Appearance/scale fitting reduced the nuisance
mismatch to approximately 1.19 on a smaller calibration split, but this remained far
outside the frozen 0.25 gate and was not promoted to a production configuration.

No factor oracle, balanced pilot, smoke condition, or main-matrix model was trained.
Consequently, H1--H4 have not been evaluated.

<!-- GENERATED:effects:START -->
```json
{}
```
<!-- GENERATED:effects:END -->

## Interpretation

This is a calibration failure, not evidence for or against long-tail geometric
memorization. The current objects remain readily distinguishable, but their occupancy,
luminance, and contrast distributions are not exchangeable enough for the intended
causal comparison. Proceeding would make object-specific rendering differences a
plausible explanation for any frequency-by-dimension effect.

## Limitations

This synthetic renderer establishes internal causality, not natural-image generality.

## Next decision

Create a new renderer version with geometry explicitly optimized for matched nuisance
statistics, then repeat calibration from scratch under a new artifact root. Do not
weaken the 0.25 threshold post hoc and do not run the oracle or generative matrix from
the failed renderer version.
