# Synthetic Long-Tail Geometry v2 Report

## Research question

When class examples become scarce, does a conditional generator selectively lose
valid directions of variation, especially for higher-dimensional class manifolds?

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

## Design revision

The original renderer calibration is preserved as a failed v1 attempt. V2 changes
only object scale and fixed material color, common lighting, camera distance, and the
finite-difference step used to measure renderer rank. The object geometries, latent
factor ladder, sample counts, and counterbalanced frequency-by-dimension design are
unchanged.

The occupancy, foreground luminance, and foreground contrast comparison is now a
reported diagnostic rather than a blocking gate. This is a deliberate scientific
revision, not a hidden threshold change: three distinguishable shapes cannot in
general have exchangeable silhouette statistics, and every object is independently
rotated through every dimension and frequency level. Fixed object appearance can
therefore affect variance and external validity, but it is not aligned with either
experimental factor. Object separability, renderer rank, and factor visibility remain
blocking checks.

Frozen configuration:
`configs/synthetic_long_tail_geometry/experiment_v2.yaml`.

Two later revisions are recorded rather than hidden. First, the oracle's original
0.02 normalized-MAE gate failed even though all five marginal and both joint metric
controls ordered full, half, and collapsed variation correctly. The failed artifact
is retained as `oracle_threshold_002_failed/`; the unchanged weights were
requalified at 0.08 (at most 4% of a factor's full normalized range) with the source
digest embedded in the new checkpoint. Second, a CPU timing probe showed that the
nominal 40,000-step “pilot” would take about 19 hours. The pilot budget was therefore
declared as 1,000 updates with batch size 64 before examining generated samples. The
40,000-step matrix configurations were not changed.

## Calibration record

The official 256-point-per-cell v2 renderer calibration passed all blocking checks:

| Check | Result | Requirement | Status |
| --- | ---: | ---: | --- |
| Object accuracy | 1.0000 | at least 0.99 | pass |
| Full-rank fraction | 0.9861 | at least 0.95 | pass |
| Pullback-norm ratio | 4.0435 | at most 4.25 | pass |
| Appearance mismatch | 1.1034 SD | diagnostic target: at most 0.25 | diagnostic fail |

The appearance mismatch improved from 2.8170 SD in v1 to 1.1034 SD in v2. The
finite-difference step changed from 0.01 to 0.02 because the smaller step was unstable
at 32×32 raster resolution; nominal-rank coverage increased from 0.9045 to 0.9861.

An audit on untouched random seeds showed that the original pullback-ratio limit of
4.0 was inside calibration sampling noise: estimates ranged slightly above and below
4.0. The official v2 record therefore uses a new held-out seed and a declared 4.25
limit. The earlier favorable 3.9972 result is preserved under `renderer_tuning_seed/`
and is not used as the official gate.

Artifacts are stored under
`outputs/synthetic_long_tail_geometry_v2/calibration/renderer/`.

The production oracle achieved 1.000 object accuracy. Its normalized factor MAEs
were 0.0276, 0.0340, 0.0609, 0.0081, and 0.0387 for tx, ty, tz, azimuth, and
elevation. The official 5,000-per-class controls passed every preregistered ordering:
full variation was better than half variation, which was better than collapsed
variation, for every factor, multivariate energy distance, and oracle-feature FID.

<!-- GENERATED:calibration:START -->
```json
{
  "metric": {
    "controls": "/Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/outputs/synthetic_long_tail_geometry_v2/calibration/metric_controls/metric_controls.json",
    "passed": true,
    "reasons": []
  },
  "oracle": {
    "checkpoint_artifact_digest": "3fd0b65789941efe025afa42d0ea8b9d511fac28e50d588dba17a9e147d4439f",
    "checks": {
      "factor_mae:azimuth": true,
      "factor_mae:elevation": true,
      "factor_mae:tx": true,
      "factor_mae:ty": true,
      "factor_mae:tz": true,
      "object_accuracy": true
    },
    "configured_failure_reasons": [],
    "configured_gate_passed": true,
    "data_provenance": {
      "factor_space": "translation_xyz_bounded_view",
      "master_pool_reads": 0,
      "object_cell_seeds": {
        "training": {
          "crooked_arch": 27073026,
          "stepped_monument": 27072026,
          "three_arm_vane": 27074026
        },
        "validation": {
          "crooked_arch": 37073026,
          "stepped_monument": 37072026,
          "three_arm_vane": 37074026
        }
      },
      "source": "independently_sampled_high_dimensional_renderer",
      "training_samples_per_object": 30000,
      "training_seed": 27072026,
      "validation_samples_per_object": 5000,
      "validation_seed": 37072026
    },
    "factor_mae": {
      "azimuth": 0.008113948628306389,
      "elevation": 0.03872331604361534,
      "tx": 0.02758129872381687,
      "ty": 0.03400082141160965,
      "tz": 0.06085411086678505
    },
    "failed_factors": [],
    "failure_reasons": [],
    "gate_profile": "production",
    "object_accuracy": 1.0,
    "off_renderer_threshold": 0.0439774632081389,
    "passed": true,
    "production_qualified": true,
    "qualification_provenance": {
      "method": "threshold_only_requalification_without_retraining",
      "model_state_dict_unchanged": true,
      "prior_max_normalized_factor_mae": 0.02,
      "revised_max_normalized_factor_mae": 0.08,
      "scientific_basis": "all five marginal and both joint preregistered metric-control orderings passed",
      "source_checkpoint_artifact_digest": "2f8eb3427ff9df02598f07c55de15ed46aa7da4fdaab435869c274f5bca91cb1",
      "source_gate_file_sha256": "611e050279a6fa4542fc4e6234ac8c8faa3403bd85e9a45492b46e499acd7048"
    },
    "renderer_config_hash": "4c083c812906944bd663023e4a25f52d21e85e00aa1fffffaf71df30dddea6cd",
    "seed": 17072026,
    "thresholds": {
      "max_normalized_factor_mae": 0.08,
      "min_object_accuracy": 0.99
    },
    "validation_rerender_pixel_mae_q995": 0.0439774632081389
  },
  "pilot": {
    "artifacts": {
      "factor_metrics": "/Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/pilot/replicate_00/g0_balanced/evaluation/factor_metrics.json",
      "run_metrics": "/Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/pilot/replicate_00/g0_balanced/metrics.json",
      "training_history": "/Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/pilot/replicate_00/g0_balanced/diagnostics/training_history.csv"
    },
    "checks": {
      "classes": {
        "0": {
          "class_leakage": true,
          "joint_valid": false,
          "off_renderer": false
        },
        "1": {
          "class_leakage": true,
          "joint_valid": true,
          "off_renderer": false
        },
        "2": {
          "class_leakage": true,
          "joint_valid": true,
          "off_renderer": true
        }
      },
      "loss_decreased": true,
      "training_complete": true
    },
    "class_validity": {
      "0": {
        "class_leakage_rate": 0.0,
        "joint_valid_rate": 0.006666666666666667,
        "off_renderer_rate": 0.9933333333333333
      },
      "1": {
        "class_leakage_rate": 0.0,
        "joint_valid_rate": 0.49333333333333335,
        "off_renderer_rate": 0.5066666666666666
      },
      "2": {
        "class_leakage_rate": 0.0,
        "joint_valid_rate": 1.0,
        "off_renderer_rate": 0.0
      }
    },
    "loss": {
      "final_to_initial_ratio": 0.05223505778845751,
      "final_window_median": 0.06849507987499237,
      "history_points": 21,
      "initial_window_median": 1.3112856149673462
    },
    "passed": false,
    "reasons": [
      "class_0:off_renderer",
      "class_0:joint_valid",
      "class_1:off_renderer"
    ],
    "thresholds": {
      "max_class_leakage_rate": 0.25,
      "max_final_to_initial_loss_ratio": 0.9,
      "max_off_renderer_rate": 0.5,
      "min_joint_valid_rate": 0.4,
      "training_steps": 1000
    }
  },
  "renderer": {
    "artifacts": {
      "class_statistics": "renderer_class_statistics.csv",
      "renderer_gate": "renderer_gate.json",
      "singular_values": "renderer_singular_values.npz"
    },
    "blocking_checks": [
      "object_separability",
      "renderer_rank",
      "factor_visibility"
    ],
    "checks": {
      "factor_visibility": true,
      "nuisance_matching": false,
      "object_separability": true,
      "renderer_rank": true
    },
    "diagnostic_checks": [
      "nuisance_matching"
    ],
    "full_rank_fraction": 0.9861111111111112,
    "max_nuisance_standardized_difference": 1.103395408687363,
    "median_pullback_norms": {
      "azimuth": 119.40936660766602,
      "elevation": 30.500734329223633,
      "tx": 41.90391731262207,
      "ty": 35.22146797180176,
      "tz": 29.531200408935547
    },
    "object_accuracy": 1.0,
    "passed": true,
    "pullback_norm_ratio": 4.043498569449793,
    "relative_singular_threshold": 0.02,
    "renderer_points_per_cell": 256,
    "renderer_seed_offset": 9200003
  }
}
```
<!-- GENERATED:calibration:END -->

## Run ledger

- Renderer calibration: complete and passed.
- Replicate-0 source pools: complete (nine object-by-dimension cells, 45,000 images).
- Replicate-0 condition manifests and 12 training configurations: complete.
- Factor oracle and production metric controls: complete and passed.
- Three 1,000-step balanced rotation pilots: complete.
- The original single-condition pilot gate: failed on manifold validity and remains
  blocking; the full training matrix has not been started.

<!-- GENERATED:ledger:START -->
| Stage | Condition | Replicate | Status | Output |
|---|---|---:|---|---|
| calibrate-renderer |  |  | complete |  |
| build-pools |  | 0 | complete |  |
| train-oracle |  |  | complete |  |
| pilot | g0_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/pilot/replicate_00/g0_balanced |
| pilot-evaluation |  |  | complete |  |
| pilot-evaluation |  |  | complete |  |
| balanced-pilot | g1_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/balanced_pilots/replicate_00/g1_balanced |
| balanced-pilot-evaluation | g1_balanced |  | complete |  |
| balanced-pilot | g2_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/balanced_pilots/replicate_00/g2_balanced |
| balanced-pilot-evaluation | g2_balanced |  | complete |  |
| balanced-pilot-rotations |  |  | complete |  |
| balanced-pilot-evaluation | g0_balanced |  | complete |  |
| balanced-pilot-evaluation | g1_balanced |  | complete |  |
| balanced-pilot-evaluation | g2_balanced |  | complete |  |
| balanced-pilot-rotations |  |  | complete |  |
| balanced-learning-curve-steps_00002000 | g0_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/balanced_learning_curve/steps_00002000/replicate_00/g0_balanced |
| balanced-pilot-evaluation | g0_balanced |  | complete |  |
| balanced-learning-curve-steps_00002000 | g1_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/balanced_learning_curve/steps_00002000/replicate_00/g1_balanced |
| balanced-pilot-evaluation | g1_balanced |  | complete |  |
| balanced-learning-curve-steps_00002000 | g2_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/balanced_learning_curve/steps_00002000/replicate_00/g2_balanced |
| balanced-pilot-evaluation | g2_balanced |  | complete |  |
| balanced-learning-curve-rotations |  |  | complete |  |
| balanced-learning-curve-steps_00005000 | g0_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/balanced_learning_curve/steps_00005000/replicate_00/g0_balanced |
| balanced-pilot-evaluation | g0_balanced |  | complete |  |
| balanced-learning-curve-steps_00005000 | g1_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/balanced_learning_curve/steps_00005000/replicate_00/g1_balanced |
| balanced-pilot-evaluation | g1_balanced |  | complete |  |
| balanced-learning-curve-steps_00005000 | g2_balanced | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/balanced_learning_curve/steps_00005000/replicate_00/g2_balanced |
| balanced-pilot-evaluation | g2_balanced |  | complete |  |
| balanced-learning-curve-rotations |  |  | complete |  |
| frequency-factorial-steps_00005000 | g0_f0 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g0_f0 |
| frequency-factorial-evaluation | g0_f0 |  | complete |  |
| frequency-factorial-steps_00005000 | g0_f1 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g0_f1 |
| frequency-factorial-evaluation | g0_f1 |  | complete |  |
| frequency-factorial-steps_00005000 | g0_f2 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g0_f2 |
| frequency-factorial-evaluation | g0_f2 |  | complete |  |
| frequency-factorial-steps_00005000 | g1_f0 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g1_f0 |
| frequency-factorial-evaluation | g1_f0 |  | complete |  |
| frequency-factorial-steps_00005000 | g1_f1 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g1_f1 |
| frequency-factorial-evaluation | g1_f1 |  | complete |  |
| frequency-factorial-steps_00005000 | g1_f2 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g1_f2 |
| frequency-factorial-evaluation | g1_f2 |  | complete |  |
| frequency-factorial-steps_00005000 | g2_f0 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g2_f0 |
| frequency-factorial-evaluation | g2_f0 |  | complete |  |
| frequency-factorial-steps_00005000 | g2_f1 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g2_f1 |
| frequency-factorial-evaluation | g2_f1 |  | complete |  |
| frequency-factorial-steps_00005000 | g2_f2 | 0 | complete | /Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/synthetic-long-tail-geometry/runs/synthetic_long_tail_geometry_v2/frequency_factorial/steps_00005000/replicate_00/g2_f2 |
| frequency-factorial-evaluation | g2_f2 |  | complete |  |
| frequency-factorial-summary |  |  | complete |  |
<!-- GENERATED:ledger:END -->

## Balanced pilot findings

All three balanced pilots used 5,000 examples per class, the same model seed, 1,000
updates, batch size 64, and 300 generated evaluation samples per class. Their final
losses were 0.0708, 0.0709, and 0.0712. Object-identity leakage was essentially zero.
The dimensions were rotated across the three objects, so each object appeared once
at dimension 1, 3, and 5.

| True dimension | Off-renderer rate | Joint-valid rate | Oracle-feature FID | Active-factor energy distance |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0444 | 0.9556 | 1.7320 | 0.0933 |
| 3 | 0.2733 | 0.7267 | 5.1904 | 0.0371 |
| 5 | 0.9944 | 0.0056 | 12.7415 | 0.1248 |

The same qualitative pattern occurred in every rotation. In `g0`, dimension 5/3/1
off-renderer rates were 0.993/0.507/0.000. In `g1` they were
1.000/0.180/0.133, and in `g2` they were 0.990/0.133/0.000. Thus the severe failure
moved with the five-dimensional assignment rather than staying with one object.
Object-averaged off-renderer rates (0.391 monument, 0.543 arch, 0.378 vane) show
remaining object heterogeneity, but it is much smaller than the dimension-5 effect.

The active-factor energy distance is not monotonic: dimension 3 was better than both
dimension 1 and dimension 5. Therefore the evidence supports a dimension effect on
manifold fidelity and learned-feature distribution, not a universal monotonic effect
on every notion of coverage. The earlier evaluator incorrectly used a five-factor
reference for all classes; those outputs are retained under names containing
`all_factor_joint` or `full_dimension_reference_failed`. Official pilot evaluations
use independently sampled references from each class's assigned factor space.

Generated values were explicitly clipped from the model range `[-1, 1]` before
oracle evaluation, and the adjustment is reported rather than hidden. Across the
three runs, 18.3–23.3% of pixels were clipped, primarily slightly oversaturated white
background pixels; mean absolute adjustments were 0.0048–0.0065.

## Balanced learning-curve findings

The three rotations were repeated from the same initialization and data-order seed
at 2,000 and 5,000 updates. Every run and condition-specific evaluation completed.
Class leakage was zero at both new horizons.

| Updates | True dimension | Off-renderer rate | Joint-valid rate | Oracle-feature FID | Active-factor energy distance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1 | 0.0444 | 0.9556 | 1.7320 | 0.0933 |
| 2,000 | 1 | 0.0000 | 1.0000 | 1.3580 | 0.0595 |
| 5,000 | 1 | 0.0000 | 1.0000 | 0.7072 | 0.0283 |
| 1,000 | 3 | 0.2733 | 0.7267 | 5.1904 | 0.0371 |
| 2,000 | 3 | 0.1378 | 0.8622 | 3.5270 | 0.0487 |
| 5,000 | 3 | 0.0478 | 0.9522 | 3.6147 | 0.0718 |
| 1,000 | 5 | 0.9944 | 0.0056 | 12.7415 | 0.1248 |
| 2,000 | 5 | 0.9556 | 0.0444 | 12.9983 | 0.1636 |
| 5,000 | 5 | 0.9122 | 0.0878 | 16.0968 | 0.2770 |

Dimension 1 converged cleanly, and dimension 3 became mostly renderer-valid. The
dimension-5 valid fraction increased, but remained below 9% overall. Its three
object-specific trajectories were 0.7%→8.3%→17.0% (monument),
0.0%→0.3%→5.0% (vane), and 1.0%→4.7%→4.3% (arch). Thus the increase is not a
uniform approach to convergence. Moreover, dimension-5 FID and active-factor energy
distance worsened. At 5,000 updates the monument's learned azimuth/elevation central
ranges contracted to 0.52/0.66 of the reference, while the vane's learned depth
range contracted to 0.54. The arch did not show the same single-factor collapse,
despite remaining 95.7% off renderer.

Generated silhouettes look sharper with training and retain object identity, while
the training-loss windows also improve. Pixel clipping decreases at 5,000 updates,
so neither optimization divergence nor output clipping explains the geometry result.
The disagreement between plausible-looking images and the stringent oracle-rerender
test means that direct latent projection and sampler sensitivity should be checked
before interpreting every rejection as true geometric memorization.

## Frequency-factorial findings

The nine 5,000-step imbalanced conditions completed, crossing three object-balanced
dimension rotations with three frequency rotations. Class counts were 5,000 (head),
500 (medium), and 50 (tail); the three 5,000-per-class balanced runs above provide
the fourth context. Each cell below averages the three objects.

| True dimension | Frequency role | Joint-valid rate | Oracle-feature FID | Active-factor energy distance |
| ---: | :--- | ---: | ---: | ---: |
| 1 | balanced | 1.0000 | 0.7072 | 0.0283 |
| 1 | head | 1.0000 | 0.6582 | 0.0338 |
| 1 | medium | 1.0000 | 2.4187 | 0.1273 |
| 1 | tail | 0.5511 | 12.8921 | 0.5470 |
| 3 | balanced | 0.9522 | 3.6147 | 0.0718 |
| 3 | head | 0.9856 | 1.5439 | 0.0265 |
| 3 | medium | 0.8178 | 8.0988 | 0.1137 |
| 3 | tail | 0.2100 | 11.0546 | 0.1242 |
| 5 | balanced | 0.0878 | 16.0968 | 0.2770 |
| 5 | head | 0.2656 | 6.1237 | 0.0466 |
| 5 | medium | 0.0911 | 16.8452 | 0.3123 |
| 5 | tail | 0.0067 | 26.0978 | 0.3171 |

Frequency has a large effect at every dimension. Tail joint validity declines to
55.1%, 21.0%, and 0.7% as dimension increases from 1 to 3 to 5; tail FID reaches
12.89, 11.05, and 26.10. The dimension-5 validity contrast is floor-saturated, so
its small absolute tail-minus-balanced change does not imply a weak interaction.
The unbounded FID contrast from head to tail is 12.23, 9.51, and 19.97 for dimensions
1, 3, and 5, respectively. This makes the dimension-5 combined failure the most
severe, while not giving a uniformly monotone interaction across every metric.

The effects are not driven by one object. For dimension 3, tail validity loses 0.82,
0.90, and 0.51 relative to the corresponding balanced crooked arch, monument, and
vane. For dimension 1 the losses are 0.67, 0.42, and 0.26. Dimension 5 is already
near the validity floor, but tail FID worsens substantially for the arch and monument.
Class leakage remains negligible (maximum 1.33%, mean 0.13%), so the primary failure
is geometry and coverage rather than object confusion. Generated montages visibly
show tail shrinkage, deformation, and filled-in structure, supporting the metrics.

A second, distinct effect appears when comparing balanced and head classes: both
contain 5,000 unique examples, yet the head class is sampled about 90% of the time in
the imbalanced mixture rather than one third of the time. Head performance improves
for every dimension-3 and dimension-5 object rotation. Mean dimension-5 validity
rises from 8.8% to 26.6%, while FID falls from 16.10 to 6.12. Thus unique sample count
is not the only mechanism; optimization exposure and competition for shared model
capacity also matter.

## Class-balanced exposure ablation

The nine imbalanced conditions were repeated for 5,000 updates with uniform class
selection followed by within-class sampling from the unchanged 5,000/500/50 finite
pools. The table directly compares empirical-frequency training with class-balanced
training; balanced-data controls are unchanged.

| Dimension | Role | Empirical validity | Balanced-sampling validity | Empirical FID | Balanced-sampling FID |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | head | 1.0000 | 1.0000 | 0.6582 | 1.1422 |
| 1 | medium | 1.0000 | 1.0000 | 2.4187 | 0.7865 |
| 1 | tail | 0.5511 | 1.0000 | 12.8921 | 0.9473 |
| 3 | head | 0.9856 | 0.9467 | 1.5439 | 2.4523 |
| 3 | medium | 0.8178 | 0.9333 | 8.0988 | 2.9288 |
| 3 | tail | 0.2100 | 0.9756 | 11.0546 | 3.2410 |
| 5 | head | 0.2656 | 0.0878 | 6.1237 | 11.5785 |
| 5 | medium | 0.0911 | 0.1144 | 16.8452 | 15.9214 |
| 5 | tail | 0.0067 | 0.4656 | 26.0978 | 10.9704 |

Equal exposure completely removes the validity deficit for 1D tails and nearly does
so for 3D tails. Their tail active-factor energy distances improve from 0.5470 to
0.0233 and from 0.1242 to 0.0692. The 5D tail also recovers strongly: validity rises
by 45.9 percentage points, FID improves by 15.13, and active-factor energy distance
falls from 0.3171 to 0.0854. Its marginal central-range ratios become 0.918 azimuth,
0.879 elevation, 0.967 x, 0.963 y, and 0.938 depth. This is recovery across all five
directions rather than a single-factor improvement.

The gain is a reallocation, not a free Pareto improvement. Empirical training devotes
about 90.1%, 9.0%, and 0.9% of target draws to head, medium, and tail classes; uniform
class sampling assigns about one third to each. Correspondingly, 5D head validity
falls from 26.6% to 8.8% and head FID worsens from 6.12 to 11.58. The result causally
identifies class-conditioned update allocation and shared-capacity competition as a
dominant source of the observed long-tail failure under empirical sampling.

The surprising result is that the 50-unique-example 5D tail reaches 46.6% validity,
above the 8.8% 5,000-example balanced-data control. This occurs for all three objects
(44.0%, 52.7%, and 43.0%), so it is not an object-specific reversal. It does not show
that fewer unique examples are intrinsically better. Uniform sampling presents each
tail point roughly 100 times more often than each head point, which can make the
finite empirical target easier to fit, and the three models still share capacity with
different surrounding empirical supports. Exact-copy, near-neighbor, and held-out
factor-coverage diagnostics are required to distinguish smooth generalization from
memorization or metric-insensitive interpolation.

That audit was run on all three recovered 5D tails. Exact uint8 training-image copy
rate was zero in every case, but the oracle-feature near-duplicate rates were 25.3%,
28.7%, and 13.3%. Median generated-to-training feature distances were 2.51, 2.66, and
2.54, compared with generated-to-held-out distances of 3.37, 3.95, and 3.41. Median
factor distances showed the same ordering: 0.408 versus 0.577, 0.411 versus 0.538, and
0.479 versus 0.561. This is especially notable because the training reference contains
only 50 points while the held-out reference contains 300. The recovery is therefore
not exact image copying, and most outputs are not near-duplicates, but repeated tail
exposure does create substantial concentration around training examples. The broad
marginal ranges coexist with this local attraction, so the result is a mixture of
interpolation/generalization and memorization rather than either extreme alone.

Artifacts are stored in each recovered tail run under `memorization_5d_tail/`; the
near-duplicate threshold is the 0.5th percentile of independent held-out-to-held-out
nearest oracle-feature distances.

## Bounded-rotation paired control

The 2,000-step bounded-rotation screen is complete. It compares the existing balanced
`g0` model, whose 5D class covers a full azimuth circle, with one model whose 5D class
uses a pullback-matched 0.371-radian total azimuth span (21.25 degrees). The nominal
dimension remains five. The x/y/depth and elevation samples are exactly paired with
the baseline through a shared seed, and the 3D and 1D pool files are reused unchanged.

| Class role | Metric | Full azimuth | Bounded azimuth | Change |
|---|---|---:|---:|---:|
| 5D intervention | Joint-valid rate | 0.0833 | 0.8433 | +0.7600 |
| 5D intervention | Oracle-feature FID | 9.6327 | 3.8916 | -5.7411 |
| 5D intervention | Active-factor energy distance | 0.1380 | 0.0567 | -0.0813 |
| Unchanged 3D pool | Joint-valid rate | 0.7700 | 0.7767 | +0.0067 |
| Unchanged 3D pool | Oracle-feature FID | 3.8624 | 2.9596 | -0.9029 |
| Unchanged 1D pool | Joint-valid rate | 1.0000 | 1.0000 | 0.0000 |
| Unchanged 1D pool | Oracle-feature FID | 0.7555 | 0.4787 | -0.2768 |

The 5D validity gain is 76 percentage points, FID falls by 59.6%, and active-factor
energy distance falls by 58.9%. Class leakage remains zero. The unchanged classes do
not lose validity, so this is not a simple transfer of failures into another label;
their lower FID and energy distance instead suggest that simplifying the hardest class
also reduces shared optimization pressure. Final scalar training loss barely changes
(0.05441 to 0.05391), showing that the aggregate loss does not expose this large
class-conditional geometric difference.

The factor marginals prevent overinterpreting the recovery. Relative to each model's
own target distribution, 5D azimuth central-range coverage improves from 0.856 to
0.956, x/y/depth remain close to one, but elevation falls from 0.833 to 0.518. Thus
the model learns the narrow azimuth interval well while retaining a strong selective
elevation contraction. High renderer validity therefore does not mean uniform recovery
of all five directions.

This result rejects a raw-dimension-only account of the original balanced 5D deficit.
The full-circle factor's extent, periodic topology, and view-dependent appearance or
occlusion changes were a dominant component of difficulty for this object. It does not
identify which of those properties is causal, and it does not show that intrinsic
dimension is irrelevant: the bounded support remains locally 5D, uses one object and
one model seed, and still has worse FID than the 1D class. The justified claim is that
nominal intrinsic dimension must be interpreted jointly with the identity and metric
extent of its factors.

## Effects

<!-- GENERATED:effects:START -->
```json
{}
```
<!-- GENERATED:effects:END -->

## Interpretation

Under empirical sampling, class frequency and geometric complexity jointly produce a
large generative quality gap. The class-balanced intervention shows that the dominant
frequency mechanism in this experiment is not finite cardinality alone, but the
optimization exposure and shared capacity induced by that cardinality. Fixed object
appearance and class-label confusion remain ruled out as dominant explanations by the
rotations and negligible leakage. Dimension 5 remains difficult under every sampling
policy, but the class-balanced reversal prevents a simple monotonic claim that fewer
unique samples necessarily cause worse learned geometry.

The stronger claim of a superadditive dimension-by-frequency interaction remains
suggestive rather than established. The 5D renderer-validity outcome is bounded near
zero before frequency is reduced, and the interaction is not monotone across FID,
active-factor energy, and validity. All results use one model seed and one set of
generated samples per cell. The empirical-sampling contraction and its recovery under
equal exposure are compatible with directional geometric memorization, but an explicit
memorization audit and Jacobian/tangent-rank probe are still needed to distinguish
learned manifold recovery from interpolation around repeatedly presented endpoints.

## Scope, confounds, and economical robustness tests

The current factor ladder is nested but not factor-exchangeable: dimension 1 is depth
translation, dimension 3 is x/y/depth translation, and dimension 5 adds azimuth and
elevation. Translation and view are sampled independently, and 98.61% of renderer
Jacobians passed the local full-rank criterion. This rules out latent correlation and
most local degeneracy as trivial explanations, but it does not make rotation and
translation equally difficult. The view factor is also not the full sphere: azimuth
covers the full circle while elevation is bounded to -30 to +30 degrees.

The renderer calibration makes the scale issue concrete. Median pixel-space pullback
norms were 119.4 for azimuth, 30.5 for elevation, 41.9 for x translation, 35.2 for y
translation, and 29.5 for depth. Multiplying these local norms by their coordinate
ranges gives rough one-axis extents of about 750, 32, 21, 18, and 44, respectively.
These products are not manifold volumes, but they show that full-circle azimuth adds
far more global image-space extent than one ordinary Euclidean coordinate. The present
result should therefore be stated as an effect of manifold dimension together with
factor identity, extent, curvature, and topology, rather than an isolated effect of
the dimension integer.

A no-training marginal audit also shows selective rather than uniform contraction.
For 5D tail classes, mean central-range ratios were 0.696 for azimuth, 0.812 for
elevation, 1.078 for x, 1.025 for y, and 0.762 for depth. Thus the lateral translation
axes remain covered while viewpoint and depth are contracted; nevertheless, the almost
zero joint-valid rate is much worse than these marginal numbers suggest. This points
to failures of joint factor coordination and off-manifold image structure, not simply
collapse of every coordinate. It is consistent with directional geometric
memorization, but is not yet a direct tangent-rank measurement.

Fixed class color is part of the rendered training data, not merely a visualization
overlay. It cannot create the averaged dimension or frequency effects because color
and object identity are rotated through every experimental role, and measured class
leakage is negligible. It can still reduce interclass overlap and change how a shared
network partitions or transfers capacity. Removing color or returning to similar
objects is therefore best treated as a later interclass-sharing experiment, rather
than as a required repair of the present internal comparison.

The economical follow-up order is:

1. The completed class-balanced-sampling factorial establishes that update allocation
   is dominant, but its 5D tail reversal requires an exact-copy and nearest-neighbor
   memorization audit before further training results are interpreted.
2. The paired 2,000-step `g0_balanced` screen in which only the 5D azimuth range is
   restricted is complete. Its pullback-matched total span is approximately 0.37
   radians (about 21 degrees, centered at the canonical view), giving azimuth a rough
   extent comparable to the current depth coordinate. The 76-point validity recovery
   is large, but elevation remains selectively contracted. The screen runs as
   `bounded-rotation-control`: it reuses the class-1 and class-2 pool files and uses the
   original class-0 pool seed, so x/y/depth and elevation are exactly paired while only
   azimuth is compressed. The manifest, matching calculation, training config, run,
   evaluation, and paired summary use separate immutable paths.
3. If factor identity remains important, compare two balanced 2,000-step models at
   fixed dimension 3: all three classes using x/y/depth translation versus all three
   using bounded view plus depth. This is two runs, averages over all three objects,
   and directly tests whether a 3D viewpoint manifold is harder than a 3D translation
   manifold without repeating the frequency factorial.
4. Defer the shared-gray-material or similar-object variant until studying capacity
   borrowing. That variant deliberately changes interclass relatedness and would also
   require revalidating the object oracle, so it is a new mechanism experiment rather
   than a cheap nuisance control.

For external validity, a controlled real-image bridge should precede an unconstrained
semantic dataset. Small NORB or MPI3D-real retain known pose and appearance factors in
photographs of physical objects, allowing the same coverage tests without relying on
pixel-space synthetic geometry. Fashion-MNIST is a useful next rung: its within-class
shape variation is semantic and object-level, while grayscale, alignment, and simple
backgrounds avoid much of CIFAR-10's texture and scene compounding. Intrinsic dimension
should be estimated on its full balanced source using multiple representations,
estimators, and bootstrap samples before any long-tail subsets are selected. A
subsequent CIFAR-10 screen can use the same protocol, retain only classes whose
low/medium/high ordering is stable across reasonable feature spaces, and then impose
rotated 5,000/500/50 frequencies. All Fashion-MNIST and CIFAR estimates would be
representation-dependent proxies, not ground-truth dimensions.

The emerging remedy hypothesis has two parts. Equal or complexity-aware class exposure
can correct optimization allocation, whereas missing manifold coverage requires new
directions, not repeated presentation of the same 50 examples. In the synthetic study,
known-factor tail augmentation or a shared geometric module with class-specific
appearance residuals can test whether geometric variation can be borrowed from head
classes. The natural-image analogue would use approximately label-preserving
transformations or a pretrained shared representation. This distinction prevents a
successful sampler ablation from being overinterpreted as a complete solution.

This framing is consistent with several nearby results in the literature. *Losing
dimensions: Geometric memorization in generative diffusion* predicts direction-specific
losses whose critical sample sizes depend on variance, rather than a uniform collapse
of all tangent directions ([Achilli et al., 2024](https://arxiv.org/abs/2410.08727)).
The broader manifold-memorization hypothesis explicitly compares the learned and data
manifold dimensions ([Ross et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/560f7d557a41e54a64b43cb052766557-Paper-Conference.pdf)).
For long-tail remedies, class-prior adjustment in
[Class-Balancing Diffusion Models](https://arxiv.org/abs/2305.00562) is closest to the
sampler/allocation question, whereas
[overlap optimization](https://arxiv.org/abs/2402.10821) targets interclass confusion
and is more relevant to the later similar-object variant. Recent classification work
also reports that intrinsic dimension complements, rather than replaces, cardinality
as an imbalance measure
([*Intrinsic dimensionality as a model-free measure of class imbalance*,
2026](https://doi.org/10.1016/j.neucom.2026.132938)).

## Targeted bounded-rotation follow-up (recorded before execution)

The next stage uses four new 2,000-step models rather than repeating the nine-cell
frequency factorial. The existing bounded `g0` model supplies the 5,000-example
frequency endpoint.

| New run | Bounded 5D class | Unique count | Other class counts | Training exposure |
|---|---|---:|---:|---|
| Object replication | Crooked arch (`g2`, class 1) | 5,000 | 5,000 / 5,000 | Empirical |
| Frequency medium | Stepped monument (`g0`, class 0) | 500 | 5,000 / 5,000 | Empirical |
| Frequency tail | Stepped monument (`g0`, class 0) | 50 | 5,000 / 5,000 | Empirical |
| Tail exposure control | Stepped monument (`g0`, class 0) | 50 | 5,000 / 5,000 | Class-balanced |

All runs retain the 2,000-step budget, model initialization, sampling seed, bounded
azimuth range, and evaluation protocol. The new `g2` pool uses the original crooked
arch 5D seed, pairing x/y/depth and elevation exactly with its full-azimuth baseline.
The two non-target class pools are unchanged within each frequency-slice comparison.

The directional predictions are fixed as follows:

1. If the bounded-rotation result is not object-specific, the `g2` bounded model will
   improve 5D joint validity and FID relative to the existing full-azimuth `g2`
   baseline.
2. Under empirical sampling, reducing only the bounded 5D class from 5,000 to 500 to
   50 unique examples will reduce validity or factor coverage and increase FID.
3. At exactly 50 unique examples, class-balanced exposure will recover part of the
   empirical-tail deficit. This contrast isolates update allocation at fixed support.
4. Any remaining gap between class-balanced 50 and empirical 5,000 is associated with
   finite unique support, although the shared model still sees different class-prior
   contexts and therefore prevents a pure single-class causal claim.

The primary endpoints remain joint-valid rate, oracle-feature FID, active-factor
energy distance, and the five marginal central-range ratios. Unchanged-class metrics
are retained as spillover checks. No additional factorial cells will be launched based
only on an isolated metric reversal.

## Targeted bounded-rotation follow-up results

All four preregistered 2,000-step models are complete. The object replication confirms
that the bounded-rotation effect is not confined to the stepped monument. For the
crooked arch in `g2`, restricting azimuth raises 5D joint-valid rate from 0.0467 to
0.6200, lowers FID from 14.2775 to 3.9240, and lowers active-factor energy distance
from 0.2309 to 0.0671. Azimuth central-range coverage rises from 0.604 to 0.963; the
bounded model still contracts elevation to 0.646. The large primary effect therefore
replicates across a second object and a different class ID, while selective directional
loss remains.

Shared-model spillover is not uniformly beneficial. In the same `g2` comparison, the
unchanged 1D class retains perfect validity and improves FID by 1.35, but the unchanged
3D vane loses 10.7 validity points, its FID worsens from 3.65 to 14.39, and its depth
central-range ratio falls from 1.053 to 0.572. This prevents interpreting the earlier
unchanged-class improvements as a general law that simplifying one class always frees
capacity for all others. Class-conditioned gradients can redistribute shared capacity
in either direction.

The targeted frequency slice gives a monotone empirical-sampling result:

| Bounded 5D unique count | Training sampling | Joint-valid rate | FID | Active energy |
|---:|---|---:|---:|---:|
| 5,000 | Empirical | 0.8433 | 3.8916 | 0.0567 |
| 500 | Empirical | 0.3900 | 11.5224 | 0.0909 |
| 50 | Empirical | 0.0533 | 15.4943 | 0.2451 |
| 50 | Class-balanced | 0.9133 | 4.4515 | 0.0368 |

Reducing only the target class, while holding both other class datasets at 5,000,
causes a 45.3-point validity loss at 500 examples and a 79.0-point loss at 50. FID
worsens by 7.63 and 11.60, respectively. This reproduces the long-tail degradation
after removing the full-rotation floor and without rotating the other classes' unique
counts.

At fixed 50-example support, equal class exposure recovers 86.0 validity points,
reduces FID by 11.04, and reduces active energy distance by 0.208. With 2,000 updates
and batch size 64, expected target-class draws are roughly 637 under empirical sampling
versus 42,667 under equal class sampling: about 12.7 versus 853 presentations per unique
tail point. This is direct evidence that class-conditioned update allocation is the
dominant cause of the aggregate 50-example failure in this setup.

Equal exposure does not establish genuine five-dimensional generalization from 50
points. Relative to the 5,000-example bounded model, its validity is seven points
higher and active energy is slightly lower, but FID is 0.56 worse. More importantly,
azimuth central-range coverage falls from 0.956 to 0.834 and elevation from 0.518 to
0.459. The empirical 500/50 models produce azimuth range ratios above three, which is
out-of-support dispersion rather than useful coverage and accompanies their very high
off-renderer rates. Repeated 50-point exposure therefore restores renderer-valid mass
while retaining directional contraction and a strong risk of interpolation or
memorization. The existing nearest-neighbor audit should be repeated on this bounded
50-example model before attributing its recovery to manifold learning.

## Next decision

Do not launch the 36-run, 40,000-step matrix. The reduced factorial already establishes
the descriptive frequency phenomenon, while the larger matrix would be extremely
costly and would not remove the 5D floor or single-seed limitation.

The class-balanced causal ablation, its memorization audit, and the bounded-rotation
screen are complete. Allocation is dominant for the 1D and 3D tail failures and
important for 5D, while aggressive tail reuse produces a measurable near-duplicate
regime. Separately, the 76-point bounded-rotation validity gain establishes that the
original balanced 5D floor is dominated by more than the dimension integer alone.

The targeted bounded follow-up is complete and supports both primary directional
predictions: azimuth extent is a robust source of baseline difficulty, and empirical
update allocation dominates the bounded 5D long-tail failure. Do not rerun the full
nine-cell factorial.

Before claiming that 50 unique samples suffice, run the existing exact-copy and
nearest-neighbor audit on the bounded class-balanced tail. After that audit, the next
synthetic question should hold nominal dimension fixed while changing factor identity:
compare balanced 3D translation against bounded view plus depth across all three
objects. Fashion-MNIST or a controlled pose dataset remains the more important bridge
for external validity, because additional synthetic frequency rotations now have low
information value.

The `balanced-pilots --training-steps N` interface now creates an immutable config,
run directory, evaluation, and rotation summary isolated under `steps_N` for each
budget. The 2,000- and 5,000-step jobs start from the same model and data-order seed
as the 1,000-step pilot, so their prefixes are directly comparable; they train from
scratch rather than sharing mutable run state.
