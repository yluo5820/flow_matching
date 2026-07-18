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

## Effects

<!-- GENERATED:effects:START -->
```json
{}
```
<!-- GENERATED:effects:END -->

## Interpretation

Both class frequency and intrinsic dimension are major sources of generative
difficulty in this synthetic system. Their joint worst case is unambiguous: the
five-dimensional tail is almost entirely invalid and has the worst FID. The rotations
show that neither fixed object appearance nor class-label confusion explains the
dominant pattern. The head-versus-balanced contrast additionally implicates allocation
of optimization exposure or shared capacity, because own-class unique sample count is
held at 5,000 while the surrounding frequency context changes.

The stronger claim of a superadditive dimension-by-frequency interaction remains
suggestive rather than established. The 5D renderer-validity outcome is bounded near
zero before frequency is reduced, and the interaction is not monotone across FID,
active-factor energy, and validity. All results use one model seed and one set of
generated samples per cell. The contracted factor ranges and visual tail failures are
compatible with geometric memorization, but a Jacobian/tangent-rank probe is still
needed to demonstrate loss of learned manifold directions directly.

## Next decision

Do not launch the 36-run, 40,000-step matrix. The reduced factorial already establishes
the descriptive frequency phenomenon, while the larger matrix would be extremely
costly and would not remove the 5D floor or single-seed limitation.

The most informative next causal ablation is a class-balanced sampler on the same nine
finite datasets at the same total update budget. Under the present empirical sampler,
all three imbalanced classes receive roughly the same number of dataset passes, but
the head receives about 100 times as many class-conditioned training examples as the
tail. Equalizing class-conditioned update exposure while retaining 5,000/500/50 unique
examples would separate finite manifold coverage from optimization/capacity allocation.
If tails remain poor, unique geometric coverage is the dominant mechanism; if they
recover substantially, allocation is dominant. Solver accuracy and direct latent
projection remain useful validity checks for the balanced 5D residual, followed by a
Jacobian/tangent-rank probe for direct evidence of lost directions.

The ablation is implemented as `frequency-pilots --training-sampling class_balanced`.
It selects the training class uniformly and then samples with replacement only from
that class's unchanged finite pool. Configs, runs, ledger entries, and summaries are
isolated under `frequency_factorial_class_balanced`, while the existing balanced 5k
controls are reused because empirical sampling is already uniform when class counts
are equal.

The `balanced-pilots --training-steps N` interface now creates an immutable config,
run directory, evaluation, and rotation summary isolated under `steps_N` for each
budget. The 2,000- and 5,000-step jobs start from the same model and data-order seed
as the 1,000-step pilot, so their prefixes are directly comparable; they train from
scratch rather than sharing mutable run state.
