# CM 16-orientation sampling screen

## Question and protocol

The prior 10k confirmation found that one of four spectrum-randomized expert
orientations outperformed the learned expert. This screen asks whether that
orientation was an isolated accident and how unusual the learned orientation
is within a larger random-orientation distribution.

The released-CM EMA checkpoint at 60,000 steps is evaluated with:

- learned, general-only, and 16 singular-subspace-randomized experts;
- 2,000 balanced generated samples per condition (20 per CIFAR-100 class);
- identical requested labels, initial noise, DDIM schedule, and guidance;
- deterministic DDIM with `skip=20` and guidance `omega=1.5`;
- one independent 100-sample endpoint-response calibration pilot per random
  expert, using a different input seed from evaluation;
- 20 KID subsets of size 500; FID is deliberately omitted at this screen scale.

The original four random experts retain their prior local-response scale.
Rotations 4–15 use the median of those scales only as an initializer; their
final fixed scales are separately calibrated by the endpoint pilot. The full
machine-readable result is
[`cm_sampling_orientation16_kid2k_summary.json`](cm_sampling_orientation16_kid2k_summary.json).
The complete server artifact is:

```text
/root/autodl-tmp/runs/imbdiff_matrix60k/
  cm_sampling_orientation16_kid2k
```

## Primary result

Lower KID is better.

| Quantity | KID |
| --- | ---: |
| Learned expert | 0.016868 |
| General-only | 0.017469 |
| Random mean | 0.017244 |
| Random median | 0.017240 |
| Best random | 0.015921 |
| Worst random | 0.018828 |
| Learned gain vs general | +0.000601 |
| Learned advantage vs random mean | +0.000376 |

The learned expert ranks **5th of 17** when it and the 16 randomized experts
are treated as candidates. Four random experts (25%) have lower KID, while 12
have higher KID. Thus learned is better than a typical response-matched random
orientation, but it is not uniquely optimal.

Twelve of the 16 randomized experts also improve on general-only. This shows
that a substantial part of the benefit is generic to adding a calibrated
low-rank correction, although orientation affects how large that benefit is.

The strongest conditions were:

| Rank | Condition | KID | Gain vs general | Main-run RMS / learned RMS |
| ---: | --- | ---: | ---: | ---: |
| 1 | Random 5 | 0.015921 | +0.001548 | 0.916 |
| 2 | Random 12 | 0.015987 | +0.001482 | 1.032 |
| 3 | Random 0 | 0.016183 | +0.001286 | 0.997 |
| 4 | Random 8 | 0.016183 | +0.001286 | 0.923 |
| 5 | Learned | 0.016868 | +0.000601 | 1.000 |

Random 0 therefore remains strong in the larger screen, but it is not a lone
outlier: two new orientations beat it and a third nearly ties it. Each of the
four leading random experts has lower KID than learned in all 20 paired KID
subset draws. These subset comparisons quantify evaluator Monte Carlo
variation for the fixed generated sets, not independent training or
orientation uncertainty.

## Frequency groups

| Group | Learned | General | Random mean | Learned gain vs general | Learned advantage vs random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Many | 0.016775 | 0.017385 | 0.017157 | +0.000611 | +0.000382 |
| Medium | 0.014784 | 0.015481 | 0.015255 | +0.000698 | +0.000471 |
| Few | 0.020330 | 0.020791 | 0.020592 | +0.000461 | +0.000262 |

Learned ranks fifth in every group. Its Few gain remains smaller than its Many
gain:

\[
G_{\mathrm{Few}}-G_{\mathrm{Many}}=-0.000149.
\]

The larger orientation set therefore strengthens, rather than weakens, the
conclusion that this checkpoint does not exhibit a tail-selective explicit
expert effect.

## What predicts random-orientation quality?

Endpoint calibration transferred reasonably to the main seed: randomized
endpoint RMS ranged from 0.916 to 1.033 times learned RMS. More importantly,
quality gain has essentially no monotonic relationship with response
magnitude:

| Descriptor vs KID gain | Spearman \(\rho\) | Unadjusted \(p\) |
| --- | ---: | ---: |
| Endpoint RMS ratio | -0.018 | 0.948 |
| Applied response scale | -0.112 | 0.680 |
| Cosine to learned residual | +0.376 | 0.151 |
| RMS distance from learned residual | -0.444 | 0.085 |
| Mid-high spectral fraction | +0.397 | 0.128 |
| High spectral fraction | +0.429 | 0.097 |

This provides no evidence that response amplitude explains the orientation
spread at the resolution of this screen. There are weak exploratory tendencies
for orientations closer to the learned residual, and for residuals with more
mid/high-frequency energy, to perform better. Neither is statistically
resolved with 16 fixed orientations, the tests are correlated, and no
multiple-comparison correction was applied. They are hypotheses for a
designed intervention, not conclusions.

## Updated interpretation

The larger control changes the emphasis of the 10k result:

1. the explicit expert is useful relative to removing it;
2. learned orientation is better than a typical random orientation;
3. exact learned orientation is neither necessary nor optimal in this
   checkpoint-level screen;
4. the benefit is not tail-selective;
5. endpoint magnitude does not explain why some randomized experts work
   better;
6. the large spread across equal-rank, equal-spectrum, response-calibrated
   orientations makes subspace orientation a scientifically meaningful
   capacity-allocation variable.

The experiment does **not** show that a random expert would be better if used
during training: the randomized factors are post-training interventions, and
the favorable orientations were selected retrospectively from the evaluation
distribution. The result instead motivates analyzing or controlling which
functional directions CM places in its expert subspace.

## Next decision

A full 10k FID run over all 16 random orientations would be expensive and
would mainly refine their ranking. The more informative next experiment is a
small designed subspace intervention motivated by the weak descriptor trends:
preserve rank and response scale while deliberately varying alignment to the
learned expert and spatial-frequency response. That can test causally whether
either property predicts useful capacity, instead of screening more random
draws.
