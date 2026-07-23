# CM matched 10k-sample FID confirmation

## Protocol

This is the confirmatory run promoted from the bounded matched-sampling
screen. It uses the released-CM EMA checkpoint at 60,000 training steps on
CIFAR-100-LT IR100.

The six conditions are learned, general-only, and four
singular-subspace-randomized experts. Each condition uses:

- 10,000 balanced generated samples (100 per class);
- identical requested labels and initial Gaussian noise;
- deterministic DDIM with `skip=20` and guidance `omega=1.5`;
- the same four random expert orientations used by the local and 2k screens;
- local-response plus independent endpoint-response scale calibration.

The full machine-readable output is
[`cm_sampling_intervention_fid10k_summary.json`](cm_sampling_intervention_fid10k_summary.json).
The complete server artifact is:

```text
/root/autodl-tmp/runs/imbdiff_matrix60k/
  cm_sampling_intervention_fid10k_r4
```

## Validation

- all expert factors were restored bit-exactly;
- the learned endpoint RMS relative to general-only is `0.016745`;
- randomized endpoint RMS values are
  `[0.017372, 0.017470, 0.016599, 0.016504]`;
- the worst random-versus-learned RMS mismatch is about 4.3%.

The random-control comparison is therefore not explained by a gross
full-trajectory response-magnitude difference.

## Overall result

Lower FID/KID is better. Positive gain/advantage means learned is better.

| Condition or contrast | FID | KID |
| --- | ---: | ---: |
| Learned | 27.0116 | 0.016700 |
| General-only | 27.9420 | 0.017345 |
| Random mean | 27.6964 | 0.017069 |
| Learned gain vs general | **+0.9304** | **+0.000645** |
| Learned advantage vs random mean | **+0.6848** | **+0.000369** |

Removing the expert worsens overall FID by 0.93, or about 3.3% relative to
general-only. The paired KID gain is positive in all 20 fixed subset draws,
with a subset range `[0.000420, 0.000867]`.

Learned also beats the mean randomized expert. Its paired KID advantage is
positive in all 20 subset draws, with range `[0.000073, 0.000617]`.

## Frequency groups

| Group | Learned FID | General FID | Random-mean FID | Gain vs general | Advantage vs random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Many | 35.3188 | 36.3416 | 36.0534 | +1.0228 | +0.7347 |
| Medium | 35.1435 | 36.1056 | 35.9065 | +0.9622 | +0.7630 |
| Few | 43.0662 | 43.8368 | 43.5811 | +0.7706 | +0.5150 |

The KID gains versus general are `+0.000570`, `+0.000625`, and
`+0.000640` for Many, Medium, and Few. Thus the tail contrast is
metric-dependent:

- Few-minus-Many FID gain: `-0.2522`;
- Few-minus-Many KID gain: `+0.000070`.

The signs disagree, and the FID effect is materially smaller for Few. This
does not establish tail-selective allocation.

The 10k learned-general endpoint residual also reproduces the earlier spatial
spectrum: `[0.6894, 0.2443, 0.0615, 0.0044]` from low through high radial
bands. Groupwise spectra remain very similar. The expert effect is therefore
predominantly low-frequency at this scale as well.

## Random-orientation result

| Condition | FID | KID | FID relative to learned |
| --- | ---: | ---: | ---: |
| Learned | 27.0116 | 0.016700 | — |
| Random 0 | **26.7783** | **0.016166** | **0.2332 better** |
| Random 1 | 27.7511 | 0.017141 | 0.7395 worse |
| Random 2 | 28.5501 | 0.018015 | 1.5385 worse |
| Random 3 | 27.7061 | 0.016955 | 0.6945 worse |

Random 0 beats learned in all 20 paired KID subset draws; its
random-minus-learned range is `[-0.000848, -0.000159]`. Randoms 1 and 2 are
consistently worse, while random 3 is weaker and mixed over subset draws.

Three of four random experts beat general-only on FID, although by much less
than learned on average. Random 2 is worse than general-only.

## Conclusion

The first result—learned outperforming general-only—is a high-sample
confirmation of the paper's component ablation, not a new contribution by
itself.

The matched random controls add the scientifically useful information:

1. some expert branch effect is generic to adding a calibrated low-rank
   correction, because the mean random expert and three individual random
   experts improve over general-only on FID;
2. learned orientation is better than a typical random orientation, because
   it beats their mean by 0.685 FID;
3. learned orientation is not uniquely privileged at this checkpoint and
   sampling seed, because one fixed random orientation beats it on both FID
   and KID;
4. there is no metric-consistent evidence that the useful expert effect is
   stronger for tail classes.

Together with the residual visualization and local probe, the most defensible
current account is a useful, mostly low-frequency, stage-composed correction.
It is not evidence that \(\theta_e\) is a clean tail-knowledge store.

## Limitations and next decision

- one training checkpoint and one matched sampling-noise seed;
- four random orientations are insufficient to estimate the orientation
  distribution precisely;
- KID subset ranges quantify evaluator Monte Carlo variation, not
  checkpoint/training uncertainty;
- FID has no paired confidence interval here;
- endpoint calibration is close but not exact;
- the FID Inception classifier does not measure CIFAR-100 requested-class
  correctness.

Before changing CM, the most focused next experiment is a larger randomized
orientation screen at the inexpensive 2k/KID scale. It would estimate how
exceptional random 0 is and whether quality correlates with subspace,
frequency, or class-group response properties. This directly informs whether
the next capacity-decomposition study should preserve learned orientation or
primarily preserve rank and response scale.
