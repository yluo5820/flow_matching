# ImbDiff-CM matched causal expert intervention

> **Archived:** the CM-specific implementation was removed on 2026-07-24. This
> document preserves the completed research record; its commands are not active.


**Status:** completed on the released-CM 60k seed-0 checkpoint,
2026-07-23.

Primary output:

```text
/root/autodl-tmp/runs/imbdiff_matrix60k/cm_intervention_screen
```

## Question

The controlled K1/K2 result found semantic structure in local expert responses,
but not more structure than was already present in the activation or preserved
by a spectrum-matched random adapter. It therefore did not identify a
defensible tail-aligned direction for selective removal.

The first causal question is narrower:

> Does the learned orientation of the complete expert branch improve the exact
> released-CM prediction target, relative to both removing the expert and
> replacing it with equal-spectrum random orientations? Is any improvement
> larger for Few than Many classes?

This is an intervention on model parameters, not another representation
decoder.

## Matched conditions

All conditions reuse the same:

- held-out images and labels from the completed K1/K2 manifest;
- Gaussian-noise rows and discrete timesteps;
- released-CM endpoint-transfer target;
- model checkpoint, general weights, evaluation mode, and numerical runtime.

The conditions are:

1. **learned:** original \(\theta_g+\theta_e\);
2. **general:** original \(\theta_g\), evaluated with `use_cm=False`;
3. **spectrum-random:** replace every learned factor product
   \(B_lA_l=U_l\Sigma_lV_l^\top\) with
   \(Q_l\Sigma_lR_l^\top\), using independent deterministic orthonormal
   subspaces and four random repeats.

Each random intervention preserves, layer by layer:

- factor rank;
- every singular value of the official \(B_lA_l\) matrixization;
- Frobenius norm and stable rank;
- parameter count and adapted-layer locations.

The checkpoint is never rewritten. Every intervention records factor digests
and restores all factors bit-exactly. A zero-factor intervention must reproduce
`use_cm=False` within the active dtype tolerance.

## Primary endpoints

For each held-out row and timestep, let \(T\) be the exact released target and
let \(P_L,P_G,P_R\) denote learned, general, and random predictions:

\[
G_L = \operatorname{MSE}(P_G,T)-\operatorname{MSE}(P_L,T),
\]

\[
G_R = \operatorname{MSE}(P_G,T)-\mathbb E_R\operatorname{MSE}(P_R,T),
\]

\[
A_{L>R} =
\mathbb E_R\operatorname{MSE}(P_R,T)-\operatorname{MSE}(P_L,T).
\]

Positive \(G_L\) means the learned expert helps relative to removing it.
Positive \(A_{L>R}\) means the learned singular directions matter beyond the
same low-rank singular spectrum.

Report these endpoints for all, Many, Medium, and Few classes, plus:

- class-clustered 95% bootstrap intervals;
- Few-minus-Many contrasts;
- per-class means;
- learned/random prediction-displacement RMS;
- learned/random radial spectral fractions;
- random-versus-learned displacement cosine;
- variation across random rotations.

Classes, rather than individual image/timestep rows, are the bootstrap unit.

Parameter-spectrum matching does not guarantee equal displacement at the final
model output. As a prespecified sensitivity analysis, for each timestep and
random repeat compute one scalar from prediction displacements only:

\[
s_{t,r}
=
\frac{\operatorname{RMS}(P_L-P_G)}
{\operatorname{RMS}(P_{R_r}-P_G)}.
\]

Then evaluate \(P_G+s_{t,r}(P_{R_r}-P_G)\). This uses no target information and
tests random direction at the learned displacement magnitude. It is not claimed
to be a realizable parameter intervention and therefore complements rather
than replaces the spectrum-preserving weight intervention.

## Interpretation matrix

| Observation | Supported interpretation |
| --- | --- |
| \(G_L>0\), \(A_{L>R}>0\), no Few-Many difference | learned expert orientation is useful, but not specifically tail capacity |
| \(G_L>0\), \(A_{L>R}>0\), Few-Many interval \(>0\) | evidence for tail-selective learned expert orientation |
| \(G_L>0\), \(A_{L>R}\approx0\) | generic low-rank correction/energy is sufficient at this endpoint |
| \(G_L\approx0\) despite sampling gains | local transferred-target MSE misses the generative mechanism; run trajectory/sampling intervention |
| parameter-random is worse but response-matched random is not | apparent orientation advantage is explained by functional perturbation magnitude |

No local prediction result is reported as an FID or generation-quality result.
End-to-end sampling is promoted only after this screen identifies a stable
effect worth spending compute on.

## Validation gates

- `probe_inputs` exactly reproduces the noisy inputs and target in
  `probe_terms`;
- zero factors with `use_cm=True` reproduce `use_cm=False`;
- direct SVD verifies every randomized \(BA\) spectrum;
- all expert factors are restored bit-exactly after every repeat;
- the saved K1/K2 manifest fixes all data and noise choices;
- output includes every random repeat rather than only its mean.

## Result

All validation gates passed:

- zero factors with `use_cm=True` matched `use_cm=False` exactly
  (`max_abs=0`);
- the worst randomized-\(BA\) singular-spectrum error was `8.70e-7`;
- every factor restoration was bit-exact;
- four random rotations were retained separately;
- response matching used one target-free scalar per timestep/repeat.

The all-class paired endpoints are:

| Diffusion t | Sampling role | Learned gain vs general | Learned advantage vs response-matched random | Interpretation |
| ---: | --- | ---: | ---: | --- |
| 100 | late / low noise | +6.20e-5 (+0.077%) | +7.10e-5 (+0.089%) | small but stable learned-orientation benefit |
| 500 | middle | -2.54e-6 (-0.034%) | -3.27e-6 (-0.044%) | no learned benefit |
| 900 | early / high noise | -9.84e-7 (-0.377%) | -1.03e-6 (-0.395%) | learned expert is slightly worse locally |

Class-bootstrap intervals exclude zero for the positive \(t=100\)
response-matched advantage (`[5.48e-5, 8.79e-5]`) and for the negative
\(t=900\) advantage (`[-1.51e-6, -5.48e-7]`). The \(t=500\) effect is small;
its response-matched interval is also slightly negative
(`[-6.68e-6, -1.45e-7]`).

The unscaled random intervention produced more final displacement than the
learned expert. Mean response-matching scales were `0.664`, `0.800`, and
`0.775` at timesteps 100, 500, and 900. Matching this functional magnitude did
not remove the late low-noise advantage, but it reversed the apparent
learned-orientation advantage at middle and high-noise timesteps.

There is no tail-selective result. Across all timesteps, Few-minus-Many was:

- learned gain vs general: `-8.36e-6`, 95% CI
  `[-2.05e-5, 5.10e-6]`;
- learned advantage vs response-matched random: `-1.11e-5`, 95% CI
  `[-2.32e-5, 3.10e-6]`.

Every timestep-specific tail contrast also included zero. The point estimates
usually favored Many rather than Few classes.

The useful \(t=100\) displacement was not specifically high-frequency. Its
mean radial energy fractions from low to high were
`[0.239, 0.361, 0.313, 0.087]`, close to the random control
`[0.255, 0.347, 0.313, 0.085]`. At \(t=900\), the learned displacement was
more high-frequency than random but worsened the local target, so frequency
content alone does not identify a useful expert mechanism.

## Current conclusion

The learned expert orientation has a real but small causal effect at late,
low-noise denoising. It is not a generic advantage across the diffusion
trajectory and is not stronger for tail classes. This rejects the simple
account that \(\theta_e\) is a tail-specialized knowledge store at this
checkpoint.

The result is more consistent with a stage-specific joint correction: CM
training changes both \(\theta_g\) and \(\theta_e\), while the explicit expert
branch adds a small learned late-stage adjustment. Because local target-MSE
effects are small and can accumulate differently through sampling, the next
justified experiment is an end-to-end matched sampling intervention measuring
groupwise generative quality. It compares learned, general-only, and
spectrum-random experts under identical class labels, initial noise, DDIM
schedule, and classifier-free guidance.

The existing TensorFlow-FID-compatible Inception model predicts 1,008 ImageNet
labels, not CIFAR-100 labels. It therefore cannot provide requested-class
accuracy. The bounded screen instead uses:

- exact paired image-space displacement and its radial spectrum;
- overall and Many/Medium/Few KID, whose finite-sample behavior is preferable
  to FID for the initial 20-samples-per-class screen;
- paired Monte Carlo contrasts over the exact KID subset draws (reported as
  estimator uncertainty, not training-run uncertainty);
- a follow-up 100-samples-per-class FID run only if the screen shows a useful
  signal.

For each random repeat, the screen averages the target-free response-match
scalars from the prior local probe over its three timesteps, then multiplies
all randomized expert \(B\) factors by that one scalar in an independent
one-sample-per-class DDIM pilot. Because local response matching need not
survive composition through the full trajectory, the pilot then multiplies
each fixed random expert scale by

\[
\frac{\operatorname{RMS}(x_L-x_G)}
{\operatorname{RMS}(x_R-x_G)}.
\]

The final evaluation uses a different initial-noise seed. Unlike the earlier
post-hoc output rescaling, this is a realizable fixed-weight intervention
throughout the evaluation trajectory. It preserves the randomized
subspace/singular-spectrum shape but calibrates its global endpoint amplitude.

The end-to-end interpretation gate is:

- learned better than general: the explicit learned expert improves generation;
- learned better than calibrated random: its learned orientation matters beyond
  response magnitude;
- a larger Few than Many gain: evidence for tail-selective allocation;
- similar gains across frequency groups: a generic correction or regularizer,
  not a tail-specific knowledge store.

## End-to-end screen result

The 20-samples-per-class, four-random-orientation screen is documented in
[`results/official_imbdiff_matrix60k/cm_sampling_intervention_screen.md`](results/official_imbdiff_matrix60k/cm_sampling_intervention_screen.md).

The learned expert has lower KID than general-only overall and in every
frequency group. It also beats the mean of four endpoint-response-matched
random experts, although one random orientation beats learned. The learned
gain versus general is smaller for Few (`+0.000461`) than Many (`+0.000611`);
Few-minus-Many is `-0.000149`. About 68.9% of learned-general endpoint
displacement energy lies in the lowest radial band and only 0.43% in the
highest.

The result supports a small useful expert correction, but not a
tail-specialized or specifically high-frequency knowledge store. A
100-samples-per-class FID confirmation is documented in
[`results/official_imbdiff_matrix60k/cm_sampling_intervention_fid10k.md`](results/official_imbdiff_matrix60k/cm_sampling_intervention_fid10k.md).

At 10,000 generated samples, learned improves over general-only by 0.930 FID
and over the mean of four endpoint-matched random experts by 0.685 FID.
However, one random orientation beats learned by 0.233 FID and by 0.000535
KID. The learned branch is therefore useful and better than a typical random
orientation, but the result does not show that its exact orientation is
uniquely necessary. FID gain is also smaller for Few than Many classes, while
the KID tail contrast has the opposite small sign, leaving no
metric-consistent evidence of tail specialization.

## Sixteen-orientation follow-up

The larger 2k/KID orientation screen is documented in
[`results/official_imbdiff_matrix60k/cm_sampling_orientation16_kid2k.md`](results/official_imbdiff_matrix60k/cm_sampling_orientation16_kid2k.md).

Learned ranks fifth among itself and 16 endpoint-response-calibrated randomized
experts. It beats the random median and 12 of 16 random orientations, but four
random orientations beat it. Random 0 remains strong, while two new rotations
are better and another nearly ties it. Twelve random orientations also improve
over general-only.

The orientation spread is not explained by endpoint amplitude: Spearman
correlation between main-run RMS ratio and KID gain is `-0.018`
(`p=0.948`). Weak, uncorrected trends associate better quality with residuals
closer to the learned residual and with more mid/high-frequency response, but
16 rotations do not resolve either hypothesis.

The updated conclusion is that rank/spectrum/response-matched capacity often
provides a generic benefit, while subspace orientation materially modulates
that benefit. CM's learned orientation is better than typical but is not
uniquely privileged, and its explicit expert effect remains smaller for Few
than Many classes. A designed alignment/frequency intervention is more
informative than simply screening additional random rotations.
