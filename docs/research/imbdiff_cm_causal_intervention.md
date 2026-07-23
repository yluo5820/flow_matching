# ImbDiff-CM matched causal expert intervention

**Status:** implemented; full released-CM 60k checkpoint run pending.

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

## Interpretation matrix

| Observation | Supported interpretation |
| --- | --- |
| \(G_L>0\), \(A_{L>R}>0\), no Few-Many difference | learned expert orientation is useful, but not specifically tail capacity |
| \(G_L>0\), \(A_{L>R}>0\), Few-Many interval \(>0\) | evidence for tail-selective learned expert orientation |
| \(G_L>0\), \(A_{L>R}\approx0\) | generic low-rank correction/energy is sufficient at this endpoint |
| \(G_L\approx0\) despite sampling gains | local transferred-target MSE misses the generative mechanism; run trajectory/sampling intervention |
| random controls are worse only because their displacement RMS is much larger | parameter-spectrum matching is insufficient; add a functional-response-matched control |

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
