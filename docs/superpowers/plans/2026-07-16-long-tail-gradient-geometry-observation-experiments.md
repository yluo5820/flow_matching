# Long-Tail Gradient-Geometry Observation Experiments Plan

**Date:** 2026-07-16
**Status:** Research design; observation and local diagnostic interventions only
**Primary system:** Class-conditional image flow matching with the existing `ImageUNetVelocity` backbone
**Primary datasets:** Fashion-MNIST-LT for the complete causal design, CIFAR-10-LT for confirmation

## Goal

Determine which of four explanations best describes long-tail learning geometry:

1. frequency status creates a stable, transferable gradient subspace;
2. apparent head/tail geometry follows semantic class identity instead;
3. geometry is mainly transient and follows checkpoint, flow timestep, or layer;
4. no reproducible low-dimensional signal exists beyond gradient noise.

The study must also determine whether any frequency effect is caused primarily by **scarcity of tail observations**, **interference from abundant classes**, or both, and whether the observed geometry predicts actual class-conditional generation failure beyond what class count already predicts.

This plan does **not** train a spectral adapter, shared/private expert, novelty router, or other corrective architecture. A small virtual parameter update is permitted only as a local functional test of an already discovered direction; it is not a method comparison.

## Central decision

The experiment should answer:

> After controlling semantic class identity, stochastic training variation, flow timestep, checkpoint, and layer, does changing only the number and relative exposure of examples from a class produce a reproducible change in its learning-demand geometry?

The central object is not a head-versus-tail spectrum from one model. It is a **within-class counterfactual contrast across models in which the same semantic class occupies different frequency ranks**.

## Why use a sequential design

Three designs were considered:

| Design | Advantage | Fatal weakness or cost | Decision |
|---|---|---|---|
| One canonical LT run followed by a detailed spectral probe | Cheap and close to the existing GS-CM plan | Cannot separate frequency from the identities of the classes assigned to the head and tail | Reject as the main design; retain only as a pipeline smoke test |
| Full factorial over dataset, imbalance ratio, mapping, seed, checkpoint, timestep, layer, and gradient representation | Maximally comprehensive | Most computation would be spent refining signals that may not exist; multiple-comparison burden would be severe | Reject |
| **Sequential counterfactual design** | Establishes reliability and frequency causality first, then spends compute on mechanism, dynamics, and confirmation | Requires explicit gates and disciplined stopping | **Use this design** |

The sequence is:

```text
measurement reliability
        ↓
frequency ↔ class counterfactual
        ↓
scarcity ↔ interference decomposition
        ↓
checkpoint × timestep × layer localization
        ↓
parameter-space ↔ function-space validation
        ↓
prediction of generative failure
        ↓
select or abandon a research direction
```

Later stages are conditional on earlier stages. A null result is a terminal scientific result, not a reason to enlarge the architecture.

---

## 1. Four outcome interpretations

The four outcomes are not forced to be mutually exclusive. In particular, a frequency-following effect may be restricted to certain checkpoints or timesteps. Report a primary explanation plus any temporal, timestep, or layer modifier.

| Outcome | Required observation pattern | Evidence against it | Research direction if supported |
|---|---|---|---|
| **A. Frequency-following geometry** | For the same class, changing frequency rank changes the gradient geometry in a consistent direction; frequency contrasts transfer to held-out classes and mappings; the signal survives function-space validation | The geometry stays with class identity when its rank changes, or disappears under cross-fitting | Study why scarcity or interference creates private learning directions; only then test a fixed spectral intervention |
| **B. Class-following geometry** | Same-class subspaces remain similar across frequency assignments; class identity explains substantially more reliable variation than count; frequency contrasts do not transfer across classes | A held-out class exhibits the same head-to-tail rotation predicted from other classes | Reframe as semantic sharing/private-capacity or donor-selection problem, not a specifically long-tail spectral mechanism |
| **C. Checkpoint/timestep-following geometry** | The signal is reproducible within a checkpoint/timestep cell but rotates or disappears across adjacent training stages or flow-time strata; interactions dominate a stationary main effect | A common subspace transports across most checkpoints and timesteps | Study dynamic optimization or timestep-localized failure; reject a single fixed basis unless a stable interval exists |
| **D. No stable signal** | Split-half and seed reliability are near their nulls, or all cross-fitted class/frequency/dynamic effects are practically negligible | A signal replicates across probe halves, mappings, and at least one second dataset | Stop spectral-capacity work; return to statistical estimation, path ambiguity, memorization, or evaluation failure modes |

### Outcome variables

Each outcome is evaluated using five kinds of evidence:

1. raw gradient magnitude;
2. normalized mean-gradient direction;
3. centered sample-gradient covariance and effective rank;
4. subspace transport on held-out classes, mappings, and probe samples;
5. local function change and final generation error.

No conclusion may rest only on an in-sample generalized eigenvalue or a visually suggestive spectrum.

---

## 2. Causal experimental unit

### 2.1 Frequency-rank intervention

For $C=10$ classes, define the exponential count schedule

\[
n_r=\max\left(1,\left\lfloor N\rho^{r/(C-1)}\right\rfloor\right),
\qquad r\in\{0,\ldots,C-1\},
\]

where $N$ is the full per-class training count and \(\rho\) is the minimum-to-maximum count ratio. The main study uses \(\rho=0.01\), corresponding to IR 100.

Instead of assigning rank $r=c$, construct ten mappings

\[
r_m(c)=(3c+m)\bmod 10,
\qquad m\in\{0,\ldots,9\}.
\]

Because 3 is coprime to 10:

- every mapping uses every frequency rank exactly once;
- every semantic class occupies every frequency rank exactly once across mappings;
- class identity and frequency rank are exactly balanced in the complete Fashion-MNIST design.

### 2.2 Nested within-class subsets

Create one deterministic random ordering of the original training examples inside each class. Reserve the first 500 examples per class as a balanced diagnostic pool that is excluded from every training mapping. Define $N$ from the remaining examples, then give a class assigned count $n_r$ the first $n_r$ examples from the remaining class-local ordering. Therefore, when a class changes from tail to head, its head dataset contains its tail dataset plus additional examples.

This nested construction makes the intervention interpretable as adding observations rather than replacing one random subset with another.

### 2.3 Paired training randomness

For each training seed:

- initialize all ten frequency-mapping models from the same parameter state;
- use the same optimizer configuration and step budget;
- pair source-noise, timestep, classifier-free dropout, and minibatch RNG streams where the different dataset sizes permit it;
- hold augmentation and dequantization procedures fixed by keyed sample seeds;
- disable early stopping for the causal runs.

Raw parameter subspaces are compared only **within a paired initialization seed**. Across independent seeds, aggregate scalar invariant statistics and function-space effects rather than directly averaging parameter vectors.

### 2.4 Dataset schedule

| Stage | Dataset | Conditions | Training seeds | Purpose |
|---|---|---:|---:|---|
| Pipeline smoke test | Fashion-MNIST | balanced, canonical IR100 | 1 | Verify artifacts, pairing, rank recovery, and null controls |
| Causal discovery | Fashion-MNIST | 10 Latin frequency mappings + balanced | 3 | Orthogonally estimate class and frequency effects |
| Mechanism decomposition | Fashion-MNIST | conditional 12-condition scarcity/interference design | 3 | Separate missing tail data from abundant-class interference |
| Dataset confirmation | CIFAR-10 | mappings \(m\in\{0,2,4,6,8\}\) + balanced | 3 | Test whether the observation survives a more heterogeneous image domain |
| Dose response | Fashion-MNIST and CIFAR-10 | IR10 subset of five mappings | 2 | Test monotonicity with imbalance severity; run only after an IR100 effect |

The CIFAR confirmation set is fixed before viewing CIFAR gradient results. It must not select mappings, classes, layers, or timesteps that looked favorable on Fashion-MNIST.

### 2.5 Compute envelope

The unconditional discovery stage contains 33 Fashion-MNIST trainings: ten mappings under three seeds plus three balanced controls. With 16 Probe-A and 16 Probe-B microbatches, its full six-checkpoint/five-timestep/ten-class probe contains approximately 317,000 backward passes; all six layer gradients are extracted in each pass.

The scarcity/interference stage is conditional and contains 72 Fashion-MNIST trainings: 12 dataset conditions × two samplers × three seeds. Probe only the frequency-sensitive cells locked in Observation 1, plus one early, one middle, and one late/timestep negative-control cell. Do not repeat the full Stage-1 grid.

The confirmation stage contains 18 CIFAR-10 trainings: five mappings and one balanced condition under three seeds. Begin with the locked Fashion-MNIST cells and expand to the complete checkpoint/timestep grid only if the CIFAR effect replicates.

Thus the expensive combinatorial probing occurs once, on the inexpensive discovery dataset. Later stages confirm or explain a locked observation rather than reopen model selection.

---

## 3. Common measurement protocol

### 3.1 Balanced probe data

Split the reserved 500-example-per-class diagnostic pool deterministically and once:

- **Probe-A:** 250 examples per class for direction discovery and rank selection;
- **Probe-B:** 250 examples per class for held-out geometric evaluation;

Keep the official balanced test set untouched as the **generation reference** for feature-based, coverage, and memorization-aware evaluation. It may be split internally for real-real metric calibration, but it is never used to discover a direction, choose a rank, select a checkpoint/timestep/layer, or set a stopping threshold.

If a dataset cannot reserve 500 training examples per class, reserve 10% of each original class with a minimum of 100 examples, split it equally into Probe-A and Probe-B, and reduce $N$ accordingly. Probe examples never enter training.

### 3.2 Paired flow tuples

For every probe example, precompute a manifest containing:

- target-image index;
- deterministic dequantization draw;
- source-noise seed;
- flow timestep stratum and timestep draw;
- microbatch assignment;
- Probe-A or Probe-B membership.

Use the same manifest for every mapping, checkpoint, and seed. This prevents different $(x_0,t,x_1)$ tuples from masquerading as geometric changes.

### 3.3 Flow timestep strata

Use five strata with equal probe allocation:

\[
[0.02,0.10],\quad[0.10,0.30],\quad[0.30,0.70],\quad
[0.70,0.90],\quad[0.90,0.98].
\]

Within a stratum, draw timesteps from a fixed keyed uniform sequence. Do not draw from the training logit-normal distribution for the primary probe; that would undersample some regions and confound prevalence with geometric importance.

### 3.4 Checkpoints

For the current 20,000-step Fashion-MNIST budget, save checkpoints at:

\[
500,\ 1{,}000,\ 3{,}000,\ 7{,}000,\ 13{,}000,\ 20{,}000.
\]

Also save step 0 as a pipeline control, but exclude it from the primary layer comparison. The current model zero-initializes the output convolution, so gradients in earlier layers can be structurally zero at initialization.

For datasets with a different budget, preserve the fractions 2.5%, 5%, 15%, 35%, 65%, and 100%. In addition to step, record balanced Probe-B loss, so a secondary analysis can distinguish chronological stage from merely reaching a particular loss level.

### 3.5 Layers

Probe shared convolutional weights at six network locations:

1. `input_block.conv2.weight` — early image features;
2. `down2_block.conv2.weight` — encoder;
3. `middle.conv2.weight` — bottleneck;
4. `up1_block.conv2.weight` — early decoder;
5. `up0_block.conv2.weight` — late decoder;
6. `output_block.2.weight` — output map.

Exclude `class_embedding.weight` from the primary analysis because its sparse per-class rows make class separation nearly tautological. Analyze `class_projection.weight` only as a conditioning-path sentinel.

### 3.6 Gradient rows

For every dataset/mapping/seed/checkpoint/timestep/class cell:

- form 16 fixed microbatches of 8 Probe-A examples and 16 corresponding Probe-B microbatches;
- compute the ordinary flow-matching loss, not the CM loss;
- collect all six layer gradients in the same backward pass;
- store the loss, raw gradient norm, normalized gradient sketch, and centered-gradient statistics;
- keep raw and normalized analyses separate.

For a row gradient $g_i$, store

\[
\|g_i\|_2,\qquad
u_i=\frac{g_i}{\|g_i\|_2+\epsilon}.
\]

For each cell, distinguish:

- the uncentered second moment (M=\frac1B\sum_i u_i u_i^\top), which contains the mean learning direction;
- the centered covariance (C=\frac1B\sum_i(u_i-\bar u)(u_i-\bar u)^\top), which measures directional variability;
- the directional concentration
  \[
  \kappa=\left\|\frac1B\sum_i u_i\right\|_2,
  \]
  which detects the possibility that an apparent tail spectrum is merely higher directional noise.

### 3.7 Storage and sketches

Do not store every full gradient row. Use:

- exact sample-space Gram matrices for within-cell spectra;
- one fixed signed CountSketch or Rademacher projection per layer for cross-condition comparisons;
- exact raw norms and losses;
- full gradients only for a preregistered 5% validation subset.

Begin with 4,096 sketch coordinates per large layer and the exact dimension for layers smaller than 4,096 parameters. On the validation subset, require absolute cosine error below 0.02 and normalized subspace-overlap error below 0.03. Double the sketch dimension if either criterion fails.

For a selected direction used in a local functional test, recompute and materialize the exact full direction from its original probe rows.

---

## 4. Observation 0 — establish the noise ceiling

This is a prerequisite, not one of the four substantive interpretations.

### Question

Is there a reproducible gradient direction or subspace inside a fixed class/mapping/checkpoint/timestep/layer cell?

### Measurements

For candidate ranks $k\in\{1,2,4,8,16,32\}$, compute normalized projection overlap

\[
S(U,V)=\frac{1}{k}\|U^\top V\|_F^2
\]

between:

- two disjoint halves of Probe-A;
- Probe-A and Probe-B;
- two independently drawn source-noise replicas for the same target examples;
- repeated training seeds, summarized in function space rather than by directly aligning parameters.

Record principal angles, effective rank, directional concentration, and mean-direction cosine. Compare against:

- shuffled class labels;
- shuffled mapping/rank labels;
- random subspaces with matched rank and sketch dimension;
- probe tuples with deliberately unpaired timestep/noise draws.

### Reliability gate

A cell is *measurable* only if its Probe-A/Probe-B overlap exceeds the 99th percentile of its matched permutation null and the result repeats in at least two of three training seeds. All later explained-variance quantities are divided by this split-half reliable variance, producing reliability-normalized effects.

If no two adjacent non-output network stages contain measurable cells after increasing Probe-A/B rows from 16 to 32, classify the proposed *network-wide* stable geometry as Outcome D and stop model-wide spectral analysis. A measurable output-layer-only result may still be reported as an explicitly layer-localized observation, but it cannot justify a decoder-wide or network-wide capacity claim.

---

## 5. Observation 1 — frequency-following versus class-following geometry

### Question

When a semantic class moves from head to tail across the ten mappings, does its geometry follow its identity or its count?

### 5.1 Balanced functional ANOVA

Let $z_{m,c,s,t,\ell}$ denote a mean-gradient sketch, centered-covariance projection, raw log norm, or subspace projection representation for mapping $m$, class $c$, checkpoint $s$, timestep stratum $t$, and layer \(\ell\). Fit, within each paired initialization seed,

\[
z_{m,c,s,t,\ell}
=\mu
+a_c
+b_{r_m(c)}
+d_s
+e_t
+f_\ell
+ (b\!\times\!d)_{r,s}
+ (b\!\times\!e)_{r,t}
+ (b\!\times\!f)_{r,\ell}
+\varepsilon.
\]

Because every class occupies every rank exactly once, $a_c$ and $b_r$ are identifiable without relying on semantic similarity assumptions.

Report cross-validated, reliability-normalized partial $R^2$ for:

- class identity;
- frequency rank or continuous (log n_r);
- checkpoint;
- timestep;
- layer;
- frequency-by-checkpoint, frequency-by-timestep, and frequency-by-layer interactions.

Use leave-one-mapping-out validation for class effects and leave-one-class-out validation for frequency effects. Bootstrap entire class and mapping blocks, not individual pairwise similarities.

### 5.2 Within-class frequency contrasts

For every class $c$, form a counterfactual contrast at a fixed checkpoint/timestep/layer:

\[
d_c
=\mathbb E[\bar u_{m,c}\mid r_m(c)\in\text{tail}]
-\mathbb E[\bar u_{m,c}\mid r_m(c)\in\text{head}].
\]

Use ranks 0–2 as head, 3–6 as medium, and 7–9 as tail for categorical summaries; use continuous $\log n_r$ for the primary estimate.

Learn a frequency-contrast subspace from five discovery classes by SVD of their $d_c$ vectors. Lock the rank using Probe-A only, then measure how much of the contrast energy for the other five classes lies in that subspace using Probe-B. Swap the class halves and average the two directions of cross-fitting.

A frequency subspace is scientifically interesting only if it transfers to semantic classes excluded from its construction.

### 5.3 Same-class and same-rank transport

Compute two matched transport statistics:

- **class transport:** subspace similarity for the same class across different frequency ranks;
- **rank transport:** subspace similarity for different classes occupying the same frequency rank.

Match checkpoint, timestep, layer, initialization seed, and probe tuple. Compare both against the reliability ceiling rather than against the theoretical maximum of one.

### Decision for Outcomes A and B

Support **Outcome A** only if all of the following hold:

1. frequency explains at least 10% of reliable held-out variation and its 95% block-bootstrap interval excludes zero;
2. the frequency-contrast subspace transfers to held-out classes above the 99th percentile of the rank-permutation null;
3. the sign of the effect agrees across all three Fashion-MNIST seeds;
4. at least one local function-space test in Observation 4 confirms that the direction changes tail loss selectively;
5. the effect is reproduced on CIFAR-10 before making a general claim.

Support **Outcome B** when class identity explains at least twice the reliable variation explained by frequency, class transport is positive, and the held-out frequency-contrast test fails or is practically negligible.

If both class and frequency survive, report a mixed result rather than forcing a binary label. The relevant question then becomes whether frequency changes a stable semantic subspace, creates a new subspace, or only changes its magnitude.

---

## 6. Observation 2 — scarcity versus abundant-class interference

Run this stage only if Observation 1 finds a reproducible frequency effect or a strong frequency-by-stage interaction.

### Question

Does a tail class change because it has few examples, because other classes are seen much more often, or because both conditions interact?

### Dataset design

Let $n$ be the IR100 tail count and $N$ the full class count. Create:

- **LL:** every class has $n$ examples;
- **HH:** every class has $N$ examples;
- **LH(c):** anchor class $c$ has $n$ examples and all other classes have $N$.

There are 12 conditions: LL, HH, and one LH condition for each of the ten classes. Use the same nested class-local subsets and paired initialization seeds as Observation 1.

For anchor class $c$:

- **interference contrast:** LH(c) minus LL, holding the data of $c$ fixed while adding other-class data;
- **scarcity contrast:** LH(c) minus HH, holding other-class data fixed while removing data from $c$;
- **scale control:** HH minus LL, where relative frequencies remain balanced.

### Sampling controls

Train each condition under two samplers:

1. natural example-proportional sampling;
2. class-balanced sampling with the same number of total optimizer steps.

The comparison distinguishes dataset support from optimization exposure:

- an effect present under both samplers points toward finite tail support or representation competition;
- an effect disappearing under class-balanced sampling points toward gradient exposure imbalance;
- an effect appearing only under natural sampling and growing with head count is evidence for majority domination;
- a class-balanced LL-versus-HH gap reflects absolute sample size rather than long-tail imbalance.

Record the realized number of examples and optimizer exposures per class; nominal dataset counts are not sufficient.

### Analysis

Apply the same Probe-A/Probe-B geometry measurements to the anchor class and measure:

\[
I_c=\operatorname{Geom}_c(\mathrm{LH}(c))
-\operatorname{Geom}_c(\mathrm{LL}),
\]

\[
Q_c=\operatorname{Geom}_c(\mathrm{LH}(c))
-\operatorname{Geom}_c(\mathrm{HH}).
\]

Here `Geom` is evaluated separately for raw norm, reliable directional variance, frequency-contrast energy, and final generation error. Do not collapse these into one score before inspecting whether their signs agree.

### Interpretation

| Pattern | Mechanism most consistent with it | Consequence |
|---|---|---|
| Large $I_c$, small $Q_c$ | Abundant-class interference dominates | Capacity isolation, gradient balancing, or exposure correction becomes plausible |
| Small $I_c$, large $Q_c$ | Missing tail support dominates | Prefer posterior-predictive estimation, safe sharing, or augmentation over private capacity |
| Both large | Scarcity and interference interact | Any method needs both statistical sharing and optimization protection |
| Neither stable | The frequency result from Observation 1 was likely mediated by another factor | Revisit checkpoint/timestep interaction or stop |

---

## 7. Observation 3 — checkpoint, timestep, and layer localization

### Question

Is a discovered signal a persistent model property or a transient feature of a particular training stage, flow-time region, or network layer?

### Temporal transport matrix

For every measurable cell, compute subspace transport across all checkpoint pairs while holding mapping, class, timestep, and layer fixed. Produce:

- a six-by-six checkpoint overlap matrix;
- a five-by-five timestep overlap matrix;
- a six-by-six layer overlap matrix;
- corresponding function-space overlap matrices for selected directions.

Normalize each cross-cell overlap by the geometric mean of its two split-half reliabilities:

\[
S_{\mathrm{rel}}(U,V)
=\frac{S(U,V)}{\sqrt{S(U,U')S(V,V')}}.
\]

Clip only for display, not for statistical analysis.

### Dynamic effects

Estimate:

- persistence length in training steps: the largest checkpoint separation with transport above half the within-cell reliability ceiling;
- timestep bandwidth: the largest flow-time separation satisfying the same criterion;
- layer localization: frequency-explained reliable variance by architectural stage;
- emergence and disappearance: the first and last checkpoints at which the frequency contrast passes the held-out test.

Repeat key comparisons at matched balanced-probe loss levels to distinguish training chronology from merely having different residual errors.

### Decision for Outcome C

Support **Outcome C** when cells are internally reliable but either:

- frequency-by-checkpoint or frequency-by-timestep explains more reliable variance than the stationary frequency main effect; or
- adjacent-checkpoint/timestep transport falls below 50% of the within-cell reliability ceiling in at least two seeds; or
- the sign of local tail selectivity reverses over training or flow time.

Interpretation must be specific:

- early-only signal → optimization acquisition problem;
- late-only signal → residual correction or overfitting problem;
- high-noise-only signal → coarse semantic allocation;
- low-noise-only signal → class-detail or memorization problem;
- decoder-only signal → output/detail specialization;
- network-wide signal → global optimization geometry.

A fixed spectral basis is ruled out if the relevant subspace does not transport across the intended fine-tuning interval.

---

## 8. Observation 4 — parameter-space robustness and local functional meaning

Run this stage only on cells selected by the preregistered cross-fitting procedure, never by visual inspection of Probe-B.

### 8.1 Gradient representations

Repeat the selected analyses with:

1. ordinary Euclidean gradients;
2. row-normalized Euclidean gradients;
3. diagonal empirical-Fisher-whitened gradients;
4. a function-space signature.

For a parameter direction $d$, define its function-space signature on a fixed balanced anchor set $\mathcal A$ as

\[
\Phi(d)
=\left[J_\theta f_\theta(x_a,t_a,c_a)d\right]_{a\in\mathcal A}.
\]

Use the same anchors for every mapping and include all five timestep strata. Similarity between $\Phi(d)$ values asks whether two parameter directions cause similar first-order changes to the learned vector field, even if raw coordinates differ.

The claim is robust only if its qualitative outcome survives Fisher whitening or function-space comparison. A result visible only in ordinary coordinates is a parameterization observation, not yet a model mechanism.

### 8.2 Local virtual update

For a cross-fitted subspace $U$, construct a descent direction using Probe-A only:

\[
d=-UU^\top \bar g_{\mathrm{target}},
\]

where `target` means held-out semantic classes assigned to the relevant frequency status, not the samples later used for evaluation.

Apply $d$ to a copy of the checkpoint at relative layerwise step sizes

\[
\|\eta d_\ell\|_2/\|\theta_\ell\|_2
\in\{10^{-5},3\times10^{-5},10^{-4}\}.
\]

Evaluate Probe-B losses for every class and timestep. Accept the largest step only if loss changes remain in the local regime: doubling the step may change the first-order prediction by at most 10% relative error.

Report:

- target-frequency loss decrease;
- non-target loss change;
- worst-class harm;
- cosine between predicted first-order and observed finite-step changes;
- the same statistics for random, joint-PCA, and class-following subspaces of equal rank.

Define target benefit and non-target harm as

\[
B_T=-\Delta L_T,\qquad
H_N=\max(0,\Delta L_N).
\]

Call an update *functionally selective* only if the lower 95% block-bootstrap bound for $B_T$ is positive, its selectivity $B_T-H_N$ exceeds the 99th percentile of matched random-subspace controls, and the upper 95% bound for $H_N$ is less than half of $B_T$. Lock this rule before evaluating Probe-B.

This test establishes whether the geometry has local functional meaning. It does not establish that prolonged training in the subspace will work.

---

## 9. Observation 5 — does geometry explain generation failure?

The geometric result is relevant to long-tail generation only if it is associated with an actual generative failure mode.

### Balanced generation

At the final checkpoint of every causal mapping run, generate 5,000 samples per class with paired source seeds and fixed solver settings. For the six checkpoint trajectories of a smaller preregistered subset of mappings, generate 1,000 samples per class to measure when final-quality differences emerge.

Use the existing balanced evaluation path and report per class:

- class-conditional feature distance, with real-real calibration;
- precision and recall or their available feature-space analogues;
- classifier consistency and entropy;
- within-class diversity;
- nearest-neighbor distance to training examples and to held-out real examples;
- duplicated or near-duplicated sample rate.

Do not treat a lower training or probe flow-matching loss as sufficient evidence of better generation.

### Geometry predictors

For each class/mapping/checkpoint, compute cross-fitted predictors:

- gradient norm;
- directional concentration;
- reliable effective rank;
- frequency-contrast energy;
- class-specific residual energy;
- model-relative novelty
  \[
  \nu_c
  =1-\frac{\operatorname{tr}(U_{-c}^\top C_cU_{-c})}
  {\operatorname{tr}(C_c)+\epsilon},
  \]
  where $U_{-c}$ is a shared subspace estimated without class $c$.

Fit nested out-of-sample predictors of final class-wise failure:

1. semantic class identity only;
2. class identity plus (log n_c);
3. class identity, (log n_c), and geometric predictors;
4. the same model with checkpoint/timestep interactions.

Use leave-one-class-and-one-mapping-out validation. The geometry is explanatory only if model 3 improves held-out prediction beyond model 2 and the improvement repeats on CIFAR-10.

This analysis directly tests the useful part of the AC-CM intuition: whether model-relative unexplained learning demand is more informative than frequency count, without building an online router.

---

## 10. Statistical protocol

### 10.1 Units and uncertainty

- The training run, frequency mapping, and semantic class are experimental blocks.
- Probe microbatches estimate within-run measurement noise; they are not independent training replicates.
- Bootstrap mappings and classes as blocks, nested inside training seed.
- Report seed-level effects individually before their aggregate.
- Use max-statistic permutation correction across the preregistered checkpoint × timestep × layer grid.

### 10.2 Cross-fitting

Use separate partitions for:

- subspace discovery: Probe-A;
- rank selection: an internal split of Probe-A;
- primary geometric evaluation: Probe-B;
- generation evaluation: the disjoint generation-reference set;
- semantic transport: held-out classes;
- frequency transport: held-out mappings.

No direction, rank, checkpoint, timestep, or layer may be selected using Probe-B or generation metrics.

### 10.3 Rank reporting

Always report the complete stable spectrum for $k\le32$. For local virtual updates, select $k\in\{1,2,4,8,16,32\}$ on the Probe-A validation split and lock it before Probe-B evaluation. Also report the smallest $k$ containing 90% of the cross-fitted contrast energy.

### 10.4 Practical null

Failure to reject zero is not evidence for Outcome D. Declare a practically null frequency effect only when:

- the upper 95% confidence bound is below 10% of reliable held-out gradient variation on both datasets; and
- frequency-contrast transport is below the preregistered smallest meaningful overlap; and
- the local functional selectivity is no larger than the best matched random-subspace control.

Calibrate the smallest meaningful overlap in the pipeline smoke test as the overlap required to predict at least a 1% relative held-out tail-loss change under a valid local step. Lock this value before the full mapping study.

### 10.5 Negative and positive controls

Negative controls:

- permuted frequency ranks within each class across mappings;
- shuffled class labels in probe rows;
- random class partitions;
- unpaired source/timestep draws;
- random subspaces and joint-PCA subspaces at matched rank;

Positive controls:

- class-embedding gradients should reveal class identity but are excluded from the primary claim;
- output-layer gradients at very early training should be measurable despite zero initialization upstream;
- an artificial condition-specific linear residual inserted into a toy model should be recovered at its known rank.

---

## 11. Decision matrix

Use this matrix after Fashion-MNIST discovery and again after CIFAR confirmation.

| Reliable within cells? | Frequency transport? | Class transport? | Dynamic interaction? | Functionally selective? | Conclusion |
|---|---|---|---|---|---|
| No | — | — | — | — | **D: no measurable stable geometry** |
| Yes | Yes | No or weaker | Weak | Yes | **A: stable frequency-induced geometry** |
| Yes | Yes | No or weaker | Strong | Yes only in localized cells | **A+C: dynamic frequency geometry** |
| Yes | No | Yes | Any | Class-selective only | **B: semantic private geometry** |
| Yes | No | No | Strong | Localized | **C: transient error-correction geometry** |
| Yes | No | No | Weak | No | **D: stable variance exists but it is unrelated to the proposed factors** |
| Yes | Yes | Yes | Any | Yes | Mixed class and frequency mechanism; quantify additive versus rotational change before designing a method |

### Method-selection consequences

| Confirmed mechanism | Next research hypothesis worth testing | Ideas to defer or reject |
|---|---|---|
| Stable frequency-induced subspace caused by interference | Fixed cross-fitted spectral intervention; compare against exposure balancing and gradient surgery | Online novelty routing until fixed-basis causality is established |
| Frequency effect caused mainly by scarcity | Posterior-predictive tail flow, safe empirical-Bayes sharing, uncertainty-aware augmentation | Private capacity as the primary solution |
| Semantic class-following geometry | Safe donor selection, shared/private semantic residual analysis | Calling the effect a long-tail-specific spectral mechanism |
| Timestep-localized frequency effect | Time-conditioned intervention or path analysis | One global basis or one global router target |
| Checkpoint-localized effect | Training-stage intervention, optimizer/exposure study | Permanent architectural experts |
| No stable geometry | Study path ambiguity, memorization, evaluation bias, or finite-sample estimation | GS-CM, CR-CM, and AC-CM in their current forms |

---

## 12. Staged execution and stopping rules

### Stage 0 — pipeline validation

- [ ] Verify nested per-class subsets and frequency-map balance.
- [ ] Verify paired probe manifests produce identical $(x_0,t,x_1)$ tuples across models.
- [ ] Verify checkpoint restoration reproduces saved balanced-probe loss.
- [ ] Verify exact and sketched gradient similarities meet the sketch-error tolerances.
- [ ] Verify all label/rank permutation controls return null effects.
- [ ] Verify the artificial low-rank positive control recovers its planted dimension.

**Stop:** do not start the causal runs until every item passes.

### Stage 1 — Fashion-MNIST causal mapping study

- [ ] Train ten frequency mappings under three paired initialization seeds plus three balanced controls.
- [ ] Collect Probe-A/Probe-B measurements at all six checkpoints, five timestep strata, and six layers.
- [ ] Establish reliability ceilings and run the class-versus-frequency analysis.
- [ ] Lock the discovery analysis code, rank rule, and practical-null threshold.

**Stop as Outcome D:** if measurement remains unreliable after doubling probe rows.
**Skip mechanism decomposition:** if frequency and all frequency interactions are practically null.
**Continue:** if class, frequency, or dynamic effects survive cross-fitting.

### Stage 2 — mechanism decomposition

- [ ] Run LL, HH, and ten LH(c) conditions with natural sampling.
- [ ] Repeat with class-balanced sampling.
- [ ] Estimate scarcity and interference contrasts class by class.
- [ ] Test whether the mechanism agrees with the effect observed in the exponential schedule.

**Stop the interference hypothesis:** if adding other-class data at fixed anchor data produces no reliable geometric or generative change.
**Stop the scarcity hypothesis:** if removing anchor data at fixed other-class data produces no reliable change.

### Stage 3 — dynamics and function-space validation

- [ ] Produce checkpoint, timestep, and layer transport matrices.
- [ ] Recompute selected directions with Fisher whitening.
- [ ] Compute function-space JVP signatures.
- [ ] Run locked local virtual updates and matched subspace controls.

**Reject a fixed basis:** if transport is below half the reliability ceiling across the intended update interval.
**Reject a parameter-space mechanism:** if ordinary-coordinate effects disappear in both Fisher-whitened and function-space analyses.

### Stage 4 — generation linkage and CIFAR confirmation

- [ ] Generate balanced samples and evaluate class-wise quality, coverage, and memorization.
- [ ] Test whether geometry predicts failure beyond class count.
- [ ] Repeat the locked causal protocol on the five preregistered CIFAR mappings.
- [ ] Run IR10 dose response only for effects confirmed at IR100.

**Promote a research direction:** only after an observation survives cross-fitting, local functional validation, generation linkage, and CIFAR confirmation.
**Report a dataset-specific finding:** if Fashion-MNIST survives but CIFAR does not.
**Report a null:** if the practical-null criterion is met on both datasets.

---

## 13. Artifact and reproducibility contract

Every run should save:

```text
runs/long_tail_geometry/<dataset>/<condition>/seed_<seed>/
  config.yaml
  metadata.json
  frequency_map.json
  subset_manifest.npz
  probe_manifest.npz
  checkpoints/
    step_000500.pt
    step_001000.pt
    step_003000.pt
    step_007000.pt
    step_013000.pt
    step_020000.pt
  diagnostics/
    gradient_rows.parquet
    gradient_sketches.npz
    gram_matrices.npz
    reliability.csv
    factor_effects.csv
    subspace_transport.csv
    functional_updates.csv
  evaluation/
    class_metrics.json
    reference_calibration.json
```

The aggregate study should save:

```text
runs/long_tail_geometry/aggregate/
  preregistration.yaml
  run_registry.csv
  exclusion_log.csv
  outcome_decision.json
  tables/
  plots/
```

`preregistration.yaml` must contain the mappings, seeds, checkpoint steps, timestep strata, probed layers, ranks, null generators, bootstrap units, practical-null threshold, and stopping rules before Stage 1 begins.

No run is silently removed. Failed or corrupted runs appear in `exclusion_log.csv` with a reason decided without viewing their outcome metrics.

---

## 14. Required tables and figures

The final observation report must include:

1. class × mapping count matrix demonstrating orthogonality;
2. raw norm, concentration, and effective-rank curves by count;
3. split-half reliability heatmaps by checkpoint/timestep/layer;
4. partial reliable variance attributed to class, frequency, checkpoint, timestep, and layer;
5. same-class versus same-rank transport;
6. held-out class frequency-contrast spectrum and permutation null;
7. LL/HH/LH scarcity-versus-interference contrasts;
8. checkpoint and timestep transport matrices;
9. Euclidean, Fisher-whitened, and function-space agreement;
10. local virtual-update selectivity against random, joint-PCA, and class-following controls;
11. out-of-sample improvement from geometry when predicting class-wise generation failure;
12. the completed four-outcome decision matrix, including negative results.

Every heatmap must display its reliability mask. Unreliable cells are gray rather than assigned zero.

---

## 15. Implementation boundary for a later engineering plan

The observation protocol will eventually require isolated components for:

- frequency-map and nested-subset construction in `fm_lab/data/long_tail.py`;
- intermediate checkpoint retention in `fm_lab/training/trainer.py` and `fm_lab/utils/checkpoints.py`;
- paired probe-manifest construction;
- gradient collection, sketch validation, and sample-space Gram analysis;
- cross-fitted factor and subspace analysis;
- local virtual updates on model copies;
- balanced per-class generation evaluation and aggregate reporting.

These components should not be implemented until this research design is accepted. When implementation begins, each component needs its own tests, config schema, exact artifact contract, and smoke-run gate. Existing user changes and CM experiments must remain untouched.

## Final criterion

The study succeeds if it leaves us with one of two things:

1. a reproducible, causal observation precise enough to justify one class of intervention and rule out others; or
2. a strong null result that prevents further engineering around an unsupported spectral story.

It does not succeed merely because a head/tail eigenspectrum looks separated.
