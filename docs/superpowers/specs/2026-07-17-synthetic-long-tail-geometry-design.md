# Synthetic Long-Tail Geometry Experiment

**Date:** 2026-07-17
**Status:** Approved for implementation planning
**Scope:** Synthetic-first discovery experiment for long-tail conditional image
generation

## 1. Purpose

The experiment asks a narrower question than whether long-tail generation has
worse aggregate image quality:

> When examples from a class become scarce, does the model selectively lose
> valid directions of variation, and is this loss worse when the class has a
> higher-dimensional data manifold?

The design deliberately separates three quantities:

- class cardinality, which controls the available evidence;
- calibrated data dimension, which controls the number of independently
  visible factors the model must recover;
- learned effective geometry, measured through factor coverage and independent
  local-geometry probes.

This is a causal discovery experiment in a controlled renderer family. It is
not intended to establish immediate generality to natural images. Natural-image
experiments come later, after the synthetic study identifies a precise effect
worth testing.

The corrected Fashion-MNIST benchmark already shows the expected aggregate
ordering: its balanced run is substantially better than the long-tail run. The
present experiment therefore studies the mechanism behind that loss rather
than trying to re-establish the aggregate result.

## 2. Scientific framing

Let (d_{\mathrm{data},c}(x)) denote the local dimension of the known renderer
manifold for class (c), and let (d_{\mathrm{model},c}(x)) denote an estimated
effective dimension of the learned distribution. Define the geometric deficit

\[
\Delta d_c(x) = d_{\mathrm{data},c}(x) - d_{\mathrm{model},c}(x).
\]

Low data dimension is not synonymous with memorization. A model can correctly
learn a genuinely low-dimensional class. The relevant failure is a learned
dimension or factor range below the calibrated data geometry. Conversely, a
high-dimensional class is expected to demand more distinct observations to
recover all of its factors.

The deterministic flow map is formally full-rank under regular ODE
assumptions. Its Jacobian spectrum is therefore interpreted as an effective
contraction probe, not as literal manifold rank. Factor recovery and FM-FLIPD
provide complementary evidence.

## 3. Hypotheses

### H1: Frequency and data dimension interact

Reducing the number of unique class examples will degrade known-factor
coverage more strongly for higher-dimensional classes. Evidence for H1 is an
interaction between \(\log_{10}(n_c)\) and calibrated data dimension, not merely
an overall head-tail quality difference.

### H2: Scarcity causes directional geometric memorization

As sample count falls, the learned distribution will contract along one or
more valid renderer directions before, or without, producing exact copies of
training images. The predicted signature is jointly reduced factor range,
weaker tangent alignment or effective Jacobian singular values, and a positive
dimension deficit. Exact-copy statistics may remain low at the first point
where this signature appears.

### H3: Shared directions may be protected by capacity borrowing

In a high-dimensional tail class, directions also varied by other classes may
be preserved better than class-private directions. This comparison is
exploratory because factor identity and sharing level cannot be made completely
exchangeable in the first experiment. A later sharing ablation is required for
a causal capacity-borrowing claim.

### H4: Long-tail context can alter a well-sampled class

The same 5,000-example object-dimension cell may behave differently when its
two companion classes are balanced versus scarce. This contrast measures
shared-model transfer or interference independently of that class's own
cardinality.

## 4. Synthetic object classes

The experiment uses three asymmetric, silhouette-distinct objects. The marked
cube is excluded. The current abstract statue and offset monument are not used
together unchanged because their stacked block structure and face coloring are
visually confusable.

The object set is:

1. **Stepped monument:** a vertical, offset stack with an asymmetric cap.
2. **Crooked arch:** unequal legs joined by an off-center lintel, with a large
   negative-space opening.
3. **Three-arm vane:** a compact core with three unequal arms and an offset fin,
   producing a broad radial or diagonal silhouette.

All objects use the same analytic rendering backend, material response,
lighting, background, and absence of texture. Object identity is reinforced by
a fixed uniform hue, with hues separated by 120 degrees at common OKLCH
lightness 0.70 and chroma 0.12. Color does not vary within a class and therefore
does not add a manifold factor.

Object sizes are calibrated so their held-out distributions of foreground
occupancy, luminance, and contrast differ by no more than 0.25 pooled standard
deviations. On an independent reference set, an object oracle must exceed 99%
classification accuracy. If either gate fails, object geometry, scale, or hue
is revised before any generative training.

## 5. Known factor ladder

Every active coordinate is sampled independently. Inactive coordinates remain
at their canonical zero or nominal-camera value.

| Level | Active factors | Nominal dimension |
| --- | --- | ---: |
| Low | camera-axis object translation \(z\) | 1 |
| Medium | object translation \((x,y,z)\) | 3 |
| High | translation \((x,y,z)\), camera azimuth, camera elevation | 5 |

Initial ranges are fixed as follows:

- \(x,y \sim U[-0.25,0.25]\) world units;
- \(z \sim U[-0.75,0.75]\) world units around camera distance 4.0;
- azimuth is uniform on \([-180^\circ,180^\circ)\);
- elevation is area-uniform over \([-30^\circ,30^\circ]\), implemented by
  sampling uniformly in sine elevation.

Coordinates are normalized to unit range for geometric comparisons. Azimuth is
represented by sine and cosine wherever an ordinary Euclidean loss would
otherwise introduce a discontinuity.

The factor ladder is nested intentionally:

- depth is shared by all three dimension levels;
- lateral translation is shared by medium and high levels;
- viewpoint variation is private to the high level.

This creates a first probe of shared versus private direction preservation.
It does not by itself prove capacity borrowing.

## 6. Renderer calibration gates

Before model training, a dense independent renderer sample establishes the
true visible geometry. The existing pullback-metric and render-map Jacobian
infrastructure is reused.

After normalizing factor coordinates to unit range:

- at least 95% of calibration points must have numerical renderer-Jacobian rank
  equal to the nominal dimension, using a relative singular-value threshold of
  0.02;
- the ratio between the largest and smallest median per-factor pullback norm
  must not exceed 4;
- deliberately full-range, half-range, and collapsed-factor reference samples
  must be ordered correctly by every proposed coverage metric.

Factor ranges, object scale, and camera parameters may be adjusted only during
this calibration stage. They are frozen before the balanced training pilot.
Failure of a gate blocks later stages rather than being silently accepted.

## 7. Counterbalanced factorial design

Let the objects be A, B, and C. Object identity is counterbalanced against
nominal dimension with three cyclic mappings:

| Geometry mapping | A | B | C |
| --- | --- | --- | --- |
| G0 | high | medium | low |
| G1 | medium | low | high |
| G2 | low | high | medium |

Frequency is independently rotated:

| Frequency mapping | A | B | C |
| --- | ---: | ---: | ---: |
| F0 | 5,000 | 500 | 50 |
| F1 | 500 | 50 | 5,000 |
| F2 | 50 | 5,000 | 500 |

Crossing G0--G2 with F0--F2 produces nine imbalanced conditions and places
every object-dimension-frequency combination in the design once. Each geometry
mapping also has a balanced \((5{,}000,5{,}000,5{,}000)\) control. The complete
matrix therefore has 12 joint-model conditions per replicate.

Three paired replicate bundles are run, for 36 trained models in the main
matrix. Within a bundle, every condition shares renderer conventions and
nested master pools. Between bundles, both the master-pool seed and model seed
change. Conclusions are limited to reproducibility in this renderer family;
the three objects are controls for identity, not a random sample of all object
categories.

## 8. Dataset construction and storage

Images are rendered at \(32\times32\) RGB. This matches CIFAR-10 resolution and
keeps the experiment fast. Resolution is increased only if the renderer or
oracle calibration gates fail after reasonable factor-range and object-scale
adjustment.

For each replicate and each of the nine object-dimension cells, create one
5,000-example master training pool. The 500- and 50-example sets are immutable
nested prefixes of that pool after one seeded permutation. Frequency rotations
refer to indices in the same pool rather than copying images.

Training pools are stored as compact uint8 arrays with factor-coordinate and
seed manifests, not individual PNG files. One replicate requires about 132 MiB
uncompressed; three require about 396 MiB before metadata. Independent
calibration and held-out references are rendered deterministically on demand
and are never eligible for training.

Every artifact records:

- renderer, data, and model seeds;
- object, dimension, and frequency mapping identifiers;
- exact training indices and factor coordinates;
- normalized configuration hash;
- code revision and checkpoint identifier.

## 9. Training protocol

The main experimental unit is one shared class-conditional image flow-matching
model trained on all three classes in a condition. The existing corrected
image flow-matching baseline is reused, with only the input resolution and
number of classes changed. Architecture, objective, optimizer, batch size,
time sampling, conditioning mechanism, solver, and sampling settings are
identical across conditions.

No crop, flip, rotation, or color augmentation is allowed because it would
alter a controlled factor. Training uses ordinary empirical sampling without a
class-balanced sampler; class frequency is the treatment.

A balanced G0 pilot selects the smallest checkpoint budget \(S_*\) that, for
every active factor of all three classes, has normalized Wasserstein error at
most 0.05 and a 5--95% range ratio between 0.80 and 1.20. Object leakage and
off-renderer rate must each be at most 1%. Update budget is increased before
model capacity is changed; any capacity change and every failed pilot setting
is recorded. Only balanced data may be used for this tuning. After the pilot
passes, all hyperparameters are frozen before any long-tail result is
inspected.

Every condition is trained to the same \(S_*\) optimizer updates. It also saves
a matched-example-pass checkpoint. If

\[
E_* = S_* B / 15{,}000,
\]

where \(B\) is batch size, the matched-pass checkpoint for a dataset of size
\(N\) is saved at \(\lfloor E_*N/B\rfloor\) updates. This permits both
equal-compute and equal-per-example-exposure comparisons without training a
second model.

Single-class models, representation-sharing interventions, and long-tail
hyperparameter sweeps are excluded from the main matrix. They are targeted
follow-ups only if the first results justify them.

## 10. Measurement hierarchy

### 10.1 Independent factor oracle

A separate multi-task oracle is trained only on a large independent stream of
balanced renderer samples. It predicts object identity and all five potential
factor coordinates. Its held-out requirements are:

- object accuracy at least 99%;
- normalized mean absolute error at most 0.02 for every active scalar factor;
- circular azimuth error computed in sine-cosine form.

The predicted factors are re-rendered. The 99.5th percentile held-out
re-render residual defines the off-renderer threshold. Generated samples above
that threshold remain in the results and contribute to the off-renderer rate;
they are never silently discarded.

### 10.2 Primary distributional measurements

For 5,000 generated samples per class and checkpoint, report:

- normalized one-dimensional Wasserstein error for every active factor;
- generated-to-reference 5--95% factor-range ratio;
- joint factor energy distance, which detects missing combinations and
  artificial dependencies;
- object leakage and off-renderer rates.

Factor metrics are always displayed beside leakage and off-renderer rates so a
small valid subset cannot make a poor model appear to cover the manifold.

### 10.3 Local geometric measurements

On 256 deterministic held-out queries per class:

- compute the calibrated renderer tangent directions;
- estimate the leading effective output subspace and singular spectrum of the
  learned flow map;
- report per-factor tangent projection, principal angles, and contraction;
- estimate FM-FLIPD on held-out real renders and generated samples;
- report \(\Delta d=d_{\mathrm{data}}-d_{\mathrm{model}}\).

The learned flow-map spectrum is explicitly labeled an effective contraction
measure. No claim of literal rank loss is made from it alone.

### 10.4 Memorization and quality checks

Report nearest-training and nearest-held-out distances in factor space and in
oracle feature space, exact or near-duplicate rate, oracle-feature
precision/recall, and oracle-feature FID. FID-like metrics are secondary sanity
checks rather than primary evidence in this controlled low-resolution setting.

## 11. Analysis and falsification

The trained model is the unit of replication. Thousands of generated images
reduce measurement error but are not treated as thousands of independent
experiments.

Primary effects are estimated with paired object-dimension contrasts and the
fixed-effects model

\[
y \sim \log_{10}(n) * d + C(\mathrm{object}) + C(\mathrm{replicate}).
\]

The balanced-versus-long-tail context contrast is analyzed separately for
cells with 5,000 examples. A 95% percentile interval is computed with 10,000
paired hierarchical bootstrap draws: resample replicate bundles, then resample
the nine object-dimension cells within each selected bundle. Report effect
sizes, the three per-replicate values, and the pooled interval; do not rely on
a single aggregate significance value.

H1 is supported in this synthetic system only if the frequency-dimension
interaction has the predicted direction in all three replicate bundles and
the pooled paired interval excludes zero. Otherwise it is reported as
inconclusive or contrary, not rescued by a favorable subset.

H2 is supported only when distributional factor loss agrees with at least one
independent local-geometry probe and cannot be explained entirely by class
leakage or off-renderer failure. Evidence of exact copying without a geometric
deficit is reported as ordinary sample memorization, not geometric
memorization.

H3 remains exploratory regardless of its initial effect size. A causal claim
requires a follow-up that changes whether the same factor is available in
other classes while holding the target class fixed.

The principal falsifying outcomes are:

- no frequency-dimension interaction after object counterbalancing;
- factor loss without corroborating local geometric change;
- geometric estimates that fail the known full/half/collapsed controls;
- apparent coverage driven by class leakage or off-renderer samples;
- balanced high-dimensional failure, which indicates inadequate rendering,
  model capacity, or training rather than a long-tail effect.

## 12. Staged execution and gates

1. **Renderer and object calibration:** implement the three shapes, freeze
   appearance, verify separability, matched nuisance statistics, Jacobian rank,
   and factor visibility.
2. **Metric calibration:** verify factor oracle accuracy and the ordered
   full-range, half-range, and collapsed-factor controls.
3. **Training pilot:** train balanced G0, select \(S_*\), and verify recovery of
   all five high-dimensional factors. Freeze all settings.
4. **Pipeline smoke test:** execute one imbalanced mapping end-to-end, including
   manifests, checkpoints, metrics, and report generation. This result is used
   only to validate plumbing.
5. **Main matrix:** run all 12 conditions in each of three paired replicate
   bundles.
6. **Analysis:** generate the preregistered factor, geometry, memorization, and
   context contrasts and update the living report.
7. **Follow-up decision:** choose a sharing ablation, natural-image validation,
   or improvement experiment only after interpreting the main matrix.

A failed gate stops downstream execution. Reruns use a new immutable run
directory and record the reason for the deviation. Completed raw metrics and
manifests are never overwritten.

## 13. Implementation boundaries

The implementation should extend existing components rather than create a
second experimental framework:

- analytic objects and rendering under `fm_lab.geometry_explorer`;
- product factor spaces and pullback diagnostics already used by known-factor
  calibration;
- deterministic cyclic mappings and nested pools based on
  `fm_lab.data.long_tail`;
- existing class-conditional image flow-matching training and explicit
  checkpoint support;
- `FMJacobianSpectrumEstimator` and `FMFLIPDEstimator` for model-side geometry.

New code is divided into bounded units:

1. object definitions and calibrated material registry;
2. factor-ladder renderer and master-pool builder;
3. factorial condition manifest generator;
4. independent factor oracle and validation gates;
5. training orchestrator adapter;
6. distributional, local-geometry, and memorization evaluators;
7. deterministic result aggregation and report generation.

Each unit consumes and emits explicit manifests. Evaluation must be rerunnable
from saved checkpoints without regenerating training data.

## 14. Testing requirements

Unit tests cover:

- deterministic rendering and distinct object registrations;
- factor bounds, periodic coordinates, and inactive-coordinate invariance;
- exact G and F mappings;
- counts \((5{,}000,500,50)\), nested prefixes, and disjoint references;
- stable configuration hashes and seed replay;
- matched-pass checkpoint arithmetic;
- oracle circular losses and gate failures;
- correct ordering of full, half-range, and collapsed metric controls;
- invalid generated samples remaining visible in summaries.

Integration tests cover a tiny master pool, one short balanced training run,
checkpoint reload, evaluation, aggregation, and deterministic report output.
The full experiment is launched only after this end-to-end smoke test passes.

## 15. Living report contract

Implementation begins a living research report at
`docs/research/synthetic_long_tail_geometry_report.md`. It contains:

- question and literature context;
- frozen hypotheses and experimental design;
- renderer, oracle, and training calibration records;
- a run ledger with configuration hashes and status;
- automatically generated tables and figure links;
- deviations and failed gates;
- observations separated from interpretation;
- conclusions, limitations, and the next experimental decision.

The report is updated after every stage. Statements that preceded an
experiment are retained separately from post-result interpretation so later
observations cannot silently rewrite the hypothesis.

## 16. Relevant literature

- Cagri Eser et al., *Intrinsic Dimensionality as a Model-Free Measure of Class
  Imbalance*: <https://arxiv.org/abs/2511.10475>
- Ross et al., *A Geometric Framework for Understanding Memorization in
  Generative Models*: <https://arxiv.org/abs/2411.00113>
- Achilli et al., *Losing Dimensions: Geometric Memorization in Generative
  Diffusion*: <https://arxiv.org/abs/2410.08727>

These works motivate dimension as complementary to cardinality and distinguish
data geometry from learned geometry. The causal renderer design, nested factor
ladder, object-frequency counterbalancing, and shared/private direction test are
the operational contribution of this experiment.
