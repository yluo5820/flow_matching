# ImbDiff-CM mechanism and reconstruction research program

**Status:** implementation-ready research specification, drafted 2026-07-23.

This document turns the completed ImbDiff-CM reproduction and checkpoint probe
into a staged research program. Its purpose is not merely to improve the reported
FID. It asks whether the mechanism proposed for Capacity Manipulation (CM) is
actually realized by the trained network, which parts of that mechanism matter,
and whether the same effect can be obtained more systematically or cheaply.

The completed 60k reproduction and its limitations are documented in
[`docs/official_imbdiff_matrix.md`](../official_imbdiff_matrix.md). The first
checkpoint probe and its results are documented in
[`docs/research/imbdiff_cm_mechanism_probe.md`](imbdiff_cm_mechanism_probe.md).
Those results are prior evidence for this program, not conclusions that this
program is designed to reproduce.

## 1. Scientific scope

The original CM paper reports ablations in which the consistency term, diversity
term, and low-rank decomposition are all useful. Those ablations establish that
the released combination improves the evaluated generative score. They do not
by themselves establish:

1. what information is represented by the general and expert parameter groups;
2. how the different objective terms allocate gradients and optimizer updates;
3. whether low rank is essential, or whether any separately controlled capacity
   would work;
4. whether the released class-frequency coefficient is a uniquely meaningful
   allocation rule, a generic regularizer, or one point in a broader family;
5. whether the second network evaluation required by CM can be removed without
   losing its benefit.

The central target is therefore a causal account connecting:

```text
class frequency and difficulty
        -> signed CM coefficient
        -> gradients and optimizer updates
        -> learned expert/general subspaces
        -> semantic, geometric, and spectral content
        -> class-conditional generative behavior
```

The program explicitly allows a negative answer to the paper's proposed story.
An FID improvement can be real even if the learned decomposition is not a clean
general-versus-tail-expert split.

## 2. Established evidence and current boundary

The controlled 60k matrix reproduced the useful direction of the released CM
objective at reduced scale and verified that the vendored implementation works
inside this repository. The completed checkpoint probe found:

- CM strongly contracts the capacity-on/off prediction distance relative to an
  identical capacity-only architecture;
- the final local correction is not reliably larger or more useful for Few
  classes than for Many classes;
- the consistency and diversity gradients are nearly opposite because they
  multiply the same branch-distance gradient;
- under the empirical IR100 training distribution, the exposure-weighted
  positive head coefficient mass greatly exceeds the negative tail mass;
- checkpoint gradients do act directly on LoRA parameters, especially early,
  but the final total-objective expert gradient fraction is small and not
  preferentially larger for Few classes;
- the local output correction is not selectively high-frequency for tail
  classes in the preregistered Fourier probe.

These observations rule out a simple argument based only on final correction
magnitude. They do **not** rule out:

- a small expert correction that has a large cumulative rollout effect;
- class information encoded in intermediate expert activations rather than in
  final output magnitude;
- transient specialization earlier in training;
- tail knowledge represented by a low-dimensional direction that is small in
  norm but causally important;
- regularization of the shared branch as the dominant source of the FID gain;
- an interaction between the LoRA product parameterization and Adam dynamics.

The existing 60k matrix has one training seed. Held-out probe seeds quantify
probe-manifest sensitivity, not independent training uncertainty.

## 3. Exact mechanism to be studied

For a sample from class \(y\), the released CM trainer evaluates:

\[
h_1=f_{\theta_g+\theta_e}(x_t,t,y), \qquad
h_2=f_{\theta_g}(x_t,t,y),
\]

and defines one branch distance:

\[
D(x_t,t,y)=\|h_1-h_2\|_2^2.
\]

Let \(p_y\) be the empirical class probability and

\[
q_y=\frac{1/p_y}{\sum_c 1/p_c}.
\]

The released auxiliary term can be written as:

\[
L_{\mathrm{CM}}
=
\omega_{\mathrm{released}}(y)D,
\qquad
\omega_{\mathrm{released}}(y)
=
C\left(\lambda_{\mathrm{con}}p_y-\lambda_{\mathrm{div}}q_y\right).
\]

Thus \(L_{\mathrm{con}}\) and \(L_{\mathrm{div}}\) are not two distinct
feature-selecting functions. They are positive and negative class-dependent
weights on the same \(D\). The primary mechanistic object is the signed
coefficient \(\omega(y)\) acting through the shared branch-distance gradient.

For each adapted convolution:

\[
\Delta W_l=B_lA_l,\qquad
z_l^g=W_l^g*h_{l-1},\qquad
e_l=\Delta W_l*h_{l-1},\qquad
z_l^{\mathrm{full}}=z_l^g+e_l.
\]

The local response \(e_l\) is a well-defined expert-only signal. In contrast,
\(\theta_e\) alone is not a standalone U-Net: nonlinearities, normalization,
skip connections, and the shared feature hierarchy remain necessary. The
program therefore studies the expert at three levels:

1. effective expert kernels \(\Delta W_l\);
2. local expert responses \(e_l\);
3. expert-induced changes in the complete vector field and sampling trajectory.

## 4. Competing hypotheses

The experiments must distinguish the following explanations.

### H1 — general-plus-tail-expert decomposition

\(\theta_g\) contains shared visual knowledge and most head-class knowledge,
while \(\theta_e\) contains class-specific tail knowledge. Tail-aligned expert
directions should be semantically identifiable and causally necessary for tail
generation.

### H2 — shared representation with residual tail refinement

Most semantic structure is transferred through \(\theta_g\). The expert stores
only residual shape, texture, or high-frequency information needed by tail
classes. Its contribution should appear late in the sampling trajectory or at
high-resolution decoder layers.

### H3 — frequency-stratified separation without general knowledge

\(\theta_g\) primarily serves high-exposure classes and \(\theta_e\) primarily
serves low-exposure classes, but neither branch has the clean semantic meaning
claimed by the general/expert terminology. Cross-class transfer should be weak,
and head/tail subspaces should be more separable than semantically organized.

### H4 — generic consistency regularization

The main benefit comes from constraining two nearby parameterizations and
regularizing the shared network. Expert content need not be tail-specific.
A magnitude-matched class-independent positive coefficient should recover much
of the gain.

### H5 — optimization-parameterization effect

The low-rank product, zero initialization, Adam preconditioning, and signed
coefficient jointly alter the optimization trajectory even if no stable
tail-specific expert representation appears at the end. Early gradient and
effective-update dynamics should predict the eventual gain better than final
expert content.

### H6 — stochastic consistency effect

The released trainer evaluates \(h_1\) and \(h_2\) in separate forward passes.
With training dropout enabled, the distance contains both capacity difference
and different dropout realizations. Some of the benefit may therefore be
stochastic consistency regularization rather than capacity allocation.

No single scalar statistic is accepted as evidence for H1 or H2. Tail-specific
knowledge requires reproducible semantic or structural selectivity plus a
matched causal intervention.

## 5. Track K — what knowledge is learned by the expert?

The completed full/general output comparison is not repeated as the primary
experiment. The paper already contains a related ablation, and the existing
checkpoint probe shows that total correction magnitude is not explanatory.

### K1. Layerwise expert-response atlas

Record the local LoRA response

\[
e_l(x_t,t,y)=(B_lA_l)*h_{l-1}
\]

for every adapted convolution, using fixed held-out examples, noise, timesteps,
and checkpoint weights. Record both magnitude-sensitive and normalized
descriptors:

- response RMS and response/general-activation ratio;
- channel selectivity and spatial sparsity;
- channel-pooled and spatially pooled representations;
- low-pass, high-pass, and fixed wavelet-band energy;
- deterministic random sketches of the normalized response;
- layer, resolution, class, superclass, frequency group, and diffusion time.

Use disjoint train/test rows for cross-validated linear probes of:

- CIFAR-100 fine class;
- CIFAR-100 coarse superclass;
- Many/Medium/Few group;
- low-pass semantic features;
- high-pass or texture-sensitive features.

Required controls:

- label-permutation nulls preserving class multiplicity;
- matched random feature sketches;
- magnitude-normalized probes, so CM contraction cannot by itself determine the
  result;
- replication across checkpoints and at least two trained seeds before a
  representation claim.

Interpretation:

- frequency-group predictability without fine-class predictability supports
  frequency-stratified separation rather than class knowledge;
- fine- and coarse-class predictability in early or middle-resolution expert
  responses supports semantic tail knowledge;
- selectivity confined to high-resolution, late-time responses supports
  residual detail refinement.

### K2. Class-conditioned expert subspaces

For layer \(l\) and class \(y\), estimate a response covariance or low-rank
sketch:

\[
C_y^{(l)}
=
\mathbb E_{x,t\mid y}
\left[
\bar e_l\bar e_l^\top
\right],
\]

where \(\bar e_l\) is a normalized response representation. Compare:

- tail-tail, head-head, and head-tail projection overlap;
- principal angles between class subspaces;
- within-superclass versus across-superclass overlap;
- stability across checkpoints and training seeds;
- relationship between subspace overlap and class-frequency distance.

This yields a class transfer graph. Semantically related head-tail pairs sharing
expert or general subspaces support knowledge transfer. A split determined only
by class frequency supports capacity separation without a general-knowledge
interpretation.

### K3. Semantic-spectral sampling-trajectory decomposition

For fixed class labels and initial noise, record at selected sampling steps:

\[
\delta v_t
=
f_{\theta_g+\theta_e}(x_t,t,y)
-
f_{\theta_g}(x_t,t,y).
\]

Do not summarize it only by RMS or Fourier fraction. Project the direction onto:

- fixed wavelet or radial Fourier bands;
- fine-class and coarse-class classifier directions;
- low-pass shape-sensitive and high-pass texture-sensitive feature spaces;
- sampling-state displacement accumulated over early, middle, and late stages.

Predictions:

- early or middle semantic displacement indicates that the expert contributes
  object identity or coarse structure;
- late, high-resolution, high-frequency displacement with little semantic
  change indicates residual detail;
- no stable class-conditioned direction supports a regularization account.

### K4. Causal ablation of tail-aligned directions

Extract candidate expert directions from K1/K2 or class-conditioned gradient
covariance. Apply reversible, energy-matched interventions:

- remove top tail-aligned directions;
- remove head-aligned directions;
- remove deterministic random directions;
- rotate the expert subspace while preserving singular values;
- retain parameter count, effective rank, and perturbation norm.

Evaluate per-class FID, recall, precision, requested-class accuracy, and coarse
semantic accuracy. A tail-knowledge claim requires that removing tail-aligned
directions damages tail behavior more than matched random or head-aligned
removals, and that this result replicates across trained seeds.

### K5. Controlled head-to-tail transfer

Use the CIFAR-100 superclass hierarchy to select related and unrelated
head-tail pairs. Hold tail support fixed while changing the exposure of:

- semantically related head classes;
- frequency-matched but semantically unrelated head classes.

Measure changes in:

- tail generalization and generative recall;
- expert-response subspace;
- head-tail gradient alignment;
- dependence on tail-aligned expert directions.

This is the most direct test of knowledge borrowing. It is deferred until K1-K4
identify stable directions worth intervening on.

## 6. Track D — learning and optimizer dynamics

Checkpoint probing cannot recover the actual sequence of updates. This track
instruments a new training run while preserving the official objective.

### D1. Loss-term gradients on the actual data stream

At sparse, preregistered steps, expose the graph tensors for:

\[
L_{\mathrm{base}},\quad
L_{\mathrm{con}},\quad
L_{\mathrm{div}},\quad
L_{\mathrm{CM}},\quad
L_{\mathrm{total}}.
\]

Record their gradients with respect to:

\[
\theta_g,\quad \{A_l\},\quad \{B_l\}.
\]

Summarize by class, Many/Medium/Few group, layer, and diffusion-time stratum:

- raw norm and parameter-normalized RMS;
- pairwise cosine and conflict rate;
- expert/general gradient-energy fraction;
- exposure-weighted signed contribution;
- cumulative class-conditioned gradient path.

The diagnostic must use the same batch, noise, dropout draws, endpoint transfer,
and graph as the optimizer step. Replaying a fresh random objective is not
accepted.

### D2. Product-factorization dynamics

For \(\Delta W=BA\) and
\(G=\partial L/\partial\Delta W\):

\[
\nabla_A L=B^\top G,\qquad
\nabla_B L=GA^\top.
\]

The release initializes \(A\) randomly and \(B=0\), implying an asymmetric
initial phase:

\[
\nabla_A L=0,\qquad \nabla_B L\ne0.
\]

Keep the raw factor quantities because this asymmetry is part of the actual
optimization:

\[
\|A\|,\ \|B\|,\ \|\nabla_A L\|,\ \|\nabla_B L\|.
\]

Also report scale-stabilized sensitivities:

\[
\|A\|\|\nabla_A L\|,\qquad
\|B\|\|\nabla_B L\|,
\]

and the singular spectra and alignment of \(A\), \(B\), and \(BA\).

### D3. Effective expert update and optimizer action

Raw gradients are not optimizer updates. Snapshot selected parameters before
and after Adam steps and compute:

\[
\Delta W_e^{(s)}
=
B_{s+1}A_{s+1}-B_sA_s
\]

with decomposition:

\[
\Delta W_e
=
B\Delta A+\Delta B A+\Delta B\Delta A.
\]

Record:

- raw-gradient direction;
- Adam-preconditioned parameter update;
- effective \(\Delta W_e\);
- stepwise path length and net displacement;
- raw-model and EMA-model displacement;
- class-conditioned contribution to cumulative updates.

The effective update is invariant to merely relabeling the two factor norms,
while the raw factor dynamics expose optimization bias. Both are required.

### D4. LoRA tangent-space utilization

For selected layers, materialize or estimate the unconstrained convolutional
gradient \(G_l\) and its projection onto the tangent space of \(B_lA_l\):

\[
\rho_l
=
\frac{\|\Pi_{T_{B_lA_l}}G_l\|}{\|G_l\|}.
\]

Compare \(\rho_l\) for head and tail samples and for each loss term. This asks
whether low rank selectively admits tail gradients or merely restricts all
classes equally.

### D5. Dynamics-to-knowledge linkage

Save class-conditioned expert update subspaces at several training stages and
compare them with the final K1-K4 semantic and spectral directions. A proposed
mechanism is supported only if the directions receiving tail-specific updates
later become causally relevant tail-content directions.

## 7. Track C — is low rank essential?

All alternatives use the same CM coefficient and base objective. The initial
comparison matches:

- trainable expert parameter count;
- adapted layers and resolutions;
- initial zero functional perturbation;
- optimizer, learning-rate schedule, and update budget;
- as far as possible, forward FLOPs and initialization scale.

Parameter matching is necessary but not sufficient; compute and function-scale
differences are reported separately.

### C1. LoRA expert

The released \(BA\) factorization is the reference.

### C2. Fixed random-subspace expert

Use

\[
\Delta w=Pz
\]

with fixed deterministic \(P\) and trainable coordinates \(z\). The
implementation must use an implicit or structured orthogonal projection if a
dense \(P\) would dominate memory. This tests learned low-rank matrix structure
against generic low-dimensional residual capacity.

### C3. Sparse-coordinate expert

Use

\[
\Delta W=M\odot S
\]

with a fixed deterministic mask containing the matched number of trainable
coordinates. This is low-dimensional but generally not low-rank.

### C4. Channel-subset expert

Reserve a matched number of output channels or channel blocks for additive
expert updates. This gives a more explicit modular partition and permits direct
activation analysis.

Decision:

- similar behavior across alternatives implies that separately controlled
  capacity, not low rank, is the main ingredient;
- LoRA outperforming matched alternatives with stronger K/D evidence supports
  a specific low-rank mechanism;
- a channel expert outperforming LoRA supports explicit modular separation;
- performance without K/D specialization supports a regularization rather than
  knowledge-allocation account.

The official released implementation remains immutable as the reference.
Alternative capacity modules are added as sibling experimental models, not by
silently modifying the vendored baseline.

## 8. Track W — a general study of the class coefficient

Write the objective as:

\[
L=L_{\mathrm{base}}+\mathbb E_{y\sim p}
\left[\omega(y)D_y\right].
\]

Every schedule reports:

- \(\mathbb E_p[\omega]\);
- \(\mathbb E_p[|\omega|]\);
- per-group signed exposure mass;
- maximum per-sample magnitude and clipping rate;
- gradient RMS induced in general and expert parameters.

This prevents coefficient shape, global strength, and sampling exposure from
being conflated.

### W1. Magnitude-matched constant contraction

\[
\omega(y)=c>0.
\]

Choose \(c\) to match the released
\(\mathbb E_p[|\omega|]\). This isolates generic branch-consistency
regularization.

### W2. Centered released coefficient

\[
\omega_{\mathrm{center}}(y)
=
\omega_{\mathrm{released}}(y)
-
\mathbb E_p[\omega_{\mathrm{released}}].
\]

Rescale it to match the released expected absolute magnitude. This preserves the
released class ordering while removing its global contraction component.

### W3. Target-distribution coefficient

Let \(q(y)\) denote the desired allocation distribution and define:

\[
\boxed{
\omega_q(y)
=
\lambda\left(1-\frac{q(y)}{p(y)}\right)
}
\]

so that:

\[
\mathbb E_{p}[\omega_q]=0.
\]

Use the interpretable family:

\[
q_\tau(y)
=
\frac{p_y^\tau}{\sum_c p_c^\tau},
\qquad 0\le\tau\le1.
\]

- \(\tau=1\): no redistribution;
- \(\tau=0\): uniform target allocation;
- intermediate \(\tau\): partial balancing.

Clip extreme per-sample coefficients only as a declared variance-control
operation and report its effect. Match \(\mathbb E_p[|\omega|]\) when comparing
against the release.

### W4. Difficulty-aware target allocation

Introduce difficulty through the target distribution, rather than appending an
uninterpretable term to the released coefficient:

\[
q_{\tau,\kappa}(y)
\propto
p_y^\tau\exp(\kappa d_y).
\]

Candidate preregistered \(d_y\) values are:

- held-out denoising generalization gap;
- class-conditioned gradient variance or conflict;
- stable intrinsic-dimension or anisotropy descriptor;
- learning progress or loss plateau;
- effective sample support.

Intrinsic dimension is one hypothesis, not a privileged ground truth. Each
difficulty estimate must be computed on a frozen split and tested for stability
before it enters the schedule.

### W5. Adaptive or meta-learned coefficient

Only after W1-W4, parameterize a monotone spline or small network:

\[
\omega_\phi(y,s,t)
=
h_\phi(\log n_y,d_y,L_y(s),t)
\]

and optimize it against a balanced validation objective. This is more flexible
but more expensive and less interpretable, so it is not the first intervention.

### Minimal first coefficient matrix

The first screen contains:

1. released CM;
2. magnitude-matched constant positive coefficient;
3. magnitude-matched centered released coefficient;
4. target-distribution coefficient with \(\tau=0\);
5. target-distribution coefficient with \(\tau=0.5\).

This is sufficient to distinguish global contraction, released class ordering,
zero-mean allocation, and balancing strength before adding difficulty.

## 9. Track E — reducing training cost

The released CM trainer performs two complete U-Net evaluations and
backpropagates through both. The low-rank parameter overhead is small; the
duplicated network evaluation dominates the observed near-twofold cost.

Efficiency methods are separated by whether they preserve the exact objective.

### E0. Profile and account

Before optimization, record on a fixed GPU:

- data-loading, forward, backward, optimizer, and sampling time;
- full versus general forward time;
- peak allocated and reserved memory;
- images or updates per second;
- cost by encoder, middle, and decoder when profiling permits.

Report wall time and quality together. A speedup without a matched FID and
per-group comparison is not accepted.

### E1. Shared encoder, dual decoder

The released configuration adapts the upsampling path. Head, downsampling path,
middle path, and skip features are structurally identical for full and general
branches. Compute them once, then evaluate two decoder branches.

This requires an explicit dropout contract. Shared-prefix activations imply a
shared dropout realization in the common path and therefore do not reproduce
the exact stochastic released objective. Compare both objectives before using
this as the new default.

### E2. Fused dual decoder

Represent full/general as a branch or concatenated-batch dimension to improve
GPU utilization. FLOPs remain similar in the dual decoder, but wall time may
decrease. Combine this with E1.

### E3. Stratified auxiliary-batch subsampling

Compute the base objective on the complete batch and compute \(D\) on a
class-stratified subset. Correct for inclusion probability so the auxiliary
gradient remains unbiased. Screen 100%, 50%, and 25% auxiliary fractions.

### E4. Temporal auxiliary subsampling

Compute CM only with probability \(\rho\):

\[
\widehat L
=
L_{\mathrm{base}}
+
\frac{I_s}{\rho}L_{\mathrm{CM}},
\qquad
I_s\sim\mathrm{Bernoulli}(\rho).
\]

The expected auxiliary gradient is unchanged, although variance and the Adam
trajectory change. Screen \(\rho=1,0.5,0.25\).

### E5. Local expert-response surrogate

Approximate the global branch distance with:

\[
D_{\mathrm{local}}
=
\sum_l a_l
\left\|
(B_lA_l)*h_{l-1}
\right\|^2.
\]

This removes the second U-Net evaluation but changes the objective. Calibrate
the layer weights against occasional exact \(D\) evaluations and treat it as a
new method, not an implementation optimization.

Do not prioritize cached general predictions, because \(x_t\), noise, and time
change every step. Do not silently stop gradients through the general branch;
that changes the allocation mechanism.

## 10. Dropout confound and paired-forward contract

The current paper-close configuration uses dropout \(0.1\). The released
trainer calls the full and general models sequentially without replaying the
random-number-generator state. During training:

\[
D_{\mathrm{released}}
=
\left\|
f_{\theta_g+\theta_e}^{m_1}
-
f_{\theta_g}^{m_2}
\right\|^2,
\]

where \(m_1\) and \(m_2\) are generally different dropout masks. A cleaner
capacity distance would use:

\[
D_{\mathrm{paired}}
=
\left\|
f_{\theta_g+\theta_e}^{m}
-
f_{\theta_g}^{m}
\right\|^2.
\]

The first new diagnostic compares:

1. independent training-mode masks, faithful to the release;
2. paired training-mode masks;
3. evaluation mode with dropout disabled.

Use a checkpoint and fixed batch to decompose:

- distance attributable to dropout alone when \(\Delta W=0\);
- distance attributable to the expert under paired masks;
- interaction between dropout and expert;
- gradient variance and cosine under the three modes.

If independent-mask distance dominates, the stochastic-consistency mechanism
becomes a separate explanatory track and E1 must not be described as an exact
optimization of the released objective.

## 11. Implementation architecture

### 11.1 Preserve the official reference

The vendored release and existing `official_imbdiff_*` methods remain the
faithful baseline. New objectives and capacity modules receive new config names
and metadata. Every checkpoint records:

- implementation family;
- coefficient schedule and normalization;
- capacity decomposition;
- dropout pairing mode;
- auxiliary sampling rate;
- trainable parameter count and measured compute.

### 11.2 Structured objective terms

Refactor the official wrapper, without changing its scalar output, so one
forward computation can optionally expose graph-connected tensors:

```text
base_per_sample
distance_per_sample
coefficient_per_sample
consistency_per_sample
diversity_per_sample
total_per_sample
full_prediction
general_prediction
```

Tests must show scalar and gradient equality with the current released path for
fixed random draws. Diagnostics must never rerun the objective with new noise or
dropout when claiming to describe an optimizer step.

### 11.3 Sparse training observer

Add an optional training observer that is inactive by default and runs only at
configured steps. Its contract needs three moments:

1. graph inspection before ordinary backward;
2. parameter and gradient inspection after backward;
3. parameter and optimizer-state inspection after `optimizer.step`.

Snapshot only expert parameters and preregistered representative general layers
to control memory. Write diagnostics incrementally so a later failure does not
discard completed steps.

### 11.4 Expert-response hooks

Add official-CM-specific hooks for every active `Conv2d_LoRA`. Given the input
activation, compute the local expert convolution with \(B_lA_l\) and no base
weight or bias. Store compact online summaries and deterministic sketches rather
than every dense activation.

The hook layer must verify numerically that:

\[
\mathrm{Conv}(h,W+BA)-\mathrm{Conv}(h,W)
=
\mathrm{Conv}(h,BA)
\]

within the active dtype tolerance.

### 11.5 Reversible expert interventions

Implement bit-exact restoration for:

- singular-direction removal;
- subspace rotation;
- matched-random removal.

Prefer functional state overrides or a factor reconstruction whose effective
\(BA\) is verified before sampling. Never mutate the stored checkpoint.
Deterministic perturbation manifests record direction digests, layer names,
rank, energy, and random seeds.

### 11.6 Coefficient schedule interface

Introduce a schedule object that maps class metadata to a coefficient vector.
The interface returns both coefficients and audit metadata. The released
schedule must reproduce the current vector exactly. Centering, target-ratio,
normalization, clipping, and difficulty inputs are explicit transformations,
not hidden config behavior.

### 11.7 Capacity module interface

Create sibling experimental capacity convolutions with a common:

```text
forward(input, use_capacity)
effective_expert_weight()
capacity_metadata()
```

contract. Before training, write an audit table containing parameter count,
adapted layers, initial output perturbation, rank or sparsity, and measured
forward cost. The official LoRA implementation remains the comparison anchor.

### 11.8 Code ownership and reuse

Keep the implementation split by scientific responsibility:

| Area | Existing foundation | Planned owner |
| --- | --- | --- |
| faithful objective and model | `fm_lab/integrations/official_imbdiff_cm.py` | structured terms, paired-dropout contract, released coefficient compatibility |
| completed checkpoint probe | `fm_lab/diagnostics/imbdiff_cm_probe.py` | shared manifest/checkpoint restoration only; do not overload it with all later analyses |
| knowledge analysis | paired manifest, Fourier summary | new `fm_lab/diagnostics/imbdiff_cm_knowledge.py` and `run_imbdiff_cm_knowledge_probe.py` |
| training dynamics | trainer loop and checkpoint schema | new `fm_lab/diagnostics/imbdiff_cm_dynamics.py` plus a minimal optional trainer observer |
| coefficient families | released class counts and weights | new `fm_lab/training/imbdiff_cm_coefficients.py` |
| capacity alternatives | official `Conv2d_LoRA` API | new experimental capacity modules outside `third_party/` |
| causal directions | `diagnostics/probes/subspaces.py`, `controls.py`, and `perturbations.py` | CM-specific effective-\(BA\) adapters and sampling CLI |
| evaluation | existing ImbDiff feature cache and classwise metrics | reuse without changing metric definitions |

Add one focused test module per new diagnostic or mechanism. Keep exact-release
compatibility tests in `tests/test_official_imbdiff_cm.py`; keep scientific
summary tests separate from model algebra tests. No new experimental method may
reuse the unqualified `released_cm` method name.

The implementation dependency graph is:

```text
structured CM terms
    -> paired-dropout diagnostic
    -> sparse training observer
    -> dynamics probe

expert-response hooks
    -> response atlas and subspaces
    -> trajectory decomposition
    -> causal expert interventions

coefficient interface
    -> coefficient audit
    -> minimal schedule matrix
    -> difficulty-aware and adaptive schedules

capacity interface
    -> matched-capacity audit
    -> alternative decomposition matrix

profiling
    -> stochastic auxiliary subsampling
    -> shared/fused branch implementation
    -> local surrogate
```

## 12. Phased implementation plan

### Phase 0 — freeze schemas and baselines

Deliverables:

- this research specification;
- a machine-readable experiment registry for method names and dependencies;
- baseline checkpoint and evaluation manifests;
- output schemas for knowledge, dynamics, coefficient, and efficiency probes.

Validation:

- documentation links resolve;
- current 60k metrics and probe outputs remain unchanged;
- no training code changes.

Compute: none.

### Phase 1 — paired-dropout diagnostic

**Implementation status (2026-07-23):** structured graph-connected CM terms,
the faithful independent-mask path, paired/disabled controls, a checkpoint CLI,
and unit controls are implemented. The real 60k checkpoint diagnostic remains
to be run before Phase 2 begins.

Implementation:

- add deterministic RNG replay for the two CM forwards;
- expose independent, paired, and disabled dropout modes to a diagnostic CLI;
- add zero-expert and planted-expert unit controls.

Validation:

- paired masks and \(\Delta W=0\) give zero branch distance;
- independent masks and dropout \(>0\) produce the expected stochastic distance;
- evaluation mode matches the existing checkpoint probe;
- no stored checkpoint is modified.

Experiment:

- one released-CM checkpoint;
- Many/Medium/Few held-out rows;
- early, middle, and late diffusion times;
- at least 20 independent dropout-mask pairs.

Decision:

- if dropout accounts for a material fraction of training-mode \(D\) or changes
  gradient direction substantially, carry paired and independent variants into
  the dynamics screen.

Compute tier: local or short server diagnostic, expected below 30 minutes.

### Phase 2 — expert knowledge probe

Implementation:

- expert-response hooks and deterministic sketches;
- CIFAR-100 fine/coarse labels in the held-out manifest;
- cross-validated probes and permutation controls;
- class-conditioned subspace summaries;
- intermediate sampling trajectory export;
- reversible direction interventions.

Validation:

- local expert response reconstructs the full-minus-general preactivation;
- planted semantic and spectral controls are recovered;
- label permutations return chance-level probe performance;
- interventions restore parameters bit-exactly;
- identical manifests reproduce identical artifacts.

Experiment order:

1. K1/K2 on existing 20k, 40k, and 60k released-CM checkpoints;
2. K3 on a small fixed class/noise panel;
3. K4 only for directions stable in K1/K2;
4. repeat primary findings on a second trained seed before claiming knowledge
   decomposition.

Compute tier:

- K1/K2 checkpoint analysis should be kept below 30 minutes per command;
- trajectory generation and causal FID runs are server/user jobs if projected
  above 30 minutes.

### Phase 3 — training-dynamics instrumentation

Implementation:

- graph-connected structured objective terms;
- sparse training observer;
- raw \(A/B\) gradient and norm logging;
- scale-stabilized sensitivity logging;
- pre/post-Adam snapshots and effective \(\Delta W_e\);
- selected-layer tangent-space utilization;
- incremental NPZ/CSV/JSON output.

Validation:

- instrumented-off training is bitwise or tolerance-equivalent to the current
  implementation;
- instrumented steps preserve the ordinary optimizer update;
- finite-difference checks validate selected raw gradients;
- effective update reconstruction matches direct \(BA\) difference;
- synthetic low-rank controls recover known tangent projections.

Experiment:

- a short 2k dynamics smoke run;
- one 30k mechanism run with logs dense in the first 1k steps and sparse later;
- only after inspection, a 60k confirmation or second seed.

Compute tier:

- unit and 2k smoke tests local/short server;
- 30k and 60k runs are handed to the user on the CUDA server.

### Phase 4 — coefficient mechanism screen

Implementation:

- schedule interface and audit metadata;
- released, constant, centered, and target-ratio schedules;
- magnitude matching and clipping reports;
- config generator for the minimal matrix.

Validation:

- released schedule exactly matches current coefficients and loss;
- centered and target-ratio schedules satisfy
  \(\mathbb E_p[\omega]\approx0\);
- matched schedules satisfy the declared
  \(\mathbb E_p[|\omega|]\);
- coefficient plots and exposure-weighted masses are generated before training.

Screen:

| Schedule | Purpose |
| --- | --- |
| released | faithful reference |
| constant positive | generic consistency control |
| centered released | remove global contraction |
| target ratio, \(\tau=0\) | uniform target allocation |
| target ratio, \(\tau=0.5\) | partial balancing |

Use one 30k seed for screening. Promote no more than two alternatives to a 60k,
multi-seed confirmation. Evaluate macro and per-frequency-group endpoints, not
FID alone.

Compute tier: all full screens are CUDA server/user jobs.

### Phase 5 — matched-capacity decomposition

Implementation order:

1. sparse-coordinate expert;
2. channel-subset expert;
3. fixed random-subspace expert after an implicit-projection feasibility test.

Validation:

- parameter and layer audits;
- zero initial functional perturbation;
- isolated capacity-on/off unit tests;
- gradient reaches only intended capacity coordinates;
- measured compute and memory are reported.

Screen:

- released LoRA;
- sparse-coordinate;
- channel-subset;
- fixed random subspace.

Use the best supported coefficient from Phase 4 and retain the released
coefficient as a secondary reference if the winner differs. A decomposition
claim requires both quality and K/D mechanism evidence.

Compute tier: smoke tests local/short server; 30k and 60k matrices are
CUDA server/user jobs.

### Phase 6 — efficiency frontier

Implementation order:

1. stratified auxiliary-batch subsampling;
2. temporal auxiliary subsampling;
3. shared encoder with dual/fused decoder;
4. local expert-response surrogate.

Validation:

- inclusion-probability correction gives an unbiased Monte Carlo auxiliary
  gradient in a fixed-batch test;
- shared-prefix outputs match the declared paired-dropout objective;
- profiler output is reproducible on the same device;
- no method is labeled exact when it changes the stochastic objective.

Screen:

- auxiliary batch fraction \(1, 0.5, 0.25\);
- temporal rate \(1, 0.5, 0.25\);
- selected combined settings only after individual effects are known.

Report a Pareto frontier of wall time, peak memory, FID, macro classwise FID,
Few-class recall, and the mechanism markers established in Phases 2-3.

Compute tier: profiling and short screens on the same CUDA device; promoted
30k/60k runs are user jobs.

### Phase 7 — difficulty-aware allocation and transfer

Only after the frequency mechanism is understood:

- freeze stable difficulty estimators;
- test \(q_{\tau,\kappa}\);
- compare frequency-only, difficulty-only, and combined targets;
- run the controlled related-head versus unrelated-head transfer study;
- consider meta-learned \(\omega_\phi\) only if fixed families remain
  systematically inadequate.

This phase reconnects CM with the earlier geometry program without treating
intrinsic dimension as a sufficient scalar explanation.

## 13. Experiment and compute policy

Use three compute tiers:

### Tier A — correctness

- CPU or local-device unit tests;
- tiny official U-Net;
- 1-10 batches;
- expected below 5 minutes.

### Tier B — mechanism smoke

- existing checkpoint probes or at most 2k training steps;
- fixed small manifests;
- expected below 30 minutes;
- may be run inline.

### Tier C — scientific outcome

- 30k or 60k training;
- generative evaluation and causal sampling;
- multiple trained seeds where a claim depends on training randomness;
- handed to the user when projected above 30 minutes.

Do not launch a full matrix before:

1. exact algebra/unit tests pass;
2. a Tier-B smoke produces finite, interpretable artifacts;
3. expected outcomes and rejection criteria are written down;
4. the coefficient, parameter, compute, and data contracts are audited.

## 14. Output contract

Every probe or experiment writes:

- resolved config and git commit;
- checkpoint provenance and weight type;
- dataset split and manifest digest;
- device, dtype, mixed precision, and channels-last settings;
- coefficient vector and normalization audit;
- capacity parameter/layer audit;
- structured row-level or class-level CSV;
- compact JSON summary;
- NPZ or safetensors only for arrays too large for CSV;
- a generated Markdown report containing interpretation boundaries.

Training-dynamics artifacts are incremental and checkpointed. Dense activation
maps are not committed; summaries, schemas, manifests, and selected figures are.

Server-only large artifacts remain under the run directory. A compact report and
artifact manifest are committed for cross-device continuity.

## 15. Decision rules

### Knowledge decomposition

Support for tail-specific expert knowledge requires all of:

1. cross-validated class or semantic information in expert responses beyond
   permutation controls;
2. stronger or qualitatively different structure for tail classes;
3. stability across checkpoints and trained seeds;
4. a direction-specific causal ablation that selectively damages tail behavior.

Magnitude, Fourier fraction, or full/general FID alone is insufficient.

### Gradient routing

Support for the proposed routing mechanism requires:

1. tail-specific effective expert updates on the actual training stream;
2. exposure-weighted cumulative effects, not only equal-class held-out probes;
3. alignment between those update directions and final tail-content directions;
4. a distinction from a magnitude-matched generic consistency control.

### Low-rank necessity

Low rank is considered necessary only if LoRA outperforms matched alternatives
and exhibits mechanism evidence not shared by them. A rank sweep alone is not
sufficient because rank simultaneously changes parameter count, expressivity,
initialization, and optimization geometry.

### Coefficient meaning

The released class schedule is considered mechanistically meaningful only if it
outperforms a magnitude-matched constant coefficient and its effect persists
after global mean and exposure are controlled.

### Efficiency

An efficiency method is promoted only if it improves wall time or memory on the
same hardware while preserving the relevant quality and mechanism endpoints.
Approximate objectives are labeled as new methods, not engineering-equivalent
implementations.

## 16. Immediate next implementation slice

The first code change should be deliberately narrow:

1. add graph-preserving structured CM terms without changing the scalar loss;
2. add paired versus independent dropout control;
3. add a diagnostic CLI and exact unit tests;
4. run the checkpoint-level dropout decomposition;
5. only then implement the expert-response hook and knowledge probe.

This order resolves a potentially important confound before building the more
expensive interpretation stack and creates the structured loss interface needed
by the later training observer.

## References

- Capacity Manipulation for Imbalanced Image Generation:
  <https://openreview.net/forum?id=wSGle6ag5I>
- Authors' ImbDiff-CM release:
  <https://github.com/Feng-Hong/ImbDiff-CM>
- LoRA: Low-Rank Adaptation of Large Language Models:
  <https://arxiv.org/abs/2106.09685>
- Network Dissection: Quantifying Interpretability of Deep Visual
  Representations:
  <https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html>
- Testing with Concept Activation Vectors:
  <https://proceedings.mlr.press/v80/kim18d.html>
- Class-Balanced Loss Based on Effective Number of Samples:
  <https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html>
- Meta-Weight-Net:
  <https://proceedings.neurips.cc/paper/2019/hash/e58cc5ca94270acaceed13bc82dfedf7-Abstract.html>
