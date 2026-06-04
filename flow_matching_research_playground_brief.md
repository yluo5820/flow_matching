# Flow Matching Research Playground: Ambiguity, Solver Sensitivity, and Latent Geometry

**Status:** implementation brief for a research agent  
**Project stage:** exploratory research infrastructure, not final paper implementation  
**Date:** 2026-06-04  
**Primary goal:** build a small-scale, highly controllable laboratory for flow matching / rectified flow experiments that can test whether a paper-worthy structural phenomenon exists.

---

## 0. Executive Summary

We want to start a research project on diffusion models, rectified flow, and flow matching that can eventually become a paper. The immediate goal is **not** to train a frontier-scale image model, nor to simply implement another flow matching variant. The immediate goal is to build a custom research playground that gives us fine-grained control over:

1. probability paths,
2. source-target couplings,
3. hidden endpoint/path variables,
4. velocity targets,
5. learned deterministic vector fields,
6. numerical solvers,
7. path/field/solver diagnostics,
8. latent or manifold-like geometric structure.

The central research hypothesis is:

> Flow matching becomes difficult when a chosen stochastic or conditional path is poorly compressible into a deterministic, numerically stable Markovian vector field.

This failure may appear as:

- **velocity ambiguity:** several incompatible hidden source-target/path velocities are plausible at the same spacetime point \((x,t)\);
- **solver sensitivity:** the learned velocity field produces noticeably different generated distributions under different black-box samplers or step schedules;
- **geometry mismatch:** the path ignores the latent geometry of the data representation, causing off-manifold excursions, high conditional uncertainty, or unnecessary curvature/stiffness.

The playground should let us test whether these three phenomena are genuinely linked.

---

## 1. Conceptual Background

### 1.1 Flow matching viewpoint

In flow matching / conditional flow matching, one usually defines a path between a source sample \(x_0 \sim p_0\) and a data sample \(x_1 \sim p_1\), samples an intermediate point \(x_t\), and trains a neural vector field \(v_\theta(x,t)\) to match a target velocity \(u_t\).

For a random path process,

\[
X_t = \psi_t(Z),
\]

where \(Z\) may contain source point, target point, coupling variable, class condition, posterior completion, or other hidden path information, define

\[
U_t = \dot X_t.
\]

The Bayes-optimal deterministic FM vector field is

\[
v^\star(x,t)=\mathbb{E}[U_t \mid X_t=x].
\]

This is important because the learned field is not the full random path law. It is a deterministic projection of that law onto spacetime.

### 1.2 Velocity ambiguity

The conditional law

\[
\mathcal{L}(U_t \mid X_t=x)
\]

may be broad or multi-modal. If so, a deterministic MSE-trained vector field must average incompatible velocity directions.

A basic ambiguity diagnostic is

\[
\mathcal A(t)
=
\mathbb{E}\left[\operatorname{Tr}\operatorname{Cov}(U_t\mid X_t)\right].
\]

But we should not stop at this metric. The research goal is to test whether this ambiguity has structural consequences:

- irreducible FM regression error,
- field curvature,
- stiffness,
- endpoint uncertainty,
- poor few-step sampling,
- high sensitivity to solver choice.

### 1.3 Solver sensitivity

A trained generative flow is not only the exact ODE

\[
\frac{d x_t}{dt} = v_\theta(x_t,t).
\]

In practice, generation is a **black-box numerical procedure**:

\[
x_1 \approx S[v_\theta](x_0),
\]

where \(S\) may be Euler, Heun, midpoint, RK4, Dormand-Prince, a stabilized solver, or a custom sampler with a chosen timestep schedule.

The project should **not** train specifically for one fixed solver. Instead, treat the solver as a black-box member of a sampler class and measure how robust a learned flow is across solver choices.

A possible solver-sensitivity functional is:

\[
\Delta_{\mathfrak S,K}(v)
=
\sup_{S_1,S_2\in\mathfrak S_K}
D\left((S_1[v])_\#p_0,\ (S_2[v])_\#p_0\right),
\]

where:

- \(\mathfrak S_K\) is a class of solvers/schedules with approximately \(K\) neural function evaluations;
- \(D\) is MMD, sliced Wasserstein distance, Wasserstein distance, FID proxy, precision/recall, or another distributional discrepancy.

The important question is not “which solver wins?” but:

> Do some paths/couplings/geometries produce vector fields that are inherently more stable across a family of solvers?

### 1.4 Geometry mismatch

Explicit Riemannian flow matching exists, but our target is broader and more practical. We are not primarily interested in doing flow matching on known manifolds such as \(SO(3)\) unless the application demands it.

Instead, we care about latent or representation geometries such as:

- data concentrated near a low-dimensional manifold;
- data concentrated near a spherical shell;
- radial/angular decomposition in autoencoder latents;
- tangent/normal decomposition near a data manifold;
- semantic/nuisance anisotropy;
- frequency-domain anisotropy;
- low-pass/high-pass structure in image-like data.

The working hypothesis is:

> A path that ignores the dominant geometry of the representation space may increase conditional velocity ambiguity and solver sensitivity.

---

## 2. Nearby Prior Art and Overlap Risks

The agent should treat the following as **must-review prior art** before making novelty claims.

### 2.1 Core references

- Flow Matching for Generative Modeling
- Conditional Flow Matching / TorchCFM
- Rectified Flow
- DPM-Solver / DPM-Solver++
- EDM design-space analysis
- Stochastic Interpolants
- Riemannian Flow Matching
- Discrete Flow Matching

### 2.2 Relevant implementation libraries

Use these as references or optional components, but do not make the playground a thin wrapper around them.

- **TorchCFM / Conditional Flow Matching**  
  GitHub: `atong01/conditional-flow-matching`  
  Useful for CFM/OT-CFM baselines and implementation conventions.

- **Meta `flow_matching`**  
  GitHub: `facebookresearch/flow_matching`  
  Useful for modern continuous and discrete FM abstractions, image/text examples, and library structure.

### 2.3 Direct overlap with velocity ambiguity

Several recent works already discuss or exploit multi-modal / ambiguous velocities. The project must not merely rediscover this.

Relevant works include:

- **Variational Rectified Flow Matching**  
  Directly models multi-modal velocity vector fields and argues that MSE-trained rectified/FM fields average ambiguous directions.

- **Hierarchical Rectified Flow**  
  Models not only location but also velocity, acceleration, and higher-order structure through a hierarchy of ODEs.

- **Stable Velocity**  
  Studies flow matching from a velocity-target variance perspective.

- **Posterior-Augmented Flow Matching**  
  Replaces sparse single-target supervision by posterior-augmented target expectations.

- **Condition-dependent Source Flow Matching**  
  Studies source design and intrinsic variance terms in FM.

- **Flows Don’t Cross in High Dimension**  
  Warns against interpreting the problem too literally as geometric line crossing in high dimension.

Therefore, the playground should test a stronger thesis:

> The important object is not literal path crossing but conditional-law non-identifiability: how much hidden path information is lost when projecting a stochastic path process onto a deterministic Markovian vector field.

### 2.4 Direct overlap with solver-aware sampling

Existing work already covers many algorithmic solver directions.

Relevant works include:

- DPM-Solver and DPM-Solver++: exploit special semi-linear diffusion ODE structure.
- STORK: structure-independent stabilized solver for diffusion and flow models.
- From Euler to Dormand-Prince: ODE solver benchmarking for flow matching.
- SharpEuler / Sharpness-Aware Sampling: training-free timestep schedule calibration for FM.
- Bespoke solvers for generative flow models.
- Shortcut models / consistency-style methods.
- Curvature minimization for ODE generative models.
- Optimal acceleration transport / Lagrangian flow matching.

Therefore, the playground should **not** focus on “train for Euler” or “add a curvature penalty” as the main idea. Instead, it should measure solver sensitivity generically and ask whether ambiguity or geometry predicts it.

### 2.5 Direct overlap with geometric / latent FM

Explicit manifold-aware and latent-geometry-aware FM already exists.

Relevant works include:

- Riemannian Flow Matching.
- Riemannian Variational Flow Matching.
- Pullback Flow Matching.
- Flow Matching is Adaptive to Manifold Structures.
- Geometry-Aware Image Flow Matching.
- Aligning Latent Geometry for Spherical Flow Matching in Image Generation.
- Low-Pass Flow Matching.

Therefore, the playground should avoid the weak claim “manifold structure matters.” The stronger question is:

> Does geometry mismatch create measurable conditional ambiguity and solver sensitivity, even when the model can in principle learn the target distribution?

---

## 3. Research Questions to Verify

The playground should be designed to answer the following questions.

### Q1. Does conditional velocity ambiguity predict learned-field difficulty?

For different paths and couplings, measure:

- conditional velocity covariance;
- multi-modality of conditional velocity;
- Bayes regression error;
- training loss floor;
- learned vector-field smoothness;
- field curvature;
- endpoint uncertainty.

Question:

> Does high ambiguity consistently correspond to harder deterministic FM learning?

### Q2. Does ambiguity predict solver sensitivity?

For the same trained model, sample using several black-box solvers and schedules.

Question:

> Do low-ambiguity paths/couplings produce generated distributions that are more stable across solvers at fixed NFE?

### Q3. Does geometry-aware path design reduce ambiguity?

Compare Euclidean linear interpolation against geometry-aware alternatives.

Question:

> Does respecting latent geometry reduce conditional velocity ambiguity, off-manifold excursion, curvature, or solver sensitivity?

### Q4. Is literal path crossing the wrong primitive?

In high dimensions, exact path intersections may be rare. But conditional ambiguity can still occur if many hidden endpoints/path variables are plausible near the same intermediate region.

Question:

> Can conditional-law overlap explain the same failures that “path crossing” was meant to explain, while remaining meaningful in high dimensions?

### Q5. Are ambiguity, curvature, and solver sensitivity separable?

It may turn out that:

- ambiguity predicts training difficulty but not solver sensitivity;
- curvature predicts solver sensitivity but not ambiguity;
- geometry mismatch predicts neither.

The lab should be designed to falsify the unifying thesis, not only confirm it.

---

## 4. Playground Design Principles

### 4.1 Fine-grained control

The code must make these objects independently swappable:

- dataset / target distribution;
- source distribution;
- source-target coupling;
- interpolation path;
- target velocity;
- model architecture;
- training objective;
- solver;
- timestep schedule;
- diagnostic suite.

### 4.2 Diagnostics are first-class citizens

This is not just a model-training library. The diagnostic pipeline is the core research instrument.

Every experiment should save:

- configuration;
- trained checkpoint;
- generated samples;
- intermediate trajectories;
- target velocities;
- learned velocities;
- diagnostic statistics;
- plots;
- raw arrays needed for later analysis.

### 4.3 Small-scale before large-scale

Start with 2D and low-dimensional controlled distributions where the true path law can be sampled densely and inspected. Only later move to MNIST/CIFAR/latent image settings.

### 4.4 Solver is black-box

The solver interface should expose only:

```python
solve(v_fn, x0, t_grid, **kwargs) -> trajectory_or_final_state
```

Diagnostics may compare many solvers, but training should initially remain solver-agnostic unless a later experiment explicitly tests solver-aware variants.

### 4.5 Avoid premature method invention

The first milestone is not a new algorithm. The first milestone is to establish whether the structural chain is real:

```text
path / coupling / source choice
        ↓
conditional velocity law
        ↓
deterministic representability
        ↓
curvature / stiffness / solver sensitivity
        ↓
few-step generation quality
```

---

## 5. Proposed Software Architecture

Recommended package name:

```text
fm_lab
```

Suggested repository structure:

```text
fm_lab/
  README.md
  pyproject.toml
  configs/
    toy/
    image/
    latent/
  fm_lab/
    data/
      base.py
      toy_2d.py
      manifold_toys.py
      image.py
      latent.py
    sources/
      base.py
      gaussian.py
      spherical.py
      lowpass.py
      learned.py
    couplings/
      base.py
      independent.py
      minibatch_ot.py
      sorted_1d.py
      class_conditional.py
    paths/
      base.py
      linear.py
      rectified.py
      spherical.py
      tangent_normal.py
      lowpass.py
    models/
      mlp.py
      unet_small.py
      dit_tiny.py
      wrappers.py
    training/
      losses.py
      trainer.py
      callbacks.py
    solvers/
      base.py
      euler.py
      heun.py
      midpoint.py
      rk4.py
      dopri.py
      external.py
    diagnostics/
      ambiguity.py
      curvature.py
      jacobian.py
      solver_sensitivity.py
      geometry.py
      metrics.py
    experiments/
      run_train.py
      run_diagnostics.py
      run_sampling.py
      sweep.py
    plotting/
      vector_fields.py
      trajectories.py
      diagnostics.py
    utils/
      config.py
      logging.py
      seeding.py
      checkpoints.py
  scripts/
    run_toy_baseline.sh
    run_solver_sensitivity.sh
    run_geometry_toy.sh
  notebooks/
    00_sanity_checks.ipynb
    01_ambiguity_toys.ipynb
    02_solver_sensitivity.ipynb
    03_geometry_mismatch.ipynb
```

Use PyTorch as the primary backend.

---

## 6. Core Interfaces

### 6.1 Dataset / target distribution

```python
class TargetDistribution:
    def sample(self, n: int, device=None) -> torch.Tensor:
        ...

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        ...

    def metadata(self) -> dict:
        ...
```

Required toy targets:

- Gaussian mixture;
- two moons;
- checkerboard;
- concentric circles / annulus;
- Swiss roll;
- low-dimensional manifold embedded in high dimension;
- sphere/spherical shell;
- mixture of thin manifolds.

### 6.2 Source distribution

```python
class SourceDistribution:
    def sample(self, n: int, device=None) -> torch.Tensor:
        ...
```

Required sources:

- standard Gaussian;
- Gaussian with tunable variance;
- spherical shell source;
- low-pass source for image-like data;
- class-conditional source placeholder.

### 6.3 Coupling

```python
class Coupling:
    def pair(self, x0: torch.Tensor, x1: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        ...
```

Required couplings:

- independent random coupling;
- minibatch optimal transport coupling;
- class-conditional coupling;
- toy analytic coupling if available;
- reflow-generated coupling placeholder.

### 6.4 Path

```python
class Path:
    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        ...

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        ...
```

Required paths:

- linear interpolation;
- variance-preserving / diffusion-like path if easy;
- rectified-flow path;
- spherical interpolation;
- tangent-normal synthetic path;
- low-pass/frequency-aware path placeholder.

### 6.5 Velocity model

```python
class VelocityModel(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        ...
```

Initial models:

- small MLP for toy data;
- residual MLP;
- small U-Net for image toy data;
- optional tiny DiT later.

### 6.6 Solver

```python
class Solver:
    name: str

    def solve(
        self,
        v_fn,
        x0: torch.Tensor,
        t_grid: torch.Tensor,
        return_trajectory: bool = False,
        **kwargs
    ) -> torch.Tensor:
        ...
```

Required solvers:

- Euler;
- Heun;
- midpoint;
- RK4;
- Dormand-Prince or torchdiffeq wrapper;
- optional external/STORK-style wrapper later.

### 6.7 Diagnostics

Diagnostics should be callable independently of training.

```python
class Diagnostic:
    def compute(self, experiment_state, **kwargs) -> dict:
        ...
```

Core diagnostics are listed below.

---

## 7. Required Diagnostics

### 7.1 Conditional velocity ambiguity

Estimate:

\[
\mathbb{E}\left[\operatorname{Tr}\operatorname{Cov}(U_t\mid X_t)\right].
\]

Approximation methods:

1. **Grid/binning** for 2D:
   - bin \(x_t\) spatially;
   - compute empirical covariance of target velocities within each bin;
   - aggregate by bin mass.

2. **k-nearest neighbors** for low-D:
   - for each sampled \(x_t\), find nearby \(x_t'\);
   - estimate local covariance of corresponding \(u_t'\).

3. **Kernel regression**:
   - weight nearby samples by Gaussian/RBF kernel;
   - compute weighted covariance.

4. **Learned posterior estimator** later:
   - train an auxiliary network to predict endpoint or velocity distribution.

Outputs:

- ambiguity over time \(A(t)\);
- spatial heatmap of ambiguity;
- velocity covariance eigenvalues;
- top ambiguous regions;
- comparison across paths/couplings.

### 7.2 Conditional velocity multi-modality

Estimate whether \(U_t \mid X_t \approx x\) is unimodal or multi-modal.

Possible methods:

- Gaussian mixture model BIC/AIC on local velocity samples;
- local clustering of target velocities;
- entropy of velocity direction histogram;
- angular dispersion;
- ratio between first and second covariance eigenvalues.

Outputs:

- local multi-modality score;
- examples of regions with incompatible velocity directions;
- plots of local velocity clouds.

### 7.3 Bayes regression gap / irreducible target variance

For sampled pairs \((X_t,U_t)\), estimate:

\[
\mathbb{E}\|U_t - \mathbb{E}[U_t\mid X_t]\|^2.
\]

This is the irreducible MSE of deterministic FM under the chosen path law.

Approximation:

- kNN conditional mean;
- grid conditional mean;
- auxiliary high-capacity regressor as approximate Bayes predictor.

### 7.4 Learned-field curvature / material acceleration

For learned \(v_\theta(x,t)\), estimate material acceleration:

\[
a_\theta(x,t)
=
\partial_t v_\theta(x,t)
+
J_x v_\theta(x,t)\,v_\theta(x,t).
\]

Implementation options:

- exact autograd for small MLPs;
- finite-difference approximation for larger models;
- directional Jacobian-vector product using `torch.autograd.functional.jvp`.

Outputs:

- average \(\|a_\theta\|^2\) over trajectories;
- time profile of acceleration;
- spatial heatmap in 2D;
- comparison against ambiguity profile.

### 7.5 Jacobian / stiffness diagnostics

Estimate:

- spectral norm of \(J_x v_\theta\);
- Frobenius norm of \(J_x v_\theta\);
- divergence \(\nabla\cdot v_\theta\);
- local Lipschitz proxy;
- stiffness proxy along trajectories.

Implementation:

- exact Jacobian for 2D/low-D;
- power iteration / JVP-VJP estimates for higher-D;
- finite differences as fallback.

### 7.6 Solver sensitivity

For a fixed trained \(v_\theta\), sample with multiple solvers and timestep schedules.

Compute distributional differences between generated samples:

- MMD;
- sliced Wasserstein;
- Wasserstein-2 for low-D if feasible;
- coverage / precision-recall proxy;
- mode mass error for known mixtures;
- FID-like metric for image experiments later.

Define:

\[
\Delta_{\mathfrak S,K}(v_\theta)
=
\max_{S_i,S_j\in \mathfrak S_K}
D(\hat p_{S_i}, \hat p_{S_j}).
\]

Solvers/schedules:

- Euler uniform;
- Euler nonlinear time grid;
- Heun;
- midpoint;
- RK4;
- Dormand-Prince fixed tolerance;
- optional stabilized solver.

NFEs to test:

- 1, 2, 4, 8, 16, 32, 64.

Outputs:

- solver-sensitivity matrix;
- NFE-quality curves;
- relation between ambiguity and sensitivity;
- relation between curvature and sensitivity.

### 7.7 Geometry mismatch diagnostics

For manifold or latent experiments, estimate:

- distance to known manifold;
- radial deviation for spherical shell data;
- normal/tangent velocity decomposition;
- off-manifold path length;
- angular vs radial velocity magnitude;
- frequency-domain energy over time for image-like data.

Examples:

For spherical shell:

\[
\text{radial deviation}(x_t)=\left|\|x_t\|-r\right|.
\]

For known manifold with projection \(\Pi_{\mathcal M}\):

\[
d_{\mathcal M}(x_t)=\|x_t-\Pi_{\mathcal M}(x_t)\|.
\]

For tangent/normal decomposition:

\[
u_t = P_T u_t + P_N u_t.
\]

Outputs:

- off-manifold excursion curves;
- radial/angular decomposition;
- tangent/normal ambiguity;
- comparison of linear vs geometry-aware paths.

---

## 8. Initial Experiment Matrix

### 8.1 Experiment Group A: ambiguity toy experiments

Purpose:

> Test whether conditional velocity ambiguity predicts deterministic FM difficulty.

Datasets:

- two moons;
- checkerboard;
- Gaussian mixture;
- concentric circles;
- annulus;
- Swiss roll;
- low-dimensional manifold embedded in 10D or 50D.

Paths/couplings:

- independent linear interpolation;
- minibatch OT linear interpolation;
- rectified/reflow coupling;
- spherical path where applicable;
- tangent-normal synthetic path where applicable.

Metrics:

- ambiguity \(A(t)\);
- multi-modality score;
- Bayes regression gap;
- training loss;
- learned curvature;
- sample quality at different NFEs.

Expected plots:

- \(A(t)\) vs \(t\);
- \(A(t)\) vs curvature;
- ambiguity heatmaps;
- local velocity cloud plots;
- sample quality vs ambiguity.

### 8.2 Experiment Group B: solver sensitivity

Purpose:

> Test whether ambiguity and curvature predict black-box solver robustness.

Train a model once for each path/coupling, then sample with many solvers.

Solvers:

- Euler;
- Heun;
- midpoint;
- RK4;
- Dormand-Prince;
- optional external solver wrapper.

Metrics:

- pairwise generated-distribution distance across solvers;
- mode coverage;
- mode leakage;
- MMD;
- sliced Wasserstein;
- NFE-quality curve.

Key analysis:

- Does lower ambiguity imply lower solver spread?
- Does curvature explain solver spread better than ambiguity?
- Does OT/reflow reduce solver sensitivity?
- Are some flows solver-robust despite high ambiguity?

### 8.3 Experiment Group C: geometry mismatch

Purpose:

> Test whether geometry-aware path design reduces ambiguity and solver sensitivity.

Controlled datasets:

- points on a circle;
- points on an annulus;
- points on a sphere/spherical shell;
- Swiss roll;
- low-dimensional manifold embedded in high dimension.

Paths:

- Euclidean linear path;
- spherical interpolation;
- radial/angular decomposed path;
- tangent-normal synthetic path;
- low-pass/frequency-aware path for image-like data later.

Metrics:

- off-manifold excursion;
- normal velocity magnitude;
- tangent/normal ambiguity;
- solver sensitivity;
- sample quality.

Expected result if hypothesis is true:

- Geometry-aware paths should reduce off-manifold excursion.
- If geometry mismatch is relevant, this should also reduce conditional velocity ambiguity and/or solver sensitivity.

### 8.4 Experiment Group D: high-dimensional non-crossing sanity check

Purpose:

> Test whether literal path crossing disappears in high dimensions while conditional ambiguity remains meaningful.

Construct:

- Gaussian mixtures in dimensions 2, 10, 50, 100;
- embedded low-D manifolds in high-D;
- independent linear interpolation.

Measure:

- exact or near path intersections;
- nearest-neighbor density of intermediate points;
- conditional velocity variance;
- endpoint posterior entropy;
- learned-field behavior.

Goal:

> Show that literal crossing is not the right concept, but conditional-law overlap / non-identifiability still exists.

---

## 9. Implementation Milestones

### Milestone 1: minimal 2D CFM playground

Deliverables:

- toy distributions;
- Gaussian source;
- independent coupling;
- linear path;
- MLP velocity model;
- FM training loop;
- Euler/Heun/RK4 solvers;
- trajectory plotting;
- generated sample plotting.

Success criterion:

- Can train a basic flow from Gaussian to two moons/checkerboard and visualize trajectories.

### Milestone 2: ambiguity diagnostics

Deliverables:

- grid/bin ambiguity estimator;
- kNN ambiguity estimator;
- local velocity cloud visualizer;
- Bayes regression gap estimator;
- ambiguity time-profile plots.

Success criterion:

- Can compare ambiguity of independent vs OT coupling on at least two toy datasets.

### Milestone 3: solver sensitivity suite

Deliverables:

- unified solver interface;
- multiple solvers and schedules;
- pairwise generated-distribution distance matrix;
- NFE-quality curves;
- solver sensitivity summary.

Success criterion:

- Can report whether two trained models with similar training loss differ in solver sensitivity.

### Milestone 4: curvature and Jacobian diagnostics

Deliverables:

- material acceleration estimator;
- Jacobian norm estimator;
- stiffness proxy;
- curvature-vs-ambiguity plots.

Success criterion:

- Can test whether high ambiguity regions correlate with high curvature or stiffness.

### Milestone 5: geometry experiments

Deliverables:

- sphere/shell/annulus datasets;
- spherical path;
- radial/angular diagnostics;
- tangent/normal synthetic diagnostics;
- off-manifold excursion plots.

Success criterion:

- Can compare Euclidean linear paths and geometry-aware paths on controlled geometric datasets.

### Milestone 6: optional image/latent extension

Deliverables:

- MNIST or CIFAR-10 low-res setup;
- simple autoencoder or pretrained latent loader;
- linear vs spherical/radial-angular latent path;
- image/latent diagnostics.

Success criterion:

- Can test whether the toy findings survive in a more practical latent setting.

---

## 10. Recommended Development Order

1. Build the minimal 2D training loop from scratch.
2. Add TorchCFM / Meta flow_matching only as reference baselines, not as the core architecture.
3. Implement diagnostics before adding many model architectures.
4. Implement independent vs OT coupling.
5. Implement solver sensitivity.
6. Add geometry-aware paths.
7. Only then move to image/latent experiments.

Do not start with a large U-Net or DiT. The first goal is scientific controllability.

---

## 11. Configuration Design

Use YAML or Hydra-style configs.

Example:

```yaml
experiment:
  name: moons_linear_independent
  seed: 0
  output_dir: runs/moons_linear_independent

data:
  name: two_moons
  n_train: 100000
  noise: 0.05

source:
  name: gaussian
  dim: 2
  std: 1.0

coupling:
  name: independent

path:
  name: linear

model:
  name: mlp
  hidden_dim: 256
  depth: 4
  activation: silu
  time_embedding: sinusoidal

training:
  batch_size: 1024
  steps: 50000
  lr: 1.0e-4
  time_sampling: uniform

solvers:
  names: [euler, heun, midpoint, rk4]
  nfes: [4, 8, 16, 32, 64]

diagnostics:
  ambiguity:
    methods: [grid, knn]
    t_values: [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
  curvature:
    enabled: true
  solver_sensitivity:
    metrics: [mmd, sliced_wasserstein]
```

---

## 12. Logging and Reproducibility Requirements

Every run should save:

```text
run_dir/
  config.yaml
  metrics.json
  checkpoint.pt
  samples/
  trajectories/
  diagnostics/
    ambiguity_time.csv
    curvature_time.csv
    solver_sensitivity.csv
    jacobian_stats.csv
  plots/
    samples.png
    vector_field.png
    trajectories.png
    ambiguity_heatmap_t*.png
    solver_sensitivity_matrix.png
```

Requirements:

- fixed random seeds;
- deterministic mode where feasible;
- save exact git commit hash;
- save library versions;
- save raw diagnostic arrays, not only plots;
- make all plots regenerable from saved artifacts.

---

## 13. Evaluation Metrics

### 13.1 For known toy distributions

Use:

- MMD;
- sliced Wasserstein;
- mode coverage;
- mode mass error;
- nearest-neighbor distance to target samples;
- visual inspection.

### 13.2 For image / latent experiments

Use later:

- FID or clean-FID if available;
- feature-space MMD;
- reconstruction/decoder quality if using autoencoder latents;
- precision/recall;
- NFE-quality curve.

### 13.3 For research hypotheses

Primary metrics:

- conditional velocity ambiguity;
- Bayes regression gap;
- solver sensitivity;
- curvature/material acceleration;
- geometry mismatch/off-manifold excursion.

---

## 14. Important Design Warnings

### 14.1 Do not overclaim novelty

Many nearby ideas already exist. The playground is designed to discover whether a sharper contribution exists.

### 14.2 Do not equate ambiguity with literal crossing

In high dimensions, paths may almost never exactly intersect. Use conditional-law overlap and local non-identifiability instead.

### 14.3 Do not hard-code one solver into the theory

Solver sensitivity should be generic and black-box. We can compare solvers, but the main object should be robustness across a solver class, not optimization for a single solver.

### 14.4 Do not treat geometry as a decorative manifold trick

The practical question is whether geometry reduces ambiguity, solver sensitivity, or off-manifold inefficiency. Geometry is useful only if it explains or improves these structural quantities.

### 14.5 Do not let image-scale training dominate early work

Large models can validate but should not define the contribution.

---

## 15. Possible Paper Directions Depending on Results

### If ambiguity strongly predicts curvature and solver sensitivity

Potential paper:

> **Flow Matchability: Ambiguity, Curvature, and Numerical Stability in Flow Matching**

Contribution:

- define conditional-law ambiguity;
- show it predicts deterministic representability and solver robustness;
- propose path/coupling diagnostics;
- maybe propose a principled ambiguity-reducing path/coupling method.

### If ambiguity predicts training difficulty but not solver sensitivity

Potential paper:

> **Deterministic Representability in Flow Matching**

Contribution:

- theory of when stochastic path laws compress into deterministic Markovian vector fields;
- analysis of irreducible FM target variance;
- comparison of deterministic vs multi-modal/hierarchical velocity models.

### If geometry-aware paths reduce ambiguity

Potential paper:

> **Geometry-Induced Ambiguity in Flow Matching**

Contribution:

- show Euclidean paths create avoidable ambiguity in latent/manifold-like data;
- propose geometry-aware diagnostic and path construction;
- validate on controlled manifolds and latent image representations.

### If solver sensitivity behaves independently

Potential paper:

> **Black-Box Numerical Robustness of Flow-Matching Generative Models**

Contribution:

- define solver sensitivity as a distributional robustness metric;
- benchmark paths and couplings;
- analyze which field properties predict robustness.

---

## 16. Immediate Task List for the Agent

### Phase 1: setup

- [ ] Create repository `fm_lab`.
- [ ] Set up PyTorch project with `pyproject.toml`.
- [ ] Add config system.
- [ ] Add deterministic seeding utilities.
- [ ] Add logging and run directories.

### Phase 2: toy FM baseline

- [ ] Implement toy datasets.
- [ ] Implement Gaussian source.
- [ ] Implement independent coupling.
- [ ] Implement linear path.
- [ ] Implement MLP velocity model.
- [ ] Implement FM loss.
- [ ] Implement training loop.
- [ ] Implement Euler sampler.
- [ ] Visualize generated samples and trajectories.

### Phase 3: paths and couplings

- [ ] Add minibatch OT coupling.
- [ ] Add rectified/reflow coupling placeholder.
- [ ] Add spherical path.
- [ ] Add tangent-normal synthetic path.

### Phase 4: diagnostics

- [ ] Implement ambiguity estimator by grid/binning.
- [ ] Implement ambiguity estimator by kNN.
- [ ] Implement conditional velocity multi-modality visualizer.
- [ ] Implement Bayes regression gap estimator.
- [ ] Implement curvature/material acceleration estimator.
- [ ] Implement Jacobian/stiffness diagnostics.
- [ ] Implement solver sensitivity matrix.

### Phase 5: solver suite

- [ ] Implement Heun.
- [ ] Implement midpoint.
- [ ] Implement RK4.
- [ ] Implement Dormand-Prince or torchdiffeq wrapper.
- [ ] Add multiple timestep schedules.
- [ ] Add NFE-quality sweep.

### Phase 6: controlled experiments

- [ ] Run independent vs OT coupling on two moons.
- [ ] Run independent vs OT coupling on checkerboard.
- [ ] Run linear vs spherical path on sphere/shell.
- [ ] Run dimension sweep for crossing vs ambiguity.
- [ ] Generate first analysis report.

---

## 17. First Minimum Viable Experiment

The first experiment should be very small but complete.

### Dataset

Two moons.

### Compare

1. independent coupling + linear path;
2. minibatch OT coupling + linear path.

### Train

Same MLP architecture and same training budget.

### Diagnose

For both:

- ambiguity \(A(t)\);
- Bayes regression gap;
- learned curvature;
- generated samples with Euler/Heun/RK4 at NFE = 4, 8, 16, 32;
- solver sensitivity matrix.

### First question

> Does the lower-ambiguity path/coupling, if any, also have lower curvature and lower solver sensitivity?

If yes, continue. If no, inspect whether ambiguity estimator is wrong, whether curvature is the mediating variable, or whether the central thesis needs revision.

---

## 18. Final Note to the Agent

This project is exploratory and theory-motivated. The implementation should prioritize **inspectability** over benchmark performance.

The point is not to produce a beautiful FM demo. The point is to build an instrument that can reveal whether ambiguity, geometry, and numerical stability are genuinely linked in flow matching.

If the data contradicts the hypothesis, preserve the negative result carefully. A useful playground is one that can kill bad ideas early.
