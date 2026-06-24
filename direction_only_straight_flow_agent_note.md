# Agent Note: Direction-Only Straight Flow with Independent Coupling

## 0. Purpose of this experiment

We want to explore a middle ground between ordinary flow matching and exact straight transport-map learning.

Ordinary flow matching learns a time-dependent vector field

\[
v_\theta(t,x)
\]

and samples by integrating

\[
\frac{dX_t}{dt}=v_\theta(t,X_t).
\]

This preserves local learnability, but trajectories may be curved, which hurts few-step sampling.

At the other extreme, exact straight constant-speed flow imposes

\[
X_t=(1-t)X_0+tT(X_0),
\]

which is equivalent to learning a global transport map \(T\). This gives one-step generation in principle, but risks collapsing the problem back into direct map learning or distillation. It removes much of the local-dynamics advantage that makes flow matching and diffusion trainable.

The current experiment weakens exact straightness. We only require that each trajectory keeps a fixed **direction**, while allowing its **speed/magnitude** to vary over time and position. In other words, trajectories should lie on straight lines, but they need not move at constant speed.

The intended class is:

\[
\frac{dF_t(a)}{dt}
=
s_\theta(t,F_t(a),a)\,n_\phi(a),
\qquad
F_0(a)=a.
\]

Here:

- \(a=X_0\) is the Lagrangian label, i.e. the source/noise point.
- \(n_\phi(a)\in \mathbb S^{d-1}\) is a fixed direction attached to that source point.
- \(s_\theta(t,x,a)\in\mathbb R\), or optionally \(\mathbb R_{\geq 0}\), is the instantaneous scalar speed.
- \(F_t(a)\) is the current position of the particle that started at \(a\).

This construction guarantees that the particle starting at \(a\) always moves parallel to the same direction \(n_\phi(a)\). Its trajectory is therefore contained in the line

\[
a+\mathbb R n_\phi(a).
\]

The accumulated displacement is still obtained by integration:

\[
F_t(a)
=
a+
\left(
\int_0^t s_\theta(\tau,F_\tau(a),a)\,d\tau
\right)n_\phi(a).
\]

So the model still learns local instantaneous dynamics through \(s_\theta\), while the angular part of the motion is fixed by \(n_\phi(a)\).

For now, the implementation should use **independent coupling**:

\[
X_0\sim p_0,
\qquad
X_1\sim p_{\mathrm{data}},
\qquad
X_0\perp X_1.
\]

Do not add teacher coupling, OT coupling, or distillation in the first implementation. Independent coupling is expected to be difficult, but that is precisely why it is a useful first stress test.

---

## 1. Why use a Lagrangian label?

Standard flow matching is Eulerian. It learns

\[
v_\theta(t,x)
\]

from examples

\[
(X_t,t)\mapsto U_t.
\]

The model sees only the current position and time. It does not know which source point \(X_0\) produced that point. Consequently, the learned field approximates

\[
v^*(t,x)=\mathbb E[U_t\mid X_t=x].
\]

This conditional averaging is the source of several issues we are investigating: velocity variance, curved marginal trajectories, and ambiguity in local direction.

In this experiment we deliberately keep the source label

\[
a=X_0.
\]

The learned field is no longer purely Eulerian. It is label-conditioned:

\[
v_\theta(t,x,a)=s_\theta(t,x,a)n_\phi(a).
\]

This is a fair tradeoff for the current research goal. During generation, we always know the source label because we start from

\[
a\sim p_0.
\]

Therefore sampling can carry \(a\) throughout the trajectory. We do not need to recover \(a\) from \(x\) during generation.

This differs from ordinary conditional flow matching. It should be described as **label-conditioned flow matching** or **Lagrangian flow matching**, not as a standard Eulerian CNF field.

The benefit is that direction-straightness is guaranteed by construction. The cost is that the learned dynamics are label-dependent.

---

## 2. Why not use a soft straightness regularizer?

A soft regularizer would penalize something like

\[
P_{v^\perp}
\left(
\partial_t v+(v\cdot\nabla)v
\right),
\]

where \(P_{v^\perp}\) projects onto the subspace perpendicular to \(v\). This penalizes normal acceleration and encourages direction-straightness.

However, regularization-based approaches often underperform or become difficult to tune. The model can satisfy the main flow-matching objective while mostly ignoring the regularizer, or the regularizer can fight the supervised velocity objective.

Instead, this experiment hard-wires direction-straightness by parameterization. The direction is fixed per source label:

\[
n_\phi(a).
\]

The speed is learned locally:

\[
s_\theta(t,x,a).
\]

Thus no separate direction-straightness penalty is needed to make trajectories lie on lines. This is the central modeling choice.

---

## 3. Independent coupling as the first stress test

The first implementation should use independent source-target samples:

\[
a=X_0\sim p_0,
\qquad
X_1\sim p_{\mathrm{data}},
\qquad
X_0\perp X_1.
\]

With linear conditional interpolation,

\[
X_t=(1-t)X_0+tX_1,
\qquad
U_t=X_1-X_0.
\]

Under independent coupling, for a fixed source label \(a\), the endpoint \(X_1\) varies over the full data distribution. Thus the target displacement

\[
U_t=X_1-a
\]

may point in many different directions for the same or nearby \(a\). Since our model attaches only one direction \(n_\phi(a)\) to each source label, it will likely learn a compromise direction.

This is not a bug in the first experiment. It is the intended stress test.

The first question is not whether independent coupling is ideal. It is:

> If we impose direction-only straightness while using the cheapest independent coupling, how badly does the direction constraint conflict with the standard flow-matching target?

This should reveal whether independent coupling is already too incoherent for this model class, or whether the local speed network can compensate enough to learn something useful.

Do not prematurely improve the coupling. Keep the first study simple and interpretable.

---

## 4. Model components

Use two networks or two heads.

### 4.1 Direction network

The direction network takes the source label \(a\) and outputs an unconstrained vector

\[
q_\phi(a)\in\mathbb R^d.
\]

Normalize it to obtain

\[
n_\phi(a)=\frac{q_\phi(a)}{\|q_\phi(a)\|+\varepsilon}.
\]

This gives a point on the sphere:

\[
n_\phi(a)\in\mathbb S^{d-1}.
\]

This normalization is preferred over explicit Riemannian optimization for the first implementation, because \(n_\phi(a)\) is a neural network output, not a finite table of free sphere parameters. Backpropagation through normalization is simple and standard.

If one later experiments with a finite set of explicit directions, then Riemannian gradient descent on the sphere is also easy. For a sphere-valued parameter \(n\), Euclidean gradient \(g\) is projected to the tangent space by

\[
g_{\mathrm{tan}}=(I-nn^\top)g,
\]

then updated and retracted:

\[
\tilde n=n-\eta g_{\mathrm{tan}},
\qquad
n_{\mathrm{new}}=\frac{\tilde n}{\|\tilde n\|}.
\]

But this is not necessary for the first neural implementation.

### 4.2 Speed network

The speed network takes

\[
(t,x,a)
\]

and outputs a scalar

\[
s_\theta(t,x,a).
\]

The velocity prediction is

\[
\hat u_\theta(t,x,a)=s_\theta(t,x,a)n_\phi(a).
\]

There are two choices.

#### Signed speed

\[
s_\theta(t,x,a)\in\mathbb R.
\]

This allows motion in either orientation along the line defined by \(n_\phi(a)\). Then \(n\) and \(-n\) represent the same geometric line, with speed sign absorbing orientation.

This is the recommended first setting because it is easier to optimize.

#### Nonnegative speed

\[
s_\theta(t,x,a)\geq 0,
\]

for example

\[
s_\theta=\operatorname{softplus}(g_\theta).
\]

This enforces monotone motion in the direction \(n_\phi(a)\), but it makes the orientation of \(n_\phi(a)\) important and may be harder to train.

Recommended plan: start with signed speed, then test nonnegative speed as a later ablation.

---

## 5. Training objective with independent coupling

Given an independent-coupling training sample:

\[
a=X_0\sim p_0,
\qquad
x_1=X_1\sim p_{\mathrm{data}},
\qquad
t\sim\mathrm{Unif}(0,1),
\]

construct the standard linear conditional interpolant:

\[
x_t=(1-t)a+tx_1,
\]

with target velocity:

\[
u_t=x_1-a.
\]

The label-conditioned model predicts:

\[
\hat u_\theta(t,x_t,a)=s_\theta(t,x_t,a)n_\phi(a).
\]

The basic supervised objective is:

\[
\mathcal L_{\mathrm{vec}}
=
\mathbb E
\left[
\left\|
s_\theta(t,x_t,a)n_\phi(a)-u_t
\right\|^2
\right].
\]

This objective is compatible with flow-matching samples, but it is not standard Eulerian CFM because the model sees \(a\). The purpose is to test whether keeping the Lagrangian label avoids some of the conditional averaging problem while still learning local speed.

---

## 6. Why learn direction and speed separately?

The raw vector MSE

\[
\left\|
s_\theta n_\phi-u_t
\right\|^2
\]

couples direction and speed. This is mathematically fine but can be unstable or hard to interpret.

For a fixed unit direction \(n\), the best signed speed under squared error is

\[
s^*=u_t\cdot n.
\]

Then

\[
u_t = (u_t\cdot n)n + u_t^\perp.
\]

The residual decomposes into:

- a speed/projection error;
- a perpendicular direction error.

Therefore it is useful to expose this decomposition explicitly.

For signed speed, a natural speed loss is:

\[
\mathcal L_{\mathrm{speed}}
=
\mathbb E
\left[
\left(
s_\theta(t,x_t,a)
-
u_t\cdot n_\phi(a)
\right)^2
\right].
\]

For direction, since signed speed makes \(n\) and \(-n\) equivalent as geometric lines, use an orientation-invariant alignment loss:

\[
\mathcal L_{\mathrm{dir}}
=
\mathbb E
\left[
1-
\frac{(u_t\cdot n_\phi(a))^2}{\|u_t\|^2+\varepsilon}
\right].
\]

Equivalently, this penalizes the squared sine of the angle between \(u_t\) and \(n_\phi(a)\).

A combined objective could be:

\[
\mathcal L
=
\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}
+
\lambda_{\mathrm{speed}}\mathcal L_{\mathrm{speed}}.
\]

Optionally include the raw vector loss as a diagnostic or auxiliary term:

\[
\mathcal L_{\mathrm{vec}}
=
\mathbb E
\left[
\left\|
s_\theta n_\phi-u_t
\right\|^2
\right].
\]

For the first implementation, prefer the decomposed direction + speed loss because it helps diagnose whether failures come from direction learning or speed learning. Also log the raw vector loss for comparison with ordinary flow matching.

---

## 7. Expected behavior under independent coupling

Independent coupling is deliberately challenging for this model.

For each source point \(a\), the target endpoint \(x_1\) is sampled independently from the data distribution. Thus the displacement

\[
u_t=x_1-a
\]

can point in many directions for the same source region. The model, however, assigns one direction

\[
n_\phi(a)
\]

to each source label.

Therefore, under independent coupling, \(n_\phi(a)\) will likely learn a dominant or average line direction rather than a true endpoint-specific displacement direction.

This can fail in several ways:

- direction loss remains high;
- speed network produces large signed speeds to compensate for poor direction;
- generated endpoints do not cover the data distribution;
- trajectories are straight by construction but point toward poor regions.

These outcomes are still informative. They measure the incompatibility between independent coupling and direction-only straightness.

If the experiment fails under independent coupling, do not immediately interpret the direction-only parameterization as useless. The failure may simply mean that independent coupling is too incoherent for a one-direction-per-source model.

---

## 8. Sampling procedure

At inference time:

1. Sample

\[
a\sim p_0.
\]

2. Compute direction

\[
n_\phi(a).
\]

3. Initialize

\[
F_0(a)=a.
\]

4. Integrate scalar-speed constrained dynamics:

\[
\frac{dF_t(a)}{dt}
=
s_\theta(t,F_t(a),a)n_\phi(a).
\]

The trajectory remains on the line

\[
a+\mathbb R n_\phi(a).
\]

Equivalently, one may track scalar progress

\[
\rho(t,a)
\]

with

\[
F_t(a)=a+\rho(t,a)n_\phi(a),
\]

and

\[
\frac{d\rho(t,a)}{dt}
=
s_\theta(t,a+\rho(t,a)n_\phi(a),a).
\]

This reduces the dynamics to a one-dimensional ODE per sample, although the speed network still evaluates at the high-dimensional current point \(x=F_t(a)\).

This is an important advantage over ordinary vector-field integration: only scalar progress changes, while direction is fixed.

---

## 9. What not to do in the first implementation

Do not try to make this a fully Eulerian field \(v(t,x)\) immediately. To define

\[
v(t,x)
\]

from the Lagrangian model, one must recover

\[
a=F_t^{-1}(x),
\]

then set

\[
v(t,x)
=
s_\theta(t,x,a)n_\phi(a).
\]

This requires inversion of the flow map and introduces the same kind of complication as OFM. For generation, inversion is unnecessary because the sample path carries \(a\).

Do not begin with a Burgers residual or direction-straightness regularizer. The point of this experiment is to enforce direction-straightness by parameterization, not by soft penalty.

Do not begin with Riemannian optimization over neural outputs. Normalize the direction-network output. Riemannian GD can be tested later if directions are represented as explicit finite parameters rather than neural outputs.

Do not force nonnegative speed at first. Signed speed avoids orientation ambiguity and makes the first training run easier.

Do not add teacher coupling, OT coupling, minibatch OT coupling, or rectified-flow pairing in the first implementation. Start with independent coupling to keep the experiment simple and to measure the baseline incompatibility.

---

## 10. Diagnostics to log

The experiment should log more than sample quality. We need to understand whether the model is learning the intended decomposition.

### Direction alignment

\[
\cos^2(u_t,n_\phi(a))
=
\frac{(u_t\cdot n_\phi(a))^2}{\|u_t\|^2+\varepsilon}.
\]

Report mean and distribution over training samples.

### Perpendicular velocity residual

\[
\|u_t-(u_t\cdot n_\phi(a))n_\phi(a)\|^2.
\]

This measures how much of the target velocity cannot be represented by the learned line direction.

### Speed prediction error

\[
\left(
s_\theta(t,x_t,a)-u_t\cdot n_\phi(a)
\right)^2.
\]

This isolates scalar speed learning.

### Raw vector error

\[
\|s_\theta n_\phi-u_t\|^2.
\]

Useful for comparing against ordinary FM losses.

### Direction diversity

Monitor the distribution of pairwise similarities

\[
n_\phi(a_i)\cdot n_\phi(a_j).
\]

This detects whether the direction network collapses to similar directions for many source points.

### Speed magnitude

Monitor

\[
|s_\theta(t,x_t,a)|
\]

and its distribution. Large speeds may indicate that the model is compensating for bad direction choices.

### Trajectory curvature

The constructed trajectories should be geometrically straight by design. Numerically verify:

\[
\frac{\|P_{n^\perp}\dot F_t\|}{\|\dot F_t\|+\varepsilon}
\approx 0.
\]

This should be near zero, up to numerical error.

### Endpoint quality

Evaluate the final samples

\[
F_1(a)
\]

by whatever distributional metrics are available in the playground: toy Wasserstein/MMD, FID-like metric, coverage, class/mode coverage, or visual inspection.

### NFE sensitivity

Compare sample quality under different numbers of integration steps. Since the path direction is fixed, the method should be less sensitive to angular integration error. But scalar speed integration error may still matter.

---

## 11. Baselines

Use independent coupling for all first-stage baselines.

Compare against:

1. ordinary Eulerian CFM with independent coupling;
2. exact constant-speed straight transport-map version, if already available;
3. direction-only label-conditioned FM with raw vector MSE;
4. direction-only label-conditioned FM with decomposed direction/speed loss;
5. ablation of speed input: \(s(t,x,a)\) versus \(s(t,a)\) versus \(s(t,x)\), if easy.

The most important comparison is:

\[
\text{ordinary Eulerian CFM with independent coupling}
\]

versus

\[
\text{direction-only label-conditioned FM with independent coupling}.
\]

This isolates the effect of retaining the Lagrangian label and fixing direction per source.

---

## 12. Research hypotheses

The experiment is designed to test the following hypotheses.

### Hypothesis 1: Direction-straightness is a useful intermediate constraint

Exact straightness may be too strong because it collapses learning to a transport map. Direction-only straightness is weaker: it fixes the angular part but leaves local speed learnable.

Expected benefit: fewer-step sampling may improve relative to ordinary curved flow, while training should remain easier than direct map learning.

### Hypothesis 2: Independent coupling is a meaningful stress test

Independent coupling is likely too incoherent for a one-direction-per-source model, but this is exactly what we want to measure first.

Expected result: if the method struggles, the failure should appear clearly in direction alignment and perpendicular residual diagnostics.

### Hypothesis 3: Label conditioning reduces conditional averaging

Standard CFM learns

\[
\mathbb E[U_t\mid X_t=x].
\]

The label-conditioned model learns a structured approximation to velocity conditioned also on source identity:

\[
U_t \mid X_0=a.
\]

This may reduce the averaging problem that causes curved mean fields, although independent coupling may still leave too much endpoint variation.

### Hypothesis 4: The product form is manageable if direction is normalized

Because

\[
\|n_\phi(a)\|=1,
\]

the model avoids arbitrary scaling degeneracy. Speed controls magnitude; direction controls orientation. This makes the product

\[
s_\theta n_\phi
\]

much more stable than a generic multiplication of two unconstrained networks.

---

## 13. Failure modes to watch for

### Direction collapse

The direction network might learn similar directions for many \(a\), especially under independent coupling. Monitor diversity of

\[
n_\phi(a).
\]

### Compromise directions

For each \(a\), independent coupling exposes the model to many possible target endpoints. A single \(n_\phi(a)\) may become a mean or compromise direction that is not useful for generation.

### Speed compensation

The speed network may try to compensate for poor directions by producing large signed speeds. Monitor speed magnitudes.

### Poor endpoint distribution

Even if trajectories are straight, the final distribution may not match data. This means the constraint is too restrictive under independent coupling, or that the training signal is insufficient.

### Overdependence on label

Since \(s_\theta\) sees \(a\), it may use \(a\) heavily and ignore \(x\). This is not necessarily wrong, but it should be diagnosed. Ablate speed inputs if convenient.

### Non-monotone motion

With signed speed, a sample can move forward and backward along the same line. This may be acceptable initially. Later test nonnegative speed if monotonicity is desired.

---

## 14. Suggested staged implementation

### Stage 1: Toy low-dimensional data

Use 2D mixtures, moons, rings, or Gaussian-to-mixture examples. These make directions and trajectories visible.

Use independent coupling and linear interpolants:

\[
x_t=(1-t)a+tx_1,
\qquad
u_t=x_1-a.
\]

Train the label-conditioned direction-speed model.

### Stage 2: Compare losses

Compare raw vector MSE versus decomposed direction/speed loss.

The decomposed loss should provide better diagnostics and may train more stably.

### Stage 3: Compare to ordinary FM

Train ordinary Eulerian CFM on the same independent-coupling data and compare:

- trajectory curvature;
- endpoint quality;
- NFE sensitivity;
- mode coverage;
- direction alignment diagnostics.

### Stage 4: Ablations

Test:

- signed speed versus nonnegative speed;
- \(s(t,x,a)\) versus \(s(t,a)\) versus \(s(t,x)\);
- normalization of direction output with different \(\varepsilon\);
- exact constant-speed straight map, if available.

Leave coherent coupling experiments for later. Do not mix them into the first-stage implementation.

---

## 15. Summary of the core model

The core model is:

\[
n_\phi(a)=\frac{q_\phi(a)}{\|q_\phi(a)\|+\varepsilon},
\]

\[
\hat u_\theta(t,x,a)
=
s_\theta(t,x,a)n_\phi(a).
\]

Training samples use independent coupling:

\[
a=X_0\sim p_0,
\qquad
x_1=X_1\sim p_{\mathrm{data}},
\qquad
X_0\perp X_1,
\]

with linear interpolation:

\[
x_t=(1-t)a+tx_1,
\qquad
u_t=x_1-a.
\]

The preferred decomposed loss for signed speed is:

\[
\mathcal L_{\mathrm{dir}}
=
\mathbb E
\left[
1-
\frac{(u_t\cdot n_\phi(a))^2}{\|u_t\|^2+\varepsilon}
\right],
\]

\[
\mathcal L_{\mathrm{speed}}
=
\mathbb E
\left[
\left(
s_\theta(t,x_t,a)-u_t\cdot n_\phi(a)
\right)^2
\right],
\]

\[
\mathcal L
=
\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}
+
\lambda_{\mathrm{speed}}\mathcal L_{\mathrm{speed}}.
\]

Sampling integrates:

\[
\frac{dF_t(a)}{dt}
=
s_\theta(t,F_t(a),a)n_\phi(a),
\qquad
F_0(a)=a.
\]

This hard-wires direction-only straightness while preserving local speed learning. The first experiment should deliberately use independent coupling to measure how much the direction-only constraint conflicts with the cheapest standard flow-matching setup.
