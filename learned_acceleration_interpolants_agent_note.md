# Agent Note: Low-Order Learned Acceleration Interpolants for Straightening Flow Matching

## 0. Purpose of this experiment

This experiment is a restricted, variance-analysis-motivated version of the learned-interpolant idea in *Learning Straight Flows by Learning Curved Interpolants*. The goal is not to reproduce the full general method from that paper. Instead, we want to test a more specific hypothesis derived from our own analysis:

> Standard linear conditional interpolants have zero acceleration. If the induced marginal vector field is curved because conditional velocity variance creates an effective stress term, then adding a small learned acceleration term to the conditional interpolant may partially cancel that effect and produce a straighter learned flow.

The core question is:

\[
\text{Can a low-order learned acceleration correction improve straightness and few-step generation under independent coupling?}
\]

This is intentionally narrower than learning an arbitrary interpolant

\[
I_\psi(t,x_0,x_1).
\]

The restriction is important. A fully general learned interpolant becomes hard to distinguish from existing learned-interpolant methods. Our version should be understood as a controlled experiment: we add the **first nontrivial nonlinear correction** to the standard linear interpolant and test whether it helps.

The intended first-order research outcome is not necessarily state-of-the-art performance. The intended outcome is to determine whether the “conditional acceleration can compensate variance-induced curvature” hypothesis has empirical support.

---

## 1. Background motivation

A standard flow matching construction samples

\[
x_0\sim p_0,\qquad x_1\sim p_1,\qquad t\sim \mathrm{Unif}(0,1),
\]

then uses the linear interpolant

\[
x_t=(1-t)x_0+t x_1
\]

with target velocity

\[
u_t=x_1-x_0.
\]

The learned marginal vector field is

\[
v_t(x)=\mathbb E[u_t\mid x_t=x].
\]

Even though the sample-wise interpolants are linear, the learned marginal flow can be curved because the conditional average velocity field need not satisfy the straightness condition

\[
\partial_t v_t+(v_t\cdot\nabla)v_t=0.
\]

Our analysis used the conditional-flow moment identity. For a general conditional interpolant

\[
X_t=I_t(X_0,X_1),
\]

define

\[
U_t=\dot X_t,\qquad A_t=\ddot X_t,
\]

and

\[
C_t(x)=\operatorname{Cov}(U_t\mid X_t=x).
\]

Then the marginal field satisfies

\[
p_t
\left(
\partial_t v_t+(v_t\cdot\nabla)v_t
\right)
=
p_t\mathbb E[A_t\mid X_t=x]
-
\nabla\cdot(p_tC_t).
\]

For linear interpolants,

\[
A_t=0,
\]

so there is no acceleration term available to balance the covariance-stress term

\[
\nabla\cdot(p_tC_t).
\]

This motivates adding a learned nonzero conditional acceleration. However, estimating

\[
p_t,\quad C_t,\quad \nabla\cdot(p_tC_t)
\]

directly is too hard. Therefore this experiment does not attempt to solve the moment identity. Instead, it uses the identity only as a design principle: add a controlled, low-order acceleration mode and test whether it reduces curvature or improves low-NFE generation.

---

## 2. Why not use the direction-only parameterization?

We already tested a direction-only straight-flow parameterization with a Lagrangian direction

\[
n_\phi(x_0)
\]

and local speed

\[
s_\theta(t,x,x_0).
\]

The method was technically working, but with independent coupling it failed structurally. For each source point \(x_0\), independent coupling pairs it with many unrelated targets \(x_1\), so there is no coherent single direction to learn. The oracle alignment was already poor, and the trained model approached that oracle, which indicates the failure was not primarily optimization.

This acceleration-interpolant experiment is different. It does **not** require a single direction per source point. The learned correction may depend on the pair \((x_0,x_1)\). Therefore it is much better matched to independent coupling, where the pairwise directions are diverse.

So the transition is:

\[
\text{direction-only model: attach one line to each source point}
\]

versus

\[
\text{acceleration interpolant: learn a pair-dependent curved path}.
\]

The latter is more compatible with independent source-target pairing.

---

## 3. Relationship to *Learning Straight Flows by Learning Curved Interpolants*

The paper *Learning Straight Flows by Learning Curved Interpolants* proposes to learn flexible interpolants so that the induced optimal flow-matching vector field becomes straight. This is close to our motivation, but their method is more general and more complex.

Their core difficulty is bilevel:

\[
\text{interpolant } I_\psi
\quad\Rightarrow\quad
\text{induced optimal field } v_\psi^*
\quad\Rightarrow\quad
\text{straightness of } v_\psi^*.
\]

Changing the interpolant changes both:

1. the training points \(x_t=I_\psi(t,x_0,x_1)\);
2. the target velocities \(\dot I_\psi(t,x_0,x_1)\);
3. the population optimal vector field \(v_\psi^*\).

The paper handles this with analytic expressions for the induced optimal vector field and practical approximations involving learned vector fields, stop-gradients, invertible interpolant parameterizations, and low-NFE evaluation.

We should learn from the paper's experimental logic but not immediately reproduce its full machinery. Our experiment should be positioned as:

> A low-order acceleration-restricted approximation to general learned curved interpolants.

The point is to test whether a simple acceleration correction is enough before moving to fully general learned interpolants.

---

## 4. Core ansatz: second-order acceleration interpolant

Let

\[
\Delta=x_1-x_0.
\]

The standard linear interpolant is

\[
I_0(t,x_0,x_1)=x_0+t\Delta.
\]

The proposed second-order acceleration interpolant is

\[
\boxed{
I_\psi(t,x_0,x_1)
=
x_0+t\Delta+t(1-t)A_\psi(x_0,x_1).
}
\]

This satisfies the endpoint constraints:

\[
I_\psi(0,x_0,x_1)=x_0,
\]

\[
I_\psi(1,x_0,x_1)=x_1.
\]

The velocity target becomes

\[
\dot I_\psi(t,x_0,x_1)
=
\Delta+(1-2t)A_\psi(x_0,x_1).
\]

The acceleration is

\[
\ddot I_\psi(t,x_0,x_1)
=
-2A_\psi(x_0,x_1).
\]

Thus \(A_\psi\) is a learned pair-dependent constant acceleration coefficient. This is the first nontrivial correction to linear interpolation.

### Why this is a good first version

This version is minimal:

- it preserves endpoint positions;
- it introduces nonzero acceleration;
- it has only one learned correction vector per pair;
- the temporal structure is fixed;
- it does not learn a fully arbitrary function of \(t,x_0,x_1\).

It gives the cleanest possible test of the hypothesis:

\[
\text{Does adding a learned acceleration mode improve marginal straightness?}
\]

---

## 5. Optional endpoint-velocity-preserving variant

The quadratic ansatz changes endpoint velocities:

\[
\dot I_\psi(0)=\Delta+A_\psi,
\]

\[
\dot I_\psi(1)=\Delta-A_\psi.
\]

This may be fine, but it could create undesirable endpoint behavior. An alternative is to use a bump with zero derivative at both endpoints:

\[
h(t)=t^2(1-t)^2.
\]

Then

\[
I_\psi(t,x_0,x_1)
=
x_0+t\Delta+h(t)A_\psi(x_0,x_1).
\]

This satisfies

\[
h(0)=h(1)=0,
\qquad
h'(0)=h'(1)=0.
\]

So the correction is concentrated in the middle of the path and does not alter endpoint velocities.

This variant is not literally second-order in \(t\), but it is still a single learned acceleration-mode correction. Use it as an ablation if the true quadratic version creates endpoint instability or very large endpoint velocities.

Recommended order:

1. Start with \(h(t)=t(1-t)\).
2. If unstable, test \(h(t)=t^2(1-t)^2\).
3. Do not begin with arbitrary learned \(h(t)\).

---

## 6. Higher-order controlled extensions

If the second-order correction is too weak, we may try a small polynomial basis:

\[
I_\psi(t)
=
x_0+t\Delta+\sum_{k=1}^K h_k(t)A_{\psi,k}(x_0,x_1),
\]

where every basis function satisfies

\[
h_k(0)=h_k(1)=0.
\]

A possible basis:

\[
h_1(t)=t(1-t),
\]

\[
h_2(t)=t(1-t)(2t-1),
\]

\[
h_3(t)=t(1-t)(2t-1)^2.
\]

This keeps the method restricted. Increasing \(K\) adds a small number of temporal acceleration modes; it does not turn the interpolant into a fully general neural function of \(t\).

The point of this hierarchy is:

\[
K=0:
\quad
\text{standard linear CFM};
\]

\[
K=1:
\quad
\text{single acceleration mode};
\]

\[
K=2,3:
\quad
\text{slightly richer but still restricted acceleration expansion}.
\]

Do not jump directly to a general interpolant

\[
I_\psi(t,x_0,x_1)
\]

unless the low-order hierarchy clearly fails and the project deliberately decides to reproduce the full learned-interpolant direction.

---

## 7. Model components

There are two learned components.

### 7.1 Acceleration network

The acceleration network predicts

\[
A_\psi(x_0,x_1)\in\mathbb R^d.
\]

For the first experiment, \(A_\psi\) should not depend on \(t\). If it depends on \(t\), the path becomes closer to a general learned interpolant and the interpretation becomes less clean.

The acceleration network is pair-dependent. This is crucial. Under independent coupling, each pair \((x_0,x_1)\) has a different displacement. The correction should be allowed to respond to that pair.

Possible input choices:

\[
(x_0,x_1),
\]

or

\[
(x_0,\Delta),
\qquad
\Delta=x_1-x_0.
\]

Use whichever is more natural in the existing playground. The conceptual requirement is only that the correction is pair-dependent and time-independent at the first stage.

### 7.2 Vector field network

The vector field network learns

\[
v_\theta(t,x).
\]

Unlike the direction-only method, this remains an ordinary Eulerian vector field. It does not receive the source label \(x_0\). This is important because we are trying to stay close to standard flow matching.

The vector field is trained on samples

\[
x_t=I_\psi(t,x_0,x_1)
\]

with velocity targets

\[
u_t=\dot I_\psi(t,x_0,x_1).
\]

---

## 8. Basic flow-matching loss

Given independent coupling:

\[
x_0\sim p_0,
\qquad
x_1\sim p_1,
\qquad
x_0\perp x_1,
\]

sample

\[
t\sim \mathrm{Unif}(0,1).
\]

Construct

\[
x_t=I_\psi(t,x_0,x_1),
\]

\[
u_t=\dot I_\psi(t,x_0,x_1).
\]

Train the vector field with

\[
\boxed{
\mathcal L_{\mathrm{FM}}(\theta,\psi)
=
\mathbb E
\left[
\|v_\theta(t,x_t)-u_t\|^2
\right].
}
\]

This is the natural CFM loss for the current interpolant.

However, this alone is not enough. If we only minimize \(\mathcal L_{\mathrm{FM}}\) over both \(\theta\) and \(\psi\), the acceleration network may learn degenerate paths that make the regression task easier without necessarily improving straightness or generation. Therefore, the training must be structured carefully.

---

## 9. Straightness objective

The desired reason for learning \(A_\psi\) is to make the induced marginal field straighter. The vector-field straightness condition is the inviscid Burgers residual:

\[
R_B(t,x)
=
\partial_t v_\theta(t,x)
+
D_xv_\theta(t,x)v_\theta(t,x).
\]

Define

\[
\boxed{
\mathcal L_{\mathrm{Burgers}}
=
\mathbb E
\left[
\|R_B(t,I_\psi(t,x_0,x_1))\|^2
\right].
}
\]

This measures whether the learned vector field has constant velocity along its own trajectories.

This is not the exact objective from the learned-interpolants paper, because we are not computing the exact induced optimal field \(v_\psi^*\). We are using the learned \(v_\theta\) as a tractable proxy. This is a deliberate simplification.

The total objective can be:

\[
\mathcal L
=
\mathcal L_{\mathrm{FM}}
+
\lambda_B\mathcal L_{\mathrm{Burgers}}
+
\lambda_A\mathcal L_{\mathrm{acc}},
\]

where

\[
\mathcal L_{\mathrm{acc}}
=
\mathbb E\|A_\psi(x_0,x_1)\|^2.
\]

The acceleration penalty is important because without it the interpolant may use extremely large corrections that leave the meaningful region of the data/noise interpolation and create pathological velocities.

---

## 10. Why the optimization is delicate

This experiment involves coupled optimization because \(\psi\) affects:

1. the training point \(x_t\);
2. the target velocity \(u_t\);
3. the distribution of points at which \(v_\theta\) is trained;
4. the points at which the Burgers residual is evaluated.

This is the simplified version of the bilevel issue in *Learning Straight Flows by Learning Curved Interpolants*.

The ideal objective would choose \(\psi\) so that the population-optimal vector field induced by \(I_\psi\),

\[
v_\psi^*(t,x)
=
\mathbb E[\dot I_\psi(t,x_0,x_1)\mid I_\psi(t,x_0,x_1)=x],
\]

is straight. But directly computing and differentiating through \(v_\psi^*\) is hard. Therefore we use \(v_\theta\) as an approximate learned proxy.

This means that training dynamics matter. If \(v_\theta\) has not yet fit the current interpolant, then the Burgers residual of \(v_\theta\) may not be a reliable signal for updating \(A_\psi\).

Therefore prefer staged or alternating training over naive fully joint optimization.

---

## 11. Recommended training schedule

### Stage 0: Baseline linear CFM

Train ordinary linear CFM:

\[
I_0(t)=x_0+t(x_1-x_0).
\]

Record:

- training loss;
- sample quality at multiple NFEs;
- trajectory curvature;
- Burgers residual;
- mode coverage or other playground metrics.

This baseline is essential. The acceleration interpolant must be compared against the same model and same independent coupling.

### Stage 1: Initialize acceleration near zero

Initialize \(A_\psi\) so that the interpolant is close to linear at the beginning:

\[
A_\psi(x_0,x_1)\approx 0.
\]

This prevents the early training dynamics from being dominated by arbitrary curved paths. The method should start as standard CFM and gradually learn useful acceleration.

### Stage 2: Fit vector field to current interpolant

For a number of steps, update only \(\theta\) using

\[
\mathcal L_{\mathrm{FM}}.
\]

Keep \(\psi\) fixed or update it very weakly. This lets the vector field track the current interpolant before using its Burgers residual to modify the interpolant.

### Stage 3: Alternating updates

Alternate between:

#### Vector-field update

Update \(\theta\) using

\[
\mathcal L_{\mathrm{FM}}
+
\lambda_B\mathcal L_{\mathrm{Burgers}}.
\]

This trains \(v_\theta\) to fit the current interpolant while also encouraging straightness.

#### Interpolant/acceleration update

Update \(\psi\) primarily using

\[
\mathcal L_{\mathrm{Burgers}}
+
\lambda_A\mathcal L_{\mathrm{acc}}.
\]

The goal of \(A_\psi\) is not merely to reduce the velocity regression error. The goal is to produce an interpolant whose learned marginal field is straighter.

This distinction is important. If \(\psi\) is optimized too aggressively through \(\mathcal L_{\mathrm{FM}}\), it may learn degenerate paths that make the current \(v_\theta\) look good without improving the final flow.

### Stage 4: Optional joint fine-tuning

Once both networks are stable, optionally fine-tune using the full loss:

\[
\mathcal L
=
\mathcal L_{\mathrm{FM}}
+
\lambda_B\mathcal L_{\mathrm{Burgers}}
+
\lambda_A\mathcal L_{\mathrm{acc}}.
\]

This stage should be treated as optional. The main interpretable result should come from the staged/alternating regime.

---

## 12. Stop-gradient recommendations

The agent should include stop-gradient ablations. The exact implementation details can be chosen according to the playground, but the conceptual recommendations are:

### Stop-gradient through target velocity when updating \(\psi\)

When updating \(\psi\), consider preventing the model from exploiting the target velocity

\[
u_t=\dot I_\psi
\]

to trivially reduce \(\mathcal L_{\mathrm{FM}}\).

This can be done by using \(\mathcal L_{\mathrm{FM}}\) mainly for \(\theta\), and using \(\mathcal L_{\mathrm{Burgers}}\) for \(\psi\).

### Stop-gradient through \(v_\theta\) when updating \(\psi\)

When updating \(\psi\), one can treat \(v_\theta\) as a fixed critic/diagnostic and move the interpolation points to reduce the measured Burgers residual. This approximates the idea that \(v_\theta\) is the current estimate of the induced marginal field.

This is not exact, but it may stabilize training.

### Compare against fully joint optimization

Run a small ablation where both \(\theta\) and \(\psi\) are updated jointly from the full loss. If this is unstable or degenerates, the staged approach is justified.

---

## 13. Degeneracy and sanity checks

The learned acceleration can cheat. The experiment should log and guard against the following.

### 13.1 Large acceleration

Log:

\[
\|A_\psi(x_0,x_1)\|.
\]

Also log relative acceleration magnitude:

\[
\frac{\|A_\psi(x_0,x_1)\|}{\|x_1-x_0\|+\varepsilon}.
\]

If this ratio becomes very large, the path may be leaving the intended interpolation regime.

### 13.2 Large path deviation

For

\[
I_\psi(t)=x_0+t\Delta+h(t)A_\psi,
\]

log:

\[
\|h(t)A_\psi\|.
\]

Compare it with:

\[
\|\Delta\|.
\]

The correction should not dominate unless that is explicitly being tested.

### 13.3 Large velocity target

The velocity is

\[
u_t=\Delta+h'(t)A_\psi.
\]

Log:

\[
\|u_t\|.
\]

If the learned acceleration creates large velocities, low-NFE sampling may worsen even if the Burgers residual decreases.

### 13.4 Endpoint correctness

Always verify:

\[
I_\psi(0)=x_0,
\qquad
I_\psi(1)=x_1.
\]

This should hold analytically if the schedule \(h\) satisfies endpoint conditions, but verify numerically.

### 13.5 Interpolant collapse

Check whether many different pairs produce similar intermediate points or whether the intermediate distribution becomes pathological. In low-dimensional toy settings, visualize intermediate samples.

### 13.6 Straightness without quality

The model may reduce Burgers residual but produce worse samples. This is not success. The target is improved few-step generation or trajectory geometry without sacrificing full-step quality.

---

## 14. Evaluation metrics

Use both standard generation metrics and geometry diagnostics.

### 14.1 Sample quality

Evaluate generated samples from integrating

\[
\frac{dX_t}{dt}=v_\theta(t,X_t)
\]

from \(t=0\) to \(t=1\).

Measure sample quality at multiple NFEs:

\[
\text{NFE}=1,2,4,8,16,\ldots
\]

The key test is low-NFE behavior. A straightening method should help few-step generation.

### 14.2 Full-NFE quality

Also evaluate at high NFE. If high-NFE quality collapses, the interpolant is not useful even if it improves some straightness metric.

### 14.3 Burgers residual

Measure:

\[
\mathbb E
\left[
\|
\partial_t v_\theta(t,X_t)
+
D_xv_\theta(t,X_t)v_\theta(t,X_t)
\|^2
\right].
\]

Evaluate this both on:

1. interpolant samples \(X_t=I_\psi(t,x_0,x_1)\);
2. generated ODE trajectories from the learned model.

The second is especially important because the learned sampler may visit regions different from the training interpolants.

### 14.4 Trajectory curvature

For generated trajectories, estimate curvature or direction change. For example, compare consecutive velocity directions along sampled ODE paths.

A simple diagnostic:

\[
1-
\frac{v(t_i,X_{t_i})\cdot v(t_{i+1},X_{t_{i+1}})}
{\|v(t_i,X_{t_i})\|\|v(t_{i+1},X_{t_{i+1}})\|+\varepsilon}.
\]

Lower values mean less turning.

### 14.5 Acceleration magnitude

Log the learned conditional acceleration:

\[
\|\ddot I_\psi\|.
\]

For the quadratic ansatz:

\[
\|\ddot I_\psi\|=2\|A_\psi\|.
\]

### 14.6 Comparison against linear CFM

Every metric must be compared to the linear CFM baseline under the same independent coupling, same architecture budget, and same training budget.

---

## 15. Baselines

At minimum compare:

1. **Linear CFM baseline**

\[
I(t)=x_0+t(x_1-x_0).
\]

2. **Quadratic acceleration interpolant**

\[
I_\psi(t)=x_0+t\Delta+t(1-t)A_\psi.
\]

3. **Endpoint-velocity-preserving bump**

\[
I_\psi(t)=x_0+t\Delta+t^2(1-t)^2A_\psi.
\]

4. **Higher-order small basis**, only if the second-order version is inconclusive.

Optional baselines:

5. Linear CFM plus Burgers regularizer, to test whether hard path modification beats soft vector-field regularization.

6. Ordinary learned-interpolant variant, if already available, to test whether the restricted method approaches the benefit of general learned interpolants.

---

## 16. What not to do initially

Do not implement a fully general interpolant

\[
I_\psi(t,x_0,x_1)
\]

as the first experiment. That would erase the distinction from the existing learned-interpolants paper.

Do not make \(A_\psi\) depend on \(t\) initially. Time dependence turns the method into a more general interpolant and weakens the interpretation.

Do not rely only on \(\mathcal L_{\mathrm{FM}}\) when training \(A_\psi\). The acceleration network exists to improve straightness, not merely to make velocity regression easier.

Do not optimize the exact moment identity involving

\[
p_t,\quad C_t,\quad \nabla\cdot(p_tC_t).
\]

Those quantities are too hard to estimate in high dimension. Use the identity as motivation, not as a direct loss.

Do not claim that this solves the bilevel problem exactly. It is a restricted surrogate.

---

## 17. Experimental hypotheses

### Hypothesis 1: Linear CFM lacks acceleration and therefore cannot compensate covariance-induced curvature

The standard linear interpolant has

\[
\ddot I=0.
\]

If curvature is partly caused by covariance-stress effects, then adding a learned acceleration term may reduce curvature.

### Hypothesis 2: A single acceleration mode may capture dominant curvature

The quadratic ansatz supplies:

\[
\ddot I=-2A_\psi(x_0,x_1).
\]

This may be enough to correct the dominant bending tendency of the marginal field.

### Hypothesis 3: Low-order acceleration may improve few-step generation

If trajectories become straighter, low-NFE generation should improve more than high-NFE generation.

### Hypothesis 4: If low-order acceleration fails, full learned interpolants may be necessary

A failure of the quadratic or low-order hierarchy is still informative. It means the covariance-stress structure is too complex for a single pair-dependent acceleration mode.

---

## 18. Success criteria

The experiment should be considered promising only if it shows at least one of the following under fair comparison with linear CFM:

1. better low-NFE sample quality;
2. lower trajectory curvature;
3. lower Burgers residual;
4. similar full-NFE quality but improved few-step quality;
5. interpretable learned acceleration patterns in toy settings.

The method should not be considered successful if:

1. it only reduces training loss but not sampling quality;
2. it requires huge acceleration magnitudes;
3. it performs no better than linear CFM with the same compute;
4. it only works when the acceleration network becomes so expressive that the method effectively becomes a general learned interpolant.

---

## 19. Recommended first experiment

Use independent coupling only.

Start with low-dimensional toy distributions where trajectories can be visualized.

Train:

\[
I_\psi(t)=x_0+t(x_1-x_0)+t(1-t)A_\psi(x_0,x_1).
\]

Use:

\[
\mathcal L_{\mathrm{FM}}
+
\lambda_B\mathcal L_{\mathrm{Burgers}}
+
\lambda_A\mathcal L_{\mathrm{acc}}.
\]

Compare against standard linear CFM.

Log:

- sample quality at multiple NFEs;
- Burgers residual;
- trajectory curvature;
- \(\|A_\psi\|\);
- velocity target magnitude;
- visualized paths in 2D.

If results are unstable, test:

\[
h(t)=t^2(1-t)^2.
\]

If results are weak but stable, test \(K=2\) polynomial basis.

If no clear benefit appears after \(K=2\) or \(K=3\), this direction should likely be stopped or treated as negative evidence.

---

## 20. Summary

The method to implement is:

\[
\boxed{
I_\psi(t,x_0,x_1)
=
x_0+t(x_1-x_0)+h(t)A_\psi(x_0,x_1).
}
\]

The first default is:

\[
h(t)=t(1-t).
\]

This is a low-order learned acceleration interpolant. It is motivated by the moment identity showing that conditional acceleration can, in principle, cancel covariance-induced curvature in the marginal vector field.

The purpose is to test whether a simple acceleration correction captures some of the benefit of learned curved interpolants without the complexity of learning a general interpolant.
