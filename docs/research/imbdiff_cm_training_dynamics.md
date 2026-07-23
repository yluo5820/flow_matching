# ImbDiff-CM live training-dynamics experiment

**Status:** instrumentation and 2k CUDA smoke validated; 30k mechanism run
prepared.

## Scientific question

The completed checkpoint and sampling interventions show that the explicit CM
expert improves generation, but does not behave as a clean tail-knowledge
store. They cannot reveal how that endpoint was reached.

This experiment observes the faithful released-CM optimizer stream directly:

> Which loss components and class-frequency groups update the general weights,
> the LoRA factors \(A/B\), and the effective expert kernel \(BA\); how does
> Adam transform those gradients; and what functional and spatial-frequency
> changes result from one optimizer step?

It is designed to distinguish transient capacity allocation from generic
regularization and LoRA/Adam parameterization effects.

## Fidelity contract

At preregistered steps, the observer retains the graph created by the ordinary
released-CM training call. Component and class-conditioned gradients therefore
use the exact:

- CIFAR-100-LT minibatch and augmentations;
- Gaussian noise and discrete diffusion timesteps;
- OC endpoint-transfer draw;
- independent dropout masks from the two official forward passes;
- loss reduction that feeds the optimizer.

The observer uses `torch.autograd.grad`, which does not write `parameter.grad`.
The normal backward, clipping, Adam step, scheduler, and EMA update then run
unchanged. A controlled test verifies that the instrumented and uninstrumented
Adam steps produce identical parameters.

Pre/post functional measurements use the same live noisy batch, timesteps, and
conditioning, but disable dropout for both evaluations. This deliberately
isolates the deterministic effect of the parameter update rather than mixing
it with a new dropout mask.

Compiled training is rejected for this experiment. The faithful matrix already
uses compilation off.

## Measurements

### Live loss-component gradients

For

\[
L_{\mathrm{base}},\quad L_{\mathrm{con}},\quad L_{\mathrm{div}},
\quad L_{\mathrm{CM}},\quad L_{\mathrm{total}},
\]

record global gradient norm, parameter-normalized RMS, and expert-gradient
energy fraction for:

- all general parameters;
- all \(A\) factors;
- all \(B\) factors;
- \(A\) and \(B\) jointly.

On five representative adapted layers, also retain componentwise \(A/B\)
gradient norms, scale-stabilized sensitivities
\(\lVert A\rVert\lVert\nabla_A L\rVert\) and
\(\lVert B\rVert\lVert\nabla_B L\rVert\), the induced effective-kernel
direction, and pairwise component-gradient cosine/conflict.

### Class and diffusion-time contributions

The total per-sample objective is masked on the live graph and divided by the
original batch size. The resulting gradients are literal additive
contributions to that optimizer batch, not gradients from replayed balanced
batches. Each row stores both this exposure-weighted contribution and the
corresponding group-mean gradient obtained by dividing by its observed batch
fraction. The former describes what the optimizer actually receives; the
latter helps compare per-example pressure across unequal-frequency groups.

They are recorded for:

- Many, Medium, and Few classes;
- late/low-noise, middle, and early/high-noise timestep thirds.

For selected layers, their effective expert-gradient directions are accumulated
over observation steps. Head-tail and time-stratum alignment is measured on
the selected adapted capacity.

Whole-batch classifier-free dropout is retained. Rows from an unconditional
batch are marked; their coefficient remains label-dependent while their model
prediction is unconditional and must be interpreted separately.

### Raw factor and Adam dynamics

Before and after each observed Adam step, record:

\[
\Delta W_e=B_{s+1}A_{s+1}-B_sA_s
\]

and verify:

\[
\Delta W_e
=B_s\Delta A+\Delta B A_s+\Delta B\Delta A.
\]

The output includes:

- raw \(A/B\) gradients and parameter updates;
- scale-stabilized sensitivities;
- effective expert and corresponding general-kernel update norms;
- general/expert update cosine;
- cosine between raw gradient descent in effective-weight space and the
  realized Adam update;
- effective-kernel rank, spectral norm, Frobenius norm, nuclear norm, and
  stable rank;
- raw and EMA update magnitudes;
- net displacement and sparse-observation chord path length.

The chord path joins observation points and is a lower bound on the complete
stepwise optimizer path.

### Functional and spatial-frequency update

On the same noisy batch, evaluate full and general-only predictions immediately
before and after Adam:

\[
\Delta f_{\mathrm{full}}=f_{\theta_{s+1}}-f_{\theta_s},
\]

\[
\Delta f_g=f_{\theta^g_{s+1}}-f_{\theta^g_s},
\]

\[
\Delta f_e
=
\Delta f_{\mathrm{full}}-\Delta f_g.
\]

For all, class-frequency, and diffusion-time scopes, record RMS, direction
cosines, and four radial Fourier-energy bands. Here \(\Delta f_e\) means the
change in the full/general functional gap; it remains a nonlinear,
stage-composed expert effect rather than an assumption of additive networks.

## Artifacts

Each run writes incrementally under `cm_dynamics/`:

| File | Content |
| --- | --- |
| `manifest.json` | schedule, selected layers, fidelity and interpretation boundaries |
| `gradient_components.csv` | global loss-component gradients |
| `layer_gradient_components.csv` | selected-layer \(A/B\) component gradients |
| `gradient_alignments.csv` | selected-layer loss-component conflicts |
| `conditioned_gradients.csv` | global frequency/time contributions |
| `conditioned_layer_gradients.csv` | cumulative selected-layer effective directions |
| `conditioned_alignments.csv` | head-tail and timestep-stratum alignment |
| `layer_updates.csv` | realized Adam, \(BA\), general-weight, EMA, and rank dynamics |
| `functional_updates.csv` | full/general/expert-effect RMS and Fourier spectra |
| `summary.json` | completion and observation-step audit |

## Preregistered interpretation

| Observation | Interpretation |
| --- | --- |
| Few induces a larger normalized expert contribution and a stable cumulative expert direction, without the same general-weight effect | evidence for tail-directed allocation during training |
| Many dominates exposure-weighted general and expert paths | frequency exposure overwhelms the proposed tail routing |
| class groups mainly conflict in the general branch while the expert admits a coherent Few direction | low rank may protect tail updates from shared-capacity conflict |
| expert gradients are present but Adam \(BA\) updates are small or poorly aligned | product parameterization/optimizer suppresses their functional action |
| \(BA\) updates are substantial but \(\Delta f_e\) is small | expert changes are absorbed by nonlinear/shared representation geometry |
| \(\Delta f_g\) dominates while the final model still improves | generic shared-branch regularization is the stronger account |
| early \(B\)-only learning later becomes balanced \(A/B\) learning | predicted zero-\(B\) LoRA optimization phase |
| tail-specific high-frequency \(\Delta f_e\) develops reproducibly | support for residual tail-refinement, pending a later causal test |

No single-step gradient norm establishes stored knowledge. A mechanism requires
a stable temporal pattern linked to the already measured endpoint behavior.

## Configurations

The 2k smoke configuration is:

```text
configs/cifar100_lt/autodl_dynamics/
  cifar100_lt_ir100_official_released_cm_dynamics_smoke2k.yaml
```

It observes 13 logarithmically spaced steps from 1 through 2,000. Its purpose
is to validate CUDA memory, runtime overhead, file schemas, initialization
asymmetry, and update reconstruction—not to establish late-training dynamics.

The promoted 30k mechanism configuration is:

```text
configs/cifar100_lt/autodl_dynamics/
  cifar100_lt_ir100_official_released_cm_dynamics_30k.yaml
```

It observes nine logarithmically spaced initialization steps through 256,
followed by every 500 steps through 30,000 (69 observations total). This denser
late schedule is necessary because a batch from CIFAR-100-LT IR100 can contain
zero Few examples; isolated observations cannot support stable tail-routing
claims. The 2k smoke has already validated overhead and numerical correctness.
