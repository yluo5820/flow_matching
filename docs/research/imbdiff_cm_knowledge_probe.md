# ImbDiff-CM expert-knowledge probe

**Status:** spectrum-controlled K1/K2 result on the released-CM 60k seed-0
checkpoint, completed 2026-07-23.

This experiment asks what structure is present in the local response of the
trained CM expert branch. It does not repeat the paper's full-versus-general
output ablation. For every active official `Conv2d_LoRA` layer it records:

\[
e_l(x_t,t,y)=(B_lA_l)*h_{l-1},
\]

with dropout disabled. Compact normalized sketches describe the complete
response and fixed low-pass and high-pass components. Two-way cross-fit ridge
probes use disjoint held-out CIFAR-100 images to predict fine class, coarse
superclass, and Many/Medium/Few group. Class-conditioned response subspaces are
compared by projection overlap and principal angle.

The primary output remains on the server under:

```text
/root/autodl-tmp/runs/imbdiff_matrix60k/cm_knowledge_k1_k2_spectrum_controlled
```

Its manifest SHA-256 is
`1dac18e2a5805f42a55a07a44b05c7b39e63bf2d6b031c68c73e803960a2bf7f`.
It contains four held-out test images per fine class, timesteps 100, 500, and
900, all 27 active LoRA convolutions, and 32-dimensional sketches. The run
produced 32,400 response rows and 729 controlled linear-probe conditions in
approximately 90 seconds on one RTX PRO 6000 Blackwell Server Edition.

## Validity checks

The preceding six-class/four-layer smoke produced exactly the preregistered
response, probe, and subspace row counts. The primary all-class run also
produced its exact expected counts.

For every layer, the hook independently recomputed the general convolution and
the local expert convolution and checked:

\[
\operatorname{Conv}(h,W+BA)-\operatorname{Conv}(h,W)
\approx
\operatorname{Conv}(h,BA).
\]

The maximum relative RMS reconstruction error was `2.28624e-4`, or about
0.023%. The error is consistent with different CUDA convolution accumulation
paths and is small relative to the response; there is no sign that the hook is
capturing the wrong layer or branch.

Permutation-label and dimension/statistics-matched random-feature nulls are at
chance for all three linear-probe targets.

The fixed random adapter preserves every singular value of the learned
`lora_B @ lora_A` product while randomizing its left and right singular
subspaces. Across the 27 layers, the maximum relative singular-spectrum error
was `8.07e-7`, the maximum effective-weight RMS mismatch was `2.04e-7`, and
the learned/random product stable ranks agreed to numerical precision.

## K1 result: semantic information is strongly decodable

The table reports the median accuracy across all 27 layers and three timesteps,
followed by the best observed accuracy. Null values are means across the same
conditions.

| Target | Response component | Median | Best | Permutation null | Random-feature null |
| --- | --- | ---: | ---: | ---: | ---: |
| Fine class | Full | 0.430 | 0.858 | 0.010 | 0.010 |
| Fine class | Low-pass | 0.503 | 0.890 | 0.010 | 0.010 |
| Fine class | High-pass | 0.043 | 0.318 | 0.010 | 0.010 |
| Coarse superclass | Full | 0.418 | 0.688 | 0.050 | 0.050 |
| Coarse superclass | Low-pass | 0.455 | 0.680 | 0.050 | 0.050 |
| Coarse superclass | High-pass | 0.110 | 0.310 | 0.050 | 0.050 |
| Frequency group | Full | 0.504 | 0.784 | 0.333 | 0.334 |
| Frequency group | Low-pass | 0.532 | 0.782 | 0.334 | 0.334 |
| Frequency group | High-pass | 0.370 | 0.477 | 0.332 | 0.335 |

Fine- and coarse-class information is therefore not confined to a single
selected layer. It is present across much of the expert-response hierarchy.
The low-pass response is consistently at least as informative as the complete
response, while high-pass response is much weaker. Mean fine-class accuracy is
highest at timestep 500 (`0.576` full and `0.620` low-pass), rather than being
restricted to the final low-noise endpoint.

This contradicts the narrow hypothesis that the expert response primarily
contains late high-frequency texture correction. At this checkpoint, its
locally decodable organization is predominantly semantic and low-frequency.

Frequency-group accuracy is not independent evidence of frequency allocation:
in this CIFAR-100-LT construction, frequency group is a deterministic function
of fine-class identity. A response that identifies classes can consequently
identify frequency thirds without representing frequency as a separate factor.

## K2 result: response subspaces follow semantic hierarchy

Within-superclass class-pair subspace overlap exceeds across-superclass overlap
in all 27 adapted layers:

- mean within-minus-across overlap: `+0.02832`;
- median: `+0.02794`;
- layer range: `+0.01021` to `+0.05263`;
- positive layers: `27/27`.

The saved 200-permutation coarse-label control is positive in every layer; all
individual one-sided permutation p-values equal or improve on `0.00498`.

In contrast, the relationship between response-subspace overlap and absolute
log class-frequency distance is effectively zero:

- mean Spearman correlation across layers: `+0.00071`;
- median: `+0.00392`;
- range: `-0.0914` to `+0.0469`.

Thus the observed response geometry is organized by CIFAR-100 semantic
superclasses, not by proximity in training frequency.

## Tail-specific and spectral checks

The expert/general local RMS ratio is nearly identical across frequency groups.
The Few/Many ratio is:

| Timestep | Few / Many expert-to-general RMS |
| ---: | ---: |
| 100 | 1.0215 |
| 500 | 1.0155 |
| 900 | 1.0043 |

Average low- and high-frequency energy fractions are also nearly identical for
Many, Medium, and Few classes at each timestep. There is no evidence here for a
larger or selectively high-frequency tail response.

## Attribution controls: the semantics are not enriched by learned \(BA\)

The same normalized sketch and projection were applied to the learned expert,
general preactivation, spectrum-matched random-adapter response, and input
activation. Median cross-fit accuracies across the 81 layer-timestep
conditions were:

| Target | Band | Expert | General | Spectrum-matched random | Input |
| --- | --- | ---: | ---: | ---: | ---: |
| Fine class | Full | 0.4300 | 0.4950 | 0.5400 | 0.6425 |
| Fine class | Low-pass | 0.5025 | 0.5225 | 0.5375 | — |
| Fine class | High-pass | 0.0425 | 0.0525 | 0.0400 | — |
| Coarse superclass | Full | 0.4175 | 0.5050 | 0.5125 | 0.5500 |
| Coarse superclass | Low-pass | 0.4550 | 0.5175 | 0.5050 | — |
| Coarse superclass | High-pass | 0.1100 | 0.1400 | 0.1150 | — |

For fine-class full responses, the median paired expert-minus-random gap was
`-0.0825`, and expert exceeded random in only `5/81` conditions. For coarse
full responses, the median paired gap was `-0.0750`, and expert exceeded
random in `2/81` conditions. The high-pass responses were approximately tied
near their much lower accuracies.

K2 gives the same attribution result. The mean within-minus-across
superclass overlap was:

| Representation | Mean semantic overlap contrast |
| --- | ---: |
| Input activation | +0.04349 |
| General preactivation | +0.04281 |
| Spectrum-matched random response | +0.03449 |
| Learned expert response | +0.02832 |

All four representations were positive in all 27 layers and significant under
the saved per-layer permutation null. Learned expert exceeded the random
control in only `8/27` layers; its mean paired contrast was `-0.00617`.
Matching the full \(BA\) spectrum reduced the random-control advantage relative
to the preceding RMS-only control, but did not reverse it.

The convolution-kernel matrixization remains different from the official
\(BA\) matrixization. After reshaping to
`out_channels × (in_channels * kernel_area)`, median stable rank was `16.15`
for learned expert kernels and `26.08` for random kernels. This is an
observable learned orientation effect, not a mismatch in the official LoRA
factor capacity. It also did not explain the semantic-decoding gap:
layerwise Spearman correlations between this rank gap and the random-minus-
expert accuracy gap were `0.082` for fine-class full response and `-0.087`
for coarse-class full response (both nonsignificant). A future
functional-operator-spectrum control would nevertheless be stricter than the
current parameter-spectrum control.

## What can and cannot be concluded

The defensible observation is:

> Local CM expert responses contain strong fine-class and superclass structure,
> predominantly in their low-frequency component, but they do not enrich that
> structure relative to the same class-conditioned activation passed through a
> spectrum-matched random low-rank adapter. Their magnitude, spectral
> allocation, and subspace organization also show little relationship to class
> frequency.

This weakens a literal account in which \(BA\) is a separately readable store
of tail-semantic or high-frequency knowledge. It is more compatible with the
expert branch acting as a constrained correction whose usefulness only
emerges jointly with the general branch and the CM objective.

It does **not** show that \(BA\) is unimportant. The published ablation and our
60k reproduction establish that the complete CM system helps generation, while
K1/K2 only test locally readable information under normalized sketches. The
first K4 matched causal screen is now complete. It finds a small learned
expert-orientation benefit at late, low-noise \(t=100\), no benefit at
\(t=500/900\), and no tail-selective effect. The next decisive test is the
matched end-to-end sampling intervention: learned versus general-only versus
spectrum-random experts under identical labels and initial noise.

**Control implementation status:** these controls are now implemented in output
schema 3. The fixed random adapter preserves the complete singular spectrum of
the learned `lora_B @ lora_A` product, including its factor rank and Frobenius
norm, while independently randomizing the left and right singular subspaces.
It is batch-response-RMS matched before sketching. Expert, general, and random
output responses use the same deterministic sketch projection.
`summary.json` audits the product-space stable ranks and maximum
spectrum-matching error. The spectrum-controlled 60k rerun above is complete;
the observations remain attributed to local responses unless a causal
intervention specifically supports a weight-level claim.
