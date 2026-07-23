# ImbDiff-CM expert-knowledge probe

**Status:** preliminary K1/K2 result on the released-CM 60k seed-0 checkpoint,
completed 2026-07-23.

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
/root/autodl-tmp/runs/imbdiff_matrix60k/cm_knowledge_k1_k2
```

Its manifest SHA-256 is
`1dac18e2a5805f42a55a07a44b05c7b39e63bf2d6b031c68c73e803960a2bf7f`.
It contains four held-out test images per fine class, timesteps 100, 500, and
900, all 27 active LoRA convolutions, and 32-dimensional sketches. The run
produced 32,400 response rows and 729 linear-probe conditions in approximately
30 seconds on one RTX PRO 6000 Blackwell Server Edition.

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

## K1 result: semantic information is strongly decodable

The table reports the median accuracy across all 27 layers and three timesteps,
followed by the best observed accuracy. Null values are means across the same
conditions.

| Target | Response component | Median | Best | Permutation null | Random-feature null |
| --- | --- | ---: | ---: | ---: | ---: |
| Fine class | Full | 0.438 | 0.885 | 0.010 | 0.010 |
| Fine class | Low-pass | 0.490 | 0.890 | 0.010 | 0.010 |
| Fine class | High-pass | 0.040 | 0.295 | 0.010 | 0.010 |
| Coarse superclass | Full | 0.417 | 0.635 | 0.051 | 0.051 |
| Coarse superclass | Low-pass | 0.455 | 0.670 | 0.049 | 0.050 |
| Coarse superclass | High-pass | 0.118 | 0.305 | 0.050 | 0.049 |
| Frequency group | Full | 0.519 | 0.761 | 0.335 | 0.335 |
| Frequency group | Low-pass | 0.535 | 0.759 | 0.334 | 0.335 |
| Frequency group | High-pass | 0.373 | 0.455 | 0.334 | 0.333 |

Fine- and coarse-class information is therefore not confined to a single
selected layer. It is present across much of the expert-response hierarchy.
The low-pass response is consistently at least as informative as the complete
response, while high-pass response is much weaker. Mean fine-class accuracy is
highest at timestep 500 (`0.578` full and `0.624` low-pass), rather than being
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

- mean within-minus-across overlap: `+0.02622`;
- median: `+0.02578`;
- layer range: `+0.00769` to `+0.04774`;
- positive layers: `27/27`.

An exploratory 1,000-permutation coarse-label control, computed from the saved
`subspace_pairs.npz`, gives an across-layer null standard deviation of
`0.000679` and empirical one-sided `p=0.000999`. Every individual layer is
positive; all individual permutation p-values are below `0.028`.

In contrast, the relationship between response-subspace overlap and absolute
log class-frequency distance is effectively zero:

- mean Spearman correlation across layers: `-0.00191`;
- median: `-0.00118`;
- range: `-0.0951` to `+0.0534`.

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

## What can and cannot be concluded

The defensible observation is:

> Local CM expert responses contain strong fine-class and superclass structure,
> predominantly in their low-frequency component, while their magnitude,
> spectral allocation, and subspace organization show little relationship to
> class frequency.

This weakens both a tail-only capacity-separation account and a
high-frequency-detail account. It is compatible with semantic knowledge
transfer or semantic residual refinement.

It does **not** yet prove that the trained \(BA\) weights themselves store this
semantic information. The response \(e_l=BAh_{l-1}\) is a deterministic
function of the general activation \(h_{l-1}\), which is already
class-conditioned. Even a class-agnostic fixed projection can preserve
linearly decodable class information. The current random-feature null tests
whether arbitrary independent features can predict labels; it is not a
matched transformation of the same input activation.

Before repeating K1/K2 across checkpoints or interpreting \(\theta_e\) as a
semantic memory, the probe should add:

1. an input/general-activation sketch at every adapted layer;
2. a fixed random low-rank convolution of the same activation, matched in
   output shape, effective rank, and response RMS;
3. decoding and subspace comparisons for expert response versus both controls;
4. a coarse-label permutation null directly in the saved K2 report.

Only expert selectivity beyond the activation and matched-random-adapter
controls can be attributed to the learned expert transformation. Stable
directions would then be candidates for K3 trajectory projection and K4 causal
intervention.

**Control implementation status:** these controls are now implemented in output
schema 3. The fixed random adapter preserves the complete singular spectrum of
the learned `lora_B @ lora_A` product, including its factor rank and Frobenius
norm, while independently randomizing the left and right singular subspaces.
It is batch-response-RMS matched before sketching. Expert, general, and random
output responses use the same deterministic sketch projection.
`summary.json` audits the product-space stable ranks and maximum
spectrum-matching error. The spectrum-controlled 60k rerun remains pending; the
observations above deliberately remain attributed to the response rather than
the expert weights.
