# ImbDiff-CM mechanism probe

**Status:** completed on the 60k seed-0 matrix on 2026-07-22.

The 60k comparison establishes that the released CM loss is effective in our
controlled matrix. This follow-up asks a narrower mechanistic question: **what
does the CM loss make the learned expert branch do, and how does that behavior
depend on class frequency?** It is a checkpoint probe, not another generative
quality benchmark and not yet a proposed replacement loss.

## Why this is the next experiment

The paper already studies the CM coefficient, LoRA rank, removal of the
consistency and diversity losses, use of the general branch alone, and several
U-Net sizes. Repeating those ablations at 60k would provide little new causal
resolution. Our missing evidence is whether the trained low-rank branch really
acts as a tail-specialized correction, which layers carry that correction, and
whether it changes coarse structure or high-frequency detail.

The primary controlled contrast is:

- `oc_capacity_only`: identical CM-capable U-Net and OC base objective, but CM
  loss weights are zero;
- `released_cm`: the same architecture and base objective with the released CM
  consistency/diversity terms enabled.

`pure_cm` is a secondary reference because it produced the best 60k FID but
removes endpoint transfer and therefore changes the base objective.

## Paired probe contract

The command builds one immutable manifest from the balanced CIFAR-100 test
split. By default it selects one held-out image per class and probes discrete
timesteps 50, 250, 500, 750, and 950. Every checkpoint receives exactly the
same images, labels, CPU-generated noise, timesteps, and endpoint-transfer
seeds. The manifest is saved with a content digest. EMA weights are used unless
explicitly overridden.

For each checkpoint and timestep, the official objective is decomposed without
changing its algebra. With the full prediction denoted by
`eps_full = eps_general + eps_expert`, the capacity-off prediction is
`eps_general`. The probe records:

- base denoising MSE with and without the expert branch, and their difference;
- the prediction correction `eps_expert = eps_full - eps_general`;
- the corresponding displacement of the implied clean image `x0`;
- raw consistency loss, signed diversity term, and their weighted CM
  contribution;
- exact gradients of the base, consistency, signed diversity, CM, and total
  objectives with respect to the general and expert parameter groups;
- gradient norms, parameter-count-normalized RMS, and pairwise cosine
  similarities, both for all expert parameters and for every LoRA layer;
- radial Fourier energy fractions of the expert prediction correction and
  implied-`x0` displacement in four fixed bands.

Gradients are summarized separately for Many, Medium, and Few classes. Batched
vector-Jacobian products compute the three frequency groups together, so each
timestep needs three reverse passes (base, consistency, and diversity) rather
than one reverse pass per group and loss.

## Interpretation before seeing the result

Evidence for the paper's intended allocation mechanism would have all of the
following features:

1. `released_cm` has a larger Few-class expert RMS or expert/general gradient
   ratio than `oc_capacity_only`, without the same increase for Many classes.
2. Turning the expert branch on improves Few-class denoising MSE more than it
   improves Many-class MSE.
3. The diversity component supplies expert-directed gradients that are distinct
   from, or oppose, the consistency component, instead of merely rescaling the
   base gradient.
4. The effect emerges across 20k, 40k, and 60k rather than appearing only in a
   single final checkpoint.

If CM improves FID but these signatures are absent, the capacity-allocation
story is incomplete: the objective may help through regularization, changed
optimization, or a more distributed effect. If the expert correction is
frequency selective, that is a concrete observation that can motivate a later
frequency-conditioned CM intervention. A high-frequency correction alone is
not evidence of useful specialization; it must also align with denoising gain
and the Many/Medium/Few contrast.

This experiment does not identify intrinsic dimension or natural-image
manifold geometry. Fourier bands are a descriptive probe of the learned
correction, not a claim that spatial frequency is the unique relevant
geometry.

## Completed result

All nine EMA checkpoints completed with the same 100 held-out images, five
timesteps, noise rows, and OC transfer draws. The manifest SHA-256 is
`509b47cc6a136f3010cc0e87d413ffa7593162996a7b00ded134f7bfe478a1ad`.
The raw structured outputs remain under
`/root/autodl-tmp/runs/imbdiff_matrix60k/cm_mechanism_probe/`.

The primary 60k functional result is:

| Group | Capacity-only distance | Released-CM distance | Capacity-only expert MSE gain | Released-CM expert MSE gain | Capacity-only high-frequency fraction | Released-CM high-frequency fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Many | 0.002422 | 0.00001022 | 0.003043 | 0.00002656 | 0.1964 | 0.1373 |
| Medium | 0.002435 | 0.00001105 | 0.003083 | 0.00002052 | 0.1980 | 0.1351 |
| Few | 0.002421 | 0.00001110 | 0.002776 | 0.00000474 | 0.1973 | 0.1320 |

Across every paired row at 20k, 40k, and 60k, released CM had a smaller
squared capacity-on/off distance than the capacity-only control. Its distance
was only 0.40%--0.48% of the control, corresponding to an approximately
15-times smaller RMS correction. Released CM also had a smaller pointwise
expert MSE gain on more than 99% of paired rows. The full capacity-on denoising
MSE was nevertheless almost identical between the two methods at 60k. Thus
CM's large FID improvement is not explained by a larger expert correction or
better held-out one-step denoising.

The correction is not tail-specialized in the preregistered functional sense.
At 60k its squared magnitude is only 8.6% larger for Few than Many classes,
while its MSE benefit is 82% smaller for Few. Its Few-class correction also has
a slightly lower, not higher, high-frequency fraction. `pure_cm` reproduces
essentially the same local pattern despite its better generative FID.

The gradient result needs two levels of interpretation:

- At 20k, about 75% of the **CM-only gradient energy** lies in LoRA parameters,
  although LoRA contains only about 8% of the probed parameters. CM therefore
  does directly act on the expert branch.
- The CM-only gradient is already small compared with the base gradient, and
  its expert energy fraction falls to roughly 13%--19% by 60k as the branch
  contracts. Under released CM, the **total-objective expert gradient energy**
  is only about 0.01%--0.11% over the observed checkpoints and is not larger
  for Few classes (the capacity-only control spans about 0.06%--0.16%).
- Consistency and diversity gradients are nearly opposite because both are
  the same branch-distance gradient multiplied by class-dependent scalars.
  They do not independently select general and expert parameter subsets.

The released coefficient makes the imbalance more explicit. For class
probability `p_y` and normalized inverse probability `q_y`, the per-sample CM
coefficient is `100 * (p_y - 0.2 q_y)`. In this IR100 split:

| Group | Training sample mass | Mean per-class coefficient | Exposure-weighted signed coefficient mass |
| --- | ---: | ---: | ---: |
| Many | 0.8041 | +2.3440 | +2.2704 |
| Medium | 0.1620 | +0.3907 | +0.0815 |
| Few | 0.0339 | -0.3815 | -0.0091 |

Consequently, the observed training distribution supplies roughly 250 times
more signed head consistency mass than tail diversity mass. This matches the
global contraction seen in the functional probe. The defensible mechanism at
this scale is therefore **strong branch-consistency regularization with a weak
tail counterforce**, not a clean routing of head updates into general weights
and tail updates into reserved expert weights. The CM improvement remains
real; this result narrows what can explain it to cumulative regularization,
optimization of the shared branch, or a small correction whose rollout effect
is not captured by one-step MSE.

The most direct next intervention is an exposure-balanced signed coefficient:
center the expected CM coefficient under the actual training distribution,
then rescale its expected absolute magnitude to match the released loss. This
tests the mechanism without reverse labels or shuffled class frequencies and
is more targeted than another unrestricted weight sweep. Multiple held-out
manifest seeds should precede training, although the roughly 200-fold squared-
distance effect is too large and consistent to be plausibly caused by the
single-image-per-class choice alone.

## Server commands

First run a cheap functional-only smoke check on one final checkpoint:

```bash
cd /root/flow_matching
conda activate fm_lab_cuda

python -m fm_lab.experiments.run_imbdiff_cm_probe \
  --run-dir /root/autodl-tmp/runs/imbdiff_matrix60k/released_cm \
  --checkpoint-steps 60000 \
  --output-dir /root/autodl-tmp/runs/imbdiff_matrix60k/cm_probe_smoke \
  --device cuda --weights ema --timesteps 500 \
  --samples-per-class 1 --mixed-precision auto \
  --channels-last on --functional-only
```

Then run the preregistered primary and secondary comparisons:

```bash
python -m fm_lab.experiments.run_imbdiff_cm_probe \
  --run-dir /root/autodl-tmp/runs/imbdiff_matrix60k/oc_capacity_only \
  --run-dir /root/autodl-tmp/runs/imbdiff_matrix60k/released_cm \
  --run-dir /root/autodl-tmp/runs/imbdiff_matrix60k/pure_cm \
  --checkpoint-steps 20000,40000,60000 \
  --output-dir /root/autodl-tmp/runs/imbdiff_matrix60k/cm_mechanism_probe \
  --device cuda --weights ema \
  --timesteps 50,250,500,750,950 \
  --samples-per-class 1 --seed 20260722 \
  --mixed-precision auto --channels-last on
```

The output directory contains `manifest.json`, `summary.json`, checkpoint-level
JSON files, `functional_rows.csv`, `gradient_summary.csv`, and a compact
`report.md`. Checkpoint results are written incrementally, so completed work is
retained if a later checkpoint fails.
