# ImbDiff-CM mechanism probe

**Status:** implementation complete; server measurements pending.

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
