# Official ImbDiff-CM reproduction

> **Archived:** the CM-specific implementation was removed on 2026-07-24. This
> document preserves the completed research record; its commands are not active.


This path runs the authors' released implementation as a pinned git submodule
instead of using this repository's continuous-flow adaptation.

The goal is diagnostic: determine whether the official discrete DDPM code path
can produce the expected CIFAR-100-LT behavior under our available AutoDL budget.
If this fails under the authors' implementation, the current failure is not
specific to our continuous-flow rewrite.

## What is pinned

The official repository is added as a submodule:

```text
third_party/ImbDiff-CM
```

The upstream CIFAR-100-LT IR100 configs are:

```text
third_party/ImbDiff-CM/configs/cifar100lt_ir100/oc.yaml
third_party/ImbDiff-CM/configs/cifar100lt_ir100/cm.yaml
```

They use the paper-scale budget:

```yaml
training:
  batch_size: 64
  total_steps: 300001
  warmup: 5000
  sample_step: 10000
  save_step: 100000

evaluation:
  num_images: 50000
```

The committed AutoDL screen configs keep the official model, diffusion, transfer,
and CM definitions, but change server paths plus the budget:

```text
configs/official_imbdiff_cm/autodl_screen/oc_cifar100lt_ir100_screen30k.yaml
configs/official_imbdiff_cm/autodl_screen/cm_cifar100lt_ir100_screen30k.yaml
```

The screen uses `total_steps: 30001`, not `30000`, because the official training
loop iterates over `range(total_steps)` and only writes `ckpt_30000.pt` if step
30000 is included.

## Setup on AutoDL

From the main repository checkout:

```bash
cd ~/flow_matching
git submodule update --init --recursive
conda activate fm_lab_cuda
python -m pip install -r third_party/ImbDiff-CM/requirements.txt
```

The official implementation uses torchvision's CIFAR-100 Python layout, so the
data disk should contain:

```text
/root/autodl-tmp/data/cifar100/cifar-100-python/train
/root/autodl-tmp/data/cifar100/cifar-100-python/test
```

If needed, extract AutoDL's public archive:

```bash
mkdir -p /root/autodl-tmp/data/cifar100
tar -xzf /root/autodl-pub/cifar-100/cifar-100-python.tar.gz \
  -C /root/autodl-tmp/data/cifar100
```

## Quick import smoke

```bash
cd ~/flow_matching/third_party/ImbDiff-CM
python tools/train.py --help
python tools/sample_images.py --help
python tools/extract_features.py --help
python tools/compute_metrics.py --help
```

## Train OC and CM screen runs

Run from the official submodule root:

```bash
cd ~/flow_matching/third_party/ImbDiff-CM
```

OC:

```bash
python tools/train.py \
  --config ../../configs/official_imbdiff_cm/autodl_screen/oc_cifar100lt_ir100_screen30k.yaml \
  --device cuda:0
```

CM:

```bash
python tools/train.py \
  --config ../../configs/official_imbdiff_cm/autodl_screen/cm_cifar100lt_ir100_screen30k.yaml \
  --device cuda:0
```

Expected checkpoints:

```text
/root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/oc/ckpt_30000.pt
/root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/cm/ckpt_30000.pt
```

## Sample 10k images per run

OC:

```bash
python tools/sample_images.py \
  --config ../../configs/official_imbdiff_cm/autodl_screen/oc_cifar100lt_ir100_screen30k.yaml \
  --ckpt /root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/oc/ckpt_30000.pt \
  --output_dir /root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/oc/revised_gen_images-ckpt_step-30000 \
  --num_images 10000 \
  --device cuda:0
```

CM:

```bash
python tools/sample_images.py \
  --config ../../configs/official_imbdiff_cm/autodl_screen/cm_cifar100lt_ir100_screen30k.yaml \
  --ckpt /root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/cm/ckpt_30000.pt \
  --output_dir /root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/cm/revised_gen_images-ckpt_step-30000 \
  --num_images 10000 \
  --device cuda:0
```

The output images are balanced by class: 100 images/class for 10,000 total
images.

## Optional FID/KID evaluation

The authors' evaluator expects the FID Inception checkpoint at:

```text
third_party/ImbDiff-CM/stats/pt_inception-2015-12-05-6726825d.pth
```

Install it from the PyTorch-FID release:

```bash
cd ~/flow_matching/third_party/ImbDiff-CM
mkdir -p stats
curl -L \
  -o stats/pt_inception-2015-12-05-6726825d.pth \
  https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
```

Extract real CIFAR-100 train features once:

```bash
python tools/extract_features.py \
  --mode real \
  --data_root /root/autodl-tmp/data/cifar100 \
  --feature_dir /root/autodl-tmp/runs/official_imbdiff_cm/features \
  --batch_size 512 \
  --device cuda:0
```

Extract OC generated features:

```bash
python tools/extract_features.py \
  --mode generated \
  --name OC-cifar100-100-screen30k \
  --image_dir /root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/oc/revised_gen_images-ckpt_step-30000 \
  --feature_dir /root/autodl-tmp/runs/official_imbdiff_cm/features \
  --num_images 10000 \
  --batch_size 512 \
  --device cuda:0
```

Extract CM generated features:

```bash
python tools/extract_features.py \
  --mode generated \
  --name CM-cifar100-100-screen30k \
  --image_dir /root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/cm/revised_gen_images-ckpt_step-30000 \
  --feature_dir /root/autodl-tmp/runs/official_imbdiff_cm/features \
  --num_images 10000 \
  --batch_size 512 \
  --device cuda:0
```

Compute metrics:

```bash
python tools/compute_metrics.py \
  --feature_dir /root/autodl-tmp/runs/official_imbdiff_cm/features \
  --generated_prefix OC-cifar100-100-screen30k \
  --overall_samples 10000 \
  --output /root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/oc/metrics_screen30k.json

python tools/compute_metrics.py \
  --feature_dir /root/autodl-tmp/runs/official_imbdiff_cm/features \
  --generated_prefix CM-cifar100-100-screen30k \
  --overall_samples 10000 \
  --output /root/autodl-tmp/runs/official_imbdiff_cm/cifar100lt_ir100_screen30k/cm/metrics_screen30k.json
```

If the FID weight download is slow, skip this initially and inspect the saved
sample grids under each run's `sample/` directory plus the generated PNG folder.

## Interpretation

This 30k screen is not a paper-number reproduction. It answers a narrower
question: with the official discrete DDPM implementation and our available
compute, do samples become recognizable and does CM trend better than OC?

If the official 30k screen produces recognizable images while our continuous
flow screen does not, the blocker is likely in our continuous objective/sampling
adaptation. If the official screen also produces poor images, the 30k budget is
probably too small for this specific CIFAR-100-LT setup.

## 2026-07-21 screen result and local CM debugging implication

The 30k official screen did produce recognizable CIFAR-like samples for both OC
and CM. The samples were still artifacted, and CM was not visually cleanly better
than OC at this reduced budget, but the failure mode was fundamentally different
from the local continuous-flow `source`-loss screen, which produced saturated
high-frequency noise.

That result makes the local continuous recreation the main suspect.

The first CM-specific mismatch found locally was the sampled capacity branch.
In the official CM sampler, all reverse diffusion calls use:

```python
use_cm=False
```

The official training loss computes `h1 = model(..., use_cm=True)` for the base
noise prediction loss, but it samples with `h2 = model(..., use_cm=False)`. The
capacity-off branch is therefore the deployed branch whose behavior is shaped by
the weighted CM distance term.

Before the local fix, this repository sampled capacity-enabled CM models without
passing any explicit capacity context. The local `DDPMUNet` default is
`use_capacity=True`, so CM checkpoints were sampled from the opposite branch from
the official implementation. Local sampling now resolves:

```yaml
sampling:
  capacity_branch: auto
```

to `base` / `use_capacity: false` when the objective includes a CM modifier.
Use `sampling.capacity_branch: full` only for an explicit ablation of the
capacity-on branch.

There is also a separate continuous-vs-DDPM mismatch in source/noise prediction.
The local paper-close configs used:

```yaml
path:
  name: linear
objective:
  model_output: source
  loss_space: source
```

For a forward ODE sampler from source to data, this is ill-conditioned at the
source endpoint. At `t = 0`, the correct source/noise prediction is simply the
current initial noise, so converting that prediction into a velocity gives a
zero or denominator-clamped field. DDPM epsilon prediction does not have this
same sampling semantics: it is consumed by a discrete reverse diffusion sampler
with a variance schedule and posterior update. Therefore, source/noise
prediction on the local continuous linear path is not a faithful DDPM
reformulation.

If the branch fix is not sufficient, the next CM debugging target is the loss
space. Official CM computes the capacity distance directly in native epsilon
output space:

```python
MSE(h2, h1)
```

The continuous implementation may compare converted target/source/velocity
values. That is not necessarily wrong, but it changes the time weighting and
conditioning of the CM distance term, so it should be tested as a separate
ablation rather than assumed equivalent.
