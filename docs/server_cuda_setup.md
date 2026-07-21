# CUDA server setup

This is the recommended setup for running the CIFAR/Fashion-MNIST long-tail
experiments on a Linux server whose default environment is Miniconda.

The environment file intentionally does not install PyTorch. Install the CUDA
PyTorch wheel first, then install this repository. That avoids the common failure
mode where a CPU PyTorch wheel is installed into an otherwise valid CUDA server
environment.

## 1. Create the Miniconda environment

Run from the repository root:

```bash
conda env create -f environment.server.yml
conda activate fm_lab_cuda
python -m pip install --upgrade pip
```

If the environment already exists:

```bash
conda env update -f environment.server.yml --prune
conda activate fm_lab_cuda
python -m pip install --upgrade pip
```

## 2. Install PyTorch for the selected server GPU

Default choice for recent drivers and RTX 50-series / Blackwell nodes:

```bash
python -m pip install -r requirements/torch-cuda-default.txt
```

Older-driver fallback, usually appropriate for Ada/Hopper nodes when the default
wheel reports a driver/runtime mismatch:

```bash
python -m pip install -r requirements/torch-cu126.txt
```

If both fail, use the official PyTorch selector at
<https://pytorch.org/get-started/locally/> for the server's driver and CUDA
runtime, then keep the rest of this document unchanged.

## 3. Install this project

Minimal experiment install:

```bash
python -m pip install -e ".[dev]"
```

Optional diagnostics extras for DINO/UMAP/Streamlit-style geometry work:

```bash
python -m pip install -e ".[image-diagnostics,image-embeddings]"
```

## 4. Verify CUDA and BF16 support

```bash
python - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
```

For our new training acceleration flags, BF16 is preferred when supported.
Otherwise `--mixed-precision auto` falls back to FP16 with a gradient scaler.

## 5. Stage CIFAR data on AutoDL

AutoDL exposes common datasets under `/root/autodl-pub`, but that location is
read-only and slower than the local data disk. Copy or extract CIFAR to
`/root/autodl-tmp` before training:

```bash
mkdir -p /root/autodl-tmp/data/cifar10
mkdir -p /root/autodl-tmp/data/cifar100

tar -xzf /root/autodl-pub/cifar-10/cifar-10-binary.tar.gz \
  -C /root/autodl-tmp/data/cifar10

tar -xzf /root/autodl-pub/cifar-100/cifar-100-binary.tar.gz \
  -C /root/autodl-tmp/data/cifar100
```

Verify without downloading from the internet:

```bash
test -f /root/autodl-tmp/data/cifar10/cifar-10-batches-bin/data_batch_1.bin
test -f /root/autodl-tmp/data/cifar10/cifar-10-batches-bin/test_batch.bin
test -f /root/autodl-tmp/data/cifar100/cifar-100-binary/train.bin
test -f /root/autodl-tmp/data/cifar100/cifar-100-binary/test.bin
```

The AutoDL-specific CIFAR-100 configs live under `configs/cifar100_lt/autodl/`.
They set:

- `data.root: /root/autodl-tmp/data/cifar100`
- `data.download: false`
- `experiment.output_dir: /root/autodl-tmp/runs/...`

## 6. Run a short CUDA smoke

Use the baseline first. This checks CUDA, AMP, channels-last layout, sampling,
and basic artifact writing without spending a full experiment budget.

```bash
python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/autodl/cifar100_lt_ir100_source_sloss_paperclose_baseline.yaml \
  --output-dir /root/autodl-tmp/runs/cifar100_lt_cuda_smoke/baseline_amp_channels_last \
  --device cuda \
  --steps 50 \
  --batch-size 64 \
  --mixed-precision auto \
  --channels-last on \
  --compile off \
  --n-samples 256 \
  --n-trajectories 4 \
  --nfe 4 \
  --sample-batch-size 128 \
  --plot-max-points 256
```

If that succeeds, test `torch.compile` separately:

```bash
python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/autodl/cifar100_lt_ir100_source_sloss_paperclose_baseline.yaml \
  --output-dir /root/autodl-tmp/runs/cifar100_lt_cuda_smoke/baseline_amp_channels_last_compile \
  --device cuda \
  --steps 50 \
  --batch-size 64 \
  --mixed-precision auto \
  --channels-last on \
  --compile on \
  --compile-mode reduce-overhead \
  --n-samples 256 \
  --n-trajectories 4 \
  --nfe 4 \
  --sample-batch-size 128 \
  --plot-max-points 256
```

For only 50 steps, `torch.compile` may look slower because the compile cost is
paid upfront. Treat this as a correctness smoke, not a throughput benchmark.

## 7. Inspect runtime metadata

Each run writes the selected runtime behavior into `metrics.json`:

```bash
python - <<'PY'
import json
from pathlib import Path

metrics = json.loads(Path(
    "/root/autodl-tmp/runs/cifar100_lt_cuda_smoke/"
    "baseline_amp_channels_last/metrics.json"
).read_text())
print(json.dumps(metrics["runtime"], indent=2, sort_keys=True))
PY
```

Expected CUDA smoke behavior:

- `mixed_precision.active` is `true`.
- `channels_last.active` is `true`.
- `compile.active` is `false` for the first smoke and `true` for the compile smoke.

If `torch.compile` fails or gives no throughput benefit on the cluster, leave it
off for the full paper-close runs. AMP plus channels-last are the safer default
speedup path.

## 8. Run the CIFAR-100-LT 30k screen

Run these after the CUDA smoke passes. The configs already enable AMP and
channels-last on CUDA and keep `torch.compile` disabled.

The upstream CM CIFAR-100-LT recipe uses `total_steps: 300001`, `batch_size: 64`,
and generates 50,000 evaluation images. At roughly 10 iterations/second on an
RTX 50-series node, that is an 8+ hour training job per condition before full
sampling/evaluation. Use this 30k screen first when the goal is to compare
baseline, CM without endpoint transfer, and CM with endpoint transfer under a
roughly one-hour-per-condition budget.

Baseline:

```bash
python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/autodl_screen/cifar100_lt_ir100_source_sloss_screen30k_baseline.yaml \
  --device cuda
```

CM without endpoint transfer:

```bash
python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/autodl_screen/cifar100_lt_ir100_source_sloss_screen30k_cm_no_oc.yaml \
  --device cuda
```

CM with endpoint transfer:

```bash
python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/autodl_screen/cifar100_lt_ir100_source_sloss_screen30k_cm.yaml \
  --device cuda
```

Default output directories:

```text
/root/autodl-tmp/runs/cifar100_lt_ir100_source_sloss_screen30k/baseline
/root/autodl-tmp/runs/cifar100_lt_ir100_source_sloss_screen30k/cm_no_oc
/root/autodl-tmp/runs/cifar100_lt_ir100_source_sloss_screen30k/cm_oc
```

The 30k screen uses about 1.92M training draws, or roughly 177 passes over the
CIFAR-100-LT IR100 support. It can show whether the local continuous-flow CM
implementation has the right direction of effect at practical cost, but it is
not a paper-number reproduction.

## 9. Optional full paper-close run

Only run this tier if the 30k screen is promising and the cluster budget allows
it. These configs preserve the upstream 300k-step scale.

```bash
python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/autodl/cifar100_lt_ir100_source_sloss_paperclose_baseline.yaml \
  --device cuda

python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/autodl/cifar100_lt_ir100_source_sloss_paperclose_cm_no_oc.yaml \
  --device cuda

python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/autodl/cifar100_lt_ir100_source_sloss_paperclose_cm.yaml \
  --device cuda
```

If AutoDL storage pressure becomes an issue, preserve at least each run's
`checkpoint.pt`, `metrics.json`, `config.yaml`, and final evaluation artifacts
under persistent storage before releasing the instance.
