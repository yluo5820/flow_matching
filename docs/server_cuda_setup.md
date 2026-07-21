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

## 5. Run a short CUDA smoke

Use the baseline first. This checks CUDA, AMP, channels-last layout, sampling,
and basic artifact writing without spending a full experiment budget.

```bash
python -m fm_lab.experiments.run_train \
  --config configs/cifar100_lt/cifar100_lt_ir100_source_sloss_paperclose_baseline.yaml \
  --output-dir runs/cifar100_lt_cuda_smoke/baseline_amp_channels_last \
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
  --config configs/cifar100_lt/cifar100_lt_ir100_source_sloss_paperclose_baseline.yaml \
  --output-dir runs/cifar100_lt_cuda_smoke/baseline_amp_channels_last_compile \
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

## 6. Inspect runtime metadata

Each run writes the selected runtime behavior into `metrics.json`:

```bash
python - <<'PY'
import json
from pathlib import Path

metrics = json.loads(Path(
    "runs/cifar100_lt_cuda_smoke/baseline_amp_channels_last/metrics.json"
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
