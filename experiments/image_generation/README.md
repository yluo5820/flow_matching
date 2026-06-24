# FLUX.2-klein Batch Generation

This module provides a reusable batch generation layer for FLUX.2-klein image
experiments. It loads prompts from JSONL or CSV, expands seed sweeps, saves
generated images, records per-image metadata, writes a manifest and summary CSV,
and can resume runs by skipping image paths that already exist.

It intentionally does not implement embeddings, UMAP, local intrinsic dimension,
VLM scoring, LoRA training, dataset distillation, or latent trajectory logging.

## Install

The core project can be installed as usual:

```bash
python -m pip install -e ".[dev]"
```

For image generation, install the image-generation extras plus the current
Diffusers source build if your installed Diffusers package does not expose
`Flux2KleinPipeline`:

```bash
python -m pip install -e ".[image-generation]"
python -m pip install git+https://github.com/huggingface/diffusers.git
```

Required packages are `torch`, `diffusers`, `transformers`, `accelerate`,
`safetensors`, `Pillow`, `PyYAML`, `pandas`, and `tqdm`. Depending on the local
Diffusers/Transformers stack, `sentencepiece` and `protobuf` may also be useful.

## Hugging Face Access

The default model is:

```text
black-forest-labs/FLUX.2-klein-4B
```

The 4B model card lists an Apache 2.0 license. If Hugging Face access is needed
in your environment, log in before running generation:

```bash
hf auth login
```

If a model page displays access terms, accept them in the browser for the same
Hugging Face account used by the CLI.

## Hardware

Black Forest Labs describes FLUX.2 [klein] 4B as a consumer-GPU model around
13 GB VRAM. Use `model.cpu_offload: true` when VRAM is tight. The example config
uses `cuda` and `bfloat16`; change `model.device` to `auto`, `cpu`, or `mps` only
when that is appropriate for your environment.

## Run

Dry-run without loading the model:

```bash
python experiments/image_generation/generate_flux_batches.py \
  --config configs/image_generation/flux2_klein_batch_example.yaml \
  --dry-run
```

Small sanity run:

```bash
python experiments/image_generation/generate_flux_batches.py \
  --config configs/image_generation/flux2_klein_batch_example.yaml \
  --limit-prompts 2 \
  --limit-seeds 2
```

Regenerate existing image paths and reset metadata files:

```bash
python experiments/image_generation/generate_flux_batches.py \
  --config configs/image_generation/flux2_klein_batch_example.yaml \
  --limit-prompts 2 \
  --limit-seeds 2 \
  --overwrite
```

## Outputs

The example writes to:

```text
outputs/image_generation/flux2_klein_batch_001/
  config_used.yaml
  run_log.txt
  manifest.jsonl
  summary.csv
  images/
  grids/
  metadata/
    per_image_metadata.jsonl
    failures.jsonl
```

When `output.skip_existing: true`, rerunning the same command skips image paths
that already exist. If a prompt/seed call fails, the runner writes failed rows to
both `metadata/per_image_metadata.jsonl` and `metadata/failures.jsonl`; the run
continues when `runtime.continue_on_error: true`.

## Prompt Files

JSONL rows can contain:

```json
{"prompt_id": "transparent_001", "family": "transparent_objects", "prompt": "Two human hands holding a transparent glass cube.", "tags": ["hands", "transparent", "contact"]}
```

CSV files should include at least:

```text
prompt_id,prompt,negative_prompt,tags,family,notes
```

Quote CSV tag cells when they contain commas:

```text
transparent_001,"Two hands holding a transparent cube.",,"hands,transparent,contact",transparent_objects,
```

Internally, prompts are normalized to `prompt_id`, `prompt`, `negative_prompt`,
`tags`, `family`, and `notes`. Optional fields may be empty.

## Diffusers Caveats

The FLUX.2-klein model card currently recommends:

```python
from diffusers import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
)
```

The Diffusers docs show `guidance_scale`, `height`, `width`,
`num_inference_steps`, `num_images_per_prompt`, `generator`,
`max_sequence_length`, and `negative_prompt_embeds` for
`Flux2KleinPipeline`. The source warns that height and width should be divisible
by `vae_scale_factor * 2`; for this model family the runner validates a multiple
of 16. Diffusers also documents that `guidance_scale` is ignored for step-wise
distilled models.
