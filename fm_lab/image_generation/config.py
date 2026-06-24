"""Configuration objects for batch image generation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fm_lab.utils.config import ConfigError, deep_update, load_config

DEFAULT_FLUX2_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_DIMENSION_MULTIPLE = 16


@dataclass(frozen=True)
class ModelConfig:
    repo_id: str = DEFAULT_FLUX2_KLEIN_REPO
    dtype: str = "bfloat16"
    device: str = "cuda"
    cpu_offload: bool = False
    device_map: str | dict[str, Any] | None = None
    torch_compile: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    guidance_scale: float = 3.5
    num_images_per_prompt: int = 1
    batch_size: int = 1
    max_sequence_length: int = 512
    text_encoder_out_layers: tuple[int, ...] = (9, 18, 27)
    caption_upsample_temperature: float | None = None


@dataclass(frozen=True)
class SeedConfig:
    mode: str = "range"
    start: int = 0
    count: int = 8
    explicit: list[int] = field(default_factory=list)
    master_seed: int = 123


@dataclass(frozen=True)
class PromptConfig:
    source: str = "data/prompts/example_flux_prompts.jsonl"
    format: str = "jsonl"
    id_field: str = "prompt_id"
    text_field: str = "prompt"
    negative_prompt_field: str = "negative_prompt"
    tag_field: str = "tags"
    family_field: str = "family"
    notes_field: str = "notes"
    max_prompts: int | None = None
    include_tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    include_families: list[str] = field(default_factory=list)
    exclude_families: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OutputConfig:
    root_dir: str = "outputs/image_generation"
    save_individual_images: bool = True
    save_prompt_grids: bool = True
    save_family_grids: bool = True
    save_metadata_jsonl: bool = True
    save_summary_csv: bool = True
    skip_existing: bool = True
    grid_tile_size: int = 256
    family_grid_max_images: int = 64


@dataclass(frozen=True)
class RuntimeConfig:
    dry_run: bool = False
    continue_on_error: bool = True
    log_every: int = 1


@dataclass(frozen=True)
class BatchGenerationConfig:
    experiment_name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return Path(self.output.root_dir) / self.experiment_name


def load_batch_generation_config(path: str | Path) -> BatchGenerationConfig:
    """Load and validate an image generation YAML config."""

    raw = load_config(path)
    return batch_generation_config_from_dict(raw)


def batch_generation_config_from_dict(raw: dict[str, Any]) -> BatchGenerationConfig:
    """Convert a raw config mapping into typed config sections."""

    if "experiment_name" not in raw:
        raise ConfigError("Image generation config must define experiment_name.")

    config = BatchGenerationConfig(
        experiment_name=str(raw["experiment_name"]),
        model=ModelConfig(**_section(raw, "model")),
        generation=_generation_config(_section(raw, "generation")),
        seeds=SeedConfig(**_section(raw, "seeds")),
        prompts=PromptConfig(**_section(raw, "prompts")),
        output=OutputConfig(**_section(raw, "output")),
        runtime=RuntimeConfig(**_section(raw, "runtime")),
        raw=dict(raw),
    )
    validate_batch_generation_config(config)
    return config


def apply_runtime_overrides(
    raw_config: dict[str, Any],
    *,
    dry_run: bool = False,
    limit_prompts: int | None = None,
    limit_seeds: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply small CLI testing overrides without replacing the YAML as source of truth."""

    overrides: dict[str, Any] = {}
    if dry_run:
        overrides = deep_update(overrides, {"runtime": {"dry_run": True}})
    if limit_prompts is not None:
        overrides = deep_update(overrides, {"prompts": {"max_prompts": limit_prompts}})
    if limit_seeds is not None:
        seed_config = raw_config.get("seeds", {})
        seed_mode = (
            str(seed_config.get("mode", "range")).lower()
            if isinstance(seed_config, dict)
            else "range"
        )
        if seed_mode == "explicit" and isinstance(seed_config, dict):
            explicit = list(seed_config.get("explicit", []))[:limit_seeds]
            overrides = deep_update(overrides, {"seeds": {"explicit": explicit}})
        else:
            overrides = deep_update(overrides, {"seeds": {"count": limit_seeds}})
    if overwrite:
        overrides = deep_update(overrides, {"output": {"skip_existing": False}})
    if not overrides:
        return dict(raw_config)
    return deep_update(raw_config, overrides)


def expand_seeds(config: SeedConfig) -> list[int]:
    """Normalize configured seed settings into a concrete deterministic seed list."""

    mode = config.mode.lower()
    if mode == "range":
        if config.count < 0:
            raise ConfigError("seeds.count must be non-negative.")
        return [int(config.start) + offset for offset in range(int(config.count))]
    if mode == "explicit":
        if not config.explicit:
            raise ConfigError("seeds.explicit must contain at least one seed in explicit mode.")
        return [int(seed) for seed in config.explicit]
    if mode == "random":
        if config.count < 0:
            raise ConfigError("seeds.count must be non-negative.")
        rng = random.Random(int(config.master_seed))
        return [rng.randrange(0, 2**32) for _ in range(int(config.count))]
    raise ConfigError(f"Unsupported seed mode: {config.mode!r}")


def validate_batch_generation_config(config: BatchGenerationConfig) -> None:
    """Validate constraints that can be checked without loading the model."""

    generation = config.generation
    if generation.width <= 0 or generation.height <= 0:
        raise ConfigError("generation.width and generation.height must be positive.")
    if generation.width % FLUX2_DIMENSION_MULTIPLE != 0:
        raise ConfigError(
            "FLUX.2-klein image width should be divisible by "
            f"{FLUX2_DIMENSION_MULTIPLE}; got {generation.width}."
        )
    if generation.height % FLUX2_DIMENSION_MULTIPLE != 0:
        raise ConfigError(
            "FLUX.2-klein image height should be divisible by "
            f"{FLUX2_DIMENSION_MULTIPLE}; got {generation.height}."
        )
    if generation.num_inference_steps <= 0:
        raise ConfigError("generation.num_inference_steps must be positive.")
    if generation.num_images_per_prompt <= 0:
        raise ConfigError("generation.num_images_per_prompt must be positive.")
    if generation.batch_size <= 0:
        raise ConfigError("generation.batch_size must be positive.")
    if generation.max_sequence_length <= 0 or generation.max_sequence_length > 512:
        raise ConfigError("generation.max_sequence_length must be in the range 1..512.")

    if config.output.grid_tile_size <= 0:
        raise ConfigError("output.grid_tile_size must be positive.")
    if config.output.family_grid_max_images <= 0:
        raise ConfigError("output.family_grid_max_images must be positive.")

    prompt_path = Path(config.prompts.source)
    if not prompt_path.exists():
        raise ConfigError(f"Prompt file does not exist: {prompt_path}")
    if config.prompts.format.lower() not in {"jsonl", "csv"}:
        raise ConfigError("prompts.format must be either 'jsonl' or 'csv'.")

    expand_seeds(config.seeds)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} config section must be a mapping.")
    return dict(value)


def _generation_config(values: dict[str, Any]) -> GenerationConfig:
    if "text_encoder_out_layers" in values and values["text_encoder_out_layers"] is not None:
        values["text_encoder_out_layers"] = tuple(
            int(item) for item in values["text_encoder_out_layers"]
        )
    return GenerationConfig(**values)
