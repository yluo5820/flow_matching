"""Batch generation runner for FLUX.2-klein experiments."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fm_lab.image_generation.config import BatchGenerationConfig, expand_seeds
from fm_lab.image_generation.metadata import (
    append_failures,
    append_metadata,
    build_metadata_row,
    reset_metadata_files,
    write_summary_csv,
)
from fm_lab.image_generation.pipeline_loader import load_flux_pipeline, make_torch_generator
from fm_lab.image_generation.prompt_loader import PromptRecord, load_prompts
from fm_lab.image_generation.save_utils import (
    family_grid_path,
    output_image_path,
    prepare_output_dir,
    prompt_grid_path,
    write_manifest,
)

LOGGER_NAME = "fm_lab.image_generation"


@dataclass(frozen=True)
class PlannedImage:
    prompt: PromptRecord
    seed: int
    image_index: int
    output_path: Path

    def as_manifest_row(self, config: BatchGenerationConfig) -> dict[str, Any]:
        return {
            "experiment_name": config.experiment_name,
            "model_repo_id": config.model.repo_id,
            "prompt_id": self.prompt.prompt_id,
            "family": self.prompt.family,
            "tags": self.prompt.tags,
            "seed": int(self.seed),
            "image_index": int(self.image_index),
            "prompt": self.prompt.prompt,
            "negative_prompt": self.prompt.negative_prompt,
            "output_path": str(self.output_path),
        }


def run_generation(
    config: BatchGenerationConfig,
    *,
    config_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run or dry-run a configured batch generation experiment."""

    prompts = load_prompts(config.prompts)
    seeds = expand_seeds(config.seeds)
    plan = build_plan(config, prompts, seeds)

    if config.runtime.dry_run:
        return _dry_run_summary(config, prompts, seeds, plan)

    output_dir = prepare_output_dir(config, config_path=config_path)
    logger = configure_logging(output_dir)
    if overwrite:
        reset_metadata_files(output_dir)

    write_manifest([item.as_manifest_row(config) for item in plan], output_dir)
    _log_run_header(logger, config, prompts, seeds, plan)

    try:
        pipe = load_flux_pipeline(config.model)
    except Exception:
        logger.exception("Model loading failed; stopping run.")
        raise

    if any(prompt.negative_prompt for prompt in prompts):
        logger.warning(
            "Negative prompt strings will be encoded as negative_prompt_embeds when the "
            "loaded pipeline uses classifier-free guidance. Distilled FLUX.2-klein models "
            "may ignore guidance_scale and negative conditioning."
        )
    if _pipeline_is_distilled(pipe) and config.generation.guidance_scale > 1.0:
        logger.warning(
            "Loaded pipeline reports is_distilled=true; Diffusers documents that guidance_scale "
            "is ignored for step-wise distilled models."
        )

    started = time.perf_counter()
    plan_by_prompt_seed = _group_plan_by_prompt_seed(plan)
    for prompt_index, prompt in enumerate(prompts, start=1):
        if prompt_index == 1 or prompt_index % max(1, config.runtime.log_every) == 0:
            logger.info("Prompt %d/%d: %s", prompt_index, len(prompts), prompt.prompt_id)
        for seed in seeds:
            planned_images = plan_by_prompt_seed[(prompt.prompt_id, seed)]
            _generate_prompt_seed(
                pipe=pipe,
                config=config,
                prompt=prompt,
                seed=seed,
                planned_images=planned_images,
                output_dir=output_dir,
                logger=logger,
            )

        if config.output.save_prompt_grids:
            _write_prompt_grid(config, prompt, seeds, output_dir, logger)

    if config.output.save_family_grids:
        _write_family_grids(config, plan, output_dir, logger)
    if config.output.save_summary_csv:
        summary_path = write_summary_csv(output_dir)
        logger.info("Wrote summary CSV: %s", summary_path)

    elapsed = time.perf_counter() - started
    logger.info("Finished generation run in %.2f seconds.", elapsed)
    return {
        "output_dir": output_dir,
        "num_prompts": len(prompts),
        "num_seeds": len(seeds),
        "planned_images": len(plan),
        "runtime_seconds": elapsed,
    }


def build_plan(
    config: BatchGenerationConfig,
    prompts: list[PromptRecord],
    seeds: list[int],
) -> list[PlannedImage]:
    plan: list[PlannedImage] = []
    output_dir = config.output_dir
    for prompt in prompts:
        for seed in seeds:
            for image_index in range(config.generation.num_images_per_prompt):
                plan.append(
                    PlannedImage(
                        prompt=prompt,
                        seed=seed,
                        image_index=image_index,
                        output_path=output_image_path(
                            output_dir,
                            prompt.prompt_id,
                            seed,
                            image_index,
                        ),
                    )
                )
    return plan


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "run_log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _generate_prompt_seed(
    *,
    pipe: Any,
    config: BatchGenerationConfig,
    prompt: PromptRecord,
    seed: int,
    planned_images: list[PlannedImage],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    skip_existing = config.output.skip_existing
    missing_planned = [item for item in planned_images if not item.output_path.exists()]
    if skip_existing and not missing_planned:
        logger.info("Skipping existing outputs for prompt=%s seed=%s", prompt.prompt_id, seed)
        return

    generator = make_torch_generator(seed, config.model.device)
    call_kwargs = _pipeline_call_kwargs(pipe, config, prompt, generator)
    started = time.perf_counter()
    try:
        result = pipe(**call_kwargs)
        images = list(result.images)
        if len(images) != len(planned_images):
            raise RuntimeError(
                "Pipeline returned "
                f"{len(images)} images for {len(planned_images)} planned outputs "
                f"(prompt={prompt.prompt_id}, seed={seed})."
            )
        runtime_seconds = time.perf_counter() - started
        metadata_rows: list[dict[str, Any]] = []
        for planned, image in zip(planned_images, images, strict=False):
            if skip_existing and planned.output_path.exists():
                continue
            if config.output.save_individual_images:
                planned.output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(planned.output_path)
            metadata_rows.append(
                build_metadata_row(
                    config=config,
                    prompt=prompt,
                    seed=seed,
                    image_index=planned.image_index,
                    output_path=planned.output_path,
                    status="success",
                    runtime_seconds=runtime_seconds,
                    error=None,
                )
            )
        if config.output.save_metadata_jsonl and metadata_rows:
            append_metadata(output_dir, metadata_rows)
    except Exception as exc:
        runtime_seconds = time.perf_counter() - started
        error = repr(exc)
        logger.exception("Generation failed for prompt=%s seed=%s", prompt.prompt_id, seed)
        failure_rows = [
            build_metadata_row(
                config=config,
                prompt=prompt,
                seed=seed,
                image_index=planned.image_index,
                output_path=planned.output_path,
                status="failed",
                runtime_seconds=runtime_seconds,
                error=error,
            )
            for planned in planned_images
            if not (skip_existing and planned.output_path.exists())
        ]
        if config.output.save_metadata_jsonl and failure_rows:
            append_metadata(output_dir, failure_rows)
            append_failures(output_dir, failure_rows)
        if not config.runtime.continue_on_error:
            raise


def _pipeline_call_kwargs(
    pipe: Any,
    config: BatchGenerationConfig,
    prompt: PromptRecord,
    generator: Any,
) -> dict[str, Any]:
    generation = config.generation
    call_kwargs: dict[str, Any] = {
        "prompt": prompt.prompt,
        "height": generation.height,
        "width": generation.width,
        "num_inference_steps": generation.num_inference_steps,
        "guidance_scale": generation.guidance_scale,
        "num_images_per_prompt": generation.num_images_per_prompt,
        "generator": generator,
        "max_sequence_length": generation.max_sequence_length,
        "text_encoder_out_layers": generation.text_encoder_out_layers,
    }
    if generation.caption_upsample_temperature is not None:
        call_kwargs["caption_upsample_temperature"] = generation.caption_upsample_temperature
    if prompt.negative_prompt:
        call_kwargs["negative_prompt_embeds"] = _encode_negative_prompt(
            pipe=pipe,
            negative_prompt=prompt.negative_prompt,
            num_images_per_prompt=generation.num_images_per_prompt,
            max_sequence_length=generation.max_sequence_length,
            text_encoder_out_layers=generation.text_encoder_out_layers,
        )

    allowed_parameters = set(inspect.signature(pipe.__call__).parameters)
    return {key: value for key, value in call_kwargs.items() if key in allowed_parameters}


def _encode_negative_prompt(
    *,
    pipe: Any,
    negative_prompt: str,
    num_images_per_prompt: int,
    max_sequence_length: int,
    text_encoder_out_layers: tuple[int, ...],
) -> Any:
    import torch

    device = getattr(pipe, "_execution_device", None)
    with torch.no_grad():
        negative_prompt_embeds, _ = pipe.encode_prompt(
            prompt=negative_prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
    return negative_prompt_embeds


def _write_prompt_grid(
    config: BatchGenerationConfig,
    prompt: PromptRecord,
    seeds: list[int],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if not config.output.save_individual_images:
        return
    image_paths = [
        output_image_path(output_dir, prompt.prompt_id, seed, image_index)
        for seed in seeds
        for image_index in range(config.generation.num_images_per_prompt)
    ]
    if len(image_paths) <= 1:
        return
    from fm_lab.image_generation.grids import make_image_grid

    grid_path = prompt_grid_path(output_dir, prompt.prompt_id)
    result = make_image_grid(
        image_paths,
        grid_path,
        tile_size=config.output.grid_tile_size,
    )
    if result is not None:
        logger.info("Wrote prompt grid: %s", result)


def _write_family_grids(
    config: BatchGenerationConfig,
    plan: list[PlannedImage],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if not config.output.save_individual_images:
        return
    by_family: dict[str, list[Path]] = {}
    for planned in plan:
        family = planned.prompt.family or "unknown_family"
        by_family.setdefault(family, []).append(planned.output_path)

    from fm_lab.image_generation.grids import make_image_grid

    for family, image_paths in sorted(by_family.items()):
        result = make_image_grid(
            image_paths[: config.output.family_grid_max_images],
            family_grid_path(output_dir, family),
            tile_size=config.output.grid_tile_size,
        )
        if result is not None:
            logger.info("Wrote family grid: %s", result)


def _group_plan_by_prompt_seed(
    plan: list[PlannedImage],
) -> dict[tuple[str, int], list[PlannedImage]]:
    grouped: dict[tuple[str, int], list[PlannedImage]] = {}
    for planned in plan:
        grouped.setdefault((planned.prompt.prompt_id, planned.seed), []).append(planned)
    return grouped


def _dry_run_summary(
    config: BatchGenerationConfig,
    prompts: list[PromptRecord],
    seeds: list[int],
    plan: list[PlannedImage],
) -> dict[str, Any]:
    preview_paths = [str(item.output_path) for item in plan[: min(8, len(plan))]]
    return {
        "dry_run": True,
        "experiment_name": config.experiment_name,
        "num_prompts": len(prompts),
        "num_seeds": len(seeds),
        "planned_images": len(plan),
        "output_dir": config.output_dir,
        "preview_paths": preview_paths,
    }


def _log_run_header(
    logger: logging.Logger,
    config: BatchGenerationConfig,
    prompts: list[PromptRecord],
    seeds: list[int],
    plan: list[PlannedImage],
) -> None:
    logger.info("Config experiment: %s", config.experiment_name)
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Model repo: %s", config.model.repo_id)
    logger.info("Device: %s; dtype: %s", config.model.device, config.model.dtype)
    logger.info("Prompts: %d", len(prompts))
    logger.info("Seeds: %d", len(seeds))
    logger.info("Total planned images: %d", len(plan))


def _pipeline_is_distilled(pipe: Any) -> bool:
    config = getattr(pipe, "config", None)
    if config is None:
        return False
    if isinstance(config, dict):
        return bool(config.get("is_distilled", False))
    return bool(getattr(config, "is_distilled", False))
