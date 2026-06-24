from __future__ import annotations

import json
from pathlib import Path

from fm_lab.image_generation.config import (
    SeedConfig,
    apply_runtime_overrides,
    batch_generation_config_from_dict,
    expand_seeds,
)
from fm_lab.image_generation.generation_runner import run_generation
from fm_lab.image_generation.metadata import append_metadata, build_metadata_row, write_summary_csv
from fm_lab.image_generation.prompt_loader import PromptConfig, PromptRecord, load_prompts
from fm_lab.image_generation.save_utils import output_image_path


def test_expand_seed_modes_are_deterministic() -> None:
    assert expand_seeds(SeedConfig(mode="range", start=3, count=3)) == [3, 4, 5]
    assert expand_seeds(SeedConfig(mode="explicit", explicit=[0, 17, 42])) == [0, 17, 42]

    first = expand_seeds(SeedConfig(mode="random", master_seed=123, count=4))
    second = expand_seeds(SeedConfig(mode="random", master_seed=123, count=4))

    assert first == second
    assert len(first) == 4


def test_prompt_loader_jsonl_filters_and_normalizes(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_id": "keep",
                        "family": "transparent_objects",
                        "prompt": "A transparent cube.",
                        "tags": ["transparent", "contact"],
                    }
                ),
                json.dumps(
                    {
                        "prompt_id": "drop",
                        "family": "rare_animals",
                        "prompt": "A pangolin.",
                        "tags": ["animal"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    prompts = load_prompts(
        PromptConfig(
            source=str(prompt_path),
            include_tags=["transparent"],
            exclude_families=["rare_animals"],
        )
    )

    assert [prompt.prompt_id for prompt in prompts] == ["keep"]
    assert prompts[0].negative_prompt is None
    assert prompts[0].notes is None


def test_prompt_loader_csv_splits_quoted_tags(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.csv"
    prompt_path.write_text(
        "prompt_id,prompt,negative_prompt,tags,family,notes\n"
        'p1,"A glass cube.",,"transparent,contact",transparent_objects,"quoted tags"\n',
        encoding="utf-8",
    )

    prompts = load_prompts(PromptConfig(source=str(prompt_path), format="csv"))

    assert prompts == [
        PromptRecord(
            prompt_id="p1",
            prompt="A glass cube.",
            negative_prompt=None,
            tags=["transparent", "contact"],
            family="transparent_objects",
            notes="quoted tags",
        )
    ]


def test_dry_run_builds_plan_without_model_load(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps({"prompt_id": "p1", "prompt": "A glass cube.", "tags": ["transparent"]})
        + "\n",
        encoding="utf-8",
    )
    raw_config = _raw_config(tmp_path, prompt_path)
    raw_config = apply_runtime_overrides(raw_config, dry_run=True, limit_prompts=1, limit_seeds=2)
    config = batch_generation_config_from_dict(raw_config)

    result = run_generation(config)

    assert result["dry_run"] is True
    assert result["num_prompts"] == 1
    assert result["num_seeds"] == 2
    assert result["planned_images"] == 2
    assert str(output_image_path(config.output_dir, "p1", 0, 0)) in result["preview_paths"]


def test_summary_csv_is_written_from_metadata(tmp_path: Path) -> None:
    prompt = PromptRecord(prompt_id="p1", prompt="A glass cube.", tags=["transparent"])
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(json.dumps({"prompt_id": "p1", "prompt": prompt.prompt}) + "\n")
    config = batch_generation_config_from_dict(_raw_config(tmp_path, prompt_path))
    output_dir = config.output_dir
    row = build_metadata_row(
        config=config,
        prompt=prompt,
        seed=0,
        image_index=0,
        output_path=output_image_path(output_dir, "p1", 0, 0),
        status="success",
        runtime_seconds=1.25,
        error=None,
    )

    append_metadata(output_dir, [row])
    summary_path = write_summary_csv(output_dir)

    summary = summary_path.read_text(encoding="utf-8")
    assert "experiment_name,prompt_id,family,tags,seed,image_index,status" in summary
    assert "test_flux,p1,,transparent,0,0,success" in summary


def _raw_config(tmp_path: Path, prompt_path: Path) -> dict:
    return {
        "experiment_name": "test_flux",
        "model": {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "dtype": "bfloat16",
            "device": "cpu",
        },
        "generation": {
            "width": 64,
            "height": 64,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            "batch_size": 1,
        },
        "seeds": {"mode": "range", "start": 0, "count": 3},
        "prompts": {"source": str(prompt_path), "format": "jsonl"},
        "output": {"root_dir": str(tmp_path / "outputs")},
        "runtime": {"dry_run": False},
    }
