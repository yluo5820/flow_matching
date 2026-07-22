"""Run checkpoint-based mechanism probes on official ImbDiff-CM models."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_probe import (
    ImbDiffCMProbeManifest,
    build_imbdiff_cm_probe_manifest,
    probe_imbdiff_cm_checkpoint,
    restore_imbdiff_cm_probe_checkpoint,
)
from fm_lab.experiments.factory import build_target, resolve_device
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.seeding import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe general/expert routing in official ImbDiff-CM checkpoints."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory containing checkpoint.pt and checkpoints/step_XXXXXX.pt.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        default="20000,40000,60000",
        help="Comma-separated checkpoint steps to discover in every run directory.",
    )
    parser.add_argument("--output-dir", required=True, help="Probe output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--weights",
        choices=("ema", "raw"),
        default="ema",
        help="Checkpoint weights used by the probe.",
    )
    parser.add_argument(
        "--timesteps",
        default="50,250,500,750,950",
        help="Comma-separated discrete DDPM timesteps.",
    )
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional existing probe manifest. The output manifest is reused by default.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "off", "bf16", "fp16"),
        default="auto",
    )
    parser.add_argument(
        "--channels-last",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument(
        "--functional-only",
        action="store_true",
        help="Skip gradient routing measurements for a faster smoke run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = _parse_int_list(args.checkpoint_steps, label="checkpoint steps")
    timesteps = _parse_int_list(args.timesteps, label="timesteps")
    checkpoint_paths = _discover_checkpoints(
        [Path(value) for value in args.run_dir],
        steps,
    )
    first_payload = load_checkpoint(checkpoint_paths[0], map_location="cpu")
    first_config = _checkpoint_config(first_payload)
    del first_payload
    training_target = build_target(first_config)
    class_counts = tuple(int(value) for value in training_target.class_counts)
    held_out_target = build_target(_held_out_config(first_config))
    held_out_images, held_out_labels, held_out_indices = (
        held_out_target.all_samples_with_labels(device="cpu")
    )
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = ImbDiffCMProbeManifest.load(manifest_path)
    else:
        manifest = build_imbdiff_cm_probe_manifest(
            held_out_labels.numpy(),
            held_out_indices,
            timesteps=timesteps,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        manifest.save(manifest_path)
    _validate_manifest_request(manifest, args=args, timesteps=timesteps)
    clean_images = held_out_images[
        torch.from_numpy(manifest.dataset_positions.copy())
    ].reshape(manifest.num_rows, *held_out_target.image_shape)
    device = resolve_device(args.device)

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    checkpoint_output_dir = output_dir / "checkpoints"
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint_path in checkpoint_paths:
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        config = _checkpoint_config(payload)
        _validate_shared_dataset(first_config, config)
        channels_last = _resolve_channels_last(config, args.channels_last, device)
        restored = restore_imbdiff_cm_probe_checkpoint(
            checkpoint_path,
            class_counts=class_counts,
            device=device,
            weights=args.weights,
            channels_last=channels_last,
            checkpoint_payload=payload,
        )
        del payload
        images = clean_images
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        summary, rows = probe_imbdiff_cm_checkpoint(
            restored,
            clean_images=images,
            manifest=manifest,
            class_counts=class_counts,
            mixed_precision=args.mixed_precision,
            compute_gradients=not args.functional_only,
        )
        summary["channels_last"] = channels_last
        summaries.append(summary)
        for row in rows:
            row.update(
                {
                    "method": restored.method,
                    "checkpoint_step": restored.checkpoint_step,
                    "checkpoint_path": str(restored.checkpoint_path),
                    "weights": restored.weights,
                }
            )
        all_rows.extend(rows)
        checkpoint_name = f"{restored.method}_step_{restored.checkpoint_step:06d}.json"
        _write_json(checkpoint_output_dir / checkpoint_name, summary)
        _write_combined_outputs(
            output_dir,
            summaries=summaries,
            rows=all_rows,
            manifest=manifest,
            class_counts=class_counts,
            data_metadata={
                "training": training_target.metadata(),
                "held_out": held_out_target.metadata(),
            },
        )
        print(
            f"Finished {restored.method} step {restored.checkpoint_step}: "
            f"{checkpoint_path}"
        )
        del restored
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Finished CM mechanism probe: {output_dir}")


def _discover_checkpoints(run_dirs: list[Path], steps: tuple[int, ...]) -> list[Path]:
    paths: list[Path] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"CM probe run directory does not exist: {run_dir}")
        final_path = run_dir / "checkpoint.pt"
        final_step = None
        for step in steps:
            periodic = run_dir / "checkpoints" / f"step_{step:06d}.pt"
            if periodic.is_file():
                paths.append(periodic)
                continue
            if final_path.is_file():
                if final_step is None:
                    final_payload = load_checkpoint(final_path, map_location="cpu")
                    final_step = int(final_payload.get("step", -1))
                if final_step == step:
                    paths.append(final_path)
                    continue
            raise FileNotFoundError(
                f"No checkpoint for step {step} under CM probe run {run_dir}."
            )
    if not paths:
        raise ValueError("At least one CM probe checkpoint is required.")
    return paths


def _held_out_config(config: dict[str, Any]) -> dict[str, Any]:
    held_out = copy.deepcopy(config)
    data = held_out.setdefault("data", {})
    data["train"] = False
    data["horizontal_flip"] = False
    data["dequantize"] = False
    data.pop("frequency_mapping", None)
    return held_out


def _validate_shared_dataset(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    keys = (
        "name",
        "root",
        "train",
        "normalize",
        "imbalance_type",
        "imbalance_factor",
        "subset_seed",
    )
    reference_data = reference.get("data", {}) or {}
    candidate_data = candidate.get("data", {}) or {}
    mismatches = [
        key for key in keys if reference_data.get(key) != candidate_data.get(key)
    ]
    if mismatches:
        raise ValueError(
            "CM probe checkpoints do not share one dataset contract: "
            + ", ".join(mismatches)
        )


def _validate_manifest_request(
    manifest: ImbDiffCMProbeManifest,
    *,
    args: argparse.Namespace,
    timesteps: tuple[int, ...],
) -> None:
    if manifest.timesteps != timesteps:
        raise ValueError(
            "Existing CM probe manifest timesteps differ from --timesteps. "
            "Use a new output directory or pass the original values."
        )
    if manifest.samples_per_class != int(args.samples_per_class):
        raise ValueError(
            "Existing CM probe manifest samples_per_class differs from the request."
        )


def _resolve_channels_last(
    config: dict[str, Any],
    mode: str,
    device: torch.device,
) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    runtime = (config.get("training", {}) or {}).get("channels_last", {}) or {}
    return device.type == "cuda" and bool(runtime.get("enabled", False))


def _write_combined_outputs(
    output_dir: Path,
    *,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    manifest: ImbDiffCMProbeManifest,
    class_counts: tuple[int, ...],
    data_metadata: dict[str, Any],
) -> None:
    combined = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "class_counts": list(class_counts),
        "data": data_metadata,
        "checkpoints": summaries,
    }
    _write_json(output_dir / "summary.json", combined)
    _write_csv(output_dir / "functional_rows.csv", rows)
    gradient_rows = _flatten_gradient_rows(summaries)
    if gradient_rows:
        _write_csv(output_dir / "gradient_summary.csv", gradient_rows)
    (output_dir / "report.md").write_text(
        _render_report(combined),
        encoding="utf-8",
    )


def _flatten_gradient_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        checkpoint = summary["checkpoint"]
        for timestep_result in summary["timesteps"]:
            gradients = timestep_result["gradients"]
            if gradients is None:
                continue
            for frequency_group, group_summary in gradients.items():
                row: dict[str, Any] = {
                    "method": checkpoint["method"],
                    "checkpoint_step": checkpoint["step"],
                    "timestep": timestep_result["timestep"],
                    "frequency_group": frequency_group,
                }
                for component, value in group_summary[
                    "expert_gradient_energy_fraction"
                ].items():
                    row[f"expert_energy_fraction_{component}"] = value
                for parameter_group in ("general", "expert"):
                    parameter_summary = group_summary["groups"][parameter_group]
                    row[f"{parameter_group}_num_parameters"] = parameter_summary[
                        "num_parameters"
                    ]
                    for component, values in parameter_summary["components"].items():
                        row[f"{parameter_group}_{component}_norm"] = values["norm"]
                        row[f"{parameter_group}_{component}_rms"] = values["rms"]
                    for pair, value in parameter_summary["cosines"].items():
                        row[f"{parameter_group}_cosine_{pair}"] = value
                rows.append(row)
    return rows


def _render_report(combined: dict[str, Any]) -> str:
    lines = [
        "# ImbDiff-CM mechanism probe",
        "",
        f"Manifest SHA-256: `{combined['manifest_digest']}`.",
        "",
        "This is a paired checkpoint diagnostic, not an independent training run. "
        "Positive expert MSE gain means the capacity-on prediction is closer to the "
        "training target than the general-only prediction.",
        "",
        "| Method | Step | Group | Capacity distance | Expert MSE gain | "
        "High-frequency fraction | Expert total-gradient fraction |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for checkpoint in combined["checkpoints"]:
        method = checkpoint["checkpoint"]["method"]
        step = checkpoint["checkpoint"]["step"]
        for group_name in ("many", "medium", "few"):
            functional_rows = [
                result["functional"][group_name]
                for result in checkpoint["timesteps"]
            ]
            gradients = [
                result["gradients"][group_name]
                for result in checkpoint["timesteps"]
                if result["gradients"] is not None
            ]
            expert_fraction = (
                float(
                    np.mean(
                        [
                            value["expert_gradient_energy_fraction"]["total"]
                            for value in gradients
                        ]
                    )
                )
                if gradients
                else float("nan")
            )
            lines.append(
                "| "
                f"`{method}` | {step} | {group_name} | "
                f"{_mean(functional_rows, 'capacity_distance'):.6g} | "
                f"{_mean(functional_rows, 'expert_mse_gain'):.6g} | "
                f"{_mean_nested(functional_rows, 'spectral_energy_fraction', 'high'):.4f} | "
                f"{expert_fraction:.4f} |"
            )
    lines.extend(
        [
            "",
            "Interpret gradients together with `gradient_summary.csv`; a larger expert "
            "fraction alone does not establish better allocation. The decisive pattern is "
            "whether CM changes the direction and functional benefit of the expert branch "
            "selectively for Medium/Few classes over training.",
            "",
        ]
    )
    return "\n".join(lines)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _mean_nested(rows: list[dict[str, Any]], outer: str, inner: str) -> float:
    return float(np.mean([float(row[outer][inner]) for row in rows]))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    additional = sorted({key for row in rows for key in row} - set(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *additional])
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("CM probe checkpoint is missing its experiment config.")
    return config


def _parse_int_list(value: str, *, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"Invalid comma-separated {label}: {value!r}") from error
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise ValueError(f"{label} must be a non-empty sequence of unique non-negative values.")
    return values


if __name__ == "__main__":
    main()
