"""Diagnose dropout contamination in the released ImbDiff-CM branch distance."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_dropout import probe_imbdiff_cm_dropout
from fm_lab.diagnostics.imbdiff_cm_probe import (
    ImbDiffCMProbeManifest,
    build_imbdiff_cm_probe_manifest,
    restore_imbdiff_cm_probe_checkpoint,
)
from fm_lab.evaluation.groups import frequency_ranked_groups
from fm_lab.experiments.factory import build_target, resolve_device
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.seeding import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare independent, paired, and disabled dropout in one official "
            "ImbDiff-CM checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="CM checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Diagnostic output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--weights",
        choices=("ema", "raw"),
        default="ema",
        help="Checkpoint weights used by the diagnostic.",
    )
    parser.add_argument(
        "--timesteps",
        default="100,500,900",
        help="Comma-separated discrete diffusion timesteps.",
    )
    parser.add_argument(
        "--classes",
        default="auto",
        help="'auto' for frequency-stratified classes, 'all', or comma-separated IDs.",
    )
    parser.add_argument("--classes-per-group", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--dropout-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--channels-last",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument(
        "--skip-gradients",
        action="store_true",
        help="Skip the one-repeat branch-distance gradient comparison.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be positive.")
    if args.dropout_repeats < 1:
        raise ValueError("--dropout-repeats must be positive.")
    if args.classes_per_group < 1:
        raise ValueError("--classes-per-group must be positive.")

    seed_everything(args.seed)
    checkpoint_path = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("CM dropout checkpoint is missing its experiment config.")

    training_target = build_target(config)
    class_counts = tuple(int(value) for value in training_target.class_counts)
    selected_classes = _select_classes(
        args.classes,
        class_counts=class_counts,
        classes_per_group=args.classes_per_group,
    )
    held_out_target = build_target(_held_out_config(config))
    held_out_images, held_out_labels, held_out_indices = (
        held_out_target.all_samples_with_labels(device="cpu")
    )
    selected_mask = torch.zeros_like(held_out_labels, dtype=torch.bool)
    for class_id in selected_classes:
        selected_mask |= held_out_labels == int(class_id)
    if not bool(selected_mask.any()):
        raise ValueError("Selected dropout-probe classes have no held-out images.")
    candidate_images = held_out_images[selected_mask]
    candidate_labels = held_out_labels[selected_mask]
    candidate_indices = np.asarray(held_out_indices)[selected_mask.numpy()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timesteps = _parse_int_list(args.timesteps, label="timesteps")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = ImbDiffCMProbeManifest.load(manifest_path)
        if manifest.timesteps != timesteps:
            raise ValueError(
                "Existing dropout manifest timesteps differ from --timesteps."
            )
        if manifest.samples_per_class != int(args.samples_per_class):
            raise ValueError(
                "Existing dropout manifest samples_per_class differs from the request."
            )
    else:
        manifest = build_imbdiff_cm_probe_manifest(
            candidate_labels.numpy(),
            candidate_indices,
            timesteps=timesteps,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        manifest.save(manifest_path)

    clean_images = candidate_images[
        torch.from_numpy(manifest.dataset_positions.copy())
    ].reshape(manifest.num_rows, *held_out_target.image_shape)
    device = resolve_device(args.device)
    channels_last = _resolve_channels_last(config, args.channels_last, device)
    restored = restore_imbdiff_cm_probe_checkpoint(
        checkpoint_path,
        class_counts=class_counts,
        device=device,
        weights=args.weights,
        channels_last=channels_last,
        checkpoint_payload=payload,
    )
    summary, rows = probe_imbdiff_cm_dropout(
        restored,
        clean_images=clean_images,
        manifest=manifest,
        class_counts=class_counts,
        repeats=args.dropout_repeats,
        seed=args.seed,
        compute_gradients=not args.skip_gradients,
    )
    summary["request"] = {
        "selected_classes": [int(value) for value in selected_classes],
        "classes_argument": str(args.classes),
        "classes_per_group": int(args.classes_per_group),
        "samples_per_class": int(args.samples_per_class),
        "channels_last": bool(channels_last),
        "device": str(device),
        "compute_gradients": not args.skip_gradients,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "functional_rows.csv", rows)
    if summary["gradients"]:
        _write_csv(output_dir / "gradient_summary.csv", summary["gradients"])
    (output_dir / "report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    print(f"Finished CM dropout probe: {output_dir}")


def _select_classes(
    value: str,
    *,
    class_counts: tuple[int, ...],
    classes_per_group: int,
) -> tuple[int, ...]:
    normalized = str(value).strip().lower()
    if normalized == "all":
        return tuple(range(len(class_counts)))
    if normalized != "auto":
        values = _parse_int_list(value, label="classes")
        if any(class_id < 0 or class_id >= len(class_counts) for class_id in values):
            raise ValueError("--classes contains an out-of-range class ID.")
        return values

    groups = frequency_ranked_groups(class_counts)
    selected: list[int] = []
    for group_name in ("many", "medium", "few"):
        class_ids = tuple(int(value) for value in groups[group_name])
        count = min(int(classes_per_group), len(class_ids))
        positions = np.linspace(0, len(class_ids) - 1, num=count, dtype=np.int64)
        selected.extend(class_ids[int(position)] for position in positions)
    return tuple(dict.fromkeys(selected))


def _held_out_config(config: dict[str, Any]) -> dict[str, Any]:
    held_out = copy.deepcopy(config)
    data = held_out.setdefault("data", {})
    data["train"] = False
    data["horizontal_flip"] = False
    data["dequantize"] = False
    data.pop("frequency_mapping", None)
    return held_out


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


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ImbDiff-CM dropout pairing diagnostic",
        "",
        "This is a fixed-checkpoint training-mode diagnostic. It does not retrain ",
        "the model and its squared-distance ratios are descriptive rather than an ",
        "additive causal variance decomposition.",
        "",
        "| Timestep | Group | Paired / independent | No dropout / independent | "
        "General dropout-only / independent |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary["descriptive_ratios"]:
        lines.append(
            f"| {row['timestep']} | {row['frequency_group']} | "
            f"{row['paired_to_independent_distance']:.6g} | "
            f"{row['disabled_to_independent_distance']:.6g} | "
            f"{row['dropout_only_to_independent_distance']:.6g} |"
        )
    if summary["gradients"]:
        lines.extend(
            [
                "",
                "Gradient rows use the mean branch distance from the first dropout ",
                "repeat. Cosines are relative to the released independent-mask ",
                "expert-plus-dropout condition.",
            ]
        )
    lines.extend(
        [
            "",
            f"Manifest digest: `{summary['manifest_digest']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_int_list(value: str, *, label: str) -> tuple[int, ...]:
    values = tuple(
        int(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if not values:
        raise ValueError(f"At least one {label} value is required.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} values must be unique.")
    return values


if __name__ == "__main__":
    main()
