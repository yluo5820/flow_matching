"""Run K1/K2 expert-knowledge diagnostics on an official ImbDiff-CM checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_knowledge import (
    ImbDiffCMKnowledgeManifest,
    build_imbdiff_cm_knowledge_manifest,
    probe_imbdiff_cm_knowledge,
)
from fm_lab.diagnostics.imbdiff_cm_probe import restore_imbdiff_cm_probe_checkpoint
from fm_lab.evaluation.groups import frequency_ranked_groups
from fm_lab.experiments.factory import build_target, resolve_device
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.seeding import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture layerwise expert responses and test their class, superclass, "
            "frequency-group, spectral, and subspace structure."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Official CM checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Diagnostic output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument(
        "--timesteps",
        default="100,500,900",
        help="Comma-separated discrete diffusion timesteps.",
    )
    parser.add_argument(
        "--classes",
        default="all",
        help="'all', 'auto' for a six-class smoke panel, or comma-separated IDs.",
    )
    parser.add_argument("--classes-per-group", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--subspace-rank", type=int, default=3)
    parser.add_argument(
        "--layers",
        default="all",
        help="'all' or exact comma-separated model module names.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="Evenly select at most this many active LoRA layers; zero keeps all.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--channels-last",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--skip-linear-probes", action="store_true")
    parser.add_argument("--skip-subspaces", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least two.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.sketch_dim < 2:
        raise ValueError("--sketch-dim must be at least two.")
    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be positive.")
    if args.max_layers < 0:
        raise ValueError("--max-layers must be non-negative.")

    seed_everything(args.seed)
    checkpoint_path = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("CM knowledge checkpoint is missing its experiment config.")

    training_target = build_target(config)
    class_counts = tuple(int(value) for value in training_target.class_counts)
    if len(class_counts) != 100:
        raise ValueError("The K1/K2 knowledge probe currently requires CIFAR-100.")
    selected_classes = _select_classes(
        args.classes,
        class_counts=class_counts,
        classes_per_group=args.classes_per_group,
    )
    held_out_target = build_target(_held_out_config(config))
    held_out_images, held_out_labels, held_out_indices = held_out_target.all_samples_with_labels(
        device="cpu"
    )
    selected_mask = torch.zeros_like(held_out_labels, dtype=torch.bool)
    for class_id in selected_classes:
        selected_mask |= held_out_labels == int(class_id)
    candidate_images = held_out_images[selected_mask]
    candidate_labels = held_out_labels[selected_mask]
    candidate_indices = np.asarray(held_out_indices)[selected_mask.numpy()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timesteps = _parse_int_list(args.timesteps, label="timesteps")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = ImbDiffCMKnowledgeManifest.load(manifest_path)
        if manifest.probe.timesteps != timesteps:
            raise ValueError("Existing knowledge manifest has different timesteps.")
        if manifest.probe.samples_per_class != int(args.samples_per_class):
            raise ValueError("Existing knowledge manifest has different samples_per_class.")
        if tuple(int(value) for value in np.unique(manifest.probe.labels)) != tuple(
            sorted(selected_classes)
        ):
            raise ValueError("Existing knowledge manifest has different selected classes.")
    else:
        manifest = build_imbdiff_cm_knowledge_manifest(
            candidate_labels.numpy(),
            candidate_indices,
            timesteps=timesteps,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        manifest.save(manifest_path)

    clean_images = candidate_images[
        torch.from_numpy(manifest.probe.dataset_positions.copy())
    ].reshape(manifest.probe.num_rows, *held_out_target.image_shape)
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
    layer_names = _select_layer_names(
        restored.model,
        args.layers,
        max_layers=args.max_layers,
    )
    (
        summary,
        response_rows,
        atlas,
        linear_rows,
        subspace_rows,
        subspace_pairs,
    ) = probe_imbdiff_cm_knowledge(
        restored,
        clean_images=clean_images,
        manifest=manifest,
        class_counts=class_counts,
        batch_size=args.batch_size,
        sketch_dim=args.sketch_dim,
        seed=args.seed,
        permutation_repeats=args.permutation_repeats,
        ridge_alpha=args.ridge_alpha,
        subspace_rank=args.subspace_rank,
        layer_names=layer_names,
        compute_linear_probes=not args.skip_linear_probes,
        compute_subspaces=not args.skip_subspaces,
    )
    summary["request"] = {
        "selected_classes": [int(value) for value in selected_classes],
        "classes_argument": str(args.classes),
        "classes_per_group": int(args.classes_per_group),
        "samples_per_class": int(args.samples_per_class),
        "batch_size": int(args.batch_size),
        "channels_last": bool(channels_last),
        "device": str(device),
        "selected_layers": list(layer_names) if layer_names is not None else "all",
        "max_layers": int(args.max_layers),
    }
    layer_name_by_index = {
        int(row["layer_index"]): str(row["layer_name"]) for row in summary["layers"]
    }
    for rows in (linear_rows, subspace_rows):
        for row in rows:
            row["layer_name"] = layer_name_by_index[int(row["layer_index"])]

    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "response_descriptors.csv", response_rows)
    _write_csv(output_dir / "linear_probes.csv", linear_rows)
    _write_csv(output_dir / "subspace_summary.csv", subspace_rows)
    np.savez_compressed(output_dir / "response_atlas.npz", **atlas)
    if subspace_pairs:
        np.savez_compressed(output_dir / "subspace_pairs.npz", **subspace_pairs)
    (output_dir / "report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    print(f"Finished CM K1/K2 knowledge probe: {output_dir}")


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


def _select_layer_names(
    model: torch.nn.Module,
    value: str,
    *,
    max_layers: int,
) -> tuple[str, ...] | None:
    available = tuple(
        name
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Conv2d_LoRA" and int(getattr(module, "r", 0)) > 0
    )
    if not available:
        raise ValueError("Checkpoint contains no active LoRA layers.")
    normalized = str(value).strip()
    if normalized.lower() == "all":
        selected = available
    else:
        selected = tuple(part.strip() for part in normalized.split(",") if part.strip())
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(f"--layers contains unknown active LoRA layers: {missing}")
    if int(max_layers) > 0 and len(selected) > int(max_layers):
        positions = np.linspace(
            0,
            len(selected) - 1,
            num=int(max_layers),
            dtype=np.int64,
        )
        selected = tuple(selected[int(position)] for position in positions)
    return selected


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


def _parse_int_list(value: str, *, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"{label} must be comma-separated integers.") from error
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must be a non-empty unique integer list.")
    return parsed


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(summary: dict[str, Any]) -> str:
    layer_by_index = {int(row["layer_index"]): str(row["layer_name"]) for row in summary["layers"]}
    lines = [
        "# ImbDiff-CM K1/K2 expert-knowledge probe",
        "",
        "The model was evaluated with dropout disabled. Every local expert response ",
        "was measured directly as `(B @ A) * activation`; the maximum relative ",
        "reconstruction error across layers was "
        f"`{summary['max_reconstruction_relative_rms']:.6g}`.",
        "",
        "Linear-probe values are two-way cross-fit accuracies on disjoint held-out ",
        "images. The permutation and matched-random-feature controls prevent response ",
        "magnitude or sketch dimension from being interpreted as knowledge.",
        "",
        "| Target | Feature | Best layer | t | Accuracy | Permuted | Excess |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["linear_probe_summary"]:
        lines.append(
            f"| {row['task']} | {row['feature']} | "
            f"`{layer_by_index[row['best_layer_index']]}` | "
            f"{row['best_timestep']} | {row['best_accuracy']:.4f} | "
            f"{row['best_permutation_mean']:.4f} | {row['best_excess']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Subspace summaries compare normalized response directions. A positive ",
            "within-minus-across superclass overlap is compatible with semantic ",
            "organization; a strong relationship with class-frequency distance is ",
            "compatible with frequency-stratified separation. Neither is causal until ",
            "matched direction interventions are run.",
            "",
            f"Manifest digest: `{summary['manifest_digest']}`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
