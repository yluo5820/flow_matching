"""Run matched causal expert interventions on an official ImbDiff-CM checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_intervention import (
    probe_imbdiff_cm_intervention,
)
from fm_lab.diagnostics.imbdiff_cm_knowledge import ImbDiffCMKnowledgeManifest
from fm_lab.diagnostics.imbdiff_cm_probe import restore_imbdiff_cm_probe_checkpoint
from fm_lab.experiments.factory import build_target, resolve_device
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.seeding import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare learned CM experts with general-only and spectrum-preserving "
            "random expert rotations at fixed held-out noisy inputs."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        required=True,
        help="K1/K2 knowledge manifest whose held-out rows and random draws are reused.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-repeats", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
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
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.random_repeats < 1:
        raise ValueError("--random-repeats must be positive.")
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be positive.")

    seed_everything(args.seed)
    checkpoint_path = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("CM intervention checkpoint is missing its config.")
    training_target = build_target(config)
    class_counts = tuple(int(value) for value in training_target.class_counts)
    manifest = ImbDiffCMKnowledgeManifest.load(args.manifest)
    if len(class_counts) != 100:
        raise ValueError("The intervention probe currently requires CIFAR-100.")

    held_out_target = build_target(_held_out_config(config))
    held_out_images, held_out_labels, held_out_indices = held_out_target.all_samples_with_labels(
        device="cpu"
    )
    clean_images = _manifest_images(
        manifest,
        images=held_out_images,
        labels=held_out_labels,
        original_indices=np.asarray(held_out_indices),
    ).reshape(manifest.probe.num_rows, *held_out_target.image_shape)

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
    (
        summary,
        effect_rows,
        random_rows,
        group_rows,
        class_rows,
        intervention_manifest,
    ) = probe_imbdiff_cm_intervention(
        restored,
        clean_images=clean_images,
        manifest=manifest,
        class_counts=class_counts,
        batch_size=args.batch_size,
        random_repeats=args.random_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
    )
    summary["request"] = {
        "batch_size": int(args.batch_size),
        "random_repeats": int(args.random_repeats),
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "mixed_precision": str(args.mixed_precision),
        "channels_last": bool(channels_last),
        "device": str(device),
        "seed": int(args.seed),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.save(output_dir / "knowledge_manifest.json")
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "intervention_manifest.json",
        intervention_manifest,
    )
    _write_csv(output_dir / "paired_effects.csv", effect_rows)
    _write_csv(output_dir / "random_repeat_effects.csv", random_rows)
    _write_csv(output_dir / "group_summary.csv", group_rows)
    _write_csv(output_dir / "class_summary.csv", class_rows)
    (output_dir / "report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    print(f"Finished CM causal intervention probe: {output_dir}")


def _manifest_images(
    manifest: ImbDiffCMKnowledgeManifest,
    *,
    images: torch.Tensor,
    labels: torch.Tensor,
    original_indices: np.ndarray,
) -> torch.Tensor:
    index_to_position = {
        int(original_index): position for position, original_index in enumerate(original_indices)
    }
    try:
        positions = np.asarray(
            [
                index_to_position[int(original_index)]
                for original_index in manifest.probe.original_indices
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "Knowledge manifest contains an index absent from the held-out dataset."
        ) from error
    selected_labels = labels[torch.from_numpy(positions)]
    expected_labels = torch.from_numpy(manifest.probe.labels.copy())
    if not torch.equal(selected_labels.cpu(), expected_labels):
        raise ValueError("Held-out labels do not match the knowledge manifest.")
    return images[torch.from_numpy(positions)]


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
    lines = [
        "# ImbDiff-CM matched causal expert intervention",
        "",
        "All conditions reuse identical held-out images, class labels, diffusion ",
        "noise, timesteps, and released-CM endpoint-transfer targets. Positive ",
        "`learned gain` means the learned expert reduces target MSE relative to ",
        "the general-only model. Positive `learned advantage` means it also beats ",
        "the mean of the singular-spectrum-matched random expert rotations.",
        "",
        "| t | Group | Learned gain | Random gain | Learned advantage | "
        "Response-matched advantage | 95% response-matched CI |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["group_summary"]:
        if int(row["timestep"]) != -1:
            continue
        lines.append(
            f"| all | {row['frequency_group']} | "
            f"{row['learned_gain_vs_general_mean']:.6g} | "
            f"{row['random_gain_vs_general_mean']:.6g} | "
            f"{row['learned_advantage_vs_random_mean']:.6g} | "
            f"{row['learned_advantage_vs_response_matched_random_mean']:.6g} | "
            f"[{row['learned_advantage_vs_response_matched_random_class_bootstrap_low']:.6g}, "
            f"{row['learned_advantage_vs_response_matched_random_class_bootstrap_high']:.6g}] |"
        )
    lines.extend(
        [
            "",
            "Tail selectivity is the Few-minus-Many difference. Positive values mean ",
            "the learned expert has a larger effect for tail classes.",
            "",
            "| t | Metric | Few - Many | 95% class-bootstrap CI |",
            "| ---: | --- | ---: | --- |",
        ]
    )
    for row in summary["tail_selectivity"]:
        lines.append(
            f"| {'all' if int(row['timestep']) == -1 else row['timestep']} | "
            f"{row['metric']} | {row['estimate']:.6g} | "
            f"[{row['low']:.6g}, {row['high']:.6g}] |"
        )
    lines.extend(
        [
            "",
            f"Zero-factor/use_cm-off maximum absolute mismatch: "
            f"`{summary['zero_validation_max_abs']:.6g}`.",
            f"Maximum random-intervention BA spectrum error: "
            f"`{summary['max_random_spectrum_relative_error']:.6g}`.",
            f"Bit-exact factor restoration verified: `{summary['restoration_verified']}`.",
            "",
            "These endpoints are causal for local model predictions at fixed noisy ",
            "inputs, but are not substitutes for end-to-end sampling, requested-class ",
            "accuracy, or per-class generative quality.",
            "The response-matched endpoint is a target-free final-displacement ",
            "sensitivity control; it is not itself a realizable weight intervention.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
