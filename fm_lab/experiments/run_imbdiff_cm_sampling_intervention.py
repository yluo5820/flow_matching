"""Run matched end-to-end sampling interventions for official ImbDiff-CM."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_probe import restore_imbdiff_cm_probe_checkpoint
from fm_lab.diagnostics.imbdiff_cm_sampling import (
    endpoint_response_scales,
    paired_sampling_effects,
    quality_contrasts,
    sample_matched_cm_interventions,
)
from fm_lab.evaluation.cache import FeatureCache, load_feature_cache, save_feature_cache
from fm_lab.evaluation.features import extract_inception_features
from fm_lab.evaluation.groups import frequency_ranked_groups
from fm_lab.evaluation.inception import ReferenceInceptionV3
from fm_lab.evaluation.metrics import fid_score, kid_subset_scores
from fm_lab.experiments.factory import build_target, resolve_device
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.seeding import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--random-repeats", type=int, default=2)
    parser.add_argument(
        "--random-effects",
        help=(
            "Prior intervention random_repeat_effects.csv. Its per-repeat local "
            "response scales are applied to the same random rotations."
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--endpoint-calibration-samples-per-class",
        type=int,
        default=1,
        help=(
            "Independent balanced DDIM pilot size used to match random and learned "
            "endpoint response RMS. Set to 0 to retain only local-probe calibration."
        ),
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "off", "bf16", "fp16"),
        default="off",
    )
    parser.add_argument(
        "--channels-last",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument(
        "--evaluation",
        choices=("none", "kid", "full"),
        default="kid",
        help="'kid' is the bounded screen; 'full' additionally computes official FID.",
    )
    parser.add_argument("--real-cache")
    parser.add_argument("--inception-weights")
    parser.add_argument("--kid-subsets", type=int, default=20)
    parser.add_argument("--kid-subset-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    seed_everything(args.seed)
    checkpoint_path = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("CM sampling checkpoint is missing its config.")
    target = build_target(config)
    class_counts = tuple(int(value) for value in target.class_counts)
    if len(class_counts) != 100:
        raise ValueError("The matched sampling intervention currently requires CIFAR-100.")
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
    local_response_scales = (
        load_response_scales(args.random_effects, random_repeats=args.random_repeats)
        if args.random_effects
        else {repeat: 1.0 for repeat in range(args.random_repeats)}
    )
    endpoint_calibration: dict[str, Any] = {"enabled": False}
    response_scales = local_response_scales
    if args.endpoint_calibration_samples_per_class > 0:
        calibration_input_seed = args.seed + 104_729
        calibration_payload, calibration_manifest = sample_matched_cm_interventions(
            restored,
            samples_per_class=args.endpoint_calibration_samples_per_class,
            batch_size=args.sample_batch_size,
            random_repeats=args.random_repeats,
            seed=args.seed,
            input_seed=calibration_input_seed,
            response_scales=local_response_scales,
            mixed_precision=args.mixed_precision,
        )
        calibration_payload.pop("labels")
        calibration_payload.pop("initial_noise")
        response_scales, endpoint_calibration = endpoint_response_scales(
            calibration_payload,
            base_scales=local_response_scales,
        )
        endpoint_calibration.update(
            {
                "enabled": True,
                "samples_per_class": int(args.endpoint_calibration_samples_per_class),
                "input_seed": int(calibration_input_seed),
                "labels_sha256": calibration_manifest["labels_sha256"],
                "initial_noise_sha256": calibration_manifest["initial_noise_sha256"],
            }
        )
    condition_payload, intervention_manifest = sample_matched_cm_interventions(
        restored,
        samples_per_class=args.samples_per_class,
        batch_size=args.sample_batch_size,
        random_repeats=args.random_repeats,
        seed=args.seed,
        response_scales=response_scales,
        mixed_precision=args.mixed_precision,
    )
    labels = condition_payload.pop("labels")
    initial_noise = condition_payload.pop("initial_noise")
    paired_rows, paired_groups = paired_sampling_effects(
        condition_payload,
        labels=labels,
        class_counts=class_counts,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    features_dir = output_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "labels.npy", labels.numpy())
    np.save(output_dir / "initial_noise.npy", initial_noise.numpy())
    for condition, samples in condition_payload.items():
        np.save(samples_dir / f"{condition}.npy", samples.numpy())
    intervention_manifest["response_scale_calibration"] = {
        "source": str(Path(args.random_effects).resolve()) if args.random_effects else None,
        "local_probe_scales": {
            str(key): value for key, value in sorted(local_response_scales.items())
        },
        "scales": {str(key): value for key, value in sorted(response_scales.items())},
        "endpoint_calibration": endpoint_calibration,
    }
    _write_json(output_dir / "sampling_manifest.json", intervention_manifest)
    _write_csv(output_dir / "paired_effects.csv", paired_rows)
    _write_csv(output_dir / "paired_group_summary.csv", paired_groups)

    condition_metrics: dict[str, dict[str, Any]] = {}
    if args.evaluation != "none":
        condition_metrics = _evaluate_conditions(
            condition_payload,
            labels=labels.numpy(),
            class_counts=class_counts,
            real_cache_path=Path(args.real_cache),
            inception_weights=Path(args.inception_weights),
            feature_batch_size=args.feature_batch_size,
            device=device,
            kid_subsets=args.kid_subsets,
            kid_subset_size=args.kid_subset_size,
            seed=args.seed,
            compute_fid=args.evaluation == "full",
            features_dir=features_dir,
        )
    summary = {
        "schema_version": 1,
        "checkpoint": intervention_manifest["checkpoint"],
        "request": {
            "samples_per_class": int(args.samples_per_class),
            "sample_batch_size": int(args.sample_batch_size),
            "feature_batch_size": int(args.feature_batch_size),
            "random_repeats": int(args.random_repeats),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "endpoint_calibration_samples_per_class": int(
                args.endpoint_calibration_samples_per_class
            ),
            "mixed_precision": str(args.mixed_precision),
            "channels_last": bool(channels_last),
            "evaluation": str(args.evaluation),
            "seed": int(args.seed),
            "device": str(device),
        },
        "matched_inputs": {
            "labels_sha256": intervention_manifest["labels_sha256"],
            "initial_noise_sha256": intervention_manifest["initial_noise_sha256"],
            "conditions": sorted(condition_payload),
        },
        "response_scale_calibration": intervention_manifest["response_scale_calibration"],
        "paired_output_summary": paired_groups,
        "condition_metrics": condition_metrics,
        "quality_contrasts": (quality_contrasts(condition_metrics) if condition_metrics else {}),
        "interpretation_boundary": (
            "The KID screen is a bounded triage experiment. Its matched condition "
            "differences are more informative than its absolute small-sample values. "
            "A positive signal should be confirmed with evaluation='full' and the "
            "established 100-samples-per-class FID protocol."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    print(f"Finished CM matched sampling intervention: {output_dir}")


def load_response_scales(
    path: str | Path,
    *,
    random_repeats: int,
) -> dict[int, float]:
    """Average prior response-match scales over probe timesteps, once per repeat."""

    by_repeat_timestep: dict[tuple[int, int], list[float]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["random_repeat"]), int(row["timestep"]))
            by_repeat_timestep[key].append(float(row["response_match_scale"]))
    result = {}
    for repeat in range(int(random_repeats)):
        timestep_values = []
        for (row_repeat, _), values in by_repeat_timestep.items():
            if row_repeat == repeat:
                timestep_values.append(float(np.mean(values)))
        if not timestep_values:
            raise ValueError(f"Response calibration is missing random repeat {repeat}.")
        result[repeat] = float(np.mean(timestep_values))
    return result


def _evaluate_conditions(
    condition_samples: dict[str, torch.Tensor],
    *,
    labels: np.ndarray,
    class_counts: tuple[int, ...],
    real_cache_path: Path,
    inception_weights: Path,
    feature_batch_size: int,
    device: torch.device,
    kid_subsets: int,
    kid_subset_size: int,
    seed: int,
    compute_fid: bool,
    features_dir: Path,
) -> dict[str, dict[str, Any]]:
    real = load_feature_cache(real_cache_path)
    model = ReferenceInceptionV3(inception_weights)
    groups = frequency_ranked_groups(class_counts)
    features_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for condition, samples in condition_samples.items():
        cache = extract_inception_features(
            samples,
            labels=labels,
            sample_ids=np.asarray(
                [f"{condition}:{index}" for index in range(len(labels))],
                dtype=str,
            ),
            model=model,
            batch_size=feature_batch_size,
            device=device,
            input_range=(-1.0, 1.0),
            provenance={
                "dataset": "cifar100",
                "split": "generated",
                "condition": condition,
                "extractor": "tf_fid_inception_v3",
                "weights_sha256": model.weights_sha256,
                "evaluator_version": 1,
            },
        )
        save_feature_cache(features_dir / f"{condition}.npz", cache)
        result[condition] = _condition_metrics(
            cache,
            real,
            groups=groups,
            kid_subsets=kid_subsets,
            kid_subset_size=kid_subset_size,
            seed=seed,
            compute_fid=compute_fid,
        )
    return result


def _condition_metrics(
    generated: FeatureCache,
    real: FeatureCache,
    *,
    groups: dict[str, list[int]],
    kid_subsets: int,
    kid_subset_size: int,
    seed: int,
    compute_fid: bool,
) -> dict[str, Any]:
    overall_kid = kid_subset_scores(
        generated.features,
        real.features,
        num_subsets=kid_subsets,
        max_subset_size=kid_subset_size,
        seed=seed,
    )
    result: dict[str, Any] = {
        "num_samples": int(len(generated.features)),
        "kid": float(overall_kid.mean()),
        "kid_subset_estimates": overall_kid.tolist(),
        "groups": {},
    }
    if compute_fid:
        result["fid"] = fid_score(generated.features, real.features)
    for group_index, group_name in enumerate(("many", "medium", "few")):
        generated_mask = np.isin(generated.labels, groups[group_name])
        real_mask = np.isin(real.labels, groups[group_name])
        group_kid = kid_subset_scores(
            generated.features[generated_mask],
            real.features[real_mask],
            num_subsets=kid_subsets,
            max_subset_size=kid_subset_size,
            seed=seed + group_index + 1,
        )
        group_metrics = {
            "kid": float(group_kid.mean()),
            "kid_subset_estimates": group_kid.tolist(),
        }
        if compute_fid:
            group_metrics["fid"] = fid_score(
                generated.features[generated_mask],
                real.features[real_mask],
            )
        result["groups"][group_name] = group_metrics
    return result


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--samples-per-class": args.samples_per_class,
        "--sample-batch-size": args.sample_batch_size,
        "--feature-batch-size": args.feature_batch_size,
        "--random-repeats": args.random_repeats,
        "--bootstrap-repeats": args.bootstrap_repeats,
        "--kid-subsets": args.kid_subsets,
        "--kid-subset-size": args.kid_subset_size,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError("Positive values required for " + ", ".join(invalid))
    if int(args.endpoint_calibration_samples_per_class) < 0:
        raise ValueError("--endpoint-calibration-samples-per-class cannot be negative.")
    if args.evaluation != "none" and (not args.real_cache or not args.inception_weights):
        raise ValueError(
            "--real-cache and --inception-weights are required when evaluation is enabled."
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


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ImbDiff-CM matched end-to-end sampling intervention",
        "",
        "Every condition uses the same requested labels, initial Gaussian noise, ",
        "DDIM schedule, and guidance settings. `general` zeros the expert branch. ",
        "`random_*` rotates the expert subspaces while preserving each layer's ",
        "learned singular spectrum, then applies the response scale calibrated by ",
        "the prior local intervention probe.",
        "",
        "## Paired output displacement",
        "",
        "| Group | Learned–general RMS | Random–general RMS | Learned–random RMS |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["paired_output_summary"]:
        lines.append(
            f"| {row['frequency_group']} | "
            f"{row['learned_vs_general_rms_mean']:.6f} | "
            f"{row['random_vs_general_rms_mean_mean']:.6f} | "
            f"{row['learned_vs_random_rms_mean_mean']:.6f} |"
        )
    contrasts = summary.get("quality_contrasts", {})
    if contrasts:
        lines.extend(
            [
                "",
                "## Quality contrasts",
                "",
                "Positive gain/advantage means the learned expert has lower KID/FID.",
                "",
                "| Scope | Metric | Learned | General | Random mean | Gain vs general | "
                "Advantage vs random |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric, row in contrasts.get("overall", {}).items():
            lines.append(_contrast_row("overall", metric, row))
        for group_name, metrics in contrasts.get("groups", {}).items():
            for metric, row in metrics.items():
                lines.append(_contrast_row(group_name, metric, row))
    lines.extend(
        [
            "",
            "## Interpretation gate",
            "",
            "- Learned better than general: the learned expert improves end-to-end quality.",
            "- Learned better than response-calibrated random: expert orientation, not only "
            "extra response magnitude, matters.",
            "- A larger Few than Many gain is evidence of tail-selective allocation.",
            "- Similar gains across groups support a generic correction/regularization account.",
            "",
            summary["interpretation_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def _contrast_row(scope: str, metric: str, row: dict[str, float]) -> str:
    return (
        f"| {scope} | {metric} | {row['learned']:.6f} | {row['general']:.6f} | "
        f"{row['random_mean']:.6f} | {row['learned_gain_vs_general']:.6f} | "
        f"{row['learned_advantage_vs_random_mean']:.6f} |"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
