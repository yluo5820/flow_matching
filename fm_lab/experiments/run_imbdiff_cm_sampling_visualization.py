"""Visualize paired general-only, expert-residual, and full CM samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Matched sampling intervention directory containing samples and paired_effects.csv.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root", help="Optional CIFAR-100 root for semantic class names.")
    parser.add_argument("--groups", default="many,few")
    parser.add_argument("--classes-per-group", type=int, default=3)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--residual-quantile", type=float, default=0.995)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.classes_per_group < 1 or args.samples_per_class < 1:
        raise ValueError("Class and sample counts must be positive.")
    if not 0.5 < args.residual_quantile < 1.0:
        raise ValueError("--residual-quantile must lie between 0.5 and 1.")
    groups = tuple(value.strip().lower() for value in args.groups.split(",") if value.strip())
    if not groups or any(value not in {"many", "medium", "few"} for value in groups):
        raise ValueError("--groups must contain many, medium, and/or few.")

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualizations"
    general = _load_images(run_dir / "samples" / "general.npy")
    learned = _load_images(run_dir / "samples" / "learned.npy")
    labels = np.load(run_dir / "labels.npy").astype(np.int64)
    if general.shape != learned.shape or len(general) != len(labels):
        raise ValueError("General, learned, and label arrays must align.")
    rows = _load_paired_rows(run_dir / "paired_effects.csv")
    selected = select_visualization_rows(
        rows,
        groups=groups,
        classes_per_group=args.classes_per_group,
        samples_per_class=args.samples_per_class,
    )
    residual = learned - general
    residual_scale = float(np.quantile(np.abs(residual), args.residual_quantile))
    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("The learned-general residual has no finite nonzero scale.")
    class_names = load_cifar100_class_names(args.data_root) if args.data_root else ()

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "general_expert_residual_full.png"
    render_expert_residual_grid(
        general=general,
        learned=learned,
        selected=selected,
        residual_scale=residual_scale,
        residual_quantile=args.residual_quantile,
        class_names=class_names,
        output_path=image_path,
        dpi=args.dpi,
    )
    manifest = {
        "schema_version": 1,
        "source_run": str(run_dir.resolve()),
        "source_files": {
            "general": str((run_dir / "samples" / "general.npy").resolve()),
            "learned": str((run_dir / "samples" / "learned.npy").resolve()),
            "labels": str((run_dir / "labels.npy").resolve()),
            "paired_effects": str((run_dir / "paired_effects.csv").resolve()),
        },
        "selection": {
            "groups": list(groups),
            "classes_per_group": int(args.classes_per_group),
            "samples_per_class": int(args.samples_per_class),
            "rule": (
                "Frequency-quantile classes within each group; samples nearest "
                "their class median learned-versus-general RMS."
            ),
            "rows": selected,
        },
        "residual_display": {
            "definition": "learned - general",
            "zero_color": [0.5, 0.5, 0.5],
            "quantile": float(args.residual_quantile),
            "shared_absolute_scale": residual_scale,
            "mapping": "clip(0.5 + residual / (2 * shared_absolute_scale), 0, 1)",
        },
        "output": str(image_path.resolve()),
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Finished CM expert-residual visualization: {image_path}")


def select_visualization_rows(
    rows: Sequence[dict[str, Any]],
    *,
    groups: Sequence[str],
    classes_per_group: int,
    samples_per_class: int,
) -> list[dict[str, Any]]:
    """Select frequency-quantile classes and class-median residual examples."""

    by_group_class: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_group_class[str(row["frequency_group"])][int(row["class_id"])].append(dict(row))
    result: list[dict[str, Any]] = []
    for group_name in groups:
        classes = sorted(by_group_class[group_name])
        if len(classes) < int(classes_per_group):
            raise ValueError(f"Group {group_name} lacks enough classes.")
        quantile_positions = np.rint(
            np.linspace(0, len(classes) - 1, int(classes_per_group))
        ).astype(int)
        selected_classes = [classes[position] for position in quantile_positions]
        for class_id in selected_classes:
            candidates = by_group_class[group_name][class_id]
            values = np.asarray(
                [float(row["learned_vs_general_rms"]) for row in candidates],
                dtype=np.float64,
            )
            median = float(np.median(values))
            ranked = sorted(
                candidates,
                key=lambda row: (
                    abs(float(row["learned_vs_general_rms"]) - median),
                    int(row["sample_index"]),
                ),
            )
            if len(ranked) < int(samples_per_class):
                raise ValueError(f"Class {class_id} lacks enough paired samples.")
            for within_class_rank, row in enumerate(ranked[: int(samples_per_class)]):
                result.append(
                    {
                        "frequency_group": group_name,
                        "class_id": class_id,
                        "sample_index": int(row["sample_index"]),
                        "within_class_rank": within_class_rank,
                        "class_median_residual_rms": median,
                        "sample_residual_rms": float(row["learned_vs_general_rms"]),
                    }
                )
    return result


def render_expert_residual_grid(
    *,
    general: np.ndarray,
    learned: np.ndarray,
    selected: Sequence[dict[str, Any]],
    residual_scale: float,
    residual_quantile: float,
    class_names: Sequence[str],
    output_path: str | Path,
    dpi: int,
) -> None:
    """Render general | signed residual | full for every selected paired sample."""

    output_path = Path(output_path)
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(selected)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(8.2, max(2.0, 1.68 * n_rows)),
        squeeze=False,
    )
    for row_index, selection in enumerate(selected):
        sample_index = int(selection["sample_index"])
        general_image = _display_image(general[sample_index])
        learned_image = _display_image(learned[sample_index])
        signed_residual = np.clip(
            0.5 + (learned[sample_index] - general[sample_index]) / (2.0 * float(residual_scale)),
            0.0,
            1.0,
        )
        residual_image = _channels_last(signed_residual)
        for axis, image in zip(
            axes[row_index],
            (general_image, residual_image, learned_image),
            strict=True,
        ):
            axis.imshow(image, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
        class_id = int(selection["class_id"])
        class_name = (
            class_names[class_id].replace("_", " ")
            if class_id < len(class_names)
            else f"class {class_id}"
        )
        group = str(selection["frequency_group"]).capitalize()
        rms = float(selection["sample_residual_rms"])
        axes[row_index, 0].set_ylabel(
            f"{group} · {class_id}: {class_name}\nresidual RMS {rms:.4f}",
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )
        if (
            row_index > 0
            and selected[row_index - 1]["frequency_group"] != selection["frequency_group"]
        ):
            for axis in axes[row_index]:
                axis.spines["top"].set_visible(True)
                axis.spines["top"].set_linewidth(2.0)
                axis.spines["top"].set_color("0.15")
    axes[0, 0].set_title("General only", fontsize=11)
    axes[0, 1].set_title(
        "Expert residual: full − general\n"
        f"gray = 0, shared ±q{100 * residual_quantile:.1f} = {residual_scale:.4f}",
        fontsize=9,
    )
    axes[0, 2].set_title("Full parameters", fontsize=11)
    fig.suptitle(
        "ImbDiff-CM paired expert effect at identical class labels and initial noise",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0.15, 0.0, 1.0, 0.975), h_pad=0.35, w_pad=0.25)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_cifar100_class_names(data_root: str | Path) -> tuple[str, ...]:
    """Load CIFAR-100 names from either binary or Python-format data."""

    root = Path(data_root)
    text_candidates = (
        root / "cifar-100-binary" / "fine_label_names.txt",
        root / "fine_label_names.txt",
    )
    for path in text_candidates:
        if path.is_file():
            names = tuple(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if len(names) == 100:
                return names
    pickle_candidates = (
        root / "cifar-100-python" / "meta",
        root / "meta",
    )
    for path in pickle_candidates:
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        raw_names = payload.get(b"fine_label_names", payload.get("fine_label_names"))
        if raw_names is None:
            continue
        names = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_names
        )
        if len(names) == 100:
            return names
    raise FileNotFoundError(f"Could not find CIFAR-100 class names under {root}.")


def _load_paired_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_images(path: Path) -> np.ndarray:
    images = np.load(path).astype(np.float32)
    if images.ndim == 2 and images.shape[1] == 3 * 32 * 32:
        images = images.reshape(-1, 3, 32, 32)
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError(f"Expected CIFAR images at {path}, got {images.shape}.")
    return images


def _display_image(image: np.ndarray) -> np.ndarray:
    return np.clip((_channels_last(image) + 1.0) / 2.0, 0.0, 1.0)


def _channels_last(image: np.ndarray) -> np.ndarray:
    if image.shape != (3, 32, 32):
        raise ValueError("Expected a single CHW CIFAR image.")
    return np.transpose(image, (1, 2, 0))


if __name__ == "__main__":
    main()
