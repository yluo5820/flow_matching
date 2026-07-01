"""Human-readable labels for geometry explorer UI and diagnostics."""

from __future__ import annotations

import re

FAMILY_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "cifar10_grayscale": "CIFAR-10 grayscale",
}

FEATURE_LABELS = {
    "raw_pixels": "Raw pixels",
    "dinov2": "DINOv2 features",
}


def family_label(value: str) -> str:
    return FAMILY_LABELS.get(value, humanize_identifier(value))


def variant_label(value: str) -> str:
    if value == "original":
        return "Original"
    match = re.fullmatch(r"tail_digit(\d+)", value)
    if match:
        return f"Tail digit {match.group(1)}"
    if value == "long_tail_monotone":
        return "Monotone long tail"
    return humanize_identifier(value)


def feature_label(value: str) -> str:
    return FEATURE_LABELS.get(value, humanize_identifier(value))


def projection_view_label(
    *,
    feature_name: str,
    projection_names: dict[str, str],
) -> str:
    feature = feature_label(feature_name)
    count = len(projection_names)
    if count == 1:
        return f"{feature} · {next(iter(projection_names.values()))}"
    return f"{feature} · {count} projections"


def trajectory_view_label(*, run_id: str, solver: str, nfe: int) -> str:
    return f"{humanize_identifier(run_id)} · {humanize_identifier(solver)} · NFE {nfe}"


def metric_label(key: str) -> str:
    fm_jacobian = _fm_jacobian_metric_label(key)
    if fm_jacobian is not None:
        return fm_jacobian
    patterns = (
        (r"^mle_lid_k(\d+)$", "MLE intrinsic dimension (k={})"),
        (r"^global_mle_lid_k(\d+)$", "Global MLE intrinsic dimension (k={})"),
        (r"^mean_local_mle_lid_k(\d+)$", "Mean local MLE ID (k={})"),
        (r"^median_local_mle_lid_k(\d+)$", "Median local MLE ID (k={})"),
        (r"^participation_ratio_k(\d+)$", "Participation ratio (k={})"),
        (r"^mean_participation_ratio_k(\d+)$", "Mean participation ratio (k={})"),
        (r"^median_participation_ratio_k(\d+)$", "Median participation ratio (k={})"),
        (r"^knn_radius_k(\d+)$", "kNN radius (k={})"),
        (r"^knn_mean_distance_k(\d+)$", "Mean kNN distance (k={})"),
        (r"^ball_scaling_dim_k(\d+)$", "Ball-scaling dimension (k={})"),
        (r"^ball_scaling_r2_k(\d+)$", "Ball-scaling fit R2 (k={})"),
        (r"^pca_dim_(\d+)$", "PCA dimension ({}% variance)"),
        (r"^global_pca_dim_(\d+)$", "Global PCA dimension ({}% variance)"),
    )
    for pattern, template in patterns:
        match = re.fullmatch(pattern, key)
        if match:
            return template.format(match.group(1))
    exact = {
        "two_nn_lid": "TwoNN intrinsic dimension",
        "two_nn_lid_local": "Local TwoNN intrinsic dimension",
        "global_two_nn_lid": "Global TwoNN intrinsic dimension",
        "global_participation_ratio": "Global participation ratio",
        "correlation_dimension": "Correlation dimension",
        "ball_scaling_dim": "Ball-scaling dimension",
        "ball_scaling_r2": "Ball-scaling fit R2",
        "ball_scaling_num_radii": "Ball-scaling radius count",
        "outlier_score": "Outlier score",
        "distance_to_label_centroid": "Distance to class centroid",
        "label_agreement": "Local label agreement",
    }
    return exact.get(key, humanize_identifier(key))


def _fm_jacobian_metric_label(key: str) -> str | None:
    match = re.fullmatch(
        r"^(mean_|median_)?fm_jacobian_(participation|entropy|threshold)_rank_t(\d{4})$",
        key,
    )
    if not match:
        return None
    aggregate, kind, time_value = match.groups()
    kind_label = {
        "participation": "participation rank",
        "entropy": "entropy rank",
        "threshold": "threshold rank",
    }[kind]
    prefix = ""
    if aggregate == "mean_":
        prefix = "Mean "
    elif aggregate == "median_":
        prefix = "Median "
    return f"{prefix}FM Jacobian {kind_label} (t={int(time_value) / 1000:.3f})"


def humanize_identifier(value: str) -> str:
    text = str(value).replace("__", " ").replace("_", " ").replace("-", " ").strip()
    if not text:
        return ""
    words = []
    acronyms = {"id", "lid", "mle", "pca", "umap", "tsne", "nfe", "ot", "rgb"}
    for word in text.split():
        lower = word.lower()
        if lower in acronyms:
            words.append(lower.upper())
        elif lower == "unet":
            words.append("U-Net")
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)
