"""Run registry and exclusion log for the Observation-0 pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)

REGISTRY_COLUMNS = (
    "study_digest",
    "condition",
    "mapping_offset",
    "seed",
    "run_dir",
    "status",
    "measurement_digest",
    "exclusion_reason",
)
VALID_STATUSES = frozenset({"planned", "trained", "measured", "excluded"})


@dataclass(frozen=True)
class Observation0Run:
    study_digest: str
    condition: str
    mapping_offset: int
    seed: int
    run_dir: str
    status: str = "planned"
    measurement_digest: str = ""
    exclusion_reason: str = ""


def prepare_observation0_registry(
    preregistration: Observation0Preregistration,
    study_dir: str | Path,
) -> pd.DataFrame:
    """Lock the study protocol and create all preregistered pilot rows."""

    root = Path(study_dir)
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    preregistration.lock(aggregate / "preregistration.yaml")
    registry_path = aggregate / "run_registry.csv"
    exclusion_path = aggregate / "exclusion_log.csv"
    if registry_path.exists():
        registry = _read_registry(registry_path)
        _validate_registry_identity(registry, preregistration)
    else:
        records = [
            Observation0Run(
                study_digest=preregistration.digest,
                condition="mapping_0",
                mapping_offset=0,
                seed=seed,
                run_dir=str(root / "mapping_0" / f"seed_{seed}"),
            )
            for seed in preregistration.training_seeds
        ]
        registry = pd.DataFrame(
            [record.__dict__ for record in records],
            columns=REGISTRY_COLUMNS,
        )
        _atomic_write_csv(registry, registry_path)
    if not exclusion_path.exists():
        pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(exclusion_path, index=False)
    return registry


def update_observation0_run(
    study_dir: str | Path,
    *,
    seed: int,
    status: str,
    run_dir: str | Path | None = None,
    measurement_digest: str = "",
    exclusion_reason: str = "",
) -> pd.DataFrame:
    """Atomically update one registered seed without deleting failed runs."""

    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid Observation-0 registry status: {status}")
    root = Path(study_dir)
    registry_path = root / "aggregate" / "run_registry.csv"
    if not registry_path.exists():
        raise ValueError("Observation-0 registry does not exist; run prepare first.")
    registry = _read_registry(registry_path)
    matches = registry.index[registry["seed"] == int(seed)].tolist()
    if len(matches) != 1:
        raise ValueError(f"Training seed {seed} is not preregistered for Observation 0.")
    if status == "excluded" and not exclusion_reason.strip():
        raise ValueError("An excluded Observation-0 run requires a reason.")
    if status != "excluded" and exclusion_reason:
        raise ValueError("Only excluded Observation-0 runs may have an exclusion reason.")
    if status == "measured" and not _is_sha256(measurement_digest):
        raise ValueError("A measured Observation-0 run requires a measurement SHA-256.")
    index = matches[0]
    registry.loc[index, "status"] = status
    registry.loc[index, "measurement_digest"] = measurement_digest
    registry.loc[index, "exclusion_reason"] = exclusion_reason
    if run_dir is not None:
        registry.loc[index, "run_dir"] = str(run_dir)
    _atomic_write_csv(registry, registry_path)

    if status == "excluded":
        exclusion_path = root / "aggregate" / "exclusion_log.csv"
        exclusions = pd.read_csv(exclusion_path, keep_default_na=False)
        row = registry.loc[[index], list(REGISTRY_COLUMNS)]
        duplicate = (
            (exclusions["seed"] == int(seed))
            & (exclusions["exclusion_reason"] == exclusion_reason)
        ) if not exclusions.empty else pd.Series(dtype=bool)
        if exclusions.empty or not bool(duplicate.any()):
            exclusions = pd.concat([exclusions, row], ignore_index=True)
            _atomic_write_csv(exclusions, exclusion_path)
    return registry


def _read_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path, keep_default_na=False)
    if tuple(registry.columns) != REGISTRY_COLUMNS:
        raise ValueError("Observation-0 registry columns do not match the schema.")
    return registry


def _validate_registry_identity(
    registry: pd.DataFrame,
    preregistration: Observation0Preregistration,
) -> None:
    if set(registry["study_digest"]) != {preregistration.digest}:
        raise ValueError("Run registry belongs to a different locked preregistration.")
    if tuple(registry["seed"]) != preregistration.training_seeds:
        raise ValueError("Run registry seeds do not match the locked preregistration.")
    if set(registry["mapping_offset"]) != {0}:
        raise ValueError("Observation-0 registry contains a non-pilot mapping.")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
