"""SQLite registry for the unified geometry explorer workspace."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_WORKSPACE = Path("outputs/geometry_explorer")


@dataclass(frozen=True)
class DatasetVariantRecord:
    variant_id: str
    family: str
    variant: str
    dataset_path: Path
    row_count: int
    label_counts: dict[str, int]
    split: str


@dataclass(frozen=True)
class ProjectionViewRecord:
    view_id: str
    variant_id: str
    feature_name: str
    explorer_data_path: Path
    projection_names: dict[str, str]


@dataclass(frozen=True)
class ModelRunRecord:
    run_id: str
    variant_id: str | None
    run_dir: Path
    family: str
    variant: str


@dataclass(frozen=True)
class TrajectoryViewRecord:
    view_id: str
    run_id: str
    variant_id: str | None
    solver: str
    nfe: int
    coordinates_path: Path
    trajectory_path: Path
    generated_path: Path | None
    target_path: Path | None
    labels_path: Path | None


class GeometryRegistry:
    """Small SQLite catalog for geometry explorer artifacts."""

    def __init__(self, workspace: str | Path = DEFAULT_WORKSPACE) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.path = self.workspace / "registry.sqlite"
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def register_dataset_variant(
        self,
        *,
        variant_id: str,
        family: str,
        variant: str,
        base: str,
        split: str,
        dataset_path: str | Path,
        data_path: str | Path | None,
        labels_path: str | Path | None,
        config_path: str | Path | None,
        row_count: int,
        label_counts: dict[str, int],
        image_shape: Iterable[int] | None,
        value_range: Iterable[float] | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_variants (
                    variant_id, family, variant, base, split, dataset_path,
                    data_path, labels_path, config_path, row_count,
                    label_counts_json, image_shape_json, value_range_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(variant_id) DO UPDATE SET
                    family=excluded.family,
                    variant=excluded.variant,
                    base=excluded.base,
                    split=excluded.split,
                    dataset_path=excluded.dataset_path,
                    data_path=excluded.data_path,
                    labels_path=excluded.labels_path,
                    config_path=excluded.config_path,
                    row_count=excluded.row_count,
                    label_counts_json=excluded.label_counts_json,
                    image_shape_json=excluded.image_shape_json,
                    value_range_json=excluded.value_range_json
                """,
                (
                    variant_id,
                    family,
                    variant,
                    base,
                    split,
                    self._relative(dataset_path),
                    self._relative(data_path) if data_path else None,
                    self._relative(labels_path) if labels_path else None,
                    self._relative(config_path) if config_path else None,
                    int(row_count),
                    json.dumps(label_counts, sort_keys=True),
                    json.dumps(list(image_shape)) if image_shape is not None else None,
                    json.dumps(list(value_range)) if value_range is not None else None,
                    _timestamp(),
                ),
            )

    def register_projection_view(
        self,
        *,
        view_id: str,
        variant_id: str,
        feature_name: str,
        feature_mode: str,
        explorer_data_path: str | Path,
        output_dir: str | Path,
        projection_names: dict[str, str],
        renderer: str,
        row_count: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO projection_views (
                    view_id, variant_id, feature_name, feature_mode,
                    explorer_data_path, output_dir, projection_names_json,
                    renderer, row_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(view_id) DO UPDATE SET
                    variant_id=excluded.variant_id,
                    feature_name=excluded.feature_name,
                    feature_mode=excluded.feature_mode,
                    explorer_data_path=excluded.explorer_data_path,
                    output_dir=excluded.output_dir,
                    projection_names_json=excluded.projection_names_json,
                    renderer=excluded.renderer,
                    row_count=excluded.row_count
                """,
                (
                    view_id,
                    variant_id,
                    feature_name,
                    feature_mode,
                    self._relative(explorer_data_path),
                    self._relative(output_dir),
                    json.dumps(projection_names, sort_keys=True),
                    renderer,
                    int(row_count),
                    _timestamp(),
                ),
            )

    def register_model_run(
        self,
        *,
        run_id: str,
        run_dir: str | Path,
        variant_id: str | None,
        family: str,
        variant: str,
        config_path: str | Path | None,
        metrics_path: str | Path | None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO model_runs (
                    run_id, variant_id, family, variant, run_dir,
                    config_path, metrics_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    variant_id=excluded.variant_id,
                    family=excluded.family,
                    variant=excluded.variant,
                    run_dir=excluded.run_dir,
                    config_path=excluded.config_path,
                    metrics_path=excluded.metrics_path
                """,
                (
                    run_id,
                    variant_id,
                    family,
                    variant,
                    self._relative(run_dir),
                    self._relative(config_path) if config_path else None,
                    self._relative(metrics_path) if metrics_path else None,
                    _timestamp(),
                ),
            )

    def register_trajectory_view(
        self,
        *,
        view_id: str,
        run_id: str,
        variant_id: str | None,
        solver: str,
        nfe: int,
        coordinates_path: str | Path,
        trajectory_path: str | Path,
        generated_path: str | Path | None,
        target_path: str | Path | None,
        labels_path: str | Path | None,
        output_dir: str | Path,
        interactive_path: str | Path | None,
        n_steps: int,
        n_trajectories: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO trajectory_views (
                    view_id, run_id, variant_id, solver, nfe, coordinates_path,
                    trajectory_path, generated_path, target_path, labels_path,
                    output_dir, interactive_path, n_steps, n_trajectories, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(view_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    variant_id=excluded.variant_id,
                    solver=excluded.solver,
                    nfe=excluded.nfe,
                    coordinates_path=excluded.coordinates_path,
                    trajectory_path=excluded.trajectory_path,
                    generated_path=excluded.generated_path,
                    target_path=excluded.target_path,
                    labels_path=excluded.labels_path,
                    output_dir=excluded.output_dir,
                    interactive_path=excluded.interactive_path,
                    n_steps=excluded.n_steps,
                    n_trajectories=excluded.n_trajectories
                """,
                (
                    view_id,
                    run_id,
                    variant_id,
                    solver,
                    int(nfe),
                    self._relative(coordinates_path),
                    self._relative(trajectory_path),
                    self._relative(generated_path) if generated_path else None,
                    self._relative(target_path) if target_path else None,
                    self._relative(labels_path) if labels_path else None,
                    self._relative(output_dir),
                    self._relative(interactive_path) if interactive_path else None,
                    int(n_steps),
                    int(n_trajectories),
                    _timestamp(),
                ),
            )

    def dataset_variants(self) -> list[DatasetVariantRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM dataset_variants ORDER BY family, variant"
            ).fetchall()
        return [
            DatasetVariantRecord(
                variant_id=row["variant_id"],
                family=row["family"],
                variant=row["variant"],
                dataset_path=self.resolve(row["dataset_path"]),
                row_count=int(row["row_count"]),
                label_counts=json.loads(row["label_counts_json"] or "{}"),
                split=row["split"],
            )
            for row in rows
        ]

    def projection_views(self, variant_id: str | None = None) -> list[ProjectionViewRecord]:
        query = "SELECT * FROM projection_views"
        params: tuple[Any, ...] = ()
        if variant_id is not None:
            query += " WHERE variant_id = ?"
            params = (variant_id,)
        query += " ORDER BY feature_name, view_id"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            ProjectionViewRecord(
                view_id=row["view_id"],
                variant_id=row["variant_id"],
                feature_name=row["feature_name"],
                explorer_data_path=self.resolve(row["explorer_data_path"]),
                projection_names=json.loads(row["projection_names_json"] or "{}"),
            )
            for row in rows
        ]

    def model_runs(self, variant_id: str | None = None) -> list[ModelRunRecord]:
        query = "SELECT * FROM model_runs"
        params: tuple[Any, ...] = ()
        if variant_id is not None:
            query += " WHERE variant_id = ?"
            params = (variant_id,)
        query += " ORDER BY run_id"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            ModelRunRecord(
                run_id=row["run_id"],
                variant_id=row["variant_id"],
                run_dir=self.resolve(row["run_dir"]),
                family=row["family"],
                variant=row["variant"],
            )
            for row in rows
        ]

    def trajectory_views(
        self,
        *,
        variant_id: str | None = None,
        run_id: str | None = None,
    ) -> list[TrajectoryViewRecord]:
        clauses = []
        params: list[Any] = []
        if variant_id is not None:
            clauses.append("variant_id = ?")
            params.append(variant_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        query = "SELECT * FROM trajectory_views"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY run_id, solver, nfe"
        with self.connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [
            TrajectoryViewRecord(
                view_id=row["view_id"],
                run_id=row["run_id"],
                variant_id=row["variant_id"],
                solver=row["solver"],
                nfe=int(row["nfe"]),
                coordinates_path=self.resolve(row["coordinates_path"]),
                trajectory_path=self.resolve(row["trajectory_path"]),
                generated_path=(
                    self.resolve(row["generated_path"])
                    if row["generated_path"]
                    else None
                ),
                target_path=self.resolve(row["target_path"]) if row["target_path"] else None,
                labels_path=self.resolve(row["labels_path"]) if row["labels_path"] else None,
            )
            for row in rows
        ]

    def get_dataset_variant(self, variant_id: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_variants WHERE variant_id = ?",
                (variant_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown dataset variant: {variant_id}")
        return row

    def get_projection_view(self, view_id: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM projection_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown projection view: {view_id}")
        return row

    def get_trajectory_view(self, view_id: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM trajectory_views WHERE view_id = ?",
                (view_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown trajectory view: {view_id}")
        return row

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.workspace / path).resolve()

    def _relative(self, value: str | Path) -> str:
        path = Path(value).expanduser().resolve()
        try:
            return str(path.relative_to(self.workspace))
        except ValueError:
            return str(path)

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_variants (
                    variant_id TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    base TEXT NOT NULL,
                    split TEXT NOT NULL,
                    dataset_path TEXT NOT NULL,
                    data_path TEXT,
                    labels_path TEXT,
                    config_path TEXT,
                    row_count INTEGER NOT NULL,
                    label_counts_json TEXT NOT NULL,
                    image_shape_json TEXT,
                    value_range_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projection_views (
                    view_id TEXT PRIMARY KEY,
                    variant_id TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_mode TEXT NOT NULL,
                    explorer_data_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    projection_names_json TEXT NOT NULL,
                    renderer TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(variant_id) REFERENCES dataset_variants(variant_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id TEXT PRIMARY KEY,
                    variant_id TEXT,
                    family TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    config_path TEXT,
                    metrics_path TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trajectory_views (
                    view_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    variant_id TEXT,
                    solver TEXT NOT NULL,
                    nfe INTEGER NOT NULL,
                    coordinates_path TEXT NOT NULL,
                    trajectory_path TEXT NOT NULL,
                    generated_path TEXT,
                    target_path TEXT,
                    labels_path TEXT,
                    output_dir TEXT NOT NULL,
                    interactive_path TEXT,
                    n_steps INTEGER NOT NULL,
                    n_trajectories INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES model_runs(run_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )


def variant_id(family: str, variant: str) -> str:
    return f"{family}/{variant}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
