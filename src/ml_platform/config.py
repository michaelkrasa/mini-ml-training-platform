"""Shared configuration derived from the runtime environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PlatformSettings:
    project_root: Path
    dataset_root: Path
    images_dir: Path
    labels_dir: Path
    manifests_dir: Path
    artifacts_root: Path
    mlflow_root: Path
    mlartifacts_root: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_model_name: str
    mlflow_model_alias: str
    model_refresh_interval_seconds: int
    training_seed: int
    training_output_root: Path

    def ensure_directories(self) -> None:
        for path in (
            self.dataset_root,
            self.images_dir,
            self.labels_dir,
            self.manifests_dir,
            self.artifacts_root,
            self.mlflow_root,
            self.mlartifacts_root,
            self.training_output_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> PlatformSettings:
    project_root = _project_root()
    dataset_root = Path(os.getenv("DATASET_ROOT", project_root / "dataset")).resolve()
    artifacts_root = Path(os.getenv("ARTIFACTS_ROOT", project_root / "artifacts")).resolve()
    mlflow_root = Path(os.getenv("MLFLOW_ROOT", project_root / "mlflow")).resolve()
    mlartifacts_root = Path(os.getenv("MLARTIFACTS_ROOT", project_root / "mlartifacts")).resolve()
    training_output_root = Path(
        os.getenv("TRAINING_OUTPUT_ROOT", artifacts_root / "training")
    ).resolve()

    settings = PlatformSettings(
        project_root=project_root,
        dataset_root=dataset_root,
        images_dir=dataset_root / "images",
        labels_dir=dataset_root / "labels",
        manifests_dir=dataset_root / "manifests",
        artifacts_root=artifacts_root,
        mlflow_root=mlflow_root,
        mlartifacts_root=mlartifacts_root,
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        mlflow_experiment_name=os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "mini-ml-training-platform"
        ),
        mlflow_model_name=os.getenv(
            "MLFLOW_MODEL_NAME", "kitti-presence-classifier"
        ),
        mlflow_model_alias=os.getenv("MLFLOW_MODEL_ALIAS", "champion"),
        model_refresh_interval_seconds=int(
            os.getenv("MODEL_REFRESH_INTERVAL_SECONDS", "30")
        ),
        training_seed=int(os.getenv("TRAINING_SEED", "7")),
        training_output_root=training_output_root,
    )
    settings.ensure_directories()
    return settings
