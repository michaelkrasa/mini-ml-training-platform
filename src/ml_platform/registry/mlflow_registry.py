"""MLflow tracking and model registry utilities."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any

import mlflow
from mlflow import MlflowClient

from ml_platform.config import PlatformSettings


@dataclass(frozen=True)
class ModelRegistryPointer:
    model_name: str
    version: str
    run_id: str | None
    source: str | None
    uri: str
    alias: str | None = None


def configure_tracking(settings: PlatformSettings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def current_git_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip()


def download_run_artifact(
    client: MlflowClient, run_id: str, artifact_path: str
) -> str:
    return client.download_artifacts(run_id, artifact_path)


def wait_for_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int = 60,
) -> Any:
    for _ in range(timeout_seconds):
        model_version = client.get_model_version(model_name, version)
        status = str(getattr(model_version, "status", "")).upper()
        if not status or status.endswith("READY"):
            return model_version
        sleep(1)
    return client.get_model_version(model_name, version)


def assign_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
    version: str,
) -> None:
    try:
        client.set_registered_model_alias(model_name, alias, version)
        return
    except Exception:
        pass
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception:
        pass


def register_run_model(
    client: MlflowClient,
    run_id: str,
    model_name: str,
    alias: str | None = None,
) -> Any:
    registration = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name,
    )
    model_version = wait_for_model_version(
        client=client,
        model_name=model_name,
        version=str(registration.version),
    )
    if alias:
        assign_alias(client, model_name=model_name, alias=alias, version=model_version.version)
    return model_version


def resolve_model_pointer(
    client: MlflowClient, model_name: str, alias: str | None = None
) -> ModelRegistryPointer | None:
    if alias:
        try:
            model_version = client.get_model_version_by_alias(model_name, alias)
            return ModelRegistryPointer(
                model_name=model_name,
                version=str(model_version.version),
                run_id=getattr(model_version, "run_id", None),
                source=getattr(model_version, "source", None),
                uri=f"models:/{model_name}/{model_version.version}",
                alias=alias,
            )
        except Exception:
            pass

    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if not versions:
        return None

    production_versions = [
        version
        for version in versions
        if str(getattr(version, "current_stage", "")).lower() == "production"
    ]
    candidates = production_versions or versions
    latest = max(candidates, key=lambda item: int(item.version))
    resolved_alias = alias if alias and production_versions else None
    return ModelRegistryPointer(
        model_name=model_name,
        version=str(latest.version),
        run_id=getattr(latest, "run_id", None),
        source=getattr(latest, "source", None),
        uri=f"models:/{model_name}/{latest.version}",
        alias=resolved_alias,
    )
