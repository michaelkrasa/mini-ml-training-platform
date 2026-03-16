"""Train a lightweight perception model and register it in MLflow."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import mlflow
import numpy as np
import torch
from mlflow.models import infer_signature
from torch import nn
from torch.utils.data import DataLoader

from ml_platform.config import PlatformSettings, get_settings
from ml_platform.constants import DEFAULT_IMAGE_SIZE
from ml_platform.data.kitti import KittiScenePresenceDataset
from ml_platform.data.manifest import (
    DatasetManifest,
    build_dataset_manifest,
    load_manifest,
    manifest_records,
)
from ml_platform.registry.mlflow_registry import (
    configure_tracking,
    current_git_commit,
    register_run_model,
)
from ml_platform.training.metrics import batch_classification_stats
from ml_platform.training.model import ScenePresenceClassifier
from ml_platform.utils.io import write_json
from ml_platform.utils.seeding import set_global_seed


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 8
    batch_size: int = 8
    learning_rate: float = 1e-3
    val_ratio: float = 0.2
    image_size: int = DEFAULT_IMAGE_SIZE[0]
    num_workers: int = 0
    device: str = "auto"


def _resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataloaders(
    manifest: DatasetManifest,
    settings: PlatformSettings,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader | None]:
    image_size = (config.image_size, config.image_size)
    train_records = manifest_records(manifest, settings.dataset_root, split="train")
    val_records = manifest_records(manifest, settings.dataset_root, split="val")
    if not train_records:
        raise RuntimeError("Manifest does not contain any training records.")

    train_dataset = KittiScenePresenceDataset(train_records, image_size=image_size)
    generator = torch.Generator().manual_seed(settings.training_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )

    if not val_records:
        return train_loader, None

    val_dataset = KittiScenePresenceDataset(val_records, image_size=image_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals = {
        "loss": 0.0,
        "label_accuracy": 0.0,
        "exact_match": 0.0,
        "macro_f1": 0.0,
        "examples": 0.0,
    }

    for features, targets, _ in loader:
        features = features.to(device)
        targets = targets.to(device)
        batch_size = float(features.shape[0])

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(features)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        stats = batch_classification_stats(logits.detach(), targets.detach())
        totals["loss"] += loss.item() * batch_size
        totals["label_accuracy"] += stats["label_accuracy"] * batch_size
        totals["exact_match"] += stats["exact_match"] * batch_size
        totals["macro_f1"] += stats["macro_f1"] * batch_size
        totals["examples"] += batch_size

    example_count = max(totals["examples"], 1.0)
    return {
        "loss": totals["loss"] / example_count,
        "label_accuracy": totals["label_accuracy"] / example_count,
        "exact_match": totals["exact_match"] / example_count,
        "macro_f1": totals["macro_f1"] / example_count,
    }


def _environment_metadata() -> dict[str, str]:
    packages = {
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
        "mlflow": importlib.metadata.version("mlflow"),
        "fastapi": importlib.metadata.version("fastapi"),
    }
    return packages


def _artifact_paths(run_directory: Path) -> dict[str, Path]:
    return {
        "manifest": run_directory / "dataset_manifest.json",
        "metadata": run_directory / "model_metadata.json",
        "summary": run_directory / "training_summary.json",
        "checkpoint": run_directory / "scene_presence_model.pt",
    }


def train(config: TrainingConfig, manifest_path: Path | None = None) -> dict[str, Any]:
    settings = get_settings()
    settings.ensure_directories()
    set_global_seed(settings.training_seed)

    manifest = (
        load_manifest(manifest_path)
        if manifest_path is not None
        else build_dataset_manifest(
            settings=settings,
            seed=settings.training_seed,
            val_ratio=config.val_ratio,
            persist=True,
        )
    )

    device = _resolve_device(config.device)
    train_loader, val_loader = _build_dataloaders(manifest, settings, config)
    model = ScenePresenceClassifier(num_classes=len(manifest.class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = settings.training_output_root / run_timestamp
    run_directory.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_paths(run_directory)

    best_state = copy.deepcopy(model.state_dict())
    best_metric = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        start_time = monotonic()
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = (
            _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            if val_loader is not None
            else {}
        )
        duration_seconds = monotonic() - start_time
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_label_accuracy": train_metrics["label_accuracy"],
            "train_exact_match": train_metrics["exact_match"],
            "train_macro_f1": train_metrics["macro_f1"],
            "epoch_duration_seconds": duration_seconds,
        }
        if val_metrics:
            epoch_record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_label_accuracy": val_metrics["label_accuracy"],
                    "val_exact_match": val_metrics["exact_match"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            ranking_metric = val_metrics["loss"]
        else:
            ranking_metric = train_metrics["loss"]
        history.append(epoch_record)

        if ranking_metric < best_metric:
            best_metric = ranking_metric
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model = model.to("cpu")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(manifest.class_names),
            "image_size": [config.image_size, config.image_size],
            "dataset_fingerprint": manifest.fingerprint,
        },
        artifacts["checkpoint"],
    )

    metadata = {
        "dataset_name": manifest.dataset_name,
        "task_type": manifest.task_type,
        "class_names": list(manifest.class_names),
        "image_size": [config.image_size, config.image_size],
        "dataset_fingerprint": manifest.fingerprint,
        "dataset_root": manifest.dataset_root,
        "environment": _environment_metadata(),
        "git_commit": current_git_commit(settings.project_root),
    }
    summary = {
        "run_timestamp": run_timestamp,
        "tracking_uri": settings.mlflow_tracking_uri,
        "experiment_name": settings.mlflow_experiment_name,
        "model_name": settings.mlflow_model_name,
        "model_alias": settings.mlflow_model_alias,
        "dataset_name": manifest.dataset_name,
        "task_type": manifest.task_type,
        "dataset_fingerprint": manifest.fingerprint,
        "class_names": list(manifest.class_names),
        "device": str(device),
        "history": history,
        "training_config": asdict(config),
    }

    write_json(artifacts["manifest"], manifest.to_dict())
    write_json(artifacts["metadata"], metadata)
    write_json(artifacts["summary"], summary)

    client = configure_tracking(settings)
    run_name = f"train-{run_timestamp}-{manifest.fingerprint[:8]}"
    input_example = np.zeros(
        (1, 3, config.image_size, config.image_size), dtype=np.float32
    )
    example_output = model(torch.from_numpy(input_example)).detach().numpy()
    signature = infer_signature(input_example, example_output)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "dataset_fingerprint": manifest.fingerprint,
                "dataset_name": manifest.dataset_name,
                "git_commit": metadata["git_commit"],
                "task": manifest.task_type,
                "task_type": manifest.task_type,
                "framework": "pytorch",
            }
        )
        mlflow.log_params(
            {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "val_ratio": config.val_ratio,
                "image_size": config.image_size,
                "dataset_name": manifest.dataset_name,
                "task_type": manifest.task_type,
                "seed": settings.training_seed,
                "train_examples": manifest.split_counts()["train"],
                "val_examples": manifest.split_counts().get("val", 0),
            }
        )
        for epoch_index, record in enumerate(history, start=1):
            mlflow.log_metrics(record, step=epoch_index)

        mlflow.log_artifact(str(artifacts["manifest"]), artifact_path="manifests")
        mlflow.log_artifact(str(artifacts["metadata"]), artifact_path="metadata")
        mlflow.log_artifact(str(artifacts["summary"]), artifact_path="metadata")
        mlflow.log_artifact(str(artifacts["checkpoint"]), artifact_path="checkpoints")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        model_version = register_run_model(
            client=client,
            run_id=run.info.run_id,
            model_name=settings.mlflow_model_name,
            alias=settings.mlflow_model_alias,
        )

        summary.update(
            {
                "run_id": run.info.run_id,
                "model_version": str(model_version.version),
                "model_source": getattr(model_version, "source", None),
            }
        )
        write_json(artifacts["summary"], summary)
        mlflow.log_artifact(str(artifacts["summary"]), artifact_path="metadata")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional existing dataset manifest to reuse for training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train(
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            image_size=args.image_size,
            num_workers=args.num_workers,
            device=args.device,
        ),
        manifest_path=args.manifest_path,
    )
    print(result)


if __name__ == "__main__":
    main()
