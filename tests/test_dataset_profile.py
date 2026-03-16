from pathlib import Path

from PIL import Image

from ml_platform.config import PlatformSettings
from ml_platform.constants import SINGLE_LABEL_TASK
from ml_platform.data.manifest import build_dataset_manifest
from ml_platform.data.profile import DatasetProfile, write_dataset_profile


def _settings(root: Path) -> PlatformSettings:
    dataset_root = root / "dataset"
    artifacts_root = root / "artifacts"
    mlflow_root = root / "mlflow"
    mlartifacts_root = root / "mlartifacts"
    training_output_root = artifacts_root / "training"
    settings = PlatformSettings(
        project_root=root,
        dataset_root=dataset_root,
        images_dir=dataset_root / "images",
        labels_dir=dataset_root / "labels",
        manifests_dir=dataset_root / "manifests",
        artifacts_root=artifacts_root,
        mlflow_root=mlflow_root,
        mlartifacts_root=mlartifacts_root,
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_experiment_name="test-experiment",
        mlflow_model_name="test-model",
        mlflow_model_alias="champion",
        model_refresh_interval_seconds=30,
        training_seed=7,
        training_output_root=training_output_root,
    )
    settings.ensure_directories()
    return settings


def test_build_dataset_manifest_uses_dataset_profile(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    Image.new("RGB", (32, 32), color=(100, 50, 25)).save(
        settings.images_dir / "train_000000.png"
    )
    (settings.labels_dir / "train_000000.txt").write_text("cat\n", encoding="utf-8")
    write_dataset_profile(
        settings,
        DatasetProfile(
            dataset_name="cifar10",
            source="torchvision:CIFAR-10",
            description="test profile",
            class_names=("cat", "dog"),
            task_type=SINGLE_LABEL_TASK,
        ),
    )

    manifest = build_dataset_manifest(settings, persist=False)

    assert manifest.dataset_name == "cifar10"
    assert manifest.task_type == SINGLE_LABEL_TASK
    assert manifest.class_names == ("cat", "dog")
