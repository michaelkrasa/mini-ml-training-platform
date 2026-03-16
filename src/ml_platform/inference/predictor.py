"""Model loading and prediction logic for the inference API."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any

import mlflow
import torch
from PIL import Image, UnidentifiedImageError

from ml_platform.config import PlatformSettings
from ml_platform.constants import DEFAULT_IMAGE_SIZE
from ml_platform.data.kitti import preprocess_image
from ml_platform.registry.mlflow_registry import (
    ModelRegistryPointer,
    configure_tracking,
    download_run_artifact,
    resolve_model_pointer,
)
from ml_platform.utils.io import read_json


class PredictorNotReady(RuntimeError):
    """Raised when inference is requested before a model is available."""


@dataclass(frozen=True)
class LoadedModelState:
    pointer: ModelRegistryPointer
    class_names: tuple[str, ...]
    image_size: tuple[int, int]
    dataset_fingerprint: str | None
    loaded_at_monotonic: float


class ModelPredictor:
    def __init__(self, settings: PlatformSettings) -> None:
        self.settings = settings
        self.client = configure_tracking(settings)
        self.device = torch.device("cpu")
        self._lock = RLock()
        self._model: torch.nn.Module | None = None
        self._state: LoadedModelState | None = None
        self._last_registry_check = 0.0

    def _load_metadata(self, run_id: str | None) -> dict[str, Any]:
        if not run_id:
            return {}
        try:
            artifact_path = download_run_artifact(
                self.client, run_id=run_id, artifact_path="metadata/model_metadata.json"
            )
        except Exception:
            return {}
        return read_json(Path(artifact_path))

    def load_latest(self, force: bool = False) -> bool:
        with self._lock:
            pointer = resolve_model_pointer(
                self.client,
                model_name=self.settings.mlflow_model_name,
                alias=self.settings.mlflow_model_alias,
            )
            if pointer is None:
                if self._model is None:
                    raise PredictorNotReady(
                        "No registered model version is available. Train and register a model first."
                    )
                return False

            if (
                not force
                and self._state is not None
                and self._state.pointer.version == pointer.version
            ):
                self._last_registry_check = monotonic()
                return False

            model = mlflow.pytorch.load_model(pointer.uri, map_location=self.device)
            model.eval()
            metadata = self._load_metadata(pointer.run_id)
            image_size = tuple(metadata.get("image_size", DEFAULT_IMAGE_SIZE))
            class_names = tuple(
                metadata.get("class_names") or metadata.get("labels") or ()
            )
            if not class_names:
                raise PredictorNotReady(
                    f"Model version {pointer.version} is missing class_names metadata."
                )

            self._model = model
            self._state = LoadedModelState(
                pointer=pointer,
                class_names=class_names,
                image_size=(int(image_size[0]), int(image_size[1])),
                dataset_fingerprint=metadata.get("dataset_fingerprint"),
                loaded_at_monotonic=monotonic(),
            )
            self._last_registry_check = monotonic()
            return True

    def maybe_refresh(self) -> bool:
        if monotonic() - self._last_registry_check < self.settings.model_refresh_interval_seconds:
            return False
        return self.load_latest(force=False)

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        self.maybe_refresh()
        with self._lock:
            if self._model is None or self._state is None:
                raise PredictorNotReady(
                    "The inference service has not loaded a registered model yet."
                )
            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    tensor = preprocess_image(
                        image, image_size=self._state.image_size
                    ).unsqueeze(0)
            except UnidentifiedImageError as exc:
                raise ValueError("Uploaded file is not a supported image.") from exc

            with torch.no_grad():
                logits = self._model(tensor.to(self.device))
                probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

            labels = dict(zip(self._state.class_names, probabilities, strict=True))
            predicted_labels = [
                class_name
                for class_name, probability in labels.items()
                if probability >= 0.5
            ]
            top_predictions = sorted(
                labels.items(), key=lambda item: item[1], reverse=True
            )[:3]
            return {
                "model_name": self.settings.mlflow_model_name,
                "model_version": self._state.pointer.version,
                "model_alias": self._state.pointer.alias,
                "dataset_fingerprint": self._state.dataset_fingerprint,
                "predicted_labels": predicted_labels,
                "top_predictions": [
                    {"label": label, "probability": round(probability, 4)}
                    for label, probability in top_predictions
                ],
                "probabilities": {
                    label: round(probability, 4)
                    for label, probability in labels.items()
                },
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._state is None:
                return {
                    "ready": False,
                    "model_name": self.settings.mlflow_model_name,
                    "model_alias": self.settings.mlflow_model_alias,
                }
            return {
                "ready": True,
                "model_name": self.settings.mlflow_model_name,
                "model_alias": self._state.pointer.alias,
                "model_version": self._state.pointer.version,
                "run_id": self._state.pointer.run_id,
                "dataset_fingerprint": self._state.dataset_fingerprint,
                "class_names": list(self._state.class_names),
                "image_size": list(self._state.image_size),
            }
