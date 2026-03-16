"""Poll for dataset changes and retrain automatically."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic, sleep

from ml_platform.config import get_settings
from ml_platform.data.manifest import build_dataset_manifest
from ml_platform.registry.mlflow_registry import configure_tracking, resolve_model_pointer
from ml_platform.training.train import TrainingConfig, train


@dataclass(frozen=True)
class WatcherConfig:
    interval_seconds: int = 30
    cooldown_seconds: int = 60
    bootstrap_if_missing: bool = True
    training: TrainingConfig = TrainingConfig()


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] {message}", flush=True)


def _model_exists() -> bool:
    settings = get_settings()
    client = configure_tracking(settings)
    pointer = resolve_model_pointer(
        client,
        model_name=settings.mlflow_model_name,
        alias=settings.mlflow_model_alias,
    )
    return pointer is not None


def watch(config: WatcherConfig) -> None:
    settings = get_settings()
    trained_fingerprint: str | None = None
    last_attempt_started = 0.0

    while True:
        try:
            manifest = build_dataset_manifest(
                settings=settings,
                seed=settings.training_seed,
                val_ratio=config.training.val_ratio,
                persist=False,
            )
            current_fingerprint = manifest.fingerprint
            if trained_fingerprint is None:
                trained_fingerprint = current_fingerprint if _model_exists() else None

            needs_training = (
                trained_fingerprint is None
                or current_fingerprint != trained_fingerprint
            )
            cooldown_elapsed = (
                monotonic() - last_attempt_started >= config.cooldown_seconds
            )

            if needs_training and cooldown_elapsed:
                reason = (
                    "bootstrap because no registered model exists"
                    if trained_fingerprint is None
                    else f"dataset fingerprint changed to {current_fingerprint[:12]}"
                )
                _log(f"Starting retraining: {reason}.")
                last_attempt_started = monotonic()
                result = train(config.training)
                trained_fingerprint = current_fingerprint
                _log(
                    "Retraining completed: "
                    f"run_id={result['run_id']} model_version={result['model_version']}"
                )
            elif needs_training:
                _log("Dataset change detected, waiting for cooldown window before retraining.")
        except RuntimeError as exc:
            _log(f"Dataset unavailable: {exc}")
        except Exception as exc:
            _log(f"Watcher cycle failed: {exc}")

        sleep(config.interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = WatcherConfig(
        interval_seconds=args.interval_seconds,
        cooldown_seconds=args.cooldown_seconds,
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_ratio=args.val_ratio,
            image_size=args.image_size,
            num_workers=args.num_workers,
            device=args.device,
        ),
    )
    watch(config)


if __name__ == "__main__":
    main()
