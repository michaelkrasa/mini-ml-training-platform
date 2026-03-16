"""End-to-end validation runner for benchmark datasets."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep


def _wait_for_port(host: str, port: int, timeout_seconds: int = 30) -> None:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.settimeout(0.5)
            if candidate.connect_ex((host, port)) == 0:
                sleep(1)
                return
        sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {host}:{port} to accept connections.")


def _start_mlflow_server(project_root: Path, port: int) -> subprocess.Popen[str]:
    backend_store = project_root / "mlflow" / f"validation-{port}.db"
    artifacts_root = project_root / "mlartifacts"
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{backend_store}",
        "--default-artifact-root",
        str(artifacts_root),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "fashion-mnist", "mnist"],
        required=True,
    )
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--mlflow-port", type=int, default=5001)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    tracking_uri = f"http://127.0.0.1:{args.mlflow_port}"

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = f"validation-{args.dataset}"
    os.environ["MLFLOW_MODEL_NAME"] = f"{args.dataset}-validation-classifier"
    os.environ["TRAINING_SEED"] = "7"

    config_module = importlib.import_module("ml_platform.config")
    config_module.get_settings.cache_clear()

    validation_module = importlib.import_module("ml_platform.data.validation_datasets")
    train_module = importlib.import_module("ml_platform.training.train")

    mlflow_process: subprocess.Popen[str] | None = None
    try:
        dataset_summary = validation_module.prepare_validation_dataset(
            dataset_key=args.dataset,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        mlflow_process = _start_mlflow_server(project_root, args.mlflow_port)
        _wait_for_port("127.0.0.1", args.mlflow_port)

        training_result = train_module.train(
            train_module.TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
            )
        )

        api_module = importlib.import_module("ml_platform.inference.api")
        testclient_module = importlib.import_module("fastapi.testclient")
        sample_image = next(
            path
            for path in sorted((project_root / "dataset" / "images").iterdir())
            if path.is_file() and path.suffix.lower() == ".png"
        )
        with testclient_module.TestClient(api_module.app) as client:
            health = client.get("/healthz").json()
            with sample_image.open("rb") as handle:
                prediction = client.post(
                    "/predict",
                    files={"file": (sample_image.name, handle, "image/png")},
                ).json()

        report = {
            "dataset": dataset_summary,
            "training": training_result,
            "inference_health": health,
            "inference_prediction": prediction,
        }
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_dir = project_root / "artifacts" / "validation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{args.dataset}-validation-{timestamp}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(report)
        print(f"Validation report written to {report_path}")
    finally:
        _stop_process(mlflow_process)


if __name__ == "__main__":
    main()
