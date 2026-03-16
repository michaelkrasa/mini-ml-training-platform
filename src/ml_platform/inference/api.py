"""FastAPI service for registry-backed online inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from ml_platform.config import get_settings
from ml_platform.inference.predictor import ModelPredictor, PredictorNotReady

settings = get_settings()
predictor = ModelPredictor(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        predictor.load_latest(force=True)
    except PredictorNotReady:
        pass
    yield


app = FastAPI(
    title="Mini ML Training Platform Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, Any]:
    status = predictor.status()
    return {
        "service": "mini-ml-training-platform",
        "dataset_name": status.get("dataset_name"),
        "task_type": status.get("task_type", "classification"),
        "model": status,
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    status = predictor.status()
    return {"status": "ready" if status["ready"] else "model_missing", **status}


@app.get("/model")
def model_details() -> dict[str, Any]:
    try:
        predictor.maybe_refresh()
    except PredictorNotReady:
        pass
    return predictor.status()


@app.post("/reload")
def reload_model() -> dict[str, Any]:
    try:
        predictor.load_latest(force=True)
    except PredictorNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return predictor.status()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return predictor.predict(image_bytes)
    except PredictorNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run("ml_platform.inference.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
