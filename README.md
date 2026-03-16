# Mini ML Training Platform

This repository demonstrates the core ML lifecycle infrastructure expected from an ML platform engineer:

- reproducible training against a KITTI-style perception dataset
- experiment tracking with MLflow
- model versioning through the MLflow Model Registry
- online inference with FastAPI
- optional automatic retraining when the dataset changes

The implementation intentionally uses a lightweight perception task so the platform mechanics stay front and center. Training turns KITTI object labels into a multi-label scene classifier that predicts whether each object category is present in an image. That keeps the project small enough to run locally while still using a real perception-style dataset contract.

## Planned Architecture

```text
dataset/
  images/
  labels/
  manifests/

train job
  -> fingerprints dataset
  -> logs params and metrics to MLflow
  -> registers new model version
  -> updates champion alias

inference API
  -> resolves latest champion model
  -> serves predictions
  -> refreshes when registry changes

retraining watcher
  -> detects dataset updates
  -> reruns training
```

## Stack

- PyTorch for model training
- MLflow for experiments and model registry
- FastAPI for online inference
- Docker Compose for local platform services

## Current Status

The repository is being built in staged commits:

1. platform scaffold
2. end-to-end training and inference workflow
3. retraining automation, tests, and final documentation
