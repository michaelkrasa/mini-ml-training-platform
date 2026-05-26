# Mini ML Training Platform

This repository demonstrates the core ML lifecycle infrastructure expected from an ML platform engineer:

- reproducible training against a KITTI-style perception dataset
- experiment tracking with MLflow
- model versioning through the MLflow Model Registry
- online inference with FastAPI
- optional automatic retraining when the dataset changes
- benchmark validation against downloaded public datasets
- active-learning candidate mining from reproducible dataset manifests
- model promotion gates and image rollout manifests for closed-loop deployment
- a PyTorch `torchrun` entrypoint for multi-process distributed training experiments

The implementation intentionally uses a lightweight perception task so the platform mechanics stay front and center. Training turns KITTI object labels into a multi-label scene classifier that predicts whether each object category is present in an image. That keeps the project small enough to run locally while still using a real perception-style dataset contract.

## What It Is Good For Right Now

Today this project is strongest as an ML platform demonstration, not as a production-grade model training stack. It is good for proving that you can design and operate the core workflow around a model:

- convert raw data into a reproducible training contract
- fingerprint datasets and persist manifests for reruns
- run training jobs with tracked parameters and metrics
- register model versions and resolve the latest champion at inference time
- trigger retraining when data changes
- validate the whole path on benchmark datasets without hand-curated local files

It is not yet optimized for model quality, distributed training, feature stores, GPUs, or Kubernetes-native job execution. Those would be natural next layers.

The repository now includes pragmatic hooks for those next layers: distributed launch configuration, active data mining, model promotion policy checks, and deployment packaging metadata. They are intentionally small and runnable locally, but they mirror the control-plane responsibilities used in larger AI flywheels.

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

active mining
  -> ranks manifest samples by class rarity and scene density
  -> emits candidate sets for labeling or replay prioritization

promotion gate
  -> evaluates candidate model metrics against release thresholds
  -> writes a pass/fail report for automated selection

deployment package
  -> creates an immutable image tag from model version and dataset fingerprint
  -> records rollout targets and validation gate status
```

## Stack

- PyTorch for model training
- MLflow for experiments and model registry
- FastAPI for online inference
- Docker Compose for local platform services
- PyTorch distributed launcher for DDP-style local or multi-node training starts

## Why The Model Is Intentionally Lightweight

The platform demonstrates ML infrastructure, not state-of-the-art perception accuracy. Instead of training a full detector, the training job converts KITTI object annotations into a multi-label scene classifier that predicts which object classes are present in the frame. That keeps the code and runtime compact while still exercising the real platform concerns:

- deterministic dataset manifests
- experiment metadata
- model registration and version lookup
- inference artifact management
- retraining orchestration

## Repository Layout

```text
dataset/
  images/                 image inputs
  labels/                 KITTI label_2 style text files
  manifests/              dataset fingerprints and train/val splits

scripts/
  bootstrap_sample_dataset.py

src/ml_platform/
  data/                   KITTI parsing and manifest creation
  training/               model, metrics, and MLflow-backed train job
  registry/               MLflow registry helpers
  inference/              FastAPI service and registry-backed predictor
  orchestration/          automatic retraining watcher
```

## Quick Start

1. Install dependencies.

```bash
uv sync --extra dev
```

2. Generate a runnable local dataset, or place your own KITTI-style files under `dataset/images` and `dataset/labels`.

```bash
make sample-data
```

3. Start MLflow.

```bash
make mlflow-up
```

4. Run a training job. This fingerprints the dataset, logs metrics to MLflow, registers a model version, and moves the `champion` alias to the new version.

```bash
make train
```

5. Start the FastAPI inference service.

```bash
make serve
```

6. Query the latest registered model.

```bash
curl -X POST \
  -F "file=@dataset/images/000000.png" \
  http://127.0.0.1:8000/predict
```

7. Optionally run the automatic retraining loop. It polls for dataset fingerprint changes and retrains when the image or label set changes.

```bash
make retrain
```

8. Rank training examples for active learning or replay prioritization.

```bash
make active-mine
```

9. Dry-run a distributed PyTorch training launch. Remove `--dry-run` in `Makefile` or call the module directly to execute with `torchrun`.

```bash
make train-distributed
```

10. Package a trained model summary into rollout metadata.

```bash
make package-model SUMMARY=artifacts/training/<run>/training_summary.json
```

## Docker Workflow

Bring up the local platform stack:

```bash
docker-compose up --build mlflow inference retrainer
```

The stack uses shared local volumes in the repo so runs, artifacts, and manifests are inspectable without container-specific tooling.

## What This Project Demonstrates

- Reproducible training: every run logs the dataset fingerprint, seed, hyperparameters, Git commit, and a persisted manifest describing the train/val split.
- Experiment tracking: MLflow captures epoch metrics, artifacts, and run metadata.
- Model versioning: the training job registers each new model and updates the `champion` alias.
- Deployment: the inference API resolves the latest registered version at runtime instead of reading a hardcoded local checkpoint.
- Operations mindset: the retraining watcher treats dataset changes as an event source and re-triggers the full training and registration workflow.
- Distributed training readiness: the `train-distributed` command builds a PyTorch `torch.distributed.run` launch around the existing training module, so the same training contract can be exercised under multi-process launches.
- Smart data mining: `active-mine` ranks manifest samples by class rarity and multi-object scene density, producing a JSON candidate set that approximates active-learning/replay-buffer prioritization.
- Automated model selection: promotion gates evaluate validation metrics against explicit thresholds and produce machine-readable pass/fail reports.
- Deployment packaging: model rollout manifests bind together MLflow model version, dataset fingerprint, Git commit, validation gate, and container image tag.

## AI Flywheel Extensions

These extensions are deliberately compact, but they map to production platform concerns:

- `ml_platform.data.mining` turns petabyte-scale log-mining ideas into a deterministic manifest-ranking strategy.
- `ml_platform.training.distributed` provides the launch surface for PyTorch DDP or multi-node experiments.
- `ml_platform.deployment.gates` gives retraining jobs a policy layer for champion/candidate model selection.
- `ml_platform.deployment.package` emits immutable rollout metadata suitable for image validation and fleet-style deployment workflows.

## Using A Real KITTI Export

The runtime contract is simple:

- images go in `dataset/images`
- labels go in `dataset/labels`
- each image must have a same-stem `.txt` label file
- each label file uses KITTI object lines where the first token is the class name

The synthetic dataset generator exists only so the platform is runnable out of the box. Replacing it with a real KITTI split requires no code changes.

## Benchmark Validation

The repo can now download and convert several popular benchmark datasets into the platform's `dataset/images` plus `dataset/labels` contract:

- `mnist`
- `fashion-mnist`
- `cifar10`

Download a dataset into the local dataset workspace:

```bash
make download-dataset DATASET=mnist
```

Run a full validation flow that downloads the dataset, starts a local MLflow server, trains a model, registers it, and exercises inference end to end:

```bash
make validate-platform DATASET=mnist
```

The validation runner writes a JSON report under `artifacts/validation/` so the proof of execution is saved alongside the repo.
