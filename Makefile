PYTHON ?= python3
COMPOSE ?= docker-compose

.PHONY: sample-data download-dataset validate-platform mlflow-up mlflow-down train serve retrain test

sample-data:
	uv run $(PYTHON) scripts/bootstrap_sample_dataset.py

download-dataset:
	uv run $(PYTHON) -m ml_platform.data.validation_datasets --dataset $(DATASET)

validate-platform:
	uv run $(PYTHON) scripts/run_validation.py --dataset $(DATASET)

mlflow-up:
	$(COMPOSE) up -d mlflow

mlflow-down:
	$(COMPOSE) down

train:
	uv run $(PYTHON) -m ml_platform.training.train

serve:
	uv run uvicorn ml_platform.inference.api:app --host 0.0.0.0 --port 8000

retrain:
	uv run $(PYTHON) -m ml_platform.orchestration.retrain_watcher

test:
	uv run pytest
