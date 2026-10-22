.PHONY: setup download data features train clean-raw test lint format all

setup:      ## install env (Python 3.11 + deps + dev tools)
	uv sync

download:   ## fetch KKBox from Kaggle (needs API token; peak ~16GB disk)
	uv run churn download

data:       ## raw files -> sampled DuckDB tables + labels + label validation
	uv run churn data

features:   ## leakage-safe feature table (sql/03_features.sql)
	uv run churn features

train:      ## baselines + tuned LightGBM + calibration -> data/model/, MLflow in data/mlflow.db
	uv run churn train

clean-raw:  ## delete data/raw once the DuckDB tables are verified (~1-2GB remains)
	uv run churn clean-raw

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

all: setup data features train test
