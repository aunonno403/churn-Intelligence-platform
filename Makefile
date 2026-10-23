.PHONY: setup download data features train economics explain uplift bundle app clean-raw test lint format all

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

economics:  ## net-savings curve + break-even targeting on the test cohort
	uv run churn economics

explain:    ## SHAP importance, dependence plots and driver sentences on the test cohort
	uv run churn explain

uplift:     ## Module B: uplift vs risk targeting on the Hillstrom e-mail RCT
	uv run churn uplift

bundle:     ## small data bundle the app runs on (app/bundle/, ~4 MB)
	uv run churn bundle

app:        ## open the Streamlit app at http://localhost:8501
	uv run streamlit run app/streamlit_app.py

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

all: setup download data features train economics explain uplift bundle test
