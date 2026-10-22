"""Baselines -> LightGBM (Optuna-tuned) -> isotonic calibration, tracked in MLflow.

Data roles, all by cohort month (see config.py):
  train  fit every model; its out-of-fold scores (each month scored by a model fitted on
         the other training months) fit the isotonic calibrator
  valid  early stopping and the Optuna objective; never seen by the calibrator, so it is an
         honest check of calibration
  test   touched once, for the reported numbers

Why not calibrate on valid: one month's churn rate (Dec: 8%) then sets every probability.
Out-of-fold training scores average four months (3.8-6.4%).
"""

from __future__ import annotations

import json
import logging

import duckdb
import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline

from churn.config import PROJECT_ROOT, Settings
from churn.data.splits import TEST, TRAIN, VALID
from churn.eval.metrics import evaluate, reliability_plot
from churn.features.build import TARGET, feature_columns, load_dataset, lr_preprocessor
from churn.models.model import ChurnModel

log = logging.getLogger(__name__)
EXPERIMENT = "churn-prediction"
REGISTERED_NAME = "churn-lightgbm"


def _lgbm(settings: Settings, **params) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", random_state=settings.seed, n_jobs=-1, verbose=-1, **params
    )


def _tune(settings: Settings, X: dict, y: dict) -> tuple[dict, int]:
    """Optuna search on validation PR-AUC; each trial is a nested MLflow run."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        # metric= replaces the default log-loss, so early stopping watches PR-AUC alone
        # (with two metrics LightGBM stops as soon as either one stalls)
        model = _lgbm(settings, n_estimators=3000, metric="average_precision", **params)
        model.fit(
            X[TRAIN], y[TRAIN], eval_X=X[VALID], eval_y=y[VALID],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )  # fmt: skip
        score = average_precision_score(y[VALID], model.predict_proba(X[VALID])[:, 1])
        trial.set_user_attr("n_estimators", int(model.best_iteration_))
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params({**params, "n_estimators": model.best_iteration_})
            mlflow.log_metric("valid_pr_auc", score)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=settings.seed)
    )
    study.optimize(
        objective, n_trials=settings.optuna_trials, timeout=settings.optuna_timeout_s,
        callbacks=[lambda s, t: log.info("  trial %d: valid PR-AUC %.4f (best %.4f)",
                                         t.number, t.value, s.best_value)],
    )  # fmt: skip
    best = study.best_trial
    log.info("Optuna: %d trials, best valid PR-AUC %.4f", len(study.trials), best.value)
    return {**best.params, "subsample_freq": 1}, best.user_attrs["n_estimators"]


def train(settings: Settings) -> dict:
    with duckdb.connect(str(settings.db_path), read_only=True) as con:
        df = load_dataset(con)
    features = feature_columns(df)
    X = {s: df.loc[df.split == s, features] for s in (TRAIN, VALID, TEST)}
    y = {s: df.loc[df.split == s, TARGET].to_numpy() for s in (TRAIN, VALID, TEST)}
    log.info("rows: %s", {s: len(v) for s, v in y.items()})

    mlflow.set_tracking_uri(settings.mlflow_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(settings.data_dir / "mlartifacts").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="train") as run:
        preds: dict[str, dict[str, np.ndarray]] = {}

        # 1. majority baseline: everyone gets the training churn rate
        preds["majority"] = {s: np.full(len(y[s]), y[TRAIN].mean()) for s in (VALID, TEST)}

        # 2. logistic regression
        lr = make_pipeline(lr_preprocessor(features), LogisticRegression(max_iter=2000))
        lr.fit(X[TRAIN], y[TRAIN])
        preds["logistic"] = {s: lr.predict_proba(X[s])[:, 1] for s in (VALID, TEST)}
        log.info("logistic regression done")

        # 3. LightGBM, tuned on validation, refit on train with the best round count
        params, n_estimators = _tune(settings, X, y)
        booster = _lgbm(settings, n_estimators=n_estimators, **params).fit(X[TRAIN], y[TRAIN])
        raw = {s: booster.predict_proba(X[s])[:, 1] for s in (VALID, TEST)}
        preds["lightgbm"] = raw

        # 4. isotonic calibration fitted on out-of-fold training scores (leave one month out)
        months = df.loc[df.split == TRAIN, "cohort_month"].to_numpy()
        oof = np.full(len(y[TRAIN]), np.nan)
        for month in np.unique(months):
            held = months == month
            fold = _lgbm(settings, n_estimators=n_estimators, **params)
            fold.fit(X[TRAIN].loc[~held], y[TRAIN][~held])
            oof[held] = fold.predict_proba(X[TRAIN].loc[held])[:, 1]
        assert not np.isnan(oof).any(), "every training row needs an out-of-fold score"
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        calibrator.fit(oof, y[TRAIN])
        model = ChurnModel(booster, calibrator, features)
        preds["lightgbm_calibrated"] = {s: model.predict_proba(X[s])[:, 1] for s in (VALID, TEST)}

        metrics = {
            name: {s: evaluate(y[s], p[s]) for s in (VALID, TEST)} for name, p in preds.items()
        }

        settings.model_dir.mkdir(parents=True, exist_ok=True)
        figure = PROJECT_ROOT / "reports" / "figures" / "reliability_test.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        reliability_plot(
            y[TEST],
            {
                "LightGBM (raw)": raw[TEST],
                "LightGBM + isotonic": preds["lightgbm_calibrated"][TEST],
            },
            "Reliability on the test cohort (2017-02)",
            figure,
        )
        report = {
            "run_id": run.info.run_id,
            "rows": {s: len(v) for s, v in y.items()},
            "lightgbm_params": {**params, "n_estimators": n_estimators},
            "metrics": metrics,
        }
        (settings.model_dir / "metrics.json").write_text(json.dumps(report, indent=2))
        joblib.dump(model, settings.model_dir / "model.joblib")

        mlflow.log_params({**params, "n_estimators": n_estimators, "n_features": len(features)})
        mlflow.log_metrics(
            {f"{name}.{s}.{k}": v for name, by in metrics.items() for s, m in by.items()
             for k, v in m.items()}
        )  # fmt: skip
        mlflow.log_artifact(str(settings.model_dir / "metrics.json"))
        mlflow.log_artifact(str(figure))
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_NAME,
            # skops refuses unknown types on load; these are ours and LightGBM's
            skops_trusted_types=[
                "churn.models.model.ChurnModel",
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
            ],
        )

    _log_table(metrics)
    return report


def _log_table(metrics: dict) -> None:
    rows = [
        {"model": name, "split": s, **{k: round(v, 4) for k, v in m.items()}}
        for name, by in metrics.items()
        for s, m in by.items()
    ]
    log.info("\n%s", pd.DataFrame(rows).to_string(index=False))
