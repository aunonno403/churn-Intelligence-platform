"""Feature table (sql/03_features.sql) and the model-ready dataset."""

from __future__ import annotations

import logging

import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from churn.config import Settings
from churn.data.sql import render

log = logging.getLogger(__name__)

KEY = ["msno", "cohort_month"]
TARGET = "is_churn"
CATEGORICAL = ["last_payment_method", "city", "gender", "registered_via"]


def build_features(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    con.execute(f"CREATE OR REPLACE TABLE features AS\n{render('03_features.sql', settings)}")
    n = con.execute("SELECT count(*) FROM features").fetchone()[0]
    n_cols = len(con.execute("SELECT * FROM features LIMIT 0").description) - len(KEY)
    log.info("features: %s rows x %d features", f"{n:,}", n_cols)


def load_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    as_text = ", ".join(f"{c}::VARCHAR AS {c}" for c in CATEGORICAL)
    df = con.execute(
        f"""
        SELECT f.* REPLACE ({as_text}), l.split, l.is_churn::INT AS {TARGET}
        FROM features f JOIN labels l USING (msno, cohort_month)
        ORDER BY cohort_month, msno
        """
    ).df()
    # One category set for all splits, so codes mean the same thing in train and test.
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (*KEY, "split", TARGET)]


def lr_preprocessor(features: list[str]) -> ColumnTransformer:
    """Logistic regression needs imputed, scaled, one-hot inputs; LightGBM takes raw columns."""
    numeric = [c for c in features if c not in CATEGORICAL]
    return ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    # heavy-tailed counts and minutes: rank-gauss instead of mean/std scaling
                    QuantileTransformer(output_distribution="normal", subsample=200_000),
                ),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100),
                ),
                CATEGORICAL,
            ),
        ]
    )
