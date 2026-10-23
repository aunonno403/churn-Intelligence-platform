"""Export the small, self-contained bundle the Streamlit app runs on (app/bundle/).

The app never touches DuckDB: a free host cannot hold the 1-2 GB database. The bundle is a
seeded sample of test-month customers (features, outcome, scores), the model, and the
summary numbers every report already wrote. KKBox user ids are replaced by row numbers.
"""

from __future__ import annotations

import json
import logging
import shutil

import duckdb
import numpy as np

from churn.config import PROJECT_ROOT, Settings
from churn.features.build import KEY, TARGET, load_test_cohort

log = logging.getLogger(__name__)

BUNDLE_DIR = PROJECT_ROOT / "app" / "bundle"
SAMPLE = 20_000
FIGURES = ["reliability_test.png", "uplift_vs_risk.png", "uplift_by_fifth.png"]


def build_bundle(settings: Settings) -> None:
    test, model = load_test_cohort(settings)
    sample = test.sample(min(SAMPLE, len(test)), random_state=settings.seed)
    sample = sample.drop(columns=[*KEY, "split"]).reset_index(drop=True)
    sample.insert(0, "customer", [f"C{i:05d}" for i in range(1, len(sample) + 1)])
    sample["score"] = model.predict_raw(sample)  # ranking
    sample["churn_probability"] = model.predict(sample)  # calibrated, for money
    assert sample[TARGET].isin([0, 1]).all()

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(BUNDLE_DIR / "customers.parquet", index=False)
    shutil.copy(settings.model_dir / "model.joblib", BUNDLE_DIR / "model.joblib")
    for name in FIGURES:
        shutil.copy(PROJECT_ROOT / "reports" / "figures" / name, BUNDLE_DIR / name)

    reports = {
        name: json.loads((settings.model_dir / f"{name}.json").read_text())
        for name in ("metrics", "economics", "drivers", "uplift")
    }
    with duckdb.connect(str(settings.db_path), read_only=True) as con:
        reports["churn_by_cohort"] = (
            con.execute(
                "SELECT strftime(cohort_month, '%Y-%m') AS cohort, split, count(*) AS customers, "
                "avg(is_churn::INT) AS churn_rate FROM labels GROUP BY ALL ORDER BY 1"
            )
            .df()
            .to_dict(orient="records")
        )
        reports["retention"] = _retention(con, settings).to_dict(orient="records")
    reports["label_validation"] = json.loads(
        (settings.data_dir / "label_validation.json").read_text()
    )
    reports["settings"] = {
        "train_months": settings.train_months,
        "valid_month": settings.valid_month,
        "test_month": settings.test_month,
        "churn_window_days": settings.churn_window_days,
        "contact_cost": settings.contact_cost,
        "success_rate": settings.success_rate,
        "customer_value": settings.customer_value,
        "sample_share_of_kkbox_users": settings.sample_keep / settings.sample_mod,
        "bundle_customers": len(sample),
        "test_customers": len(test),
    }
    (BUNDLE_DIR / "summary.json").write_text(json.dumps(reports, indent=2, default=_json))
    size = sum(f.stat().st_size for f in BUNDLE_DIR.iterdir()) / 1024**2
    log.info("bundle: %s customers, %.1f MB in %s", f"{len(sample):,}", size, BUNDLE_DIR)


def _retention(con: duckdb.DuckDBPyConnection, settings: Settings):
    """Share of each first-transaction cohort with paid cover in each later month.
    Jan 2015 is left out: data starts then, so that 'cohort' is everyone already subscribed."""
    return con.execute(
        f"""
        WITH tx AS (
            SELECT msno,
                   date_trunc('month', transaction_date)::DATE AS m0,
                   date_trunc('month', least(membership_expire_date,
                                             DATE '{settings.data_end}'))::DATE AS m1
            FROM transactions
            WHERE NOT is_cancel AND membership_expire_date >= transaction_date
        ),
        first AS (SELECT msno, min(m0) AS cohort FROM tx GROUP BY msno),
        active AS (
            SELECT DISTINCT msno, unnest(generate_series(m0, m1, INTERVAL 1 MONTH))::DATE AS m
            FROM tx
        ),
        counts AS (
            SELECT f.cohort, date_diff('month', f.cohort, a.m) AS months_since, count(*) AS n
            FROM first f JOIN active a USING (msno)
            WHERE f.cohort >= DATE '2015-02-01'
            GROUP BY ALL
        )
        SELECT strftime(cohort, '%Y-%m') AS cohort, months_since,
               n / first_value(n) OVER (PARTITION BY cohort ORDER BY months_since) AS retained,
               first_value(n) OVER (PARTITION BY cohort ORDER BY months_since) AS cohort_size
        FROM counts
        ORDER BY cohort, months_since
        """
    ).df()


def _json(v):
    return v.item() if isinstance(v, np.generic) else str(v)
