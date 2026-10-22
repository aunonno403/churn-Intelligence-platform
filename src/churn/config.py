"""Central configuration. Every tunable number lives here, overridable via CHURN_* env vars."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_month(month: str) -> date:
    """'2017-02' -> date(2017, 2, 1)."""
    year, mon = month.split("-")
    return date(int(year), int(mon), 1)


def month_end(month_start: date) -> date:
    nxt = date(month_start.year + month_start.month // 12, month_start.month % 12 + 1, 1)
    return nxt - timedelta(days=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    # --- paths -------------------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    sql_dir: Path = PROJECT_ROOT / "sql"

    # --- sampling ----------------------------------------------------------------------
    # Deterministic md5-based user sample: keep users where md5(msno) % sample_mod < sample_keep.
    # 1000/10000 = 10% of users (~200k with a cohort in the label window).
    sample_mod: int = 10_000
    sample_keep: int = 1_000

    # --- label definition ----------------------------------------------------------------
    # Churn = no valid (non-cancel) renewal within `churn_window_days` after membership expiry.
    churn_window_days: int = 30
    # Last date covered by the raw KKBox files (transactions_v2 / user_logs_v2).
    data_end: date = date(2017, 3, 31)
    # How far back behavioral features may look; also bounds how much user_logs we ingest.
    feature_lookback_days: int = 90

    # --- temporal split (cohort = month in which the membership expires) ----------------
    # Embargo: every label used by an earlier split must be resolved (expiry + window) before
    # the next split's cutoff, i.e. we only train on labels that existed at prediction time.
    # Hence the unused gap months (2016-11, 2017-01) between splits.
    train_months: list[str] = ["2016-07", "2016-08", "2016-09", "2016-10"]
    valid_month: str = "2016-12"
    test_month: str = "2017-02"
    # The month for which KKBox ships official labels (train.csv) — used to validate ours.
    official_label_month: str = "2017-02"

    # --- modelling -----------------------------------------------------------------------
    seed: int = 42
    # Budgeted LightGBM search: whichever limit is hit first ends the study.
    optuna_trials: int = 50
    optuna_timeout_s: int = 1800

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "churn.duckdb"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "model"

    @property
    def mlflow_uri(self) -> str:
        return f"sqlite:///{(self.data_dir / 'mlflow.db').as_posix()}"

    @property
    def all_months(self) -> list[str]:
        return [*self.train_months, self.valid_month, self.test_month]

    @property
    def logs_start(self) -> date:
        return parse_month(self.all_months[0]) - timedelta(days=self.feature_lookback_days)

    @model_validator(mode="after")
    def _check_split(self) -> Settings:
        starts = [parse_month(m) for m in self.all_months]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError("Cohort months must be strictly increasing: train < valid < test.")

        def horizon_of(start: date) -> date:
            return month_end(start) + timedelta(days=self.churn_window_days)

        valid_start, test_start = starts[-2], starts[-1]
        if horizon_of(starts[-3]) >= valid_start or horizon_of(valid_start) >= test_start:
            raise ValueError(
                "Embargo violated: a split's labels resolve after the next split's cutoff. "
                "Leave a gap month between train/valid/test."
            )
        horizon = horizon_of(test_start)
        if horizon > self.data_end:
            raise ValueError(
                f"Test cohort's label horizon ({horizon}) exceeds data_end ({self.data_end}); "
                "labels for that cohort would be censored."
            )
        if not 0 < self.sample_keep <= self.sample_mod:
            raise ValueError("Need 0 < sample_keep <= sample_mod.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
