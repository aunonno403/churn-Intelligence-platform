"""Build churn labels for every configured cohort and check them against KKBox's official ones."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb

from churn.config import Settings, parse_month
from churn.data.splits import cohorts
from churn.data.sql import render

log = logging.getLogger(__name__)


def build_labels(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE labels (
            msno VARCHAR, cohort_month DATE, expire_date DATE, is_churn BOOLEAN, split VARCHAR
        )
        """
    )
    for c in cohorts(settings):
        sql = render(
            "02_labels.sql",
            settings,
            cutoff=c.cutoff,
            month_end=c.month_end,
            window_days=c.window_days,
        )
        con.execute(f"INSERT INTO labels SELECT *, '{c.split}' FROM (\n{sql}\n)")
        n, rate = con.execute(
            "SELECT count(*), avg(is_churn::INT) FROM labels WHERE cohort_month = ?", [c.month]
        ).fetchone()
        log.info("  cohort %s [%-5s] n=%8s churn=%.3f", c.month, c.split, f"{n:,}", rate or 0)


@dataclass
class LabelValidation:
    official_users: int  # official-label users in our sample
    in_our_cohort: int  # ...of which our cohort logic also selected
    agreement: float  # label agreement on the overlap (flattered by the low churn rate)
    churner_recall: float  # share of official churners we also label churned
    churner_precision: float  # share of our churners that are official churners
    false_negatives: int
    false_positives: int
    our_churn_rate: float
    official_churn_rate: float


def validate_against_official(
    con: duckdb.DuckDBPyConnection, settings: Settings
) -> LabelValidation:
    month = parse_month(settings.official_label_month)
    official_users = con.execute("SELECT count(*) FROM official_labels").fetchone()[0]
    n, agreement, tp, fn, fp, ours, theirs = con.execute(
        """
        SELECT count(*),
               avg((l.is_churn = o.is_churn)::INT),
               count(*) FILTER (WHERE l.is_churn AND o.is_churn),
               count(*) FILTER (WHERE NOT l.is_churn AND o.is_churn),
               count(*) FILTER (WHERE l.is_churn AND NOT o.is_churn),
               avg(l.is_churn::INT),
               avg(o.is_churn::INT)
        FROM labels l
        JOIN official_labels o USING (msno)
        WHERE l.cohort_month = ?
        """,
        [month],
    ).fetchone()
    return LabelValidation(
        official_users=official_users,
        in_our_cohort=n,
        agreement=agreement or 0.0,
        churner_recall=tp / (tp + fn) if tp + fn else 0.0,
        churner_precision=tp / (tp + fp) if tp + fp else 0.0,
        false_negatives=fn,
        false_positives=fp,
        our_churn_rate=ours or 0.0,
        official_churn_rate=theirs or 0.0,
    )
