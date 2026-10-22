"""Leakage test: no feature may change when data on or after the cohort's cutoff changes."""

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from churn.config import Settings
from churn.features.build import build_features

CUTOFFS = [date(2016, 9, 1), date(2016, 12, 1)]
START = date(2016, 4, 1)
USERS = [f"u{i}" for i in range(40)]


def _tx_rows(rng, n, lo, hi):
    rows = []
    for _ in range(n):
        u = rng.choice(USERS)
        d = lo + timedelta(days=int(rng.integers(0, (hi - lo).days)))
        plan = int(rng.choice([30, 30, 30, 90, 180]))
        price = int(rng.choice([99, 149, 180]))
        rows.append(
            (u, int(rng.integers(30, 42)), plan, price, price - int(rng.integers(0, 2)) * 50,
             bool(rng.integers(0, 2)), d, d + timedelta(days=plan), bool(rng.random() < 0.1))
        )  # fmt: skip
    return rows


def _log_rows(rng, n, lo, hi):
    return [
        (rng.choice(USERS), lo + timedelta(days=int(rng.integers(0, (hi - lo).days))),
         *map(int, rng.integers(0, 30, 6)), float(rng.normal(3000, 3000)))
        for _ in range(n)
    ]  # fmt: skip


def _db(rng) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE transactions (msno VARCHAR, payment_method_id INT, payment_plan_days INT, "
        "plan_list_price INT, actual_amount_paid INT, is_auto_renew BOOLEAN, "
        "transaction_date DATE, membership_expire_date DATE, is_cancel BOOLEAN)"
    )
    con.execute(
        "CREATE TABLE user_logs (msno VARCHAR, date DATE, num_25 INT, num_50 INT, num_75 INT, "
        "num_985 INT, num_100 INT, num_unq INT, total_secs DOUBLE)"
    )
    con.execute(
        "CREATE TABLE members (msno VARCHAR, city INT, bd INT, gender VARCHAR, "
        "registered_via INT, registration_date DATE)"
    )
    con.execute(
        "CREATE TABLE labels (msno VARCHAR, cohort_month DATE, expire_date DATE, "
        "is_churn BOOLEAN, split VARCHAR)"
    )
    con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)",
                    _tx_rows(rng, 300, START, CUTOFFS[-1]))  # fmt: skip
    con.executemany("INSERT INTO user_logs VALUES (?,?,?,?,?,?,?,?,?)",
                    _log_rows(rng, 2000, START, CUTOFFS[-1]))  # fmt: skip
    con.executemany(
        "INSERT INTO members VALUES (?,?,?,?,?,?)",
        [(u, 1, int(rng.integers(0, 60)), None, 7, date(2015, 1, 1)) for u in USERS],
    )
    con.executemany(
        "INSERT INTO labels VALUES (?,?,?,?, 'train')",
        [
            (u, c, c + timedelta(days=int(rng.integers(0, 28))), False)
            for u in USERS
            for c in CUTOFFS
        ],
    )
    return con


def _features(con, cutoff) -> pd.DataFrame:
    build_features(con, Settings())
    return (
        con.execute("SELECT * FROM features WHERE cohort_month = ? ORDER BY msno", [cutoff])
        .df()
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_features_ignore_data_on_or_after_cutoff(cutoff):
    rng = np.random.default_rng(0)
    con = _db(rng)
    before = _features(con, cutoff)
    assert len(before) > 20  # the test must actually cover users

    # flood the post-cutoff period (starting exactly on the cutoff day) with new activity
    end = cutoff + timedelta(days=90)
    con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?)",
                    _tx_rows(rng, 300, cutoff, end))  # fmt: skip
    con.executemany("INSERT INTO user_logs VALUES (?,?,?,?,?,?,?,?,?)",
                    _log_rows(rng, 2000, cutoff, end))  # fmt: skip
    con.execute("UPDATE user_logs SET total_secs = total_secs * 7 WHERE date >= ?", [cutoff])
    pd.testing.assert_frame_equal(before, _features(con, cutoff))


def test_leakage_check_is_sensitive():
    """Same data change one day *before* the cutoff must move the features."""
    rng = np.random.default_rng(0)
    con = _db(rng)
    cutoff = CUTOFFS[0]
    before = _features(con, cutoff)
    day_before = cutoff - timedelta(days=1)
    con.executemany(
        "INSERT INTO user_logs VALUES (?, ?, 0, 0, 0, 0, 50, 40, 9000)",
        [(u, day_before) for u in USERS],
    )
    assert not before.equals(_features(con, cutoff))
