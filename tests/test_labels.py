"""Label logic on hand-built transaction histories, one scenario per user."""

from datetime import date

import duckdb
import pytest

from churn.config import Settings
from churn.data.sql import render

D = date

# (msno, transaction_date, membership_expire_date, is_cancel)
TRANSACTIONS = [
    # renews on expiry day -> retained
    ("renew_on_time", D(2017, 1, 15), D(2017, 2, 14), False),
    ("renew_on_time", D(2017, 2, 14), D(2017, 3, 16), False),
    # no renewal -> churn
    ("no_renewal", D(2017, 1, 10), D(2017, 2, 10), False),
    # renews 23 days after expiry (inside 30-day window) -> retained
    ("late_renewal", D(2017, 1, 10), D(2017, 2, 10), False),
    ("late_renewal", D(2017, 3, 5), D(2017, 4, 4), False),
    # renews 38 days after expiry -> churn
    ("too_late", D(2017, 1, 10), D(2017, 2, 10), False),
    ("too_late", D(2017, 3, 20), D(2017, 4, 19), False),
    # only post-expiry activity is a cancellation -> churn
    ("cancelled", D(2017, 1, 20), D(2017, 2, 20), False),
    ("cancelled", D(2017, 2, 5), D(2017, 2, 20), True),
    # expires in March -> not in the Feb cohort
    ("expires_later", D(2017, 1, 15), D(2017, 3, 15), False),
    # first transaction after the cutoff -> unknown at cutoff, must not enter the cohort
    ("new_after_cutoff", D(2017, 2, 3), D(2017, 2, 25), False),
    # renewed *before* the cutoff: state at cutoff already expires in March -> not in cohort
    ("renewed_early", D(2017, 1, 1), D(2017, 2, 1), False),
    ("renewed_early", D(2017, 1, 25), D(2017, 3, 1), False),
    # cancels Feb 5 (expiry pulled to Feb 5), renews Mar 10: 33 days after the *new* expiry -> churn
    ("cancel_then_renew", D(2017, 1, 20), D(2017, 2, 20), False),
    ("cancel_then_renew", D(2017, 2, 5), D(2017, 2, 5), True),
    ("cancel_then_renew", D(2017, 3, 10), D(2017, 4, 9), False),
    # window is strict: 29 days after expiry renews, 30 days does not
    ("gap_29_days", D(2017, 1, 10), D(2017, 2, 10), False),
    ("gap_29_days", D(2017, 3, 11), D(2017, 4, 10), False),
    ("gap_30_days", D(2017, 1, 10), D(2017, 2, 10), False),
    ("gap_30_days", D(2017, 3, 12), D(2017, 4, 11), False),
    # same-day subscribe + cancel on the same plan: cancellation is the later event, so the
    # state at cutoff expires Feb 15 (not Mar 15) -> in cohort, no renewal -> churn
    ("same_day_cancel", D(2017, 1, 15), D(2017, 3, 15), False),
    ("same_day_cancel", D(2017, 1, 15), D(2017, 2, 15), True),
]

EXPECTED = {
    "renew_on_time": False,
    "no_renewal": True,
    "late_renewal": False,
    "too_late": True,
    "cancelled": True,
    "cancel_then_renew": True,
    "gap_29_days": False,
    "gap_30_days": True,
    "same_day_cancel": True,
}


@pytest.fixture
def labels() -> dict[str, bool]:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE transactions (msno VARCHAR, transaction_date DATE, "
        "membership_expire_date DATE, is_cancel BOOLEAN, is_auto_renew BOOLEAN, "
        "payment_method_id INTEGER, payment_plan_days INTEGER, plan_list_price INTEGER)"
    )
    con.executemany(  # every row on the same plan, so only the same-plan tie rules apply
        "INSERT INTO transactions VALUES (?, ?, ?, ?, false, 41, 30, 149)",
        [list(r) for r in TRANSACTIONS],
    )
    sql = render(
        "02_labels.sql", Settings(), cutoff=D(2017, 2, 1), month_end=D(2017, 2, 28), window_days=30
    )
    return dict(con.execute(f"SELECT msno, is_churn FROM ({sql})").fetchall())


def test_cohort_membership(labels):
    assert set(labels) == set(EXPECTED)


@pytest.mark.parametrize("msno,churn", EXPECTED.items())
def test_label(labels, msno, churn):
    assert labels[msno] is churn
