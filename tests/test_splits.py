from datetime import date

import polars as pl
import pytest
from pydantic import ValidationError

from churn.config import Settings
from churn.data.splits import TEST, TRAIN, VALID, assign_split, cohorts


def test_cohorts_are_strictly_ordered_by_split():
    cs = cohorts(Settings())
    order = {TRAIN: 0, VALID: 1, TEST: 2}
    assert [c.month for c in cs] == sorted(c.month for c in cs)
    assert [order[c.split] for c in cs] == sorted(order[c.split] for c in cs)


def test_labels_resolve_before_next_split_cutoff():
    """Embargo: at each split's cutoff, every label from earlier splits already exists."""
    cs = cohorts(Settings())
    for split, later in [(TRAIN, VALID), (VALID, TEST)]:
        last_horizon = max(c.label_horizon for c in cs if c.split == split)
        next_cutoff = min(c.cutoff for c in cs if c.split == later)
        assert last_horizon < next_cutoff


def test_every_label_is_observable():
    s = Settings()
    assert all(c.label_horizon <= s.data_end for c in cohorts(s))


def test_embargo_violation_is_rejected():
    with pytest.raises(ValidationError, match="Embargo"):
        Settings(valid_month="2016-11")


def test_label_horizon_beyond_data_end_is_rejected():
    with pytest.raises(ValidationError, match="censored"):
        Settings(test_month="2017-04")


def test_unordered_months_are_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        Settings(train_months=["2016-10", "2016-09"])


def test_assign_split():
    df = pl.DataFrame(
        {"cohort_month": [date(2016, 9, 1), date(2016, 12, 1), date(2017, 2, 1), date(2017, 1, 1)]}
    )
    out = assign_split(df, Settings()).sort("cohort_month")
    assert out["split"].to_list() == [TRAIN, VALID, TEST]  # out-of-range month dropped
