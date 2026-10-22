"""Temporal cohorts and train/valid/test assignment.

A cohort is the set of users whose membership (as known at the cutoff) expires in a given
month. The cutoff is the first day of that month: features may only use data strictly
before it, labels only data on/after it. Splits are by cohort month, never random.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from churn.config import Settings, month_end, parse_month

TRAIN, VALID, TEST = "train", "valid", "test"


@dataclass(frozen=True)
class Cohort:
    month: date  # first day of the expiry month == feature cutoff (exclusive)
    split: str
    window_days: int

    @property
    def cutoff(self) -> date:
        return self.month

    @property
    def month_end(self) -> date:
        return month_end(self.month)

    @property
    def label_horizon(self) -> date:
        """Last date whose data can influence this cohort's label."""
        return self.month_end + timedelta(days=self.window_days)


def cohorts(settings: Settings) -> list[Cohort]:
    out = [Cohort(parse_month(m), TRAIN, settings.churn_window_days) for m in settings.train_months]
    out.append(Cohort(parse_month(settings.valid_month), VALID, settings.churn_window_days))
    out.append(Cohort(parse_month(settings.test_month), TEST, settings.churn_window_days))
    return out
