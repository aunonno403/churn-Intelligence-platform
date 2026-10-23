import numpy as np
import pytest

from churn.eval.metrics import evaluate, lift_at, qini_coefficient, qini_curve


def test_lift_counts_churners_in_top_fraction():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 20% churn
    p = np.linspace(1, 0, 10)  # perfect ranking
    assert lift_at(y, p, 0.2) == 5.0  # top 2 are both churners: 100% / 20%
    assert lift_at(y, p[::-1], 0.2) == 0.0  # worst ranking


def test_constant_score_has_no_lift():
    y = np.array([1, 0, 0, 0])
    assert lift_at(y, np.full(4, 0.25), 0.25) == 1.0
    assert evaluate(y, np.full(4, 0.25))["roc_auc"] == 0.5


def test_qini_curve_by_hand():
    # top-ranked: treated responder, treated non-responder, control responder, control non-r.
    y, t = np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0])
    curve = qini_curve(y, t, np.array([4, 3, 2, 1]))
    # k=3: 1 treated responder - 1 control responder * (2 treated / 1 control) = -1
    np.testing.assert_allclose(curve, [0, 1, 1, -1, 0])


def test_qini_coefficient_is_zero_for_random_and_positive_for_oracle():
    rng = np.random.default_rng(0)
    n = 20_000
    t = rng.integers(0, 2, n)
    effect = np.where(rng.random(n) < 0.3, 0.3, 0.0)  # 30% of customers respond to treatment
    y = (rng.random(n) < 0.1 + t * effect).astype(int)
    assert abs(qini_coefficient(qini_curve(y, t, rng.random(n)))) < 0.002
    # oracle: all gain (0.5 treated x 0.3 x 0.3 = 0.045 per customer) in the first 30%,
    # so the area above the random line is 0.045 * (1 - 0.3) / 2
    assert qini_coefficient(qini_curve(y, t, effect)) == pytest.approx(0.045 * 0.35, abs=0.002)
