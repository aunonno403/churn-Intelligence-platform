import numpy as np

from churn.eval.metrics import evaluate, lift_at


def test_lift_counts_churners_in_top_fraction():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 20% churn
    p = np.linspace(1, 0, 10)  # perfect ranking
    assert lift_at(y, p, 0.2) == 5.0  # top 2 are both churners: 100% / 20%
    assert lift_at(y, p[::-1], 0.2) == 0.0  # worst ranking


def test_constant_score_has_no_lift():
    y = np.array([1, 0, 0, 0])
    assert lift_at(y, np.full(4, 0.25), 0.25) == 1.0
    assert evaluate(y, np.full(4, 0.25))["roc_auc"] == 0.5
