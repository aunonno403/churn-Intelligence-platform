import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from churn.explain.shap_utils import driver_sentence, explain
from churn.models.model import ChurnModel


def test_shap_values_add_up_to_the_raw_score():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = (X.a > 0.5).astype(int).to_numpy()
    booster = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)
    model = ChurnModel(booster, IsotonicRegression(), ["a", "b"])
    shap = explain(model, X)
    assert list(shap.columns) == ["a", "b", "base"]
    np.testing.assert_allclose(1 / (1 + np.exp(-shap.sum(axis=1))), model.predict_raw(X))


def test_driver_rule_follows_shap_and_reports_real_churn_rates():
    x = pd.Series(np.arange(100, dtype=float), name="days_inactive")
    shap = x.to_numpy() - 70  # risk rises past 70 days
    churned = (x.to_numpy() > 70).astype(int)
    churned[:10] = 1  # some churners below the cut too
    d = driver_sentence(x, shap, churned)
    assert d["rule"].startswith("days_inactive > ")
    assert 68 <= float(d["rule"].split("> ")[1]) <= 71
    assert d["churn_rate_in_group"] == 1.0
    assert 0.1 < d["churn_rate_rest"] < 0.2  # the 10 early churners out of ~70
