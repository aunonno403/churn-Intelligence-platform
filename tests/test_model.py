import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from churn.models.model import ChurnModel


def test_churn_model_scores_churners_higher():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"risk": rng.normal(size=400), "noise": rng.normal(size=400)})
    y = (X.risk + rng.normal(scale=0.5, size=400) > 1).astype(int).to_numpy()
    booster = lgb.LGBMClassifier(n_estimators=30, verbose=-1).fit(X, y)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibrator.fit(booster.predict_proba(X)[:, 1], y)
    model = ChurnModel(booster, calibrator, ["risk", "noise"])

    X_extra = X.assign(unused=1.0)[["unused", "noise", "risk"]]  # column order must not matter
    proba = model.predict_proba(X_extra)
    assert proba.shape == (400, 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_array_equal(model.predict(X_extra), proba[:, 1])
    # column 1 is churn: churners must score higher on average
    assert proba[y == 1, 1].mean() > proba[y == 0, 1].mean() + 0.3
