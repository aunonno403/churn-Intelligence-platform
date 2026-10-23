"""Known-answer test: on a simulated RCT with a known effect, both learners must find it."""

import numpy as np
import pandas as pd
import pytest

from churn.eval.metrics import qini_coefficient, qini_curve
from churn.models.uplift import TLearner, XLearner, segments


def _rct(n, rng):
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    true_uplift = np.where(X.a > 0, 0.15, -0.05)  # half helped, half put off
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.2 + t * true_uplift).astype(int)
    return X, t, y, true_uplift


@pytest.mark.parametrize("learner", [TLearner, XLearner])
def test_learner_recovers_known_uplift(learner):
    rng = np.random.default_rng(0)
    X, t, y, _ = _rct(20_000, rng)
    model = learner(seed=0).fit(X, t, y)
    X_new, t_new, y_new, truth = _rct(10_000, rng)
    pred = model.predict(X_new)
    assert pred[truth > 0].mean() == pytest.approx(0.15, abs=0.04)
    assert pred[truth < 0].mean() == pytest.approx(-0.05, abs=0.04)
    oracle = qini_coefficient(qini_curve(y_new, t_new, truth))
    assert qini_coefficient(qini_curve(y_new, t_new, pred)) > 0.8 * oracle


def test_segments():
    p0 = np.array([0.9, 0.1, 0.5, 0.5, 0.8, 0.05])
    uplift = np.array([0.0, 0.0, 0.2, -0.2, 0.005, -0.005])
    assert segments(p0, uplift).tolist() == [
        "sure thing", "lost cause", "persuadable", "sleeping dog", "sure thing", "lost cause"
    ]  # fmt: skip


def test_changed_hillstrom_file_is_refused(tmp_path):
    from churn.config import Settings
    from churn.models.uplift import load_hillstrom

    path = tmp_path / "hillstrom" / "hillstrom.csv"
    path.parent.mkdir()
    path.write_text("recency,segment,visit\n1,No E-Mail,0\n")  # present, so no download
    with pytest.raises(ValueError, match="SHA-256"):
        load_hillstrom(Settings(data_dir=tmp_path))


def test_empty_arm_gives_none_not_nan():
    from churn.models.uplift import _effect

    assert _effect(np.array([1, 0]), np.array([1, 1])) == (None, None)  # no control rows
