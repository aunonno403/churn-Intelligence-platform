"""The published bundle is synthetic: it must look like the data without copying any row."""

import numpy as np
import pandas as pd

from churn.bundle import synthesize


def test_synthetic_rows_keep_the_patterns_but_copy_no_customer():
    rng = np.random.default_rng(0)
    n = 4000
    minutes = rng.gamma(2.0, 300.0, n)
    method = rng.choice(["41", "38", "36"], n, p=[0.7, 0.2, 0.1])
    churn = (rng.random(n) < np.where(method == "41", 0.02, 0.3)).astype("int32")
    X = pd.DataFrame(
        {
            "minutes_30d": minutes,
            "active_days_30d": np.clip((minutes / 40 + rng.normal(0, 2, n)).round(), 0, 30),
            "age": pd.array(np.where(rng.random(n) < 0.6, None, rng.integers(15, 60, n)), "Int32"),
            "last_payment_method": pd.Categorical(method),
            "total_paid": rng.gamma(3.0, 200.0, n) + minutes / 10,
        }
    )
    fake = synthesize(X, pd.Series(churn), 3000, seed=1)

    assert list(fake.columns) == list(X.columns)
    assert (fake.dtypes == X.dtypes).all()
    # the strong link between minutes and active days survives
    assert fake.minutes_30d.corr(fake.active_days_30d, method="spearman") > 0.8
    # so do missing values and category shares
    assert abs(fake.age.isna().mean() - 0.6) < 0.03
    assert abs((fake.last_payment_method == "41").mean() - 0.7) < 0.03
    # but no fake customer is a real one: each column's value comes from a different real row
    real_rows = set(X.astype(str).itertuples(index=False))
    assert not any(r in real_rows for r in fake.astype(str).itertuples(index=False))
