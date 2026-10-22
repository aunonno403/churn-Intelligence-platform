"""The saved model object. Kept free of training-only imports (MLflow, Optuna, DuckDB):
model.joblib records this module path, so loading the model only imports this file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression


@dataclass
class ChurnModel:
    """LightGBM scores mapped through isotonic calibration.

    Rank users with `predict_raw` (calibration ties scores and loses ranking detail);
    use `predict_proba` / `predict` wherever a real probability is needed.
    """

    booster: lgb.LGBMClassifier
    calibrator: IsotonicRegression
    features: list[str]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated churn probability. Deliberately not a 0/1 label: the cut-off comes
        from the savings curve, not a fixed 0.5."""
        return self.calibrator.predict(self.predict_raw(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1 - p, p])
