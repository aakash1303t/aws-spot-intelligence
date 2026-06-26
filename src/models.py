"""
The model ladder, lowest rung first. A model only earns its place if it beats
the rung below it.

  1. naive          y_hat(t) = y(t-1)
  2. seasonal naive y_hat(t) = y(t-24)         (same hour yesterday)
  3. SARIMA         classical trend + seasonal model, per series
  4. gbm (global)   gradient-boosted trees over lag/calendar/rolling features
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURE_COLS, CATEGORICAL


# ---- 1 & 2: baselines (vectorised, 1-step-ahead over a test window) ----
def naive(history: np.ndarray, n_test: int) -> np.ndarray:
    """Predict each test point as the value one hour earlier (actual)."""
    return history[-n_test - 1: -1]


def seasonal_naive(history: np.ndarray, n_test: int, m: int = 24) -> np.ndarray:
    """Predict each test point as the value m hours earlier (actual)."""
    return history[-n_test - m: -m]


# ---- 3: SARIMA, fit once then filtered forward over the test window ----
def sarima_rolling(y_train: np.ndarray, y_test: np.ndarray,
                   order=(1, 0, 1), seasonal=(1, 0, 1, 24)) -> np.ndarray:
    """
    Honest 1-step-ahead: fit on train, then walk through the test window feeding
    each observed value back in (state updates, no refit, no peeking at the point
    being predicted).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y_train, order=order, seasonal_order=seasonal,
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=50, method="lbfgs")
        preds = []
        for actual in y_test:
            preds.append(float(res.forecast(steps=1)[0]))   # predict next
            res = res.append([actual], refit=False)         # then reveal truth
    return np.asarray(preds)


# ---- 4: global gradient-boosting model ----
class GlobalGBM:
    """One model for all series. Lag/calendar features make it series-aware."""

    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=7,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            random_state=7,
            categorical_features=[FEATURE_COLS.index(c) for c in CATEGORICAL],
        )

    def fit(self, feat_train: pd.DataFrame):
        X = feat_train[FEATURE_COLS]
        y = feat_train["y"]
        ok = X.notna().all(axis=1) & y.notna()
        self.model.fit(X[ok], y[ok])
        return self

    def predict(self, feat: pd.DataFrame) -> np.ndarray:
        return self.model.predict(feat[FEATURE_COLS])
