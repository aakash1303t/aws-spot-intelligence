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


# ---- 3: SARIMA, one-step-ahead over the test window (leakage-free, fast) ----
def sarima(y_train: np.ndarray, y_test: np.ndarray,
           order=(1, 0, 1), seasonal=(1, 0, 1, 24)) -> np.ndarray:
    """
    Honest 1-step-ahead forecasts for the test window.

    Parameters are estimated on the TRAIN data only. Those fixed parameters are
    then applied to the full series and we read off the in-sample 1-step-ahead
    predictions over the test span (dynamic=False), each of which uses only
    actual values up to t-1. So: no parameter leakage, no peeking at the target,
    and one filter pass instead of a slow refit loop.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y_train = np.asarray(y_train, float)
    y_full = np.concatenate([y_train, np.asarray(y_test, float)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(y_train, order=order, seasonal_order=seasonal,
                         enforce_stationarity=False, enforce_invertibility=False
                         ).fit(disp=False, maxiter=50, method="lbfgs")
        full = fitted.apply(y_full)                          # reuse train params
        pred = full.get_prediction(start=len(y_train),
                                   end=len(y_full) - 1,
                                   dynamic=False).predicted_mean
    return np.asarray(pred)


# ---- 4: global gradient-boosting model ----
class GlobalGBM:
    """
    One model for all series. Lag/calendar features make it series-aware.

    It predicts the **change** from the last hour's price, not the absolute price
    (target = y - lag_1). The reconstructed forecast is therefore
    `last_price + predicted_change` — i.e. a learned correction on top of the
    naive forecast. This matters because series span very different price scales
    (cents to dollars); training on absolute price lets the expensive series
    dominate the loss and starves the cheap ones. Predicting the change puts the
    model's worst case at "predict no change" = naive, so it can't blow up on a
    cheap series the way an absolute-price model can.
    """

    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=7,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            random_state=7,
            categorical_features=[FEATURE_COLS.index(c) for c in CATEGORICAL],
        )

    def fit(self, feat_train: pd.DataFrame):
        X = feat_train[FEATURE_COLS]
        target = feat_train["y"] - feat_train["lag_1"]      # the change to learn
        ok = X.notna().all(axis=1) & target.notna()
        self.model.fit(X[ok], target[ok])
        return self

    def predict(self, feat: pd.DataFrame) -> np.ndarray:
        change = self.model.predict(feat[FEATURE_COLS])
        return feat["lag_1"].to_numpy() + change            # rebuild the price
