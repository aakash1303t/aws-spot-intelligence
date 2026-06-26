"""
Forecasting metrics.

MASE (Mean Absolute Scaled Error) is the headline metric. It divides the model's
test error by the in-sample error of a *seasonal naive* forecast (same hour, one
day earlier). The result is scale-free, so prices in cents and prices in dollars
are comparable, and MASE < 1 means the model beats that seasonal-naive benchmark.
"""
from __future__ import annotations
import numpy as np


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else np.nan


def seasonal_scale(y_train, m: int = 24) -> float:
    """Denominator for MASE: in-sample MAE of a seasonal-naive forecast."""
    y = np.asarray(y_train, float)
    if len(y) <= m:
        return np.nan
    d = np.abs(y[m:] - y[:-m])
    d = d[~np.isnan(d)]
    return float(d.mean()) if len(d) else np.nan


def mase(y_true, y_pred, y_train, m: int = 24) -> float:
    scale = seasonal_scale(y_train, m)
    if not scale or np.isnan(scale):
        return np.nan
    return mae(y_true, y_pred) / scale
