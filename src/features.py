"""
Feature engineering for short-horizon spot-price forecasting.

Every feature is built from values at time <= t-1 to predict the price at t,
so a 1-step-ahead forecast never sees its own target. We build features per
series (one instance_type in one AZ) and stack them for a single global model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12, 24, 48, 168]      # hours: recent + daily + weekly
ROLL = [6, 24, 168]                        # rolling-window sizes (hours)


def series_id(df: pd.DataFrame) -> pd.Series:
    return df["instance_type"] + "@" + df["az"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame with one row per (series, hour); target = spot_price."""
    df = df.copy()
    df["sid"] = series_id(df)
    df = df.sort_values(["sid", "timestamp"])
    g = df.groupby("sid", sort=False)["spot_price"]

    feat = pd.DataFrame(index=df.index)
    feat["sid"] = df["sid"].values
    feat["timestamp"] = df["timestamp"].values
    feat["y"] = df["spot_price"].values

    # lag features
    for L in LAGS:
        feat[f"lag_{L}"] = g.shift(L).values

    # rolling stats of *past* values only (shift(1) before rolling => no leakage)
    for W in ROLL:
        past = g.shift(1)
        feat[f"rmean_{W}"] = past.rolling(W, min_periods=max(2, W // 4)).mean().values
        feat[f"rstd_{W}"] = past.rolling(W, min_periods=max(2, W // 4)).std().values

    # short-term momentum
    feat["diff_1"] = (g.shift(1) - g.shift(2)).values
    feat["diff_24"] = (g.shift(1) - g.shift(25)).values

    # calendar
    ts = pd.to_datetime(feat["timestamp"])
    feat["hour"] = ts.dt.hour.values
    feat["dow"] = ts.dt.dayofweek.values
    feat["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int).values

    # static descriptors (help a global model tell series apart)
    feat["family"] = df["family"].astype("category").cat.codes.values
    feat["vcpu"] = df["vcpu"].values
    feat["ram_gib"] = df["ram_gib"].values
    feat["on_demand"] = df["on_demand"].values

    return feat


FEATURE_COLS = (
    [f"lag_{L}" for L in LAGS]
    + [f"rmean_{W}" for W in ROLL] + [f"rstd_{W}" for W in ROLL]
    + ["diff_1", "diff_24", "hour", "dow", "is_weekend",
       "family", "vcpu", "ram_gib", "on_demand"]
)
CATEGORICAL = ["hour", "dow", "is_weekend", "family"]
