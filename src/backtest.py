"""
Walk-forward backtest harness.

Split: the last TEST_HOURS of every series are held out. Models only ever see
data before each point they predict — this harness enforces that split so we
can add rungs (seasonal naive, SARIMA, GBM) on top without changing the rules.

Run:  python src/backtest.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from metrics import mase, mae
from models import naive, seasonal_naive

TEST_HOURS = 24 * 14   # hold out the last 14 days


def load() -> pd.DataFrame:
    df = pd.read_parquet("data/spot_prices.parquet")
    df["sid"] = df["instance_type"] + "@" + df["az"]
    return df.sort_values(["sid", "timestamp"])


def evaluate_naive(df: pd.DataFrame, n_test: int = TEST_HOURS) -> pd.DataFrame:
    """Per-series MASE/MAE for the naive (last-value) forecast."""
    rows = []
    for sid, g in df.groupby("sid", sort=False):
        y = g["spot_price"].to_numpy()
        if len(y) < n_test + 24 + 1:
            continue
        y_train, y_test = y[:-n_test], y[-n_test:]
        pred = naive(y, n_test)                       # y(t-1) for each test t
        rows.append({
            "sid": sid,
            "family": g["family"].iloc[0],
            "instance_type": g["instance_type"].iloc[0],
            "naive_mae": mae(y_test, pred),
            "naive_mase": mase(y_test, pred, y_train, m=24),
        })
    return pd.DataFrame(rows)


def evaluate_snaive(df: pd.DataFrame, n_test: int = TEST_HOURS, m: int = 24) -> pd.DataFrame:
    """Per-series MASE/MAE for the seasonal-naive (same hour yesterday) forecast."""
    rows = []
    for sid, g in df.groupby("sid", sort=False):
        y = g["spot_price"].to_numpy()
        if len(y) < n_test + m + 1:
            continue
        y_train, y_test = y[:-n_test], y[-n_test:]
        pred = seasonal_naive(y, n_test, m)            # y(t-24) for each test t
        rows.append({
            "sid": sid,
            "family": g["family"].iloc[0],
            "snaive_mae": mae(y_test, pred),
            "snaive_mase": mase(y_test, pred, y_train, m=m),
        })
    return pd.DataFrame(rows)


def main():
    df = load()
    n_series = df["sid"].nunique()

    r1 = evaluate_naive(df)
    r2 = evaluate_snaive(df)
    combined = r1.merge(r2[["sid", "snaive_mae", "snaive_mase"]], on="sid")
    combined.to_csv("outputs/rung2_snaive.csv", index=False)

    beats = (combined.snaive_mase < combined.naive_mase).sum()
    print(f"series evaluated : {len(combined)} / {n_series}")
    print(f"test window      : last {TEST_HOURS} h ({TEST_HOURS // 24} days)\n")
    print(f"{'model':<16}{'mean MASE':>12}{'median MASE':>14}")
    print(f"{'naive':<16}{r1.naive_mase.mean():>12.3f}{r1.naive_mase.median():>14.3f}")
    print(f"{'seasonal naive':<16}{r2.snaive_mase.mean():>12.3f}{r2.snaive_mase.median():>14.3f}\n")
    print(f"seasonal naive beats naive on {beats} / {len(combined)} series")
    print("\nReading it: if seasonal naive's MASE is higher, the daily cycle adds less")
    print("than short-term persistence at a 1-hour horizon — last value still wins.")


if __name__ == "__main__":
    main()
