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
from models import naive

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


def main():
    df = load()
    n_series = df["sid"].nunique()
    res = evaluate_naive(df)
    res.to_csv("outputs/rung1_naive.csv", index=False)

    print(f"series evaluated : {len(res)} / {n_series}")
    print(f"test window      : last {TEST_HOURS} h ({TEST_HOURS // 24} days)\n")
    print("naive baseline — MASE (vs seasonal-naive in-sample scale):")
    print(f"  mean   {res.naive_mase.mean():.3f}")
    print(f"  median {res.naive_mase.median():.3f}")
    print(f"  range  {res.naive_mase.min():.3f} – {res.naive_mase.max():.3f}\n")
    print("by family (mean MASE):")
    print(res.groupby("family").naive_mase.mean().round(3).to_string())
    print("\nInterpretation: MASE ~1 means the naive forecast is about as good as")
    print("seasonal-naive; >1 means worse. This is the bar the next rungs must beat.")


if __name__ == "__main__":
    main()
