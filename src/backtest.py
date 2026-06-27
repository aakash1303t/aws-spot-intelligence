"""
Walk-forward backtest harness — the full ladder, rungs 1-4.

Split: the last TEST_HOURS of every series are held out. Models only ever see
data before each point they predict — this harness enforces that split so each
rung is judged under identical rules.

  rung 1  naive           y(t) = y(t-1)            — all 110 series
  rung 2  seasonal naive  y(t) = y(t-24)           — all 110 series
  rung 3  SARIMA          per-series fit           — 5 representative series
  rung 4  GBM (global)    one model, all series    — all 110 series

SARIMA is fit per series and is heavy, so it runs on one representative series
per family. Everything else runs on all 110. The leaderboard has a full-110 view
(rungs 1, 2, 4) and a fair head-to-head on the shared 5 series (all rungs).

Run:  python src/backtest.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from metrics import mase, mae
from features import build_features
from models import naive, seasonal_naive, sarima, GlobalGBM

TEST_HOURS = 24 * 14   # hold out the last 14 days

REPRESENTATIVE = [
    "t3.medium@us-east-1a",    # burst
    "m5.xlarge@us-east-1a",    # general
    "c5.xlarge@us-east-1a",    # compute
    "r5.xlarge@us-east-1a",    # memory
    "g4dn.xlarge@us-east-1a",  # gpu
]


def load() -> pd.DataFrame:
    df = pd.read_parquet("data/spot_prices.parquet")
    df["sid"] = df["instance_type"] + "@" + df["az"]
    return df.sort_values(["sid", "timestamp"])


def evaluate_naive(df: pd.DataFrame, n_test: int = TEST_HOURS) -> pd.DataFrame:
    """Rung 1 — naive (last-value) forecast."""
    rows = []
    for sid, g in df.groupby("sid", sort=False):
        y = g["spot_price"].to_numpy()
        if len(y) < n_test + 24 + 1:
            continue
        y_train, y_test = y[:-n_test], y[-n_test:]
        pred = naive(y, n_test)
        rows.append({
            "sid": sid, "family": g["family"].iloc[0],
            "instance_type": g["instance_type"].iloc[0],
            "naive_mae": mae(y_test, pred),
            "naive_mase": mase(y_test, pred, y_train, m=24),
        })
    return pd.DataFrame(rows)


def evaluate_snaive(df: pd.DataFrame, n_test: int = TEST_HOURS, m: int = 24) -> pd.DataFrame:
    """Rung 2 — seasonal-naive (same hour yesterday) forecast."""
    rows = []
    for sid, g in df.groupby("sid", sort=False):
        y = g["spot_price"].to_numpy()
        if len(y) < n_test + m + 1:
            continue
        y_train, y_test = y[:-n_test], y[-n_test:]
        pred = seasonal_naive(y, n_test, m)
        rows.append({
            "sid": sid,
            "snaive_mae": mae(y_test, pred),
            "snaive_mase": mase(y_test, pred, y_train, m=m),
        })
    return pd.DataFrame(rows)


def evaluate_sarima(df: pd.DataFrame, sids=REPRESENTATIVE, n_test: int = TEST_HOURS) -> pd.DataFrame:
    """Rung 3 — SARIMA on the representative subset."""
    rows = []
    for sid in sids:
        g = df[df["sid"] == sid].sort_values("timestamp")
        if g.empty:
            print(f"  ! series not found, skipping: {sid}")
            continue
        y = g["spot_price"].to_numpy()
        y_train, y_test = y[:-n_test], y[-n_test:]
        pred = sarima(y_train, y_test)
        m = mase(y_test, pred, y_train, m=24)
        rows.append({"sid": sid, "family": g["family"].iloc[0],
                     "sarima_mae": mae(y_test, pred), "sarima_mase": m})
        print(f"  fit {sid:<24} sarima MASE = {m:.3f}")
    return pd.DataFrame(rows)


def evaluate_gbm(df: pd.DataFrame, n_test: int = TEST_HOURS) -> pd.DataFrame:
    """
    Rung 4 — one global gradient-boosting model across all series.

    Build leakage-free features, train on everything before the cutoff, predict
    the held-out window, then score MASE per series. Because the model is global,
    a single fit covers all 110 series — which is what makes the predictability
    map affordable.
    """
    feat = build_features(df)
    uniq = np.sort(feat["timestamp"].unique())
    cutoff = uniq[-(n_test + 1)]                       # last n_test hours are test
    train = feat[feat["timestamp"] <= cutoff]
    test = feat[feat["timestamp"] > cutoff].copy()

    gbm = GlobalGBM().fit(train)
    test["pred"] = gbm.predict(test)

    rows = []
    for sid, g in test.groupby("sid", sort=False):
        y_train = feat.loc[(feat["sid"] == sid) & (feat["timestamp"] <= cutoff), "y"].to_numpy()
        y_test, pred = g["y"].to_numpy(), g["pred"].to_numpy()
        rows.append({
            "sid": sid,
            "gbm_mae": mae(y_test, pred),
            "gbm_mase": mase(y_test, pred, y_train, m=24),
        })
    return pd.DataFrame(rows)


def main():
    df = load()
    n_series = df["sid"].nunique()

    r1 = evaluate_naive(df)
    r2 = evaluate_snaive(df)
    print("training global GBM (rung 4) ...")
    r4 = evaluate_gbm(df)
    print(f"fitting SARIMA on {len(REPRESENTATIVE)} representative series ...")
    r3 = evaluate_sarima(df)

    # full-110 table (rungs 1, 2, 4) + predictability fields
    full = (r1.merge(r2[["sid", "snaive_mae", "snaive_mase"]], on="sid")
              .merge(r4[["sid", "gbm_mae", "gbm_mase"]], on="sid"))
    full["gbm_vs_naive_pct"] = (1 - full["gbm_mase"] / full["naive_mase"]) * 100
    full.to_csv("outputs/leaderboard_all.csv", index=False)
    full[["sid", "family", "instance_type", "naive_mase", "snaive_mase",
          "gbm_mase", "gbm_vs_naive_pct"]].to_csv("outputs/predictability.csv", index=False)

    # representative head-to-head (all rungs)
    rep = full[full["sid"].isin(r3["sid"])].merge(r3[["sid", "sarima_mase"]], on="sid")
    rep.to_csv("outputs/leaderboard_representative.csv", index=False)

    line = "-" * 64
    print(f"\n{line}\nLEADERBOARD\n{line}")
    print(f"test window: last {TEST_HOURS // 24} days   |   series: {n_series}\n")
    print("All series (rungs 1, 2, 4)     mean MASE     median MASE")
    for name, col in [("naive", "naive_mase"), ("seasonal naive", "snaive_mase"),
                      ("GBM (global)", "gbm_mase")]:
        print(f"  {name:<26}{full[col].mean():>7.3f}        {full[col].median():>7.3f}")

    gbm_wins = (full.gbm_mase < full.naive_mase).sum()
    print(f"\n  GBM beats naive on {gbm_wins} / {len(full)} series   "
          f"(mean improvement {full.gbm_vs_naive_pct.mean():.1f}%)")

    print(f"\nRepresentative 5 (all rungs, fair head-to-head)")
    show = rep.set_index("sid")[["naive_mase", "snaive_mase", "sarima_mase", "gbm_mase"]].round(3)
    print(show.to_string())
    print(f"\n  mean:  naive {rep.naive_mase.mean():.3f}   snaive {rep.snaive_mase.mean():.3f}   "
          f"sarima {rep.sarima_mase.mean():.3f}   gbm {rep.gbm_mase.mean():.3f}")
    print(line)
    print("Run  python src/predictability_map.py  to render the predictability map.")


if __name__ == "__main__":
    main()
