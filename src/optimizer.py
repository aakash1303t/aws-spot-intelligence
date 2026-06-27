"""
Optimizer — turns prices/forecasts into a decision.

Given a workload spec (minimum vCPU/RAM and allowed regions), recommend the
cheapest instance + availability zone, adjusted for interruption risk, plus a
suggested launch window. This is the layer that makes the project a product:
the forecasting core says *what the price will be*; the optimizer says
*what to run, where, and when*.

Design choices, with rationale:
  - Cost estimate = the most recent price. The forecasting analysis showed the
    naive (last-value) forecast is the strongest simple predictor at short
    horizons, so it's the honest near-term cost estimate. (Swap in the GBM's
    forecast here later for a multi-hour view.)
  - Risk = recent price volatility (coefficient of variation over VOL_WINDOW
    hours). Choppy series are likelier to spike and get reclaimed, so they score
    riskier. `risk_aversion` lets the user trade a little money for stability.
  - Launch window = the series' hour-of-day price profile, used to find the
    cheapest upcoming hour — a concrete "launch now vs wait" call.

Run:  python src/optimizer.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from backtest import load

VOL_WINDOW = 48                       # hours used for the volatility / risk proxy
US_REGIONS = ["us-east-1", "us-west-2"]


def _latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per (instance_type, az), plus a recent-volatility risk score."""
    df = df.sort_values("timestamp")
    rows = []
    for _, g in df.groupby("sid", sort=False):
        recent = g["spot_price"].to_numpy()[-VOL_WINDOW:]
        last = g.iloc[-1]
        cv = float(recent.std() / recent.mean()) if recent.mean() else np.nan
        rows.append({
            "instance_type": last["instance_type"], "az": last["az"],
            "region": last["region"], "family": last["family"],
            "vcpu": int(last["vcpu"]), "ram_gib": int(last["ram_gib"]),
            "on_demand": float(last["on_demand"]),
            "price_now": float(last["spot_price"]),
            "risk": cv,
        })
    return pd.DataFrame(rows)


def _risk_label(rn: float) -> str:
    return "low" if rn < 0.34 else "medium" if rn < 0.67 else "high"


def _launch_window(df: pd.DataFrame, instance_type: str, az: str, horizon_h: int = 12) -> str:
    """Use the series' hour-of-day average to find the cheapest upcoming hour."""
    g = df[(df["instance_type"] == instance_type) & (df["az"] == az)].copy()
    hours = pd.to_datetime(g["timestamp"]).dt.hour
    hourly = g.assign(hour=hours).groupby("hour")["spot_price"].mean()
    last_hour = int(hours.iloc[-1])
    upcoming = [(last_hour + h) % 24 for h in range(horizon_h + 1)]
    cheapest = min(upcoming, key=lambda h: hourly.get(h, np.inf))
    offset = upcoming.index(cheapest)
    if offset == 0:
        return "Launch now — already near the daily low."
    return f"Consider waiting ~{offset}h: price tends to bottom near {cheapest:02d}:00 UTC."


def recommend(df: pd.DataFrame, min_vcpu: int = 1, min_ram: int = 1,
              regions=None, risk_aversion: float = 0.5, top_n: int = 5):
    """
    Return (ranked_options, recommendation_message).

    risk_aversion=0 ranks on raw price; higher values penalise volatile options.
    """
    snap = _latest_snapshot(df)
    regions = regions or sorted(snap["region"].unique())

    elig = snap[(snap.vcpu >= min_vcpu) & (snap.ram_gib >= min_ram)
                & (snap.region.isin(regions))].copy()
    if elig.empty:
        return elig, "No instance matches that spec."

    # normalise risk to [0,1] across the eligible set so risk_aversion is intuitive
    r = elig["risk"]
    elig["risk_norm"] = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else 0.0
    elig["savings_pct"] = (1 - elig.price_now / elig.on_demand) * 100
    # risk-adjusted effective hourly cost — lower is better
    elig["score"] = elig.price_now * (1 + risk_aversion * elig["risk_norm"])

    ranked = elig.sort_values("score").head(top_n).reset_index(drop=True)
    best = ranked.iloc[0]
    timing = _launch_window(df, best["instance_type"], best["az"])
    msg = (f"Run on {best.instance_type} in {best.az} — "
           f"${best.price_now:.4f}/hr, {best.savings_pct:.0f}% cheaper than on-demand, "
           f"{_risk_label(best.risk_norm)} interruption risk. {timing}")
    return ranked, msg


if __name__ == "__main__":
    df = load()
    print("EC2 Spot Optimizer — example queries")

    queries = [
        ("General compute: >=4 vCPU, >=16 GB, US regions",
         dict(min_vcpu=4, min_ram=16, regions=US_REGIONS)),
        ("Memory-heavy: >=8 vCPU, >=32 GB, any region",
         dict(min_vcpu=8, min_ram=32)),
        ("Cost-first vs stability-first (same spec, risk_aversion 0 -> 3)",
         dict(min_vcpu=4, min_ram=8)),
    ]
    cols = ["instance_type", "az", "region", "price_now", "savings_pct", "risk_norm", "score"]

    for title, spec in queries[:2]:
        ranked, msg = recommend(df, **spec)
        print(f"\n{'='*70}\n{title}\n{'-'*70}")
        print("  →", msg, "\n")
        print(ranked[cols].round(4).to_string(index=False))

    # show how risk_aversion changes the pick (same spec, different stability weight)
    print(f"\n{'='*70}\n{queries[2][0]}\n{'-'*70}")
    flip_spec = dict(min_vcpu=4, min_ram=16, regions=US_REGIONS)
    for ra in (0.0, 3.0):
        ranked, msg = recommend(df, **flip_spec, risk_aversion=ra)
        print(f"  risk_aversion={ra}:  {msg}")
