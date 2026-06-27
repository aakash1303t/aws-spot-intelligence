"""
Predictability map.

Reads outputs/predictability.csv (written by backtest.py) and renders the
project's headline figure: which spot markets the model can forecast, and which
resist prediction. Two panels:

  A. naive MASE vs GBM MASE per series — points below the diagonal are series
     where the GBM extracts signal the baselines miss.
  B. GBM improvement over naive, by instance family — reveals that predictability
     is structural: some families forecast well, others are mostly noise.

Run:  python src/predictability_map.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NAVY, ORANGE = "#232f3e", "#ff9900"
FAMILY_COLOR = {
    "burst": "#0972d3", "general": "#ff9900", "compute": "#1d8102",
    "memory": "#8b5cf6", "gpu": "#d13212",
}


def main():
    df = pd.read_csv("outputs/predictability.csv")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "text.color": NAVY,
        "xtick.color": "#5f6b7a", "ytick.color": "#5f6b7a",
    })
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="white",
                           gridspec_kw={"width_ratios": [1, 1.05]})
    for a in ax:
        a.set_facecolor("white")

    # ---- Panel A: naive vs GBM MASE ----
    for fam, c in FAMILY_COLOR.items():
        s = df[df.family == fam]
        ax[0].scatter(s.naive_mase, s.gbm_mase, s=34, color=c, alpha=.8,
                      edgecolor="white", linewidth=.5, label=fam)
    lim = [0, max(df.naive_mase.max(), df.gbm_mase.max()) * 1.05]
    ax[0].plot(lim, lim, color=NAVY, lw=1, ls="--", alpha=.6)
    ax[0].text(lim[1] * 0.97, lim[1] * 0.9, "GBM worse", ha="right", fontsize=8,
               color="#5f6b7a", style="italic")
    ax[0].text(lim[1] * 0.55, lim[1] * 0.04, "GBM better", ha="center", fontsize=8,
               color="#5f6b7a", style="italic")
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("naive MASE"); ax[0].set_ylabel("GBM MASE")
    ax[0].set_title("Per-series: does the GBM beat naive?", fontsize=11, loc="left", color=NAVY)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")

    # ---- Panel B: improvement by family (the predictability map) ----
    order = (df.groupby("family").gbm_vs_naive_pct.median()
               .sort_values().index.tolist())
    rng = np.random.default_rng(1)
    for i, fam in enumerate(order):
        s = df[df.family == fam]
        jit = rng.uniform(-.13, .13, len(s))
        ax[1].scatter(s.gbm_vs_naive_pct, i + jit, s=30, color=FAMILY_COLOR[fam],
                      alpha=.75, edgecolor="white", linewidth=.4)
        med = s.gbm_vs_naive_pct.median()
        ax[1].plot([med, med], [i - .28, i + .28], color=NAVY, lw=2)
    ax[1].axvline(0, color=NAVY, lw=1, ls=":")
    ax[1].text(1.5, len(order) - .5, "GBM better →", fontsize=8, color="#5f6b7a", style="italic")
    ax[1].text(-1.5, len(order) - .5, "← naive better", fontsize=8, color="#5f6b7a",
               style="italic", ha="right")
    ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels(order)
    ax[1].set_xlabel("GBM improvement over naive (%)")
    ax[1].set_title("Predictability by instance family", fontsize=11, loc="left", color=NAVY)
    ax[1].set_xlim(min(df.gbm_vs_naive_pct.min() * 1.1, -20),
                   df.gbm_vs_naive_pct.max() * 1.15)

    plt.tight_layout()
    plt.savefig("outputs/predictability_map.png", dpi=130, facecolor="white")
    print("saved outputs/predictability_map.png")


if __name__ == "__main__":
    main()
