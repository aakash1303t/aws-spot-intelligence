"""
Realistic synthetic EC2 spot-price generator.

Why this exists: querying real spot prices needs a (free) AWS account, and we
want the modelling/dashboard to run end-to-end *now*. This produces data with
the same statistical fingerprints real spot prices have, so models built here
transfer directly once `ingest_aws.py` swaps in real data.

Spot-price dynamics we reproduce:
  - price floats as a fraction of on-demand (typically 30-90%), capped at on-demand
  - mean-reverting AR(1) noise (prices wander but get pulled back)
  - diurnal + weekly seasonality (demand is higher in business hours/weekdays)
  - occasional capacity-crunch spikes toward on-demand (the fat tail)
  - per-instance volatility (GPU instances are far choppier than burstable)
  - per-AZ baseline offsets (same instance differs across availability zones)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Real-ish us-east-1 on-demand hourly USD prices, plus a volatility profile per type.
# vol: relative AR(1) noise scale. crunch: probability/severity of capacity spikes.
INSTANCES = {
    # type           on_demand  vcpu  ram_gib  vol    crunch  family
    "t3.medium":   dict(od=0.0416, vcpu=2,  ram=4,   vol=0.6, crunch=0.6, fam="burst"),
    "m5.large":    dict(od=0.096,  vcpu=2,  ram=8,   vol=0.8, crunch=0.8, fam="general"),
    "m5.xlarge":   dict(od=0.192,  vcpu=4,  ram=16,  vol=0.9, crunch=0.9, fam="general"),
    "m5.2xlarge":  dict(od=0.384,  vcpu=8,  ram=32,  vol=1.0, crunch=1.0, fam="general"),
    "c5.large":    dict(od=0.085,  vcpu=2,  ram=4,   vol=0.9, crunch=0.9, fam="compute"),
    "c5.xlarge":   dict(od=0.17,   vcpu=4,  ram=8,   vol=1.0, crunch=1.0, fam="compute"),
    "r5.large":    dict(od=0.126,  vcpu=2,  ram=16,  vol=0.9, crunch=0.9, fam="memory"),
    "r5.xlarge":   dict(od=0.252,  vcpu=4,  ram=32,  vol=1.0, crunch=1.1, fam="memory"),
    "g4dn.xlarge": dict(od=0.526,  vcpu=4,  ram=16,  vol=1.6, crunch=1.8, fam="gpu"),
    "p3.2xlarge":  dict(od=3.06,   vcpu=8,  ram=61,  vol=2.2, crunch=2.6, fam="gpu"),
}

REGIONS = {
    "us-east-1":      dict(azs=["a", "b", "c"], mult=1.00),
    "us-west-2":      dict(azs=["a", "b", "c"], mult=1.03),
    "eu-west-1":      dict(azs=["a", "b"],      mult=1.08),
    "ap-southeast-2": dict(azs=["a", "b", "c"], mult=1.12),  # Sydney
}


def _seasonality(idx: pd.DatetimeIndex) -> np.ndarray:
    """Multiplicative demand factor from hour-of-day and day-of-week."""
    hod = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    # business-hours bump (peak ~14:00), weekend dip
    diurnal = 0.06 * np.sin((hod - 8) / 24 * 2 * np.pi)
    weekly = np.where(dow >= 5, -0.05, 0.02)
    return 1.0 + diurnal + weekly


def _series(n: int, od: float, vol: float, crunch: float, base_frac: float,
            rng: np.random.Generator, season: np.ndarray) -> np.ndarray:
    """One (instance, region, az) price path of length n."""
    # AR(1) mean-reverting log-noise
    phi, sigma = 0.92, 0.015 * vol
    eps = rng.normal(0, sigma, n)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + eps[t]
    base = od * base_frac * season * np.exp(z)

    # capacity crunches: rare clustered spikes toward on-demand
    spikes = np.zeros(n)
    n_events = rng.poisson(0.9 * crunch * n / (24 * 30))  # ~per-month rate
    for _ in range(n_events):
        start = rng.integers(0, n)
        dur = int(rng.integers(2, 14))
        height = rng.uniform(0.4, 1.0) * (od - base[start]) * rng.uniform(0.5, 1.0)
        decay = np.exp(-np.arange(dur) / max(dur / 3, 1))
        end = min(start + dur, n)
        spikes[start:end] += height * decay[: end - start]

    price = np.clip(base + spikes, od * 0.18, od)  # never below ~18% or above on-demand
    return np.round(price, 5)


def generate(days: int = 90, freq: str = "h", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Pinned anchor (not "now") so the dataset is identical on every machine/run.
    end = pd.Timestamp("2026-06-21 06:00", tz="UTC")
    idx = pd.date_range(end=end, periods=days * 24, freq=freq)
    season = _seasonality(idx)
    rows = []
    for inst, spec in INSTANCES.items():
        for region, rspec in REGIONS.items():
            for az in rspec["azs"]:
                base_frac = float(rng.uniform(0.30, 0.45))  # per-AZ baseline discount
                od_eff = spec["od"] * rspec["mult"]
                price = _series(len(idx), od_eff, spec["vol"], spec["crunch"],
                                base_frac, rng, season)
                rows.append(pd.DataFrame({
                    "timestamp": idx,
                    "instance_type": inst,
                    "region": region,
                    "az": f"{region}{az}",
                    "family": spec["fam"],
                    "vcpu": spec["vcpu"],
                    "ram_gib": spec["ram"],
                    "on_demand": round(od_eff, 5),
                    "spot_price": price,
                }))
    df = pd.concat(rows, ignore_index=True)
    df["savings_pct"] = (1 - df["spot_price"] / df["on_demand"]) * 100
    return df


if __name__ == "__main__":
    df = generate()
    out = "data/spot_prices.parquet"
    df.to_parquet(out, index=False)
    print(f"rows={len(df):,}  series={df.groupby(['instance_type','az']).ngroups}")
    print(f"span={df.timestamp.min()} -> {df.timestamp.max()}")
    print(f"mean savings vs on-demand: {df.savings_pct.mean():.1f}%")
    print(f"wrote {out}")
