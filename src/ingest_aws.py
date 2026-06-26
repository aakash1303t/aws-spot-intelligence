"""
Real EC2 spot-price ingestion via boto3 — run this once you have a (free) AWS account.

DescribeSpotPriceHistory is a read-only call with no charge. AWS retains roughly
the last 90 days of history, which matches what generate_synthetic.py produces, so
the downstream forecasting/optimizer code runs unchanged on this output.

Setup:
    pip install boto3
    aws configure          # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    python src/ingest_aws.py

Output schema matches the synthetic generator exactly:
    timestamp, instance_type, region, az, family, vcpu, ram_gib, on_demand,
    spot_price, savings_pct
"""
from __future__ import annotations
import datetime as dt
import pandas as pd

# Mirror the synthetic universe so models/dashboard need zero changes.
from generate_synthetic import INSTANCES, REGIONS

PRODUCT = ["Linux/UNIX"]


def fetch_region(region: str, instance_types: list[str], days: int = 90) -> pd.DataFrame:
    import boto3
    ec2 = boto3.client("ec2", region_name=region)
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    paginator = ec2.get_paginator("describe_spot_price_history")
    records = []
    for page in paginator.paginate(
        InstanceTypes=instance_types,
        ProductDescriptions=PRODUCT,
        StartTime=start,
        EndTime=end,
    ):
        for item in page["SpotPriceHistory"]:
            records.append({
                "timestamp": pd.Timestamp(item["Timestamp"]).floor("h"),
                "instance_type": item["InstanceType"],
                "region": region,
                "az": item["AvailabilityZone"],
                "spot_price": float(item["SpotPrice"]),
            })
    return pd.DataFrame.from_records(records)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach static specs + on-demand prices, compute savings, hourly resample."""
    meta = pd.DataFrame([
        {"instance_type": k, "family": v["fam"], "vcpu": v["vcpu"],
         "ram_gib": v["ram"], "on_demand": v["od"]}
        for k, v in INSTANCES.items()
    ])
    df = df.merge(meta, on="instance_type", how="left")
    # forward-fill to a regular hourly grid per series (spot reports irregularly)
    df = (df.sort_values("timestamp")
            .groupby(["instance_type", "region", "az"], group_keys=False)
            .apply(lambda g: g.set_index("timestamp")
                              .resample("h").ffill().reset_index()))
    df["savings_pct"] = (1 - df["spot_price"] / df["on_demand"]) * 100
    return df.dropna(subset=["spot_price"])


def main():
    frames = []
    for region in REGIONS:
        types = list(INSTANCES.keys())
        print(f"fetching {region} ...")
        try:
            frames.append(fetch_region(region, types))
        except Exception as e:  # region not enabled / no creds — skip gracefully
            print(f"  skipped {region}: {e}")
    if not frames:
        raise SystemExit("No data fetched. Check AWS credentials and enabled regions.")
    df = enrich(pd.concat(frames, ignore_index=True))
    out = "data/spot_prices.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}  rows={len(df):,}")


if __name__ == "__main__":
    main()
