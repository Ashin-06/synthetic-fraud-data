"""A stand-in for a bank's private transaction table: mixed types, heavy tails,
correlated columns and rare fraud of several kinds. Replace with your own table via
`load_csv` - every other module only relies on the column schema."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Schema:
    continuous: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str = "is_fraud"
    condition: str = "fraud_type"

    @property
    def columns(self) -> list[str]:
        return list(self.continuous) + list(self.categorical)


SCHEMA = Schema(
    continuous=("customer_age", "tenure_months", "amount", "hour", "device_age_days", "distance_km", "txn_count_24h", "avg_amount_30d"),
    categorical=("income_band", "city_tier", "channel", "mcc", "is_international", "fraud_type", "is_fraud"),
)

FRAUD_TYPES = ["none", "card_testing", "account_takeover", "upi_scam", "skimming"]


def generate(n: int = 30_000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(37, 12, n), 18, 85).round()
    income = np.where(age < 25, rng.choice(["low", "mid"], n, p=[0.7, 0.3]),
                      rng.choice(["low", "mid", "high", "very_high"], n, p=[0.3, 0.4, 0.22, 0.08]))
    inc_mult = pd.Series(income).map({"low": 0.6, "mid": 1.0, "high": 1.8, "very_high": 3.2}).to_numpy()
    avg30 = np.exp(rng.normal(7.0, 0.6, n)) * inc_mult
    channel = rng.choice(["upi", "card_present", "card_not_present", "netbanking", "atm"], n, p=[0.45, 0.2, 0.18, 0.1, 0.07])
    mcc = rng.choice(["grocery", "fuel", "restaurants", "electronics", "travel", "utilities", "p2p", "gaming"], n,
                     p=[0.24, 0.1, 0.16, 0.08, 0.06, 0.1, 0.22, 0.04])
    df = pd.DataFrame({
        "customer_age": age,
        "tenure_months": np.clip(rng.gamma(2, 24, n) * (age - 17) / 20, 1, 400).round(),
        "amount": (avg30 * np.exp(rng.normal(0, 0.7, n))).round(2),
        "hour": (rng.normal(14.5, 4.5, n) % 24).round(),
        "device_age_days": rng.gamma(2, 180, n).round(),
        "distance_km": np.round(rng.lognormal(1.2, 1.0, n), 1),
        "txn_count_24h": rng.poisson(2.5, n).astype(float),
        "avg_amount_30d": avg30.round(2),
        "income_band": income,
        "city_tier": rng.choice(["tier1", "tier2", "tier3"], n, p=[0.45, 0.35, 0.2]),
        "channel": channel,
        "mcc": mcc,
        "is_international": np.where(rng.random(n) < 0.03, "yes", "no"),
        "fraud_type": "none",
        "is_fraud": "0",
    })
    nf = int(n * fraud_rate)
    idx = rng.choice(n, nf, replace=False)
    kinds = rng.choice(FRAUD_TYPES[1:], nf, p=[0.3, 0.3, 0.3, 0.1])
    for k in FRAUD_TYPES[1:]:
        rows = idx[kinds == k]
        m = len(rows)
        if not m:
            continue
        df.loc[rows, "fraud_type"] = k
        df.loc[rows, "is_fraud"] = "1"
        if k == "card_testing":
            df.loc[rows, "amount"] = rng.uniform(1, 50, m).round(2)
            df.loc[rows, "txn_count_24h"] = rng.poisson(14, m)
            df.loc[rows, "channel"] = "card_not_present"
            df.loc[rows, "mcc"] = rng.choice(["gaming", "electronics"], m)
        elif k == "account_takeover":
            df.loc[rows, "device_age_days"] = rng.integers(0, 3, m)
            df.loc[rows, "amount"] = (df.loc[rows, "avg_amount_30d"] * rng.uniform(4, 15, m)).round(2)
            df.loc[rows, "channel"] = rng.choice(["netbanking", "upi"], m)
            df.loc[rows, "hour"] = rng.choice([0, 1, 2, 3, 4, 23], m)
        elif k == "upi_scam":
            df.loc[rows, "channel"] = "upi"
            df.loc[rows, "mcc"] = "p2p"
            df.loc[rows, "amount"] = (df.loc[rows, "avg_amount_30d"] * rng.uniform(3, 10, m)).round(2)
            df.loc[rows, "customer_age"] = np.clip(rng.normal(62, 8, m), 45, 85).round()
        else:  # skimming
            df.loc[rows, "channel"] = rng.choice(["atm", "card_present"], m)
            df.loc[rows, "distance_km"] = np.round(rng.uniform(150, 1500, m), 1)
            df.loc[rows, "amount"] = rng.choice([5000, 10000, 20000], m).astype(float)
    return df


def load_csv(path: str, schema: Schema = SCHEMA) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in schema.categorical:
        df[c] = df[c].astype(str)
    return df[schema.columns]


# Public domain bounds (from product knowledge, not from the data). Differentially
# private synthesizers bin values inside these ranges so the bins themselves leak nothing.
BOUNDS = {
    "customer_age": (18, 90, "linear"), "tenure_months": (0, 480, "linear"), "amount": (0.5, 2_000_000, "log"),
    "hour": (0, 23, "linear"), "device_age_days": (0, 3650, "log1p"), "distance_km": (0, 5000, "log1p"),
    "txn_count_24h": (0, 60, "linear"), "avg_amount_30d": (1, 2_000_000, "log"),
}
CATEGORIES = {
    "income_band": ["high", "low", "mid", "very_high"], "city_tier": ["tier1", "tier2", "tier3"],
    "channel": ["atm", "card_not_present", "card_present", "netbanking", "upi"],
    "mcc": ["electronics", "fuel", "gaming", "grocery", "p2p", "restaurants", "travel", "utilities"],
    "is_international": ["no", "yes"], "fraud_type": FRAUD_TYPES, "is_fraud": ["0", "1"],
}
