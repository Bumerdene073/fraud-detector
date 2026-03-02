"""
Feature Engineering — Fraud Detection
=======================================
Transforms raw transaction data into ML-ready features.

We build 6 groups of features that together give the model
a complete picture of whether a transaction is suspicious.

Key concept: The model never sees raw fields like "amount".
It sees engineered signals like "this amount is 18x the user's
normal spend AND happened at 3am AND the device is unknown."
THAT combination is what catches fraud.

Run: python training/feature_engineering.py
"""

import pandas as pd
import numpy as np
import os
import pickle

RAW_FILE       = "data/raw/transactions.csv"
PROCESSED_FILE = "data/processed/features.csv"
REFERENCE_FILE = "data/reference/training_data.csv"  # saved for drift detection later


# ─────────────────────────────────────────────────────────────────────────────
# MERCHANT RISK MAPPING
# Not all merchants are equal risk — we encode domain knowledge here
# This is called "domain-driven feature engineering"
# A Data Scientist might miss this — a banking domain expert wouldn't
# ─────────────────────────────────────────────────────────────────────────────
MERCHANT_RISK = {
    # Low risk — everyday purchases
    "grocery"       : 1,
    "restaurant"    : 1,
    "gas_station"   : 1,
    "pharmacy"      : 1,
    "healthcare"    : 1,
    "utilities"     : 1,
    # Medium risk — occasional large purchases
    "clothing"      : 2,
    "entertainment" : 2,
    # High risk — easily resaleable, often targeted by fraudsters
    "electronics"   : 3,
    "jewelry"       : 4,
    "luxury_goods"  : 4,
    "gift_cards"    : 5,   # Highest! Gift cards = untraceable cash
    "crypto_exchange": 5,  # Highest! Irreversible transactions
    "wire_transfer" : 5,   # Highest! Irreversible, often money laundering
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features():

    # ── LOAD ──────────────────────────────────────────────────────────────────
    print("Loading raw transactions...")
    df = pd.read_csv(RAW_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df)} transactions")
    print(f"  Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # ── GROUP 1: AMOUNT FEATURES ───────────────────────────────────────────────
    print("\nBuilding amount features...")

    # How many times larger is this vs their personal average?
    # Already in raw data but we enhance it
    df["amount_vs_user_avg"] = df["amount_vs_user_avg"].clip(upper=100)
    # clip at 100x — extreme outliers can destabilize the model

    # Is this amount unusually high overall (not just for this user)?
    global_p75 = df["amount"].quantile(0.75)
    global_p95 = df["amount"].quantile(0.95)
    global_p99 = df["amount"].quantile(0.99)

    df["is_high_amount"]      = (df["amount"] > global_p75).astype(int)
    df["is_very_high_amount"] = (df["amount"] > global_p95).astype(int)
    df["is_extreme_amount"]   = (df["amount"] > global_p99).astype(int)

    # Log transform of amount — reduces skew, helps linear models
    # log(4850) = 8.48 vs log(45) = 3.81 — more meaningful difference
    df["log_amount"] = np.log1p(df["amount"])

    # ── GROUP 2: VELOCITY FEATURES ─────────────────────────────────────────────
    print("Building velocity features...")

    # Flag suspicious velocity thresholds
    # In real banking: >5 in 1h or >20 in 24h triggers review
    df["is_velocity_spike_1h"]  = (df["transactions_last_1h"]  > 5).astype(int)
    df["is_velocity_spike_24h"] = (df["transactions_last_24h"] > 15).astype(int)

    # Velocity ratio — how much busier than normal is this user right now?
    # Avoid division by zero with clip
    df["velocity_ratio"] = (
        df["transactions_last_1h"] /
        (df["transactions_last_24h"] / 24).clip(lower=0.01)
    ).clip(upper=50).round(3)
    # High ratio = suddenly very active = suspicious

    # ── GROUP 3: GEOGRAPHIC FEATURES ──────────────────────────────────────────
    print("Building geographic features...")

    # Distance risk tiers — how far from home?
    # 0-10km    = local (low risk)
    # 10-100km  = regional (medium risk)
    # 100-500km = domestic travel (elevated risk)
    # 500km+    = likely international / suspicious
    df["distance_risk"] = pd.cut(
        df["distance_from_home_km"],
        bins    = [0, 10, 100, 500, float("inf")],
        labels  = [0, 1, 2, 3],
        right   = False
    ).astype(int)

    # Combined: international AND far from home = very high risk
    df["international_far"] = (
        (df["is_international"] == 1) &
        (df["distance_from_home_km"] > 500)
    ).astype(int)

    # ── GROUP 4: TIME FEATURES ─────────────────────────────────────────────────
    print("Building time features...")

    # Late night flag — 2am to 5am is the highest fraud window
    df["is_late_night"] = (
        (df["hour_of_day"] >= 2) &
        (df["hour_of_day"] <= 5)
    ).astype(int)

    # Business hours flag — 9am-5pm is lowest fraud risk
    df["is_business_hours"] = (
        (df["hour_of_day"] >= 9) &
        (df["hour_of_day"] <= 17)
    ).astype(int)

    # Weekend flag — slightly higher fraud on weekends
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # ── GROUP 5: BEHAVIORAL FEATURES ──────────────────────────────────────────
    print("Building behavioral features...")

    # Merchant risk score — domain knowledge encoded as numbers
    df["merchant_risk_score"] = df["merchant_category"].map(MERCHANT_RISK).fillna(2)

    # High risk merchant flag
    df["is_high_risk_merchant"] = (df["merchant_risk_score"] >= 4).astype(int)

    # Card not present + international = very high risk combo
    df["online_international"] = (
        (df["card_present"] == 0) &
        (df["is_international"] == 1)
    ).astype(int)

    # Failed attempts risk flag
    df["has_failed_attempts"]      = (df["failed_attempts"] > 0).astype(int)
    df["many_failed_attempts"]     = (df["failed_attempts"] >= 3).astype(int)

    # Unknown device flag
    df["unknown_device"] = (df["device_match"] == 0).astype(int)

    # ── GROUP 6: COMBINED RISK SCORE ──────────────────────────────────────────
    print("Building combined risk score...")

    # This is a hand-crafted risk score that combines multiple signals
    # Think of it as a "pre-score" that helps the ML model
    # Each signal adds to the overall risk — weighted by importance
    df["manual_risk_score"] = (
        df["is_late_night"]          * 2.0 +   # late night = 2 points
        df["is_velocity_spike_1h"]   * 2.5 +   # velocity spike = 2.5 points
        df["many_failed_attempts"]   * 2.0 +   # many failures = 2 points
        df["unknown_device"]         * 1.5 +   # unknown device = 1.5 points
        df["international_far"]      * 3.0 +   # intl + far = 3 points
        df["is_high_risk_merchant"]  * 2.0 +   # risky merchant = 2 points
        df["online_international"]   * 2.5 +   # online + intl = 2.5 points
        df["is_extreme_amount"]      * 2.0     # extreme amount = 2 points
    )

    # Normalize to 0-1 range so it doesn't dominate other features
    max_score = df["manual_risk_score"].max()
    if max_score > 0:
        df["manual_risk_score"] = (df["manual_risk_score"] / max_score).round(4)

    # ── SELECT FINAL FEATURE COLUMNS ──────────────────────────────────────────
    feature_cols = [
        # Amount features
        "amount",
        "log_amount",
        "amount_vs_user_avg",
        "is_high_amount",
        "is_very_high_amount",
        "is_extreme_amount",
        # Velocity features
        "transactions_last_1h",
        "transactions_last_24h",
        "is_velocity_spike_1h",
        "is_velocity_spike_24h",
        "velocity_ratio",
        # Geographic features
        "distance_from_home_km",
        "is_international",
        "distance_risk",
        "international_far",
        # Time features
        "hour_of_day",
        "day_of_week",
        "is_late_night",
        "is_business_hours",
        "is_weekend",
        # Behavioral features
        "card_present",
        "merchant_risk_score",
        "is_high_risk_merchant",
        "online_international",
        "failed_attempts",
        "has_failed_attempts",
        "many_failed_attempts",
        "device_match",
        "unknown_device",
        # Combined risk score
        "manual_risk_score",
        # Label — kept for training, not a feature
        "is_fraud"
    ]

    df_features = df[feature_cols].dropna()

    # ── SAVE ──────────────────────────────────────────────────────────────────
    os.makedirs("data/processed",  exist_ok=True)
    os.makedirs("data/reference",  exist_ok=True)

    df_features.to_csv(PROCESSED_FILE, index=False)

    # Save reference copy for drift detection in Step 6
    df_features.to_csv(REFERENCE_FILE, index=False)

    # Save feature column list — model serving needs exact column order
    feature_only_cols = [c for c in feature_cols if c != "is_fraud"]
    os.makedirs("serving/model", exist_ok=True)
    with open("serving/model/feature_columns.pkl", "wb") as f:
        pickle.dump(feature_only_cols, f)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\nFeature engineering complete!")
    print(f"  Input rows     : {len(df)}")
    print(f"  Output rows    : {len(df_features)}")
    print(f"  Features built : {len(feature_cols) - 1}")
    print(f"  Saved to       : {PROCESSED_FILE}")

    print(f"\nFeature group summary:")
    print(f"  Amount features    : 6")
    print(f"  Velocity features  : 5")
    print(f"  Geographic features: 5")
    print(f"  Time features      : 5")
    print(f"  Behavioral features: 9")
    print(f"  Risk score         : 1")
    print(f"  TOTAL              : 31 features")

    print(f"\nClass distribution:")
    print(f"  Legitimate : {(df_features['is_fraud']==0).sum()}")
    print(f"  Fraud      : {(df_features['is_fraud']==1).sum()}")
    print(f"  Ratio      : {(df_features['is_fraud']==0).sum() / (df_features['is_fraud']==1).sum():.0f}:1")

    print(f"\nTop fraud signals (mean value fraud vs legit):")
    fraud_df = df_features[df_features["is_fraud"] == 1]
    legit_df = df_features[df_features["is_fraud"] == 0]
    signal_cols = [
        "amount_vs_user_avg", "is_velocity_spike_1h",
        "many_failed_attempts", "unknown_device",
        "international_far", "is_late_night",
        "manual_risk_score"
    ]
    for col in signal_cols:
        fraud_mean = fraud_df[col].mean()
        legit_mean = legit_df[col].mean()
        print(f"  {col:30s}  fraud={fraud_mean:.3f}  legit={legit_mean:.3f}")


if __name__ == "__main__":
    engineer_features()