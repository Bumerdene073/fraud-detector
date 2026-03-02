"""
FastAPI Fraud Detection Server
================================
Serves the trained fraud detection model as a production REST API.
Designed for < 100ms real-time transaction scoring.

Endpoints:
  GET  /health           -> Server + model health check
  GET  /model/info       -> Model metadata (version, metrics, features)
  POST /predict          -> Score single transaction
  POST /predict/batch    -> Score multiple transactions
  POST /predict/explain  -> Score + explain WHY it's flagged
  GET  /docs             -> Auto Swagger UI

Run: uvicorn serving.app:app --reload --port 8000
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi         import FastAPI, HTTPException, Query
from pydantic        import BaseModel, Field
from typing          import List, Optional
import pickle
import numpy  as np
import os
from datetime        import datetime
import time

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Fraud Detection API",
    description = """
    Real-time fraud detection for banking, fintech, and e-commerce.
    Scores transactions in < 100ms using XGBoost trained on 10,000 transactions.

    Fraud patterns detected:
    - Velocity abuse (too many transactions too fast)
    - Geographic anomaly (impossible location jumps)
    - Amount spike (way above user's normal spend)
    - Merchant mismatch (unusual category)
    - Late night fraud (2am-5am patterns)
    - Card not present (online fraud)
    """,
    version     = "1.0.0"
)
# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# Load model + scaler + metadata + feature columns on startup
# All must be loaded together — they were saved together during training
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "serving/model"
model     = None
scaler    = None
metadata  = None
feature_cols = None

@app.on_event("startup")
def load_model():
    global model, scaler, metadata, feature_cols

    paths = {
        "model"    : os.path.join(MODEL_DIR, "fraud_model.pkl"),
        "scaler"   : os.path.join(MODEL_DIR, "scaler.pkl"),
        "metadata" : os.path.join(MODEL_DIR, "metadata.pkl"),
        "features" : os.path.join(MODEL_DIR, "feature_columns.pkl"),
    }

    # Verify all files exist before loading
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            print(f"[ERROR] Run training/train_model.py first!")
            return

    try:
        with open(paths["model"],    "rb") as f: model        = pickle.load(f)
        with open(paths["scaler"],   "rb") as f: scaler       = pickle.load(f)
        with open(paths["metadata"], "rb") as f: metadata     = pickle.load(f)
        with open(paths["features"], "rb") as f: feature_cols = pickle.load(f)

        print(f"[READY] Model loaded: {metadata['model_name']}")
        print(f"[READY] F1={metadata['f1_score']:.4f} | "
              f"ROC-AUC={metadata['roc_auc']:.4f}")
        print(f"[READY] Features: {len(feature_cols)}")
        print(f"[READY] Fraud Detection API is live!")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMA
# Every field maps directly to what feature_engineering.py produced
# Pydantic validates all types + ranges before your code runs
# ─────────────────────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    # Core transaction fields
    amount                  : float = Field(..., gt=0,          description="Transaction amount in USD")
    merchant_category       : str   = Field(...,                description="e.g. grocery, electronics, jewelry")
    is_international        : int   = Field(..., ge=0, le=1,    description="1=international, 0=domestic")
    card_present            : int   = Field(..., ge=0, le=1,    description="1=in-person, 0=online")

    # User behavioral context
    user_id                 : int   = Field(..., gt=0,          description="Customer ID")
    amount_vs_user_avg      : float = Field(..., ge=0,          description="Amount / user's historical average")
    transactions_last_1h    : int   = Field(..., ge=0,          description="How many transactions in last 1 hour")
    transactions_last_24h   : int   = Field(..., ge=0,          description="How many transactions in last 24 hours")

    # Geographic
    distance_from_home_km   : float = Field(..., ge=0,          description="Distance from user's home location in km")

    # Security signals
    failed_attempts         : int   = Field(..., ge=0,          description="Failed attempts before this transaction")
    device_match            : int   = Field(..., ge=0, le=1,    description="1=known device, 0=unknown device")

    # Time
    hour_of_day             : int   = Field(..., ge=0, le=23,   description="Hour of transaction (0-23)")

    # Fraud decision threshold (optional — defaults to 0.5)
    threshold               : float = Field(0.5, ge=0.1, le=0.9,
                                            description="Fraud probability threshold (0.1-0.9)")

    class Config:
        json_schema_extra = {
            "example": {
                "amount"                : 4850.00,
                "merchant_category"     : "electronics",
                "is_international"      : 1,
                "card_present"          : 0,
                "user_id"               : 347,
                "amount_vs_user_avg"    : 18.4,
                "transactions_last_1h"  : 12,
                "transactions_last_24h" : 25,
                "distance_from_home_km" : 8472.3,
                "failed_attempts"       : 5,
                "device_match"          : 0,
                "hour_of_day"           : 3,
                "threshold"             : 0.5
            }
        }
# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class FraudPrediction(BaseModel):
    transaction_id      : str
    user_id             : int
    is_fraud            : int
    fraud_probability   : float
    risk_level          : str
    decision            : str
    confidence          : str
    inference_ms        : float
    timestamp           : str


class BatchResponse(BaseModel):
    total_transactions  : int
    fraud_count         : int
    fraud_rate_pct      : float
    total_amount_at_risk: float
    results             : List[FraudPrediction]
    processing_ms       : float


class ExplainResponse(BaseModel):
    prediction          : FraudPrediction
    top_fraud_signals   : List[dict]
    recommendation      : str
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# Converts raw Transaction input into the exact feature vector
# the model was trained on — same order, same transformations
# This is where the Feature Store concept lives
# ─────────────────────────────────────────────────────────────────────────────
MERCHANT_RISK = {
    "grocery": 1, "restaurant": 1, "gas_station": 1,
    "pharmacy": 1, "healthcare": 1, "utilities": 1,
    "clothing": 2, "entertainment": 2,
    "electronics": 3, "jewelry": 4, "luxury_goods": 4,
    "gift_cards": 5, "crypto_exchange": 5, "wire_transfer": 5,
}

def build_feature_vector(txn: Transaction) -> np.ndarray:
    """
    Builds the exact same features as feature_engineering.py.
    This is the serving-side feature computation.
    In production this would read from a Feature Store (Feast/Redis).
    """
    import math

    amount              = txn.amount
    merchant_risk_score = MERCHANT_RISK.get(txn.merchant_category, 2)

    # Amount features
    log_amount          = math.log1p(amount)
    is_high_amount      = 1 if amount > 200  else 0
    is_very_high_amount = 1 if amount > 500  else 0
    is_extreme_amount   = 1 if amount > 2000 else 0

    # Velocity features
    is_velocity_spike_1h  = 1 if txn.transactions_last_1h  > 5  else 0
    is_velocity_spike_24h = 1 if txn.transactions_last_24h > 15 else 0
    velocity_ratio = round(
        txn.transactions_last_1h /
        max(txn.transactions_last_24h / 24, 0.01), 3
    )
    velocity_ratio = min(velocity_ratio, 50)

    # Geographic features
    dist = txn.distance_from_home_km
    if dist < 10:       distance_risk = 0
    elif dist < 100:    distance_risk = 1
    elif dist < 500:    distance_risk = 2
    else:               distance_risk = 3

    international_far = (
        1 if txn.is_international == 1 and dist > 500 else 0
    )

    # Time features
    hour            = txn.hour_of_day
    is_late_night   = 1 if 2 <= hour <= 5  else 0
    is_business_hrs = 1 if 9 <= hour <= 17 else 0
    day_of_week     = datetime.now().weekday()
    is_weekend      = 1 if day_of_week >= 5 else 0

    # Behavioral features
    is_high_risk_merchant = 1 if merchant_risk_score >= 4 else 0
    online_international  = (
        1 if txn.card_present == 0 and txn.is_international == 1 else 0
    )
    has_failed_attempts  = 1 if txn.failed_attempts > 0 else 0
    many_failed_attempts = 1 if txn.failed_attempts >= 3 else 0
    unknown_device       = 1 if txn.device_match == 0 else 0

    # Combined manual risk score
    raw_risk = (
        is_late_night          * 2.0 +
        is_velocity_spike_1h   * 2.5 +
        many_failed_attempts   * 2.0 +
        unknown_device         * 1.5 +
        international_far      * 3.0 +
        is_high_risk_merchant  * 2.0 +
        online_international   * 2.5 +
        is_extreme_amount      * 2.0
    )
    manual_risk_score = round(min(raw_risk / 18.0, 1.0), 4)

    # Build vector in EXACT same order as feature_engineering.py
    feature_vector = [
        amount,
        log_amount,
        min(txn.amount_vs_user_avg, 100),
        is_high_amount,
        is_very_high_amount,
        is_extreme_amount,
        txn.transactions_last_1h,
        txn.transactions_last_24h,
        is_velocity_spike_1h,
        is_velocity_spike_24h,
        velocity_ratio,
        dist,
        txn.is_international,
        distance_risk,
        international_far,
        hour,
        day_of_week,
        is_late_night,
        is_business_hrs,
        is_weekend,
        txn.card_present,
        merchant_risk_score,
        is_high_risk_merchant,
        online_international,
        txn.failed_attempts,
        has_failed_attempts,
        many_failed_attempts,
        txn.device_match,
        unknown_device,
        manual_risk_score,
    ]

    return np.array(feature_vector).reshape(1, -1)
# ────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# Converts raw probability into business decision
# This is where MLOps meets business logic
# ─────────────────────────────────────────────────────────────────────────────
def make_decision(probability: float, threshold: float) -> dict:
    is_fraud = int(probability >= threshold)

    if probability >= 0.8:
        risk_level  = "CRITICAL"
        decision    = "BLOCK"
        confidence  = "Very High"
    elif probability >= 0.6:
        risk_level  = "HIGH"
        decision    = "BLOCK" if is_fraud else "REVIEW"
        confidence  = "High"
    elif probability >= 0.4:
        risk_level  = "MEDIUM"
        decision    = "REVIEW"
        confidence  = "Medium"
    elif probability >= 0.2:
        risk_level  = "LOW"
        decision    = "APPROVE"
        confidence  = "High"
    else:
        risk_level  = "MINIMAL"
        decision    = "APPROVE"
        confidence  = "Very High"

    return {
        "is_fraud"  : is_fraud,
        "risk_level": risk_level,
        "decision"  : decision,
        "confidence": confidence,
    }
# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    model_ready = all([
        model is not None,
        scaler is not None,
        metadata is not None,
        feature_cols is not None
    ])
    return {
        "status"       : "healthy" if model_ready else "unhealthy",
        "model_loaded" : model_ready,
        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version"      : "1.0.0"
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — MODEL INFO
# Exposes model metadata — useful for UI teams and audits
# In banking, model versioning is a regulatory requirement
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/model/info")
def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name"    : metadata["model_name"],
        "trained_on"    : metadata["trained_on"],
        "f1_score"      : round(metadata["f1_score"],   4),
        "roc_auc"       : round(metadata["roc_auc"],    4),
        "precision"     : round(metadata["precision"],  4),
        "recall"        : round(metadata["recall"],     4),
        "inference_ms"  : round(metadata["infer_ms"],   4),
        "feature_count" : len(feature_cols),
        "features"      : feature_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — SINGLE PREDICTION
# Core endpoint — scores one transaction in real time
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=FraudPrediction)
def predict(txn: Transaction):
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run training first.")

    start   = time.time()

    # Build features → scale → predict
    X       = build_feature_vector(txn)
    X_scaled = scaler.transform(X)
    proba   = float(model.predict_proba(X_scaled)[0][1])

    infer_ms = round((time.time() - start) * 1000, 3)

    decision = make_decision(proba, txn.threshold)

    return FraudPrediction(
        transaction_id    = f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}",
        user_id           = txn.user_id,
        is_fraud          = decision["is_fraud"],
        fraud_probability = round(proba, 4),
        risk_level        = decision["risk_level"],
        decision          = decision["decision"],
        confidence        = decision["confidence"],
        inference_ms      = infer_ms,
        timestamp         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — BATCH PREDICTION
# Score multiple transactions — useful for batch processing
# MuleSoft can send last 100 transactions every 5 minutes
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(transactions: List[Transaction]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(transactions) > 500:
        raise HTTPException(status_code=400,
                            detail="Batch limit is 500 transactions")

    start   = time.time()
    results = []

    for txn in transactions:
        X        = build_feature_vector(txn)
        X_scaled = scaler.transform(X)
        proba    = float(model.predict_proba(X_scaled)[0][1])
        infer_ms = round((time.time() - start) * 1000, 3)
        decision = make_decision(proba, txn.threshold)

        results.append(FraudPrediction(
            transaction_id    = f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}",
            user_id           = txn.user_id,
            is_fraud          = decision["is_fraud"],
            fraud_probability = round(proba, 4),
            risk_level        = decision["risk_level"],
            decision          = decision["decision"],
            confidence        = decision["confidence"],
            inference_ms      = infer_ms,
            timestamp         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

    total_ms        = round((time.time() - start) * 1000, 2)
    fraud_results   = [r for r in results if r.is_fraud == 1]
    amount_at_risk  = sum(
        t.amount for t, r in zip(transactions, results) if r.is_fraud == 1
    )

    return BatchResponse(
        total_transactions   = len(results),
        fraud_count          = len(fraud_results),
        fraud_rate_pct       = round(len(fraud_results) / len(results) * 100, 2),
        total_amount_at_risk = round(amount_at_risk, 2),
        results              = results,
        processing_ms        = total_ms
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 — EXPLAIN PREDICTION (most impressive endpoint!)
# Not just "is this fraud" — but WHY the model thinks so
# This is called Explainable AI (XAI) — huge in banking (regulatory)
# Banks legally need to explain why a transaction was blocked
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict/explain", response_model=ExplainResponse)
def predict_explain(txn: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start    = time.time()
    X        = build_feature_vector(txn)
    X_scaled = scaler.transform(X)
    proba    = float(model.predict_proba(X_scaled)[0][1])
    infer_ms = round((time.time() - start) * 1000, 3)
    decision = make_decision(proba, txn.threshold)

    prediction = FraudPrediction(
        transaction_id    = f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}",
        user_id           = txn.user_id,
        is_fraud          = decision["is_fraud"],
        fraud_probability = round(proba, 4),
        risk_level        = decision["risk_level"],
        decision          = decision["decision"],
        confidence        = decision["confidence"],
        inference_ms      = infer_ms,
        timestamp         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # Build human-readable fraud signals
    signals = []

    if txn.amount_vs_user_avg > 5:
        signals.append({
            "signal"     : "Amount Spike",
            "severity"   : "HIGH",
            "detail"     : f"Transaction is {txn.amount_vs_user_avg:.1f}x "
                           f"the user's normal spend",
            "risk_points": min(txn.amount_vs_user_avg / 10, 5.0)
        })

    if txn.transactions_last_1h > 5:
        signals.append({
            "signal"     : "Velocity Abuse",
            "severity"   : "HIGH",
            "detail"     : f"{txn.transactions_last_1h} transactions in last 1 hour "
                           f"(normal: 1-3)",
            "risk_points": min(txn.transactions_last_1h / 5, 5.0)
        })

    if txn.distance_from_home_km > 500:
        signals.append({
            "signal"     : "Geographic Anomaly",
            "severity"   : "CRITICAL",
            "detail"     : f"Transaction {txn.distance_from_home_km:.0f}km from home",
            "risk_points": min(txn.distance_from_home_km / 1000, 5.0)
        })

    if txn.failed_attempts >= 3:
        signals.append({
            "signal"     : "Multiple Failed Attempts",
            "severity"   : "HIGH",
            "detail"     : f"{txn.failed_attempts} failed attempts before success "
                           f"(card testing pattern)",
            "risk_points": min(txn.failed_attempts, 5.0)
        })

    if txn.device_match == 0:
        signals.append({
            "signal"     : "Unknown Device",
            "severity"   : "MEDIUM",
            "detail"     : "Transaction from an unrecognized device",
            "risk_points": 2.0
        })

    if 2 <= txn.hour_of_day <= 5:
        signals.append({
            "signal"     : "Late Night Activity",
            "severity"   : "MEDIUM",
            "detail"     : f"Transaction at {txn.hour_of_day}:00am "
                           f"(peak fraud window: 2am-5am)",
            "risk_points": 2.0
        })

    if txn.is_international == 1 and txn.card_present == 0:
        signals.append({
            "signal"     : "Online International",
            "severity"   : "HIGH",
            "detail"     : "International online transaction — card not present",
            "risk_points": 3.0
        })

    merchant_risk = MERCHANT_RISK.get(txn.merchant_category, 2)
    if merchant_risk >= 4:
        signals.append({
            "signal"     : "High Risk Merchant",
            "severity"   : "HIGH",
            "detail"     : f"Merchant category '{txn.merchant_category}' "
                           f"has high fraud association (risk={merchant_risk}/5)",
            "risk_points": float(merchant_risk)
        })

    # Sort signals by severity
    signals.sort(key=lambda x: x["risk_points"], reverse=True)

    # Build human-readable recommendation
    if decision["decision"] == "BLOCK":
        recommendation = (
            f"BLOCK this transaction. "
            f"Fraud probability {proba*100:.1f}% exceeds threshold. "
            f"Alert customer via SMS/email. "
            f"Primary signals: {', '.join(s['signal'] for s in signals[:2])}."
        )
    elif decision["decision"] == "REVIEW":
        recommendation = (
            f"FLAG for manual review. "
            f"Fraud probability {proba*100:.1f}% requires investigation. "
            f"Contact customer to verify before processing. "
            f"Primary concern: {signals[0]['signal'] if signals else 'elevated risk'}."
        )
    else:
        recommendation = (
            f"APPROVE this transaction. "
            f"Fraud probability {proba*100:.1f}% is within acceptable range."
        )

    return ExplainResponse(
        prediction       = prediction,
        top_fraud_signals = signals,
        recommendation   = recommendation
    )