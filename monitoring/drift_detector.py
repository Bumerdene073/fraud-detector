from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a customer will churn",
    version="1.0.0"
)

# ── Load model and scaler on startup ─────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",  "models/model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Request schema ────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    tenure:           float
    monthly_charges:  float
    total_charges:    float
    contract_type:    int    # 0=monthly, 1=one_year, 2=two_year
    tech_support:     int    # 0=No, 1=Yes
    online_security:  int    # 0=No, 1=Yes

class PredictionResponse(BaseModel):
    churn_prediction: int           # 0 or 1
    churn_probability: float        # 0.0 to 1.0
    risk_level: str                 # Low / Medium / High

# ── Health check endpoint ─────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

# ── Prediction endpoint ───────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        features = np.array([[
            customer.tenure,
            customer.monthly_charges,
            customer.total_charges,
            customer.contract_type,
            customer.tech_support,
            customer.online_security
        ]])

        features_scaled = scaler.transform(features)

        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Risk level bucketing
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=round(float(probability), 4),
            risk_level=risk
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Batch prediction endpoint ─────────────────────────────────
@app.post("/predict/batch")
def predict_batch(customers: list[CustomerFeatures]):
    return [predict(c) for c in customers]