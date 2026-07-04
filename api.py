"""
api.py

Minimal FastAPI service wrapping the trained model.

Run with:  uvicorn src.api:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""

import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "models/loan_risk_model.pkl"
THRESHOLD_PATH = "models/decision_threshold.json"

state = {"model": None, "threshold": 0.5}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        state["threshold"] = json.load(f)["threshold"]
    yield
    state.clear()


app = FastAPI(title="Loan Risk Analyzer API", version="1.0", lifespan=lifespan)


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    employment_length_years: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)
    loan_amount: float = Field(..., gt=0)
    existing_debt: float = Field(..., ge=0)
    debt_to_income: float = Field(..., ge=0)
    home_ownership: str = Field(..., pattern="^(RENT|MORTGAGE|OWN)$")
    purpose: str
    num_delinquencies_2yr: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    default_probability: float
    decision: str
    threshold_used: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([application.model_dump()])
    proba = float(state["model"].predict_proba(row)[:, 1][0])
    threshold = state["threshold"]
    decision = "DENY" if proba >= threshold else "APPROVE"

    return PredictionResponse(
        default_probability=round(proba, 4),
        decision=decision,
        threshold_used=threshold,
    )
