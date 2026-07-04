"""
Unit tests covering data integrity, the preprocessing pipeline, and the API.
Run with: pytest -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.generate_data import generate
from src.api import app


# ---------- Data integrity ----------

def test_generated_data_schema():
    df = generate(n=500)
    expected_cols = {
        "age", "annual_income", "employment_length_years", "credit_score",
        "loan_amount", "existing_debt", "debt_to_income", "home_ownership",
        "purpose", "num_delinquencies_2yr", "default"
    }
    assert expected_cols.issubset(set(df.columns))


def test_generated_data_no_nulls():
    df = generate(n=500)
    assert df.isnull().sum().sum() == 0


def test_default_is_binary():
    df = generate(n=500)
    assert set(df["default"].unique()).issubset({0, 1})


def test_default_rate_is_realistic():
    # Real-world consumer loan default rates are typically 5-25%.
    # A generator producing 90% defaults (or 0%) would signal a bug.
    df = generate(n=5000)
    rate = df["default"].mean()
    assert 0.05 < rate < 0.30


def test_credit_score_within_valid_range():
    df = generate(n=500)
    assert df["credit_score"].between(300, 850).all()


# ---------- API ----------

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "age": 35, "annual_income": 60000, "employment_length_years": 5,
    "credit_score": 700, "loan_amount": 15000, "existing_debt": 8000,
    "debt_to_income": 0.25, "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation", "num_delinquencies_2yr": 0
}


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["decision"] in {"APPROVE", "DENY"}


def test_predict_rejects_invalid_credit_score(client):
    bad = dict(VALID_PAYLOAD, credit_score=200)  # below valid FICO range
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # Pydantic validation error


def test_predict_rejects_invalid_home_ownership(client):
    bad = dict(VALID_PAYLOAD, home_ownership="SPACESHIP")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_higher_risk_profile_scores_higher(client):
    low_risk = client.post("/predict", json=VALID_PAYLOAD).json()
    high_risk_payload = dict(
        VALID_PAYLOAD, credit_score=520, debt_to_income=1.9,
        num_delinquencies_2yr=4, employment_length_years=0.2
    )
    high_risk = client.post("/predict", json=high_risk_payload).json()
    assert high_risk["default_probability"] > low_risk["default_probability"]
