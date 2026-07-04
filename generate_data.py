"""
generate_data.py

Generates a synthetic loan applicant dataset.

WHY SYNTHETIC DATA (be ready to explain this in interviews):
- Real datasets like LendingClub or "Give Me Some Credit" are great, but they're
  heavily used, often static/outdated, and their true generating process is unknown
  (you can't cleanly verify your model recovered the right relationships).
- This generator encodes a *known* causal structure (income, credit score, DTI,
  employment length, loan amount -> default probability) plus realistic noise.
  This lets us sanity-check that the model & SHAP explanations recover sensible
  relationships, and it lets you swap in a real dataset later without changing
  the rest of the pipeline (same column names/schema is easy to replicate).
- In your README/interview: be upfront this is synthetic, and mention that the
  pipeline (preprocessing, evaluation, thresholding, explainability, API) is the
  reusable part -- that's the actual skill being demonstrated.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 15000


def generate(n=N):
    age = RNG.integers(21, 70, n)
    annual_income = np.clip(RNG.normal(55000, 25000, n), 12000, 300000)
    employment_length = np.clip(RNG.normal(6, 5, n), 0, 40).round(1)
    credit_score = np.clip(RNG.normal(680, 75, n), 300, 850).round(0)
    loan_amount = np.clip(RNG.normal(15000, 9000, n), 1000, 60000)
    existing_debt = np.clip(RNG.normal(12000, 10000, n), 0, 150000)
    home_ownership = RNG.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.4, 0.15])
    purpose = RNG.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business", "medical", "other"],
        n, p=[0.35, 0.2, 0.15, 0.1, 0.1, 0.1]
    )
    num_delinquencies_2yr = RNG.poisson(0.3, n)

    debt_to_income = (existing_debt + loan_amount * 0.15) / annual_income
    debt_to_income = np.clip(debt_to_income, 0, 3)

    # --- True (hidden) risk generating process ---
    # Higher score = higher default probability. Coefficients are illustrative,
    # loosely mirroring known real-world risk drivers.
    logit = (
        -3.6
        + 3.2 * debt_to_income
        - 0.011 * (credit_score - 680)
        - 0.000012 * (annual_income - 55000)
        - 0.05 * employment_length
        + 0.55 * num_delinquencies_2yr
        + 0.00002 * loan_amount
        + np.where(home_ownership == "RENT", 0.25, 0.0)
        + np.where(purpose == "small_business", 0.35, 0.0)
        + RNG.normal(0, 0.5, n)  # irreducible noise
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = RNG.binomial(1, prob_default)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "employment_length_years": employment_length,
        "credit_score": credit_score,
        "loan_amount": loan_amount.round(2),
        "existing_debt": existing_debt.round(2),
        "debt_to_income": debt_to_income.round(3),
        "home_ownership": home_ownership,
        "purpose": purpose,
        "num_delinquencies_2yr": num_delinquencies_2yr,
        "default": default,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/loan_data.csv", index=False)
    print(f"Generated {len(df)} rows -> data/loan_data.csv")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(df.head())
