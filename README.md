# Loan Risk Analyzer

A loan default risk model with a full pipeline: data generation, preprocessing,
model training with class-imbalance handling, business-cost-driven decision
thresholding, permutation-based interpretability, a served REST API, and tests.

## The problem, framed as a decision 

> Given an applicant's financial profile, estimate the probability they'll
> default, and recommend **APPROVE** or **DENY**.

The interesting part isn't "train a classifier" — it's that the two error
types have very different costs:
- **False Negative** (approve a borrower who defaults) → lose the loan principal.
- **False Positive** (deny a borrower who would've repaid) → lose the profit
  margin on that loan, but no principal loss.

Because these costs aren't equal, the "right" decision threshold is not 0.5.
This project tunes the threshold explicitly against an assumed cost matrix
(see `src/train.py`) instead of accepting the classifier default.

## Data

`data/loan_data.csv` is **synthetically generated** (`src/generate_data.py`),
not scraped or downloaded. This is a deliberate choice:

- Real datasets like LendingClub are widely reused and have an unknown true
  generating process, which makes it hard to sanity-check whether a model
  learned the right relationships or just noise.
- Here, the generator encodes a known causal structure (debt-to-income and
  credit score drive default risk, with realistic noise layered on) at a
  realistic ~18% default rate. This lets me verify the model and the
  interpretability output actually recover sensible relationships — which
  they do (see Results below).
- The rest of the pipeline (preprocessing, evaluation, thresholding, the API)
  is dataset-agnostic. Swapping in a real dataset like LendingClub only
  requires matching the column schema.

**Limitation to be upfront about:** synthetic data can't capture real-world
messiness (missing values, label noise, data entry errors, regime shifts).
A production version would need real historical data and periodic
retraining as economic conditions change.

## Approach

1. **Split before preprocessing** — train/test split happens first to avoid
   any leakage from fitting scalers/encoders on test data.
2. **Two baselines before the real model:**
   - Majority-class dummy classifier (82.4% accuracy just by always
     predicting "no default" — this is why accuracy alone is a useless
     metric here, since the classes are imbalanced ~82/18).
   - Logistic Regression with `class_weight="balanced"`.
3. **Main model:** Random Forest (`class_weight="balanced"`, depth-limited to
   reduce overfitting on this feature set).
4. **Evaluation:** ROC-AUC and PR-AUC (not accuracy), since PR-AUC is the more
   informative metric when positives are rare.
5. **Threshold tuning:** the default 0.5 cutoff is not used for the final
   decision. Instead, thresholds are swept and the one minimizing expected
   dollar cost (given the assumed cost matrix) is selected.
6. **Interpretability:** permutation importance on the full pipeline —
   measures the actual drop in model performance when a feature is shuffled,
   which is more honest than tree-based `.feature_importances_` (which is
   biased toward high-cardinality features).

## Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Dummy (majority class) | — | — (82.4% accuracy, 0% recall on defaults) |
| Logistic Regression | 0.817 | 0.588 |
| Random Forest | 0.810 | 0.577 |

At the naive 0.5 threshold, the Random Forest catches **64% of defaulters**
(precision 44%). At the **cost-optimized threshold (t=0.27)**, recall rises to
**91%** of defaulters caught, at the cost of more false positives (precision
drops to 26%) — the right trade for this cost structure, since missing a
default (~$15k loss) is assumed far more expensive than wrongly rejecting a
good applicant (~$1.5k in forgone profit).

**Expected cost comparison on the test set:**
- At threshold 0.5: **$3,532,500**
- At cost-optimized threshold 0.27: **$2,772,000**
- **Savings from threshold tuning alone: $760,500** — with no change to the
  underlying model.

This is the single biggest point worth making in an interview: *model
selection and threshold selection are two different decisions, and getting
the second one right can matter as much as the first.*

**Top predictive features** (permutation importance):
1. `debt_to_income`
2. `credit_score`
3. `annual_income`
4. `num_delinquencies_2yr`
5. `home_ownership`

This ordering matches the causal structure built into the data generator,
which confirms the model is learning real signal rather than noise.

See `reports/` for ROC curve, precision-recall curve, cost-vs-threshold plot,
and the feature importance chart.

## Project structure

```
loan-risk-analyzer/
├── data/
│   └── loan_data.csv          # generated dataset
├── src/
│   ├── generate_data.py       # synthetic data generator
│   ├── train.py                # full training + evaluation pipeline
│   └── api.py                  # FastAPI serving layer
├── models/
│   ├── loan_risk_model.pkl    # trained sklearn pipeline
│   └── decision_threshold.json
├── reports/
│   ├── metrics.json
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── cost_vs_threshold.png
│   └── feature_importance.png
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt

# regenerate data (optional, already included)
python src/generate_data.py

# train the model, produce metrics + plots
python src/train.py

# run tests
pytest -v

# serve the API
uvicorn src.api:app --reload --port 8000
# then open http://localhost:8000/docs
```

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "annual_income": 60000, "employment_length_years": 5,
    "credit_score": 700, "loan_amount": 15000, "existing_debt": 8000,
    "debt_to_income": 0.25, "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation", "num_delinquencies_2yr": 0
  }'
```

## Known limitations / what i'll do with it 

- **Fairness auditing**: haven't checked whether the model produces disparate
  outcomes across protected-adjacent variables (e.g. zip code as a proxy for
  race). This matters a lot in real lending — the Equal Credit Opportunity
  Act constrains what factors can be used. A production version needs a
  disparate impact analysis before deployment.
- **Real data**: synthetic data proves the pipeline works but not that it
  generalizes to real applicant behavior, which is messier and drifts over
  time (e.g. macroeconomic shifts change baseline default rates).
- **Model monitoring**: no drift detection or retraining trigger. In
  production I'd track prediction distribution and default rate over time
  and alert on drift.
- **Calibration**: haven't checked whether predicted probabilities are
  well-calibrated (does "30% probability" actually mean ~30% of such loans
  default?). Would add a calibration plot and consider `CalibratedClassifierCV`.
- **Cost assumptions**: the $15k/$1.5k cost figures are illustrative
  placeholders. In practice they'd come from actual loss data and vary by
  loan size/type, and the threshold would be recalculated per-segment rather
  than globally.
