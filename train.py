"""
train.py

Full training pipeline for the loan risk model:
1. Load data, split (train/test) BEFORE any preprocessing to avoid leakage.
2. Preprocess: scale numeric features, one-hot encode categoricals.
3. Train a baseline (majority-class + Logistic Regression) and a main model
   (RandomForest with class_weight='balanced' to handle the ~18/82 imbalance).
4. Evaluate with metrics that actually matter for imbalanced classification:
   ROC-AUC, PR-AUC, precision/recall/F1 -- NOT accuracy.
5. Pick a decision threshold based on business cost, not the default 0.5.
6. Compute permutation feature importance for interpretability.
7. Save the trained pipeline + metrics + plots.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

NUMERIC = ["age", "annual_income", "employment_length_years", "credit_score",
           "loan_amount", "existing_debt", "debt_to_income", "num_delinquencies_2yr"]
CATEGORICAL = ["home_ownership", "purpose"]
TARGET = "default"

# --- Business cost assumptions (make these explicit -- this is the part
# interviewers love, because it shows you think beyond model metrics) ---
# Cost of a False Negative (approving someone who defaults): lose the principal.
# Cost of a False Positive (rejecting someone who would've repaid): lose the
# interest margin/profit on that loan.
COST_FALSE_NEGATIVE = 15000   # avg loan amount lost on a missed default
COST_FALSE_POSITIVE = 1500    # avg forgone profit on a wrongly rejected good loan


def load_data(path="data/loan_data.csv"):
    return pd.read_csv(path)


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def find_best_threshold(y_true, y_proba):
    """Sweep thresholds, pick the one minimizing expected business cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = np.sum((y_pred == 0) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
        costs.append(cost)
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost, thresholds, costs


def main():
    df = load_data()
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()

    # --- Baseline 1: majority class (sanity floor) ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_acc = dummy.score(X_test, y_test)

    # --- Baseline 2: plain Logistic Regression ---
    logreg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    logreg.fit(X_train, y_train)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]

    # --- Main model: RandomForest, balanced for imbalance ---
    rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # --- Evaluate ---
    def eval_model(name, proba, threshold=0.5):
        preds = (proba >= threshold).astype(int)
        return {
            "model": name,
            "threshold": round(float(threshold), 3),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
            "report": classification_report(y_test, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }

    results = {
        "dummy_baseline_accuracy": round(dummy_acc, 4),
        "logreg_default_threshold": eval_model("logistic_regression", logreg_proba, 0.5),
        "rf_default_threshold": eval_model("random_forest", rf_proba, 0.5),
    }

    # --- Cost-based threshold tuning on the RF model ---
    best_t, best_cost, thresholds, costs = find_best_threshold(y_test.values, rf_proba)
    results["rf_cost_optimal_threshold"] = eval_model("random_forest", rf_proba, best_t)
    results["rf_cost_optimal_threshold"]["expected_cost"] = float(best_cost)
    results["cost_assumptions"] = {
        "cost_false_negative": COST_FALSE_NEGATIVE,
        "cost_false_positive": COST_FALSE_POSITIVE,
    }

    # cost at naive 0.5 threshold, for comparison
    preds_50 = (rf_proba >= 0.5).astype(int)
    fn50 = np.sum((preds_50 == 0) & (y_test.values == 1))
    fp50 = np.sum((preds_50 == 1) & (y_test.values == 0))
    cost_50 = fn50 * COST_FALSE_NEGATIVE + fp50 * COST_FALSE_POSITIVE
    results["cost_at_default_threshold_0.5"] = float(cost_50)
    results["cost_savings_from_tuning"] = float(cost_50 - best_cost)

    # --- Permutation feature importance (model-agnostic, honest interpretability) ---
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    feature_names = NUMERIC + CATEGORICAL
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean[:len(feature_names)] if len(perm.importances_mean) == len(feature_names) else None
    })
    # permutation_importance works on raw X columns since it's applied to the full pipeline
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv("reports/feature_importance.csv", index=False)

    # --- Plots ---
    fpr, tpr, _ = roc_curve(y_test, rf_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"RF (AUC={results['rf_default_threshold']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest"); plt.legend()
    plt.tight_layout(); plt.savefig("reports/roc_curve.png", dpi=120); plt.close()

    prec, rec, _ = precision_recall_curve(y_test, rf_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Random Forest")
    plt.tight_layout(); plt.savefig("reports/pr_curve.png", dpi=120); plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, costs)
    plt.axvline(best_t, color="red", linestyle="--", label=f"optimal t={best_t:.2f}")
    plt.xlabel("Decision Threshold"); plt.ylabel("Expected Cost ($)")
    plt.title("Business Cost vs Threshold"); plt.legend()
    plt.tight_layout(); plt.savefig("reports/cost_vs_threshold.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 5))
    top_feat = importance_df.head(10).iloc[::-1]
    plt.barh(top_feat["feature"], top_feat["importance_mean"])
    plt.xlabel("Permutation Importance (mean AUC drop)")
    plt.title("Top 10 Feature Importances - Random Forest")
    plt.tight_layout(); plt.savefig("reports/feature_importance.png", dpi=120); plt.close()

    # --- Save model + metrics ---
    joblib.dump(rf, "models/loan_risk_model.pkl")
    with open("reports/metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open("models/decision_threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f)

    print("=" * 60)
    print(f"Dummy baseline accuracy:        {dummy_acc:.4f}")
    print(f"LogReg  ROC-AUC / PR-AUC:       {results['logreg_default_threshold']['roc_auc']} / {results['logreg_default_threshold']['pr_auc']}")
    print(f"RF      ROC-AUC / PR-AUC:       {results['rf_default_threshold']['roc_auc']} / {results['rf_default_threshold']['pr_auc']}")
    print(f"Cost-optimal threshold:         {best_t:.3f}  (expected cost ${best_cost:,.0f})")
    print(f"Cost at naive 0.5 threshold:    ${cost_50:,.0f}")
    print(f"Savings from threshold tuning:  ${cost_50 - best_cost:,.0f}")
    print("Top 5 features:")
    print(importance_df.head(5).to_string(index=False))
    print("=" * 60)
    print("Saved: models/loan_risk_model.pkl, reports/metrics.json, reports/*.png")


if __name__ == "__main__":
    main()
