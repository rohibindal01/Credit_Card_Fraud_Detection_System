"""
main.py  ─  End-to-end training pipeline
─────────────────────────────────────────
Run:  python main.py
"""

import os
import sys
import numpy as np

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_data, explore_data, preprocess
from train       import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_autoencoder,
    ae_predict,
    ae_predict_proba,
)
from evaluate import evaluate_model, tune_threshold, explain_with_shap, compare_models

os.makedirs("notebooks", exist_ok=True)
os.makedirs("models",    exist_ok=True)


def main():
    # ── 1. Load & Explore ───────────────────────────────────────────────
    df = load_data("data/creditcard.csv")
    explore_data(df)

    # ── 2. Preprocess (SMOTE resampling) ────────────────────────────────
    X_train, X_test, y_train, y_test = preprocess(
        df, sampling_strategy="smote"
    )

    # Feature names (same columns as the dataset minus Class)
    feature_names = (
        [f"V{i}" for i in range(1, 29)] + ["NormAmount", "NormTime"]
    )

    # ── 3. Train models ─────────────────────────────────────────────────
    lr  = train_logistic_regression(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train, X_test, y_test)
    ae, threshold, _ = train_autoencoder(X_train, y_train)

    # ── 4. Evaluate ─────────────────────────────────────────────────────
    results = []

    for name, model in [
        ("Logistic Regression", lr),
        ("Random Forest",       rf),
        ("XGBoost",             xgb),
    ]:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        res = evaluate_model(name, y_test, y_pred, y_prob)
        results.append(res)

    # Autoencoder evaluation
    ae_prob = ae_predict_proba(ae, threshold, X_test)
    ae_pred = ae_predict(ae, threshold, X_test)
    res = evaluate_model("Autoencoder", y_test, ae_pred, ae_prob)
    results.append(res)

    # ── 5. Threshold tuning (XGBoost) ───────────────────────────────────
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    best_threshold = tune_threshold("XGBoost", y_test, xgb_prob)
    np.save("models/xgb_best_threshold.npy", best_threshold)

    # ── 6. SHAP Explainability ───────────────────────────────────────────
    explain_with_shap(rf,  X_test, feature_names, "Random Forest")
    explain_with_shap(xgb, X_test, feature_names, "XGBoost")

    # ── 7. Final comparison table ────────────────────────────────────────
    compare_models(results)

    print("\n🎉 Pipeline complete! Check notebooks/ for all plots.")


if __name__ == "__main__":
    main()
