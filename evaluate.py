"""
evaluate.py
-----------
Full evaluation suite:
  - Confusion matrix
  - Classification report (Precision / Recall / F1)
  - ROC-AUC & PR-AUC
  - Threshold tuning with business-impact analysis
  - SHAP explainability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)
import shap


# ──────────────────────────────────────────────
# 1. CORE METRICS
# ──────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_plots: bool = True,
    plot_dir: str = "notebooks",
) -> dict:
    """
    Print and return a dict of all key metrics.
    Also saves confusion matrix + ROC + PR plots.
    """
    print(f"\n{'='*55}")
    print(f"  📊  {model_name}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"],
                                 digits=4))

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    f1      = f1_score(y_true, y_pred)

    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  F1      : {f1:.4f}")

    if save_plots:
        _plot_confusion_matrix(model_name, y_true, y_pred, plot_dir)
        _plot_roc_pr(model_name, y_true, y_prob, plot_dir)

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
    }


def _plot_confusion_matrix(name, y_true, y_pred, plot_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{plot_dir}/cm_{safe_name}.png", dpi=150)
    plt.show()


def _plot_roc_pr(name, y_true, y_prob, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color="#2196F3",
                 label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title=f"ROC Curve — {name}")
    axes[0].legend(loc="lower right")

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    axes[1].plot(recall, precision, lw=2, color="#F44336",
                 label=f"AP = {pr_auc:.4f}")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title=f"Precision-Recall Curve — {name}")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{plot_dir}/roc_pr_{safe_name}.png", dpi=150)
    plt.show()


# ──────────────────────────────────────────────
# 2. THRESHOLD TUNING  (business impact)
# ──────────────────────────────────────────────

def tune_threshold(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    avg_fraud_amount: float = 122.0,   # $ average fraudulent transaction
    cost_fp: float = 5.0,              # $ cost to investigate a false positive
    plot_dir: str = "notebooks",
) -> float:
    """
    Sweep classification thresholds and pick the one with best F1.
    Also plots business-impact (cost) vs threshold.
    Returns optimal threshold.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, costs = [], []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, preds, zero_division=0))

        TP = int(((preds == 1) & (y_true == 1)).sum())
        FP = int(((preds == 1) & (y_true == 0)).sum())
        FN = int(((preds == 0) & (y_true == 1)).sum())

        # Cost = missed fraud + investigation cost of false alarms
        cost = FN * avg_fraud_amount + FP * cost_fp
        costs.append(cost)

    best_idx  = int(np.argmax(f1s))
    best_t    = float(thresholds[best_idx])
    best_cost_idx = int(np.argmin(costs))
    best_cost_t   = float(thresholds[best_cost_idx])

    print(f"\n── Threshold Tuning ({model_name}) ──")
    print(f"  Best F1 threshold   : {best_t:.3f}  (F1 = {f1s[best_idx]:.4f})")
    print(f"  Min-cost threshold  : {best_cost_t:.3f}  (cost = ${costs[best_cost_idx]:,.0f})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(thresholds, f1s, color="#2196F3", lw=2)
    axes[0].axvline(best_t, color="red", linestyle="--",
                    label=f"Best = {best_t:.3f}")
    axes[0].set(xlabel="Threshold", ylabel="F1-Score",
                title=f"F1 vs Threshold — {model_name}")
    axes[0].legend()

    axes[1].plot(thresholds, costs, color="#FF9800", lw=2)
    axes[1].axvline(best_cost_t, color="red", linestyle="--",
                    label=f"Min-cost = {best_cost_t:.3f}")
    axes[1].set(xlabel="Threshold", ylabel="Business Cost ($)",
                title=f"Cost vs Threshold — {model_name}")
    axes[1].legend()

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(f"{plot_dir}/threshold_{safe_name}.png", dpi=150)
    plt.show()

    return best_t


# ──────────────────────────────────────────────
# 3. SHAP EXPLAINABILITY
# ──────────────────────────────────────────────

def explain_with_shap(
    model,
    X_sample: np.ndarray,
    feature_names: list,
    model_name: str = "Model",
    n_samples: int = 200,
    plot_dir: str = "notebooks",
) -> None:
    """
    Generate SHAP summary + waterfall plot for a tree-based or linear model.
    """
    print(f"\n🔍 Generating SHAP explanations for {model_name} …")

    X_sub = X_sample[:n_samples]

    try:
        if model_name in ("Random Forest", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sub)
            # For binary RF, shap_values is a list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer = shap.LinearExplainer(model, X_sub)
            shap_values = explainer.shap_values(X_sub)

        # Summary bar plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X_sub,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(f"{plot_dir}/shap_bar_{safe_name}.png", dpi=150,
                    bbox_inches="tight")
        plt.show()

        # Beeswarm / dot plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X_sub,
            feature_names=feature_names,
            show=False,
        )
        plt.title(f"SHAP Beeswarm — {model_name}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/shap_beeswarm_{safe_name}.png", dpi=150,
                    bbox_inches="tight")
        plt.show()
        print(f"  ✅ SHAP plots saved to {plot_dir}/")

    except Exception as e:
        print(f"  ⚠️  SHAP explanation skipped: {e}")


# ──────────────────────────────────────────────
# 4. COMPARISON TABLE
# ──────────────────────────────────────────────

def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Print a ranked comparison table from a list of evaluate_model dicts.
    """
    df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    df.index = range(1, len(df) + 1)
    print("\n── Model Comparison (sorted by ROC-AUC) ──")
    print(df.to_string())

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["roc_auc"], width, label="ROC-AUC", color="#2196F3")
    ax.bar(x,         df["pr_auc"],  width, label="PR-AUC",  color="#4CAF50")
    ax.bar(x + width, df["f1"],      width, label="F1",      color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Model Comparison")
    plt.tight_layout()
    plt.savefig("notebooks/model_comparison.png", dpi=150)
    plt.show()
    print("📊 Comparison chart saved → notebooks/model_comparison.png")

    return df
