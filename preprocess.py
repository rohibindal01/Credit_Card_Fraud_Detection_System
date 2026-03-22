"""
preprocess.py
-------------
Handles all data loading, exploration, and preprocessing steps:
  - Normalization of Amount & Time
  - Train/test split
  - Class-imbalance handling (SMOTE, undersampling, oversampling)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the Kaggle credit-card fraud dataset."""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────

def explore_data(df: pd.DataFrame) -> None:
    """Print basic stats and show key visualisations."""
    print("\n── Class Distribution ──")
    counts = df["Class"].value_counts()
    print(counts)
    fraud_pct = counts[1] / counts.sum() * 100
    print(f"Fraud rate: {fraud_pct:.4f}%\n")

    # --- Figure 1: Class imbalance bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(["Legit (0)", "Fraud (1)"], counts.values,
                color=["#4CAF50", "#F44336"], edgecolor="black")
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    # --- Figure 2: Transaction amount distribution ---
    axes[1].hist(df[df["Class"] == 0]["Amount"], bins=60,
                 alpha=0.6, color="#2196F3", label="Legit")
    axes[1].hist(df[df["Class"] == 1]["Amount"], bins=60,
                 alpha=0.8, color="#F44336", label="Fraud")
    axes[1].set_title("Transaction Amount Distribution")
    axes[1].set_xlabel("Amount ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].set_xlim(0, 2500)

    # --- Figure 3: Time of transaction ---
    axes[2].hist(df[df["Class"] == 0]["Time"] / 3600, bins=48,
                 alpha=0.6, color="#2196F3", label="Legit")
    axes[2].hist(df[df["Class"] == 1]["Time"] / 3600, bins=48,
                 alpha=0.8, color="#F44336", label="Fraud")
    axes[2].set_title("Transaction Time Distribution (hours)")
    axes[2].set_xlabel("Hours")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("notebooks/eda_plots.png", dpi=150)
    plt.show()
    print("📊 EDA plots saved → notebooks/eda_plots.png")

    # --- Correlation heatmap (fraud rows only) ---
    plt.figure(figsize=(14, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                linewidths=0.3, annot=False)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("notebooks/correlation_heatmap.png", dpi=150)
    plt.show()
    print("📊 Correlation heatmap saved → notebooks/correlation_heatmap.png")


# ──────────────────────────────────────────────
# 3. PREPROCESSING
# ──────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    sampling_strategy: str = "smote",   # "smote" | "oversample" | "undersample" | "none"
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Normalise Amount & Time, split data, and handle class imbalance.

    Returns
    -------
    X_train, X_test, y_train, y_test  (numpy arrays)
    """
    # --- Scale Amount and Time ---
    scaler = StandardScaler()
    df = df.copy()
    df["NormAmount"] = scaler.fit_transform(df[["Amount"]])
    df["NormTime"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    # --- Train / test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\n── Split Info ──")
    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"  Train fraud: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")

    # --- Resample training set only ---
    if sampling_strategy == "smote":
        sampler = SMOTE(random_state=random_state)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        print(f"  After SMOTE → Train: {X_train.shape[0]:,} (fraud: {y_train.sum():,})")

    elif sampling_strategy == "oversample":
        sampler = RandomOverSampler(random_state=random_state)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        print(f"  After Oversampling → Train: {X_train.shape[0]:,} (fraud: {y_train.sum():,})")

    elif sampling_strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        print(f"  After Undersampling → Train: {X_train.shape[0]:,} (fraud: {y_train.sum():,})")

    else:
        print("  No resampling applied.")

    return X_train, X_test, y_train, y_test
