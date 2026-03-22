"""
train.py
--------
Trains and persists four models:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost
  4. Autoencoder (anomaly detection — trains on legit txns only)
"""

import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(model, name: str) -> None:
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  💾 Saved → {path}")


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)


# ──────────────────────────────────────────────
# 1. LOGISTIC REGRESSION
# ──────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, C: float = 1.0) -> LogisticRegression:
    print("\n🔵 Training Logistic Regression …")
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    save_model(model, "logistic_regression")
    return model


# ──────────────────────────────────────────────
# 2. RANDOM FOREST
# ──────────────────────────────────────────────

def train_random_forest(
    X_train, y_train,
    n_estimators: int = 200,
    max_depth: int = 12,
) -> RandomForestClassifier:
    print("\n🌲 Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    save_model(model, "random_forest")
    return model


# ──────────────────────────────────────────────
# 3. XGBOOST
# ──────────────────────────────────────────────

def train_xgboost(
    X_train, y_train,
    X_val=None, y_val=None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> XGBClassifier:
    print("\n⚡ Training XGBoost …")

    # Handle class imbalance via scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    eval_set = [(X_val, y_val)] if X_val is not None else None

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=20 if eval_set else None,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    save_model(model, "xgboost")
    return model


# ──────────────────────────────────────────────
# 4. AUTOENCODER (anomaly detection)
# ──────────────────────────────────────────────

def build_autoencoder(input_dim: int) -> keras.Model:
    """
    Encoder → Bottleneck → Decoder.
    Trained on LEGITIMATE transactions only.
    High reconstruction error → likely fraud.
    """
    inputs = keras.Input(shape=(input_dim,), name="input")

    # Encoder
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    encoded = layers.Dense(8, activation="relu", name="bottleneck")(x)

    # Decoder
    x = layers.Dense(16, activation="relu")(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    decoded = layers.Dense(input_dim, activation="linear", name="output")(x)

    autoencoder = keras.Model(inputs, decoded, name="Autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(
    X_train, y_train,
    epochs: int = 30,
    batch_size: int = 256,
) -> tuple:
    """
    Train autoencoder on legitimate transactions only.
    Returns (autoencoder_model, reconstruction_threshold).
    """
    print("\n🤖 Training Autoencoder (anomaly detection) …")

    # Train ONLY on normal transactions
    X_legit = X_train[y_train == 0]
    input_dim = X_legit.shape[1]

    ae = build_autoencoder(input_dim)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=0
        ),
    ]

    history = ae.fit(
        X_legit, X_legit,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # Determine threshold: mean + 2*std of reconstruction errors on legit txns
    X_pred = ae.predict(X_legit, verbose=0)
    mse = np.mean(np.power(X_legit - X_pred, 2), axis=1)
    threshold = float(np.mean(mse) + 2 * np.std(mse))
    print(f"  Reconstruction threshold set to: {threshold:.6f}")

    ae.save(os.path.join(MODELS_DIR, "autoencoder.keras"))
    np.save(os.path.join(MODELS_DIR, "ae_threshold.npy"), threshold)
    print(f"  💾 Saved → {MODELS_DIR}/autoencoder.keras")

    return ae, threshold, history


def ae_predict(ae_model, threshold: float, X: np.ndarray) -> np.ndarray:
    """Return binary predictions from autoencoder (1 = fraud)."""
    X_pred = ae_model.predict(X, verbose=0)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    return (mse > threshold).astype(int)


def ae_predict_proba(ae_model, threshold: float, X: np.ndarray) -> np.ndarray:
    """Return a normalised fraud probability from reconstruction error."""
    X_pred = ae_model.predict(X, verbose=0)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    # Sigmoid-like normalisation around threshold
    prob = 1 / (1 + np.exp(-(mse - threshold) / (threshold + 1e-9)))
    return prob


# ──────────────────────────────────────────────
# TRAIN ALL MODELS (convenience wrapper)
# ──────────────────────────────────────────────

def train_all(X_train, X_test, y_train, y_test):
    """Train all four models and return them in a dict."""
    models = {}

    models["Logistic Regression"] = train_logistic_regression(X_train, y_train)
    models["Random Forest"]       = train_random_forest(X_train, y_train)
    models["XGBoost"]             = train_xgboost(X_train, y_train, X_test, y_test)

    ae, threshold, _ = train_autoencoder(X_train, y_train)
    models["Autoencoder"] = (ae, threshold)

    print("\n✅ All models trained successfully!")
    return models
