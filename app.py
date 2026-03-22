"""
app.py  ─  Streamlit Fraud Detection Dashboard
───────────────────────────────────────────────
Run:  streamlit run app.py
"""

import os
import time
import random

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💳 Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .fraud-alert   { background:#FFEBEE; border-left:5px solid #F44336;
                   padding:12px; border-radius:4px; color:#B71C1C; font-size:18px; }
  .safe-alert    { background:#E8F5E9; border-left:5px solid #4CAF50;
                   padding:12px; border-radius:4px; color:#1B5E20; font-size:18px; }
  .metric-card   { background:#F3F4F6; border-radius:8px; padding:14px;
                   text-align:center; }
  .stButton>button { background:#1A237E; color:white; font-weight:bold; }
  .stButton>button:hover { background:#283593; }
</style>
""", unsafe_allow_html=True)


# ── Load Models ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models …")
def load_models():
    models = {}
    model_dir = "models"
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest":       "random_forest.pkl",
        "XGBoost":             "xgboost.pkl",
    }
    for name, fname in model_files.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    # XGBoost threshold
    threshold_path = os.path.join(model_dir, "xgb_best_threshold.npy")
    xgb_threshold  = float(np.load(threshold_path)) if os.path.exists(threshold_path) else 0.5

    return models, xgb_threshold


# ── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.image(
        "https://img.icons8.com/color/96/000000/security-checked.png",
        width=80,
    )
    st.sidebar.title("⚙️ Settings")

    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        index=2,
    )

    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.01, max_value=0.99,
        value=0.50, step=0.01,
        help="Lower → catch more fraud (more false alarms). "
             "Higher → fewer false alarms (may miss some fraud).",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Batch Upload")
    uploaded = st.sidebar.file_uploader(
        "Upload CSV (transactions)", type="csv"
    )

    return selected_model, threshold, uploaded


# ── Feature Input Form ────────────────────────────────────────────────────────

def feature_input_form():
    """Return a dict of feature values entered by the user."""
    st.markdown("### 🔢 Enter Transaction Features")

    with st.expander("ℹ️ How to use", expanded=False):
        st.write(
            "Enter the PCA-transformed features (V1–V28) along with "
            "the normalised Amount and Time. "
            "You can paste real values from the dataset, or use the "
            "**Random Sample** button to auto-fill."
        )

    if st.button("🎲 Random Sample (demo)"):
        # Generate plausible random values
        for i in range(1, 29):
            st.session_state[f"V{i}"] = round(random.gauss(0, 1), 4)
        st.session_state["NormAmount"] = round(random.uniform(-1, 4), 4)
        st.session_state["NormTime"]   = round(random.uniform(-1, 1), 4)

    cols = st.columns(5)
    features = {}
    for idx in range(1, 29):
        col = cols[(idx - 1) % 5]
        features[f"V{idx}"] = col.number_input(
            f"V{idx}",
            value=st.session_state.get(f"V{idx}", 0.0),
            format="%.4f",
            key=f"V{idx}",
        )

    c1, c2 = st.columns(2)
    features["NormAmount"] = c1.number_input(
        "Normalised Amount",
        value=st.session_state.get("NormAmount", 0.0),
        format="%.4f",
        key="NormAmount",
    )
    features["NormTime"] = c2.number_input(
        "Normalised Time",
        value=st.session_state.get("NormTime", 0.0),
        format="%.4f",
        key="NormTime",
    )
    return features


# ── Predict single transaction ────────────────────────────────────────────────

def predict_single(features: dict, model, threshold: float):
    X = np.array(list(features.values())).reshape(1, -1)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= threshold)
    return pred, prob


def show_prediction_result(pred: int, prob: float):
    col1, col2 = st.columns([1, 2])

    with col1:
        if pred == 1:
            st.markdown(
                '<div class="fraud-alert">🚨 <b>FRAUD DETECTED</b></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="safe-alert">✅ <b>TRANSACTION LEGITIMATE</b></div>',
                unsafe_allow_html=True,
            )

    with col2:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 2),
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": "Fraud Probability"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#F44336" if pred == 1 else "#4CAF50"},
                "steps": [
                    {"range": [0,  40], "color": "#E8F5E9"},
                    {"range": [40, 70], "color": "#FFF9C4"},
                    {"range": [70, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 3},
                    "thickness": 0.9,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


# ── Batch prediction ──────────────────────────────────────────────────────────

def batch_predict(df_raw: pd.DataFrame, model, threshold: float):
    expected_cols = [f"V{i}" for i in range(1, 29)] + ["NormAmount", "NormTime"]
    missing = [c for c in expected_cols if c not in df_raw.columns]

    if missing:
        # Try raw Amount / Time and normalise on the fly
        from sklearn.preprocessing import StandardScaler
        df = df_raw.copy()
        if "Amount" in df.columns:
            df["NormAmount"] = StandardScaler().fit_transform(df[["Amount"]])
            df.drop(columns=["Amount"], errors="ignore", inplace=True)
        if "Time" in df.columns:
            df["NormTime"] = StandardScaler().fit_transform(df[["Time"]])
            df.drop(columns=["Time"], errors="ignore", inplace=True)
        df.drop(columns=["Class"], errors="ignore", inplace=True)
    else:
        df = df_raw[expected_cols].copy()

    X = df.values
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = df_raw.copy()
    result["FraudProbability"] = np.round(probs * 100, 2)
    result["Prediction"]       = np.where(preds == 1, "🚨 FRAUD", "✅ Legit")
    return result, probs, preds


def show_batch_results(result_df: pd.DataFrame, probs, preds):
    n_fraud = int(preds.sum())
    n_legit = len(preds) - n_fraud

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(preds):,}")
    c2.metric("🚨 Fraud Detected",   f"{n_fraud:,}")
    c3.metric("✅ Legitimate",        f"{n_legit:,}")
    c4.metric("Fraud Rate",          f"{n_fraud/len(preds)*100:.2f}%")

    # Probability distribution
    fig = px.histogram(
        probs * 100, nbins=60,
        labels={"value": "Fraud Probability (%)", "count": "Count"},
        title="Fraud Probability Distribution",
        color_discrete_sequence=["#1A237E"],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Flagged transactions
    st.markdown(f"#### 🚨 Flagged Transactions ({n_fraud})")
    fraud_rows = result_df[result_df["Prediction"] == "🚨 FRAUD"]
    if not fraud_rows.empty:
        st.dataframe(
            fraud_rows.sort_values("FraudProbability", ascending=False)
                      .head(50),
            use_container_width=True,
        )
    else:
        st.success("No fraudulent transactions detected in this batch.")

    # Download
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Results (CSV)",
        csv, "fraud_predictions.csv", "text/csv",
    )


# ── Real-time Simulation ──────────────────────────────────────────────────────

def realtime_simulation(model, threshold: float):
    st.markdown("### ⚡ Real-time Transaction Stream Simulation")

    placeholder = st.empty()
    chart_placeholder = st.empty()
    log_placeholder = st.empty()

    n_transactions = st.slider("Transactions to simulate", 10, 200, 50, 10)

    if st.button("▶️ Start Simulation"):
        history = []
        fraud_count = 0

        for i in range(n_transactions):
            # Generate synthetic transaction
            X = np.random.randn(1, 30)

            # Inject occasional fraud-like signal
            if random.random() < 0.05:
                X[0, 0:3] -= 3.0   # strong negative V1, V2, V3

            prob = float(model.predict_proba(X)[0, 1])
            pred = int(prob >= threshold)

            if pred == 1:
                fraud_count += 1

            history.append({"txn": i + 1, "prob": round(prob * 100, 2),
                             "fraud": pred})
            df_hist = pd.DataFrame(history)

            # Live metrics
            placeholder.markdown(
                f"**Transaction #{i+1}** │ "
                f"Fraud Prob: `{prob*100:.1f}%` │ "
                f"Status: {'🚨 FRAUD' if pred else '✅ OK'} │ "
                f"Total Flagged: `{fraud_count}`"
            )

            # Rolling chart
            fig = px.line(
                df_hist, x="txn", y="prob",
                title="Live Fraud Probability Feed",
                labels={"txn": "Transaction #", "prob": "Fraud Prob (%)"},
                color_discrete_sequence=["#1A237E"],
            )
            fig.add_hline(y=threshold * 100, line_dash="dash",
                          line_color="red", annotation_text="Threshold")
            fig.update_layout(height=350)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Log (last 10)
            log_placeholder.dataframe(
                df_hist.tail(10).iloc[::-1],
                use_container_width=True,
            )
            time.sleep(0.06)

        st.success(
            f"✅ Simulation complete! "
            f"Flagged **{fraud_count}** / {n_transactions} transactions as fraud "
            f"({fraud_count/n_transactions*100:.1f}%)."
        )


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.caption("Real-time fraud detection powered by ML — XGBoost · Random Forest · Logistic Regression")

    selected_model_name, threshold, uploaded = sidebar()

    # Load models
    try:
        models, xgb_threshold = load_models()
        if not models:
            st.warning("⚠️ No trained models found. Run `python main.py` first.")
            st.stop()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        st.info("Run `python main.py` to train and save models, then restart the app.")
        st.stop()

    model = models.get(selected_model_name)
    if model is None:
        st.error(f"Model '{selected_model_name}' not found.")
        st.stop()

    # Tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Single Transaction", "📋 Batch Prediction", "⚡ Real-time Simulation"]
    )

    with tab1:
        features = feature_input_form()
        if st.button("🔍 Analyse Transaction", use_container_width=True):
            with st.spinner("Analysing …"):
                pred, prob = predict_single(features, model, threshold)
            show_prediction_result(pred, prob)

    with tab2:
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.info(f"Loaded {df_raw.shape[0]:,} transactions. Running predictions …")
            with st.spinner("Processing batch …"):
                result_df, probs, preds = batch_predict(df_raw, model, threshold)
            show_batch_results(result_df, probs, preds)
        else:
            st.info("Upload a CSV file from the sidebar to run batch predictions.")

    with tab3:
        realtime_simulation(model, threshold)


if __name__ == "__main__":
    main()
