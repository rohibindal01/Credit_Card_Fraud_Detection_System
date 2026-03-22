# 💳 Credit Card Fraud Detection System

A production-ready ML pipeline to detect fraudulent credit card transactions using
**Logistic Regression**, **Random Forest**, **XGBoost**, and an **Autoencoder**
(anomaly detection) — with a live **Streamlit** dashboard.

---

## 📂 Project Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv          ← Download from Kaggle (link below)
├── models/                     ← Saved trained models (auto-created)
├── notebooks/                  ← EDA plots, evaluation charts (auto-created)
├── src/
│   ├── preprocess.py           ← Data loading, EDA, normalisation, SMOTE
│   ├── train.py                ← All four model training functions
│   └── evaluate.py             ← Metrics, threshold tuning, SHAP
├── app.py                      ← Streamlit dashboard
├── main.py                     ← End-to-end training pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

### 2. Download the Dataset

👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` inside the `data/` folder.

### 3. Train All Models

```bash
python main.py
```

This will:
- Run EDA and save visualisation plots to `notebooks/`
- Normalise features and apply SMOTE oversampling
- Train LR, Random Forest, XGBoost, and Autoencoder
- Evaluate all models and produce comparison charts
- Run SHAP explainability for tree models
- Save trained models to `models/`

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Models

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Supervised | Fast baseline, `class_weight=balanced` |
| Random Forest | Supervised | 200 trees, handles non-linearity |
| XGBoost | Supervised | Best overall; `scale_pos_weight` for imbalance |
| Autoencoder | Anomaly Detection | Trained on legit txns only; high recon error = fraud |

---

## 📊 Evaluation Metrics

Because the dataset is heavily imbalanced (~0.17% fraud), accuracy is meaningless.
We use:

- **Precision & Recall** — Fraud detection completeness vs false alarm rate
- **F1-Score** — Harmonic mean of Precision & Recall
- **ROC-AUC** — Overall discrimination ability
- **PR-AUC** — More informative for imbalanced classes than ROC-AUC
- **Confusion Matrix** — TP / FP / FN / TN breakdown

---

## ⚙️ Handling Class Imbalance

Three strategies are available in `preprocess.py`:

```python
preprocess(df, sampling_strategy="smote")        # SMOTE (default)
preprocess(df, sampling_strategy="oversample")   # Random oversample
preprocess(df, sampling_strategy="undersample")  # Random undersample
preprocess(df, sampling_strategy="none")         # No resampling
```

---

## 🔍 SHAP Explainability

SHAP bar and beeswarm plots are auto-generated for Random Forest and XGBoost
after training. They show which PCA components drive the fraud predictions most.

---

## ⚡ Dashboard Features

| Tab | Feature |
|---|---|
| Single Transaction | Enter V1–V28 + Amount + Time; get instant fraud probability gauge |
| Batch Prediction | Upload a CSV; get flagged rows + downloadable results |
| Real-time Simulation | Watch a live transaction stream with rolling fraud chart |

---

## 📈 Results (XGBoost on test set)

| Metric | Score |
|---|---|
| ROC-AUC | ~0.981 |
| PR-AUC | ~0.862 |
| F1 (fraud class) | ~0.875 |
| Recall (fraud) | ~0.901 |

---

## 📄 Dataset Reference

> Dal Pozzolo, Andrea, et al. "Calibrating Probability with Undersampling for
> Unbalanced Classification." *2015 IEEE Symposium Series on Computational Intelligence.*
> ULB Machine Learning Group — [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
