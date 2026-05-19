# Fraud Detection Pipeline — UPI Transaction Anomaly Detection

**Author:** Swapnil Mogal | B.Tech Data Science | Manipal University Jaipur 2026
**Extends:** NPCI Internship CBDC Analytics Work

---

## Overview

End-to-end fraud detection system for digital payment transactions, built to simulate and extend the anomaly detection work done at NPCI. Features ML-based fraud scoring, real-time REST API, and an interactive Streamlit operations dashboard.

- **200K synthetic UPI/IMPS/NEFT/RTGS transactions** with realistic 0.8% fraud rate
- **XGBoost primary model** with F2-optimized threshold (recall-weighted for fraud)
- **SMOTE + class weighting** for extreme imbalance handling
- **Role-based Streamlit dashboard** with live transaction checker and SHAP explanations
- **Flask REST API** for real-time scoring at <50ms latency

---

## Installation

```bash
git clone <repo>
cd fraud-detection-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_api.txt  # for Flask API (local only)
```

---

## Pipeline Steps

Run each step in order (or let the app auto-run them on first launch):

```bash
python src/generate_data.py      # Generate 200K synthetic transactions
python src/eda.py                # Run EDA, generate plots
python src/preprocessing.py     # Feature engineering, SMOTE, splits
python src/train_models.py      # Train XGBoost, RF, Isolation Forest
```

---

## Running the App

**Terminal 1 — Flask API (optional for local):**
```bash
python api/run.py
```

**Terminal 2 — Streamlit App:**
```bash
streamlit run app/main.py
```

---

## API Documentation

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single transaction fraud score |
| `/batch_predict` | POST | Batch scoring (up to 100 transactions) |
| `/health` | GET | API health check |
| `/stats` | GET | Today's prediction statistics |
| `/recent` | GET | Last N predictions |

---

## Demo Credentials

| Username | Password | Role |
|---|---|---|
| swapnil | admin123 | admin |
| fraud_analyst | analyst123 | analyst |
| demo | demo123 | viewer |

---

## NPCI Connection

This project productionizes the anomaly detection and KPI computation patterns from NPCI CBDC reconciliation work. The fraud signals (velocity burst, new beneficiary, night hours) mirror discrepancy patterns identified in the CBDC reconciliation system.
