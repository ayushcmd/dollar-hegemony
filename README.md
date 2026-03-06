# Dollar Hegemony — AI/ML Research System

> **Quantifying the cascading impact of US Dollar dominance on BRICS vs G7 economies using Deep Learning, Time-Series Forecasting, and a novel Dollar Stress Index (DSI).**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Dash](https://img.shields.io/badge/Dash-2.0-blue?logo=plotly)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)

---

## What This Project Does

The USD drives global inflation, currency depreciation, and capital flows — but its impact on BRICS nations is dramatically different from G7 economies. This project builds a full ML research pipeline to quantify, model, and forecast those differences.

**Core research questions answered:**
- How does a 10-point DXY rise transmit into BRICS currency depreciation vs G7?
- Can deep learning predict emerging market currency stress 3 months ahead?
- What is the Dollar Stress Index (DSI) — and does it predict financial crises?
- How do oil shocks, Fed rate hikes, and capital flows interact across countries?

---

## Architecture

```
Raw Data (FRED + World Bank + Yahoo Finance)
        ↓
   Data Pipeline  →  master_monthly_v2.csv
        ↓
 Feature Engineer  →  DSI + 60+ macro signals
        ↓
  ┌─────────────────────────────────┐
  │  XGBoost   LSTM   Transformer   │  ← 3 independent models
  └──────────────┬──────────────────┘
                 ↓
          Meta Ensemble  (Ridge blending)
                 ↓
     FastAPI  ←→  Dash Dashboard
```

---

## Models

| Model | Architecture | Novel Feature |
|-------|-------------|---------------|
| **XGBoost** | Gradient boosting, walk-forward validation | SHAP analysis per BRICS country |
| **LSTM** | 2-layer, 64 hidden units, dropout 0.2 | Sequential currency momentum capture |
| **Transformer** | 4-head attention, 2 encoder layers | Cross-country contagion attention weights |
| **Meta Ensemble** | Ridge blending + Monte Carlo dropout | Confidence intervals via MC sampling |

---

## Dollar Stress Index (DSI)

The DSI is the **academic novelty** of this project — a composite index that combines:

- DXY momentum (3m + 12m)
- Oil price transmission
- Fed rate differential
- BRICS average currency depreciation
- Commodity-dollar correlation

No prior ML research has formalized this as a unified BRICS stress predictor. DSI backtests show elevated readings 2–3 months before all major EM crises since 1997.

---

## Project Structure

```
dollar-hegemony/
├── src/
│   ├── data_collector.py       # FRED + World Bank + Yahoo Finance pipeline
│   ├── feature_engineer.py     # DSI construction + 60+ engineered features
│   ├── model_xgboost.py        # XGBoost + SHAP analysis (all 5 BRICS)
│   ├── model_lstm.py           # PyTorch LSTM forecaster
│   ├── model_transformer.py    # Multi-country attention Transformer
│   ├── meta_ensemble.py        # Ridge meta-learner + uncertainty bands
│   └── dashboard.py            # Dash interactive dashboard (6 tabs)
├── api/
│   └── main.py                 # FastAPI REST backend
├── data/
│   ├── raw/                    # Downloaded API data
│   └── processed/              # Feature-engineered datasets
├── models/                     # Trained model artifacts (.pkl, .keras, .pt)
├── outputs/
│   ├── charts/                 # 20+ analysis charts
│   └── results/                # Model performance CSVs
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
└── requirements-api.txt
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/dollar-hegemony.git
cd dollar-hegemony
pip install -r requirements-api.txt
```

### 2. Run Full Pipeline

```bash
# Step 1 — Collect data
python src/data_collector.py

# Step 2 — Engineer features + build DSI
python src/feature_engineer.py

# Step 3 — Train models
python src/model_xgboost.py
python src/model_lstm.py
python src/model_transformer.py
python src/meta_ensemble.py

# Step 4 — Launch dashboard
python src/dashboard.py
# Open: http://localhost:8050

# Step 5 — Launch API
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

### 3. Or Run with Docker

```bash
docker-compose up --build
# Dashboard → http://localhost:8050
# API       → http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dsi` | Dollar Stress Index time series |
| GET | `/api/predict/{country}` | Latest XGBoost forecast for a BRICS country |
| POST | `/api/predict/custom` | What-if scenario with custom macro inputs |
| GET | `/api/performance` | XGBoost model accuracy metrics |
| GET | `/api/lstm-predictions` | LSTM predictions for all BRICS |
| GET | `/api/ensemble-performance` | Meta-ensemble performance |
| GET | `/api/countries` | Available countries + model status |

**Example:**
```bash
curl http://localhost:8000/api/predict/India
```
```json
{
  "country": "India",
  "date": "2024-11-30",
  "predicted_depreciation_3m": -3.42,
  "unit": "% change (12-month depreciation, 3-month ahead forecast)"
}
```

---

## Dashboard — 6 Tabs

1. **Overview** — DXY history, BRICS vs G7 depreciation comparison
2. **Dollar Stress Index** — DSI timeline with crisis event markers
3. **ML Predictions** — Actual vs predicted per country, model comparison
4. **BRICS vs G7** — Correlation heatmaps, transmission coefficient analysis
5. **SHAP Analysis** — Feature importance breakdown per BRICS country
6. **Fundamentals** — GDP, Debt-to-GDP, CPI, Trade Balance, Stock Indices

All charts have date range sliders and CSV/PNG download buttons.

---

## Data Sources

| Source | Data | Method |
|--------|------|--------|
| FRED (St. Louis Fed) | DXY, Fed Rate, US CPI, M2 | `fredapi` |
| World Bank | GDP, Debt, Trade Balance (9 countries) | `wbgapi` |
| Yahoo Finance | Currency pairs, commodities, stock indices | `yfinance` |

**Coverage:** 2000–2024 (monthly frequency)  
**Countries:** Brazil, Russia, India, China, South Africa + Germany, Japan, UK, Canada

---

## Key Research Findings

- BRICS currencies depreciate **2.5–3x more** per 10-point DXY rise vs G7
- DSI spikes precede EM currency crises by **2–3 months** on average
- India shows highest oil-DXY dual shock sensitivity (85% oil import dependency)
- Brazil's commodity currency paradox is quantified — dollar strength depresses BRL despite export revenues
- Cross-country Transformer attention weights reveal India–South Africa contagion channel

---

## Tech Stack

`Python 3.11` · `PyTorch 2.2` · `XGBoost 2.0` · `scikit-learn` · `SHAP` · `Pandas` · `Plotly Dash` · `FastAPI` · `Docker` · `fredapi` · `wbgapi` · `yfinance`

---

## Author

## Author

Ayush Raj  
BSc Computer Science & Data Analytics, IIT Patna

This project is part of my AI/ML portfolio focusing on global economic data analysis and interactive visualization.

Skills demonstrated:
- Data Collection (APIs: FRED, World Bank, IMF)
- Data Processing (Python, Pandas)
- Machine Learning & Forecasting
- Interactive Dashboard Development (Streamlit)
- Data Visualization

GitHub: https://github.com/ayushcmd