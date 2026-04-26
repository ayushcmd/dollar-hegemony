# Dollar Hegemony  
## Quantifying USD Dominance Across BRICS vs G7 with DSI, XGBoost, and LSTM

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Plotly Dash](https://img.shields.io/badge/Frontend-Plotly%20Dash-3F4F75)](https://plotly.com/dash/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Models](https://img.shields.io/badge/Models-XGBoost%20%7C%20LSTM-orange)](#modeling-framework)
[![Hugging Face Space](https://img.shields.io/badge/Live%20Demo-HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Ayush0826/dollar-hegemony)

An interactive macro-financial ML research system that tracks USD strength, estimates stress transmission across BRICS/G7 economies, and supports scenario-based currency depreciation analysis.

---

## Live Demo

- **Hugging Face Space:** https://huggingface.co/spaces/Ayush0826/dollar-hegemony

---

## Research Objective

This project quantifies how USD regime shifts (strength, momentum, liquidity pressure) propagate into country-level currency stress, especially for BRICS economies relative to G7 context.

---

## Key Contributions

- **Dollar Stress Index (DSI):** composite USD pressure signal
- **Country-level depreciation analytics:** 12-month impact vs USD
- **Hybrid model layer:** XGBoost + LSTM
- **Scenario simulator:** user-controlled DXY level + momentum
- **Crisis diagnostics:** country alert states and stress timelines
- **Macro context integration:** fundamentals + live BRICS-focused news

---

## Dashboard Modules

### 1) Macro
- Timeline coverage: **2000–2024**
- DXY historical chart with crisis markers (e.g., GFC, COVID)
- DSI gauge and stress-zone view
- BRICS vs G7 depreciation comparison
- Correlation/stress exploration panels

### 2) Models
- Country selector: Brazil, Russia, India, China, South Africa
- LSTM actual vs predicted panel (3-month ahead)
- Metrics table: MAE, RMSE, Directional Accuracy
- Directional accuracy comparison chart by country

### 3) Scenario Sim
- DXY level slider
- DXY 12-month momentum slider
- Quick presets: Strong Dollar / Current / Weak Dollar
- Projected 12M depreciation cards (country-wise)
- Sensitivity analysis and scenario bar chart

### 4) World Map
- Global choropleth of currency stress/depreciation
- Color mode toggle (Latest Depreciation / Alert Level)
- Period selector + country click interaction

### 5) Crisis Alerts
- Regime counters: Critical / Warning / Watch / Stable
- Country status board with depreciation + crisis-period behavior
- DSI timeline for crisis windows
- Worst depreciation events ranking
- Country crisis-history selector

### 6) Fundamentals
- Indicator selector (example: GDP Growth)
- Multi-country BRICS trend view
- Country summary cards
- BRICS stock index panel
- Cross-indicator correlation scatter analysis

### 7) News Feed
- Live BRICS/emerging-market/USD-relevant headlines
- Refresh workflow for updates
- Macro snapshot: DXY, DSI, WTI, Fed Rate, US CPI YoY

---

## Header KPIs

The app top bar displays:
- **DXY Index**
- **DSI Score**
- **Coverage (2000–2024)**
- **Alerts count**

---

## Dollar Stress Index (DSI)

### Conceptual Form

The DSI is a normalized composite of USD-linked macro-financial stress factors.

**DSI(t) = Σ (w_i × z_i(t)), for i = 1 to n**

Where:
- `z_i(t)` = standardized value of factor `i` at time `t`
- `w_i` = weight of factor `i` (fixed or calibrated)
- DSI is scaled into regime bands (low / moderate / high stress)

### Typical Factor Families
- USD broad strength proxies (e.g., DXY-linked signals)
- Global rates/liquidity environment
- Inflation and commodity stress proxies
- Risk-off behavior and FX volatility clusters
- Country-specific depreciation response terms

---

## Modeling Framework

### XGBoost
- Handles nonlinear interactions in macro + market tabular signals
- Strong baseline for country-level depreciation estimation

### LSTM
- Learns temporal dependencies in FX stress transmission
- Used for short-horizon directional and level prediction workflows

---

## Data Sources

As shown in the app and footer:
- Yahoo Finance
- FRED
- World Bank
- Live financial news feeds (News tab)

---

## Tech Stack

- **Language:** Python
- **Dashboard/UI:** Plotly Dash
- **Backend/API:** FastAPI
- **ML:** XGBoost, LSTM
- **Containerization:** Dockerfile

---

## High-Level Architecture

```text
Data Sources (Yahoo/FRED/World Bank/News)
            │
            ▼
   Data Processing + Feature Engineering
            │
            ├── DSI Computation
            ├── XGBoost Pipeline
            └── LSTM Pipeline
            │
            ▼
     Inference + Metrics + Alerts
            │
            ▼
 Plotly Dash Interface + FastAPI Endpoints
```

---

## Local Setup

### 1) Clone
```bash
git clone https://github.com/ayushcmd/dollar-hegemony.git
cd dollar-hegemony
```

### 2) Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### 3) Run (based on your repo entrypoint)
```bash
python app.py
```

or

```bash
uvicorn app.main:app --reload
```

---

## Reproducibility Notes

- Use fixed time-split train/validation/test windows
- Version processed datasets and model artifacts
- Track metrics per country and horizon
- Record scenario assumptions when reporting outputs

---

## Limitations

- Regime shifts can reduce historical model reliability
- Output quality depends on data freshness and retraining
- Research/analytics tool only — **not financial advice**

---

## Roadmap

- [ ] Add uncertainty bands (quantile/interval forecasts)
- [ ] Add explainability layer (feature attribution diagnostics)
- [ ] Extend benchmark suite and ablation reporting
- [ ] Automate retraining + monitoring workflows

---

## Author

**Ayush Raj**  
BSc CSDA, IIT Patna (Grad: Aug 2027)

- GitHub: https://github.com/ayushcmd  
- LinkedIn: https://www.linkedin.com/in/ayush08iitp  
- Portfolio: https://ayushcmd.me

---

## Citation

If you use this project in research/content, cite:

**Ayush Raj — Dollar Hegemony (GitHub + Hugging Face Space)**  
- Repo: https://github.com/ayushcmd/dollar-hegemony  
- Demo: https://huggingface.co/spaces/Ayush0826/dollar-hegemony

---

## License

MIT (or as specified in `LICENSE`)