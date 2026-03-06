"""
Dollar Hegemony - FastAPI Backend
Serves model predictions, DSI data, and performance metrics.

Endpoints:
  GET  /                          - Health check
  GET  /api/dsi                   - Dollar Stress Index time series
  GET  /api/predict/{country}     - XGBoost latest prediction for a BRICS country
  GET  /api/performance           - All model performance metrics
  GET  /api/lstm-predictions      - LSTM predictions for all BRICS
  GET  /api/ensemble-performance  - Meta ensemble performance
  GET  /api/countries             - Available BRICS countries list
  POST /api/predict/custom        - Predict with custom macro inputs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="Dollar Hegemony API",
    description="ML-powered API for Dollar Hegemony impact on BRICS economies",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "outputs", "results")

BRICS = ["Brazil", "Russia", "India", "China", "South_Africa"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features_df():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("features.csv not found")
    df = pd.read_csv(FEATURES_PATH)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")
    return df[df.index.notna()].sort_index()


def load_xgb_model(country: str):
    path = os.path.join(MODELS_DIR, f"xgboost_{country.lower()}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Schema for custom prediction
# ---------------------------------------------------------------------------

class MacroInput(BaseModel):
    country: str
    DXY: float
    Oil_WTI: float
    Gold: float
    US_Fed_Rate: float
    US_10Y_Yield: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "project": "Dollar Hegemony",
        "description": "ML API for BRICS currency stress forecasting",
        "docs": "/docs",
    }


@app.get("/api/countries")
def get_countries():
    available = []
    for country in BRICS:
        model_path = os.path.join(MODELS_DIR, f"xgboost_{country.lower()}.pkl")
        available.append({
            "name": country,
            "model_available": os.path.exists(model_path),
        })
    return {"countries": available}


@app.get("/api/dsi")
def get_dsi():
    """Return DSI time series — the Dollar Stress Index."""
    try:
        df = load_features_df()
        if "DSI" not in df.columns:
            raise HTTPException(status_code=404, detail="DSI column not found in features.csv")

        dsi = df[["DSI"]].dropna().reset_index()
        dsi["Date"] = dsi["Date"].dt.strftime("%Y-%m-%d")
        return {
            "data": dsi.to_dict(orient="records"),
            "count": len(dsi),
            "min": round(float(dsi["DSI"].min()), 2),
            "max": round(float(dsi["DSI"].max()), 2),
            "latest": {
                "date": dsi.iloc[-1]["Date"],
                "value": round(float(dsi.iloc[-1]["DSI"]), 2),
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/predict/{country}")
def predict_country(country: str):
    """
    Latest XGBoost prediction for a BRICS country.
    Uses the most recent available macro data row.
    """
    if country not in BRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid country. Choose from: {BRICS}"
        )

    model = load_xgb_model(country)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found for {country}. Run model_xgboost.py first."
        )

    try:
        df = load_features_df()
        feature_cols = model.get_booster().feature_names
        feature_cols = [c for c in feature_cols if c in df.columns]

        latest = df[feature_cols].dropna().iloc[[-1]]
        prediction = float(model.predict(latest)[0])
        latest_date = latest.index[0].strftime("%Y-%m-%d")

        return {
            "country": country,
            "date": latest_date,
            "predicted_depreciation_3m": round(prediction, 4),
            "unit": "% change (12-month depreciation, 3-month ahead forecast)",
            "features_used": feature_cols,
            "feature_values": {col: round(float(latest[col].iloc[0]), 4) for col in feature_cols},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/performance")
def get_performance():
    """XGBoost model performance for all BRICS countries."""
    path = os.path.join(RESULTS_DIR, "model_performance.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="model_performance.csv not found")

    df = pd.read_csv(path)
    df = df.replace({np.nan: None})
    return {
        "model": "XGBoost",
        "metric_description": "MAE=Mean Absolute Error, RMSE=Root Mean Squared Error, DA=Directional Accuracy %",
        "results": df.to_dict(orient="records"),
    }


@app.get("/api/lstm-predictions")
def get_lstm_predictions(country: str = None):
    """LSTM predictions — optionally filter by country."""
    path = os.path.join(RESULTS_DIR, "lstm_predictions.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="lstm_predictions.csv not found")

    df = pd.read_csv(path)
    if country:
        if country not in BRICS:
            raise HTTPException(status_code=400, detail=f"Invalid country. Choose from: {BRICS}")
        df = df[df["Country"] == country]

    df = df.replace({np.nan: None})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    return {
        "count": len(df),
        "country_filter": country,
        "data": df.to_dict(orient="records"),
    }


@app.get("/api/ensemble-performance")
def get_ensemble_performance():
    """Meta-ensemble (XGBoost + LSTM + Transformer) performance metrics."""
    path = os.path.join(RESULTS_DIR, "ensemble_performance.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ensemble_performance.csv not found")

    df = pd.read_csv(path)
    df = df.replace({np.nan: None})
    return {
        "model": "Meta Ensemble (Ridge blending XGBoost + LSTM + Transformer)",
        "results": df.to_dict(orient="records"),
    }


@app.post("/api/predict/custom")
def predict_custom(inputs: MacroInput):
    """
    Predict currency depreciation with custom macro inputs.
    Useful for 'what-if' scenarios — e.g., what if DXY rises to 115?
    """
    if inputs.country not in BRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid country. Choose from: {BRICS}"
        )

    model = load_xgb_model(inputs.country)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found for {inputs.country}"
        )

    try:
        feature_cols = model.get_booster().feature_names
        input_map = {
            "DXY": inputs.DXY,
            "Oil_WTI": inputs.Oil_WTI,
            "Gold": inputs.Gold,
            "US_Fed_Rate": inputs.US_Fed_Rate,
            "US_10Y_Yield": inputs.US_10Y_Yield,
        }

        row = {col: input_map.get(col, 0.0) for col in feature_cols}
        X = pd.DataFrame([row])
        prediction = float(model.predict(X)[0])

        return {
            "country": inputs.country,
            "predicted_depreciation_3m": round(prediction, 4),
            "unit": "% change (12-month depreciation, 3-month ahead forecast)",
            "scenario_inputs": input_map,
            "note": "What-if scenario — not based on real-time data",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
