"""
Dollar Hegemony Project - Phase 4
LSTM Model - All BRICS Countries Currency Depreciation Forecasting

WHAT THIS FILE DOES:
  - Loads your features.csv (built in Phase 1)
  - Trains a separate LSTM for each BRICS country
  - Uses a 12-month lookback window to predict 3 months ahead
  - Saves each model + generates comparison charts
  - Appends LSTM predictions to a results CSV for dashboard use

HOW TO RUN:
  pip install tensorflow scikit-learn pandas numpy matplotlib joblib
  python src/model_lstm.py

WHAT IS AN LSTM?
  LSTM = Long Short-Term Memory. It's a type of neural network designed
  for sequences (time series). Unlike XGBoost which treats each row
  independently, LSTM "remembers" patterns across time — useful for
  catching momentum effects in currency markets.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Suppress TensorFlow info logs (keeps output clean)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Fix random seeds so results are reproducible
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIG — tweak these if you want to experiment
# =============================================================================

BRICS_COUNTRIES  = ["Brazil", "Russia", "India", "China", "South_Africa"]
FORECAST_HORIZON = 3       # predict 3 months ahead
LOOKBACK         = 12      # use last 12 months as input window
EPOCHS           = 100     # max training epochs (EarlyStopping will stop earlier)
BATCH_SIZE       = 16
TEST_YEARS       = 5       # last 5 years = held-out test set

FEATURE_COLS = [
    "DXY",
    "DXY_mom_12m",
    "Oil_WTI",
    "US_Fed_Rate",
    "US_10Y_Yield",
]

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_features():
    path = "data/processed/features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "features.csv not found at: " + path + "\n"
            "Please run feature_engineer.py first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df.index.notna()].sort_index()
    print("Loaded features: " + str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns")
    print("Date range: " + str(df.index.min().date()) + " to " + str(df.index.max().date()))
    return df


# =============================================================================
# 2. BUILD SEQUENCES
#    LSTM needs data shaped as [samples, timesteps, features]
#    e.g. for each prediction we feed the last 12 months of data
# =============================================================================

def build_sequences(X_arr, y_arr, lookback):
    """
    Convert flat arrays into overlapping windows.
    X_arr: shape (n_months, n_features)
    y_arr: shape (n_months,)
    Returns Xs shape (samples, lookback, features), ys shape (samples,)
    """
    Xs, ys = [], []
    for i in range(lookback, len(X_arr)):
        Xs.append(X_arr[i - lookback : i])   # last 12 months of features
        ys.append(y_arr[i])                   # the target value at month i
    return np.array(Xs), np.array(ys)


def prepare_country_data(df, country, lookback, horizon, test_years):
    target_col = country + "_depr_12m"

    if target_col not in df.columns:
        print("  WARNING: " + target_col + " not found, skipping " + country)
        return None

    # Use available feature columns + country-specific depreciation as input
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    all_cols = available_features + [target_col]

    sub = df[all_cols].copy()
    sub = sub.ffill().bfill()

    # Shift target forward by horizon (we're predicting h months ahead)
    sub["target"] = sub[target_col].shift(-horizon)
    sub = sub.dropna()

    if len(sub) < lookback + 20:
        print("  WARNING: Not enough data for " + country + " (" + str(len(sub)) + " rows)")
        return None

    X_raw = sub[available_features].values
    y_raw = sub["target"].values
    dates = sub.index

    # Train/test split by date
    cutoff  = dates.max() - pd.DateOffset(years=test_years)
    train_mask = dates <= cutoff
    test_mask  = dates >  cutoff

    # Scale features to [0, 1] — IMPORTANT for LSTM convergence
    # Fit scaler ONLY on training data to avoid data leakage
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_raw = X_raw[train_mask]
    X_test_raw  = X_raw[test_mask]
    y_train_raw = y_raw[train_mask].reshape(-1, 1)
    y_test_raw  = y_raw[test_mask].reshape(-1, 1)

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled  = scaler_X.transform(X_test_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw).ravel()
    y_test_scaled  = scaler_y.transform(y_test_raw).ravel()

    # Build sequence windows
    X_tr_seq, y_tr_seq = build_sequences(X_train_scaled, y_train_scaled, lookback)
    X_te_seq, y_te_seq = build_sequences(X_test_scaled,  y_test_scaled,  lookback)

    test_dates = dates[test_mask][lookback:]

    print("  " + country + ": train=" + str(len(X_tr_seq)) +
          " test=" + str(len(X_te_seq)) +
          " features=" + str(len(available_features)))

    return {
        "X_train": X_tr_seq,
        "y_train": y_tr_seq,
        "X_test":  X_te_seq,
        "y_test":  y_te_seq,
        "scaler_y": scaler_y,
        "scaler_X": scaler_X,
        "test_dates": test_dates,
        "n_features": len(available_features),
    }


# =============================================================================
# 3. BUILD LSTM MODEL
#    Architecture: LSTM(64) → Dropout → LSTM(32) → Dense(1)
#    Two LSTM layers = captures both short and medium-term patterns
# =============================================================================

def build_lstm(lookback, n_features):
    model = Sequential([
        # First LSTM layer — return_sequences=True passes hidden state to next LSTM
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),   # randomly zero 20% of neurons to prevent overfitting

        # Second LSTM layer
        LSTM(32, return_sequences=False),
        Dropout(0.2),

        # Output layer — single value (the depreciation forecast)
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# =============================================================================
# 4. TRAIN + EVALUATE
# =============================================================================

def train_country(data, country):
    print("  Training LSTM for: " + country)

    model = build_lstm(LOOKBACK, data["n_features"])

    # EarlyStopping: stop training if validation loss doesn't improve for 15 epochs
    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    history = model.fit(
        data["X_train"], data["y_train"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0,   # set to 1 if you want to see epoch-by-epoch progress
    )

    epochs_ran = len(history.history["loss"])
    print("  Stopped at epoch: " + str(epochs_ran))

    # Predict on test set
    y_pred_scaled = model.predict(data["X_test"], verbose=0).ravel()

    # Inverse-transform to get real % values
    y_pred = data["scaler_y"].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = data["scaler_y"].inverse_transform(data["y_test"].reshape(-1, 1)).ravel()

    mae     = mean_absolute_error(y_true, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = (np.sign(y_pred) == np.sign(y_true)).mean() * 100

    print("  MAE: " + str(round(mae, 3)) +
          "% | RMSE: " + str(round(rmse, 3)) +
          "% | Dir.Acc: " + str(round(dir_acc, 1)) + "%")

    return {
        "model":    model,
        "history":  history,
        "y_pred":   y_pred,
        "y_true":   y_true,
        "dates":    data["test_dates"],
        "mae":      mae,
        "rmse":     rmse,
        "dir_acc":  dir_acc,
    }


# =============================================================================
# 5. CHARTS
# =============================================================================

BRICS_COLORS = {
    "Brazil":       "#009C3B",
    "Russia":       "#CC0000",
    "India":        "#FF9933",
    "China":        "#DE2910",
    "South_Africa": "#007A4D",
}


def chart_predictions(results_dict):
    """One subplot per country showing actual vs predicted."""
    n = len(results_dict)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (country, res) in zip(axes, results_dict.items()):
        color = BRICS_COLORS.get(country, "#3498db")
        ax.plot(res["dates"], res["y_true"],
                label="Actual", color="#2c3e50", linewidth=2)
        ax.plot(res["dates"], res["y_pred"],
                label="LSTM Predicted", color=color,
                linewidth=1.8, linestyle="--")
        ax.fill_between(res["dates"], res["y_true"], res["y_pred"],
                        alpha=0.12, color=color)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        ax.set_title(
            country + "  |  MAE: " + str(round(res["mae"], 2)) +
            "%  Dir.Acc: " + str(round(res["dir_acc"], 1)) + "%",
            fontsize=11, fontweight="bold", color=color
        )
        ax.set_ylabel("Depreciation %")
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("LSTM Forecasts — BRICS Currency Depreciation vs USD (3-Month Ahead)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = "outputs/charts/08_lstm_brics_predictions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: " + path)


def chart_training_loss(results_dict):
    """Training vs validation loss curves for each country."""
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (country, res) in zip(axes, results_dict.items()):
        hist = res["history"].history
        ax.plot(hist["loss"],     label="Train loss", color="#3498db", linewidth=1.5)
        ax.plot(hist["val_loss"], label="Val loss",   color="#e74c3c",
                linewidth=1.5, linestyle="--")
        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=7)

    plt.suptitle("LSTM Training Curves — Did It Learn?", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = "outputs/charts/09_lstm_training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: " + path)


def chart_brics_comparison(results_dict):
    """Bar chart comparing XGBoost vs LSTM metrics across countries."""
    # Load XGBoost results if available
    xgb_path = "outputs/results/model_performance.csv"
    xgb_df   = pd.read_csv(xgb_path) if os.path.exists(xgb_path) else pd.DataFrame()

    countries = list(results_dict.keys())
    lstm_mae  = [results_dict[c]["mae"]     for c in countries]
    lstm_da   = [results_dict[c]["dir_acc"] for c in countries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [BRICS_COLORS.get(c, "#aaa") for c in countries]
    x = np.arange(len(countries))

    # MAE comparison
    bars = ax1.bar(x, lstm_mae, color=colors, alpha=0.85, width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n") for c in countries], fontsize=9)
    ax1.set_ylabel("MAE (%)")
    ax1.set_title("LSTM — Mean Absolute Error by Country\n(lower = better)", fontweight="bold")
    for bar, val in zip(bars, lstm_mae):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(round(val, 2)) + "%", ha="center", fontsize=8)

    # Directional accuracy
    bars2 = ax2.bar(x, lstm_da, color=colors, alpha=0.85, width=0.5)
    ax2.axhline(50, color="red", linestyle="--", linewidth=1, label="Random baseline (50%)")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in countries], fontsize=9)
    ax2.set_ylabel("Directional Accuracy (%)")
    ax2.set_title("LSTM — Did It Get the Direction Right?\n(above 50% = better than random)",
                  fontweight="bold")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, lstm_da):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(round(val, 1)) + "%", ha="center", fontsize=8)

    plt.tight_layout()
    path = "outputs/charts/10_lstm_brics_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: " + path)


# =============================================================================
# 6. SAVE RESULTS
# =============================================================================

def save_results(results_dict):
    rows = []
    for country, res in results_dict.items():
        rows.append({
            "Model":        "LSTM",
            "Country":      country,
            "MAE (%)":      round(res["mae"],     3),
            "RMSE (%)":     round(res["rmse"],    3),
            "Dir_Acc (%)":  round(res["dir_acc"], 1),
        })

    lstm_df = pd.DataFrame(rows)
    lstm_df.to_csv("outputs/results/lstm_performance.csv", index=False)
    print("Saved: outputs/results/lstm_performance.csv")

    # Also save predictions to CSV (useful for dashboard integration later)
    all_preds = []
    for country, res in results_dict.items():
        pred_df = pd.DataFrame({
            "Date":    res["dates"],
            "Country": country,
            "Actual":  res["y_true"],
            "LSTM_Predicted": res["y_pred"],
        })
        all_preds.append(pred_df)

    pd.concat(all_preds).to_csv("outputs/results/lstm_predictions.csv", index=False)
    print("Saved: outputs/results/lstm_predictions.csv")

    # Save each model
    for country, res in results_dict.items():
        model_path = "models/lstm_" + country.lower() + ".keras"
        res["model"].save(model_path)
        print("Saved: " + model_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("Dollar Hegemony - Phase 4: LSTM (All BRICS Countries)")
    print("=" * 55)

    df = load_features()

    results_dict = {}

    for country in BRICS_COUNTRIES:
        print("\n--- " + country + " ---")

        data = prepare_country_data(df, country, LOOKBACK, FORECAST_HORIZON, TEST_YEARS)
        if data is None:
            continue

        res = train_country(data, country)
        results_dict[country] = res

    if not results_dict:
        print("\nERROR: No countries were trained. Check features.csv has depreciation columns.")
        return

    print("\n--- Generating Charts ---")
    chart_predictions(results_dict)
    chart_training_loss(results_dict)
    chart_brics_comparison(results_dict)

    print("\n--- Saving Results ---")
    save_results(results_dict)

    print("\n" + "=" * 55)
    print("PHASE 4 COMPLETE!")
    print("Charts: outputs/charts/08, 09, 10")
    print("Results: outputs/results/lstm_performance.csv")
    print("Models: models/lstm_*.keras")
    print("\nNext step: Add LSTM predictions to your dashboard!")
    print("=" * 55)

    # Final summary table
    print("\nLSTM Results Summary:")
    print("-" * 50)
    print(f"{'Country':<15} {'MAE':>8} {'RMSE':>8} {'Dir.Acc':>10}")
    print("-" * 50)
    for country, res in results_dict.items():
        print(f"{country:<15} {res['mae']:>7.2f}% {res['rmse']:>7.2f}% {res['dir_acc']:>9.1f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()
