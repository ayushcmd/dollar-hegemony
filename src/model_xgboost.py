"""
Dollar Hegemony Project - Phase 2
XGBoost Model — All BRICS Countries + SHAP Analysis

CHANGES FROM PREVIOUS VERSION:
  - Trains on ALL 5 BRICS countries (was hardcoded to India only)
  - Adds SHAP value analysis for each country (model interpretability)
  - Combined performance CSV covers all countries
  - Summary comparison chart across all BRICS nations
  - Fixed: model saved for every country, not just one

HOW TO RUN:
  pip install xgboost shap scikit-learn pandas numpy matplotlib joblib
  python src/model_xgboost.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.makedirs("models",           exist_ok=True)
os.makedirs("outputs/charts",   exist_ok=True)
os.makedirs("outputs/results",  exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================

# All 5 BRICS countries — previously only "India" was trained
BRICS_COUNTRIES  = ["Brazil", "Russia", "India", "China", "South_Africa"]
FORECAST_HORIZON = 3   # predict 3 months ahead

BRICS_COLORS = {
    "Brazil":       "#009C3B",
    "Russia":       "#CC0000",
    "India":        "#FF9933",
    "China":        "#DE2910",
    "South_Africa": "#007A4D",
}


# =============================================================================
# 1. LOAD DATA
#    Robust: auto-detects date column regardless of whether header is "Date"
# =============================================================================

def load_features():
    path = "data/processed/features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "features.csv not found. Run feature_engineer.py first."
        )
    df = pd.read_csv(path)

    # Auto-detect date column (first column) — handles missing "Date" header
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)
    df.index.name = "Date"

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df.index.notna()].sort_index()
    print("Loaded: " + str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns")
    print("Date range: " + str(df.index.min().date()) + " to " + str(df.index.max().date()))
    return df


# =============================================================================
# 2. BUILD X / y
# =============================================================================

def build_xy(df, country, horizon):
    target_col = country + "_depr_12m"
    if target_col not in df.columns:
        print("  SKIP: " + target_col + " not found in features.csv")
        return None, None, None

    candidates   = ["DXY", "Oil_WTI", "Gold",
                    "US_Fed_Rate", "US_10Y_Yield"]
    feature_cols = [c for c in candidates if c in df.columns]

    all_needed = feature_cols + [target_col]
    sub = df[all_needed].copy()
    sub = sub.ffill().bfill()                     # handle sparse FRED monthly gaps
    sub["target"] = sub[target_col].shift(-horizon)
    sub = sub.dropna(subset=feature_cols + ["target"])

    X = sub[feature_cols]
    y = sub["target"]

    print("  Features: " + str(feature_cols))
    print("  Valid rows: " + str(len(X)) +
          "  (" + str(X.index[0].date()) + " to " + str(X.index[-1].date()) + ")")
    return X, y, feature_cols


# =============================================================================
# 3. WALK-FORWARD VALIDATION
#    Train on everything up to 5 years ago, test on last 5 years.
#    No future data leaks into training set.
# =============================================================================

def walk_forward_validation(X, y, n_test_years=5):
    cutoff  = X.index.max() - pd.DateOffset(years=n_test_years)
    X_train = X[X.index <= cutoff]
    X_test  = X[X.index >  cutoff]
    y_train = y[y.index <= cutoff]
    y_test  = y[y.index >  cutoff]

    print("  Train: " + str(len(X_train)) + " months  |  Test: " + str(len(X_test)) + " months")

    if len(X_train) < 20 or len(X_test) < 6:
        print("  WARNING: insufficient data for validation — skipping")
        return None

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)

    preds   = model.predict(X_test)
    mae     = mean_absolute_error(y_test, preds)
    rmse    = np.sqrt(mean_squared_error(y_test, preds))
    dir_acc = (np.sign(preds) == np.sign(y_test.values)).mean() * 100

    print("  MAE: " + str(round(mae, 3)) + "%  |  "
          "RMSE: " + str(round(rmse, 3)) + "%  |  "
          "Dir.Acc: " + str(round(dir_acc, 1)) + "%")

    return {
        "model":   model,
        "preds":   pd.Series(preds,        index=X_test.index, name="predicted"),
        "actuals": y_test.rename("actual"),
        "X_test":  X_test,
        "mae": mae, "rmse": rmse, "dir_acc": dir_acc,
    }


# =============================================================================
# 4. CHARTS
# =============================================================================

def chart_predictions(results, country, horizon):
    color = BRICS_COLORS.get(country, "#3498db")
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(results["actuals"].index, results["actuals"].values,
            label="Actual", color="#2c3e50", linewidth=2)
    ax.plot(results["preds"].index, results["preds"].values,
            label="XGBoost Predicted", color=color, linewidth=1.8, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.fill_between(results["actuals"].index,
                    results["actuals"].values, results["preds"].values,
                    alpha=0.15, color=color)

    ax.set_title(
        country.replace("_", " ") + " — " + str(horizon) + "M Ahead Forecast  |  "
        "MAE: " + str(round(results["mae"], 2)) + "%  |  "
        "Dir.Acc: " + str(round(results["dir_acc"], 1)) + "%",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Depreciation vs USD (%)")
    ax.legend()
    plt.tight_layout()

    path = "outputs/charts/06_" + country.lower() + "_xgb_predictions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Chart saved: " + path)


def chart_shap(model, X_test, feature_cols, country):
    """
    SHAP (SHapley Additive exPlanations) — shows HOW MUCH each feature
    contributed to each prediction. Critical for research credibility.
    Red bars = dollar/rate drivers  |  Blue bars = other factors
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Mean absolute SHAP per feature — overall importance
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    order      = np.argsort(mean_shap)
    feat_names = [feature_cols[i] for i in order]
    vals       = mean_shap[order]

    bar_colors = [
        "#e74c3c" if any(k in f for k in ["DXY", "Fed", "Yield"]) else "#3498db"
        for f in feat_names
    ]

    ax.barh(feat_names, vals, color=bar_colors, alpha=0.85)
    ax.set_xlabel("Mean |SHAP value| — average impact on predicted depreciation (%)")
    ax.set_title(
        country.replace("_", " ") + " — SHAP Feature Importance\n"
        "Red = Dollar/Rate drivers  |  Blue = Other factors",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    path = "outputs/charts/07_" + country.lower() + "_shap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("  SHAP chart saved: " + path)

    # Save SHAP values to CSV — used by dashboard Tab 4 (ML Predictions)
    shap_df = pd.DataFrame(shap_vals, index=X_test.index, columns=feature_cols)
    shap_df.to_csv("outputs/results/shap_" + country.lower() + ".csv")


def chart_brics_comparison(all_results):
    """Bar chart comparing MAE and Directional Accuracy across all BRICS countries."""
    countries = list(all_results.keys())
    maes    = [all_results[c]["mae"]     for c in countries]
    dir_acs = [all_results[c]["dir_acc"] for c in countries]
    colors  = [BRICS_COLORS.get(c, "#aaa") for c in countries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x      = np.arange(len(countries))
    labels = [c.replace("_", "\n") for c in countries]

    # MAE
    bars1 = ax1.bar(x, maes, color=colors, alpha=0.85, width=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("MAE (%)")
    ax1.set_title("XGBoost — MAE by Country\n(lower = better)", fontweight="bold")
    for bar, val in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(round(val, 2)) + "%", ha="center", fontsize=9)

    # Directional accuracy
    bars2 = ax2.bar(x, dir_acs, color=colors, alpha=0.85, width=0.5)
    ax2.axhline(50, color="red", linestyle="--", linewidth=1.2,
                label="Random baseline (50%)")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Directional Accuracy (%)")
    ax2.set_title("XGBoost — Directional Accuracy\n(above 50% = better than random)",
                  fontweight="bold")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, dir_acs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(round(val, 1)) + "%", ha="center", fontsize=9)

    plt.suptitle("XGBoost — All BRICS Countries Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = "outputs/charts/07_brics_xgb_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Comparison chart saved: " + path)


# =============================================================================
# 5. MAIN — loop over all BRICS countries
# =============================================================================

def main():
    print("=" * 55)
    print("Dollar Hegemony - Phase 2: XGBoost — All BRICS Countries")
    print("=" * 55)

    df = load_features()
    all_results = {}
    perf_rows   = []

    for country in BRICS_COUNTRIES:
        print("\n--- " + country + " ---")

        X, y, feature_cols = build_xy(df, country, FORECAST_HORIZON)
        if X is None:
            continue

        results = walk_forward_validation(X, y)
        if results is None:
            continue

        # Save model per country
        model_path = "models/xgboost_" + country.lower() + ".pkl"
        joblib.dump(results["model"], model_path)
        print("  Model saved: " + model_path)

        # Charts
        chart_predictions(results, country, FORECAST_HORIZON)
        chart_shap(results["model"], results["X_test"], feature_cols, country)

        all_results[country] = results
        perf_rows.append({
            "Model":       "XGBoost",
            "Country":     country,
            "MAE (%)":     round(results["mae"],     3),
            "RMSE (%)":    round(results["rmse"],    3),
            "Dir_Acc (%)": round(results["dir_acc"], 1),
        })

    if not all_results:
        print("\nERROR: No countries trained. Check features.csv has *_depr_12m columns.")
        return

    # Cross-country comparison chart
    chart_brics_comparison(all_results)

    # Save combined performance CSV (picked up by dashboard + LSTM comparison)
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv("outputs/results/model_performance.csv", index=False)
    print("\nPerformance CSV saved: outputs/results/model_performance.csv")

    # Print summary table
    print("\n" + "=" * 55)
    print("PHASE 2 COMPLETE — XGBoost Results Summary")
    print("=" * 55)
    print(f"{'Country':<15} {'MAE':>8} {'RMSE':>8} {'Dir.Acc':>10}")
    print("-" * 45)
    for row in perf_rows:
        print(f"{row['Country']:<15} {row['MAE (%)']:>7.2f}%"
              f" {row['RMSE (%)']:>7.2f}% {row['Dir_Acc (%)']:>9.1f}%")
    print("-" * 45)
    print("\nNext step: python src/model_lstm.py")


if __name__ == "__main__":
    main()
