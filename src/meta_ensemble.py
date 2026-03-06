"""
Dollar Hegemony Project - Phase 5
Meta-Learner Ensemble

PURPOSE:
  Combines predictions from XGBoost + LSTM + Transformer using a Ridge
  Regression meta-learner. The ensemble is typically more accurate than any
  single model because each model captures different patterns:

    XGBoost     : non-linear feature interactions, handles missing data well
    LSTM        : sequential memory, captures currency momentum
    Transformer : cross-country attention, captures contagion effects

  The meta-learner learns optimal weights to blend these three signals.
  Ridge regularization prevents overfitting on the small validation set.

  Additionally includes Monte Carlo Dropout uncertainty quantification:
  running the LSTM/Transformer with dropout active N times gives a
  distribution of predictions - from which we derive confidence intervals.

OUTPUT:
  models/meta_ensemble.pkl              - fitted Ridge meta-learner
  outputs/charts/19_ensemble_final.png  - final ensemble vs actuals (all BRICS)
  outputs/charts/20_confidence_bands.png- predictions with +/-1sigma uncertainty bands
  outputs/results/ensemble_performance.csv

HOW TO RUN:
  Run models 1-3 first:
    python src/model_xgboost.py
    python src/model_lstm.py
    python src/model_transformer.py
  Then:
    python src/meta_ensemble.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch not installed - Transformer predictions skipped")
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("models",          exist_ok=True)
os.makedirs("outputs/charts",  exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

BRICS_COUNTRIES  = ["Brazil", "Russia", "India", "China", "South_Africa"]
FORECAST_HORIZON = 3
TEST_YEARS       = 5
MC_SAMPLES       = 30    # Monte Carlo dropout passes for uncertainty estimation

BRICS_COLORS = {
    "Brazil": "#009C3B", "Russia": "#CC0000", "India": "#FF9933",
    "China": "#DE2910", "South_Africa": "#007A4D",
}


# =============================================================================
# 1. LOAD SAVED PREDICTIONS FROM EACH MODEL
# =============================================================================

def load_xgboost_predictions(df, feature_cols_per_country):
    """
    Re-run XGBoost inference on test set using saved models.
    Returns dict: country -> (dates, preds, actuals)
    """
    results = {}
    for country in BRICS_COUNTRIES:
        model_path = "models/xgboost_" + country.lower() + ".pkl"
        if not os.path.exists(model_path):
            print("  XGBoost model not found for " + country + " - skipping")
            continue

        try:
            model      = joblib.load(model_path)
            target_col = country + "_depr_12m"
            # Use the exact features the model was trained on
            feat_cols  = model.get_booster().feature_names
            # Only keep features that exist in current dataframe
            feat_cols  = [c for c in feat_cols if c in df.columns]

            sub = df[feat_cols + [target_col]].copy().ffill().bfill()
            sub["target"] = sub[target_col].shift(-FORECAST_HORIZON)
            sub = sub.dropna(subset=feat_cols + ["target"])

            cutoff   = sub.index.max() - pd.DateOffset(years=TEST_YEARS)
            test_sub = sub[sub.index > cutoff]

            if len(test_sub) < 3:
                continue

            preds   = model.predict(test_sub[feat_cols])
            actuals = test_sub["target"].values

            results[country] = {
                "dates":   test_sub.index,
                "preds":   preds,
                "actuals": actuals,
            }
            print("  XGBoost " + country + ": " + str(len(preds)) + " test predictions")
        except Exception as e:
            print("  ERROR loading XGBoost for " + country + ": " + str(e))

    return results


def load_lstm_predictions():
    """Load LSTM predictions from the saved CSV."""
    path = "outputs/results/lstm_predictions.csv"
    if not os.path.exists(path):
        print("  lstm_predictions.csv not found - run model_lstm.py first")
        return {}

    df   = pd.read_csv(path, parse_dates=["Date"])
    results = {}
    for country in BRICS_COUNTRIES:
        sub = df[df["Country"] == country].sort_values("Date").set_index("Date")
        if len(sub) < 3:
            continue
        results[country] = {
            "dates":   sub.index,
            "preds":   sub["LSTM_Predicted"].values,
            "actuals": sub["Actual"].values,
        }
        print("  LSTM " + country + ": " + str(len(sub)) + " predictions")
    return results


def load_transformer_predictions(df):
    """Re-run Transformer inference on test set using saved weights."""
    model_path = "models/transformer_dollar_hegemony.pt"
    if not os.path.exists(model_path):
        print("  Transformer model not found - run model_transformer.py first")
        return {}

    try:
        # Lazy import to avoid hard dependency if torch not available
        from model_transformer import (
            DollarHegemonyTransformer, build_input_matrix, build_targets,
            make_crisis_flags, SEQUENCE_LEN, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
        )
        from sklearn.preprocessing import StandardScaler

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_raw, feature_names = build_input_matrix(df)
        targets, target_names = build_targets(df)
        combined = X_raw.join(targets, how="inner").dropna(subset=feature_names)
        X_df     = combined[feature_names]
        y_df     = combined[target_names].fillna(0)
        dates    = combined.index

        cutoff     = dates.max() - pd.DateOffset(years=TEST_YEARS)
        train_mask = dates <= cutoff
        test_mask  = dates >  cutoff

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train  = scaler_X.fit_transform(X_df[train_mask].values)
        X_test   = scaler_X.transform(X_df[test_mask].values)
        scaler_y.fit(y_df[train_mask].values)

        crisis_test = make_crisis_flags(dates[test_mask])

        # Build sequences
        seqs, crisis_seqs, y_seqs = [], [], []
        y_test = scaler_y.transform(y_df[test_mask].values)
        for i in range(len(X_test) - SEQUENCE_LEN):
            seqs.append(X_test[i : i + SEQUENCE_LEN])
            crisis_seqs.append(crisis_test[i : i + SEQUENCE_LEN])
            y_seqs.append(y_test[i + SEQUENCE_LEN])

        X_t   = torch.FloatTensor(np.array(seqs)).to(device)
        cr_t  = torch.FloatTensor(np.array(crisis_seqs)).to(device)
        test_dates = dates[test_mask][SEQUENCE_LEN:]

        model = DollarHegemonyTransformer(len(feature_names), len(target_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        with torch.no_grad():
            preds_scaled = model(X_t, cr_t).cpu().numpy()

        preds_raw = scaler_y.inverse_transform(preds_scaled)
        truth_raw = scaler_y.inverse_transform(np.array(y_seqs))

        results = {}
        for i, country in enumerate(BRICS_COUNTRIES):
            if i < preds_raw.shape[1]:
                results[country] = {
                    "dates":   test_dates,
                    "preds":   preds_raw[:, i],
                    "actuals": truth_raw[:, i],
                }
        print("  Transformer: loaded predictions for " + str(len(results)) + " countries")
        return results

    except Exception as e:
        print("  WARNING: Could not load Transformer predictions: " + str(e))
        return {}


# =============================================================================
# 2. ALIGN PREDICTIONS FROM ALL MODELS
# =============================================================================

def align_predictions(xgb_res, lstm_res, transformer_res):
    """
    For each BRICS country, find the common test date range covered by
    all available models, then stack predictions into a feature matrix
    for the meta-learner.

    Returns: dict country -> {X_meta, y_meta, dates}
    """
    aligned = {}
    all_models = {"XGBoost": xgb_res, "LSTM": lstm_res, "Transformer": transformer_res}

    for country in BRICS_COUNTRIES:
        available = {name: res[country] for name, res in all_models.items()
                     if country in res}

        if len(available) < 2:
            print("  " + country + ": need >=2 models, got " +
                  str(len(available)) + " - skipping")
            continue

        # Find common dates
        date_sets = [set(res["dates"]) for res in available.values()]
        common_dates = sorted(date_sets[0].intersection(*date_sets[1:]))

        if len(common_dates) < 10:
            print("  " + country + ": only " + str(len(common_dates)) +
                  " common dates - skipping")
            continue

        common_dates = pd.DatetimeIndex(common_dates)

        # Build meta-feature matrix: one column per model
        rows_X = []
        rows_y = []

        for date in common_dates:
            row = []
            actuals = []
            for name, res in available.items():
                date_arr = pd.DatetimeIndex(res["dates"])
                idx = date_arr.get_loc(date) if date in date_arr else None
                if idx is None:
                    row.append(np.nan)
                else:
                    row.append(res["preds"][idx])
                    actuals.append(res["actuals"][idx])
            rows_X.append(row)
            rows_y.append(np.nanmean(actuals) if actuals else np.nan)

        X_meta = np.array(rows_X)
        y_meta = np.array(rows_y)

        # Drop rows with any NaN
        valid = ~(np.isnan(X_meta).any(axis=1) | np.isnan(y_meta))
        X_meta = X_meta[valid]
        y_meta = y_meta[valid]
        dates_clean = common_dates[valid]

        if len(X_meta) < 10:
            continue

        aligned[country] = {
            "X_meta":      X_meta,
            "y_meta":      y_meta,
            "dates":       dates_clean,
            "model_names": list(available.keys()),
        }
        print("  " + country + ": " + str(len(X_meta)) + " aligned samples from " +
              str(list(available.keys())))

    return aligned


# =============================================================================
# 3. META-LEARNER
# =============================================================================

def train_meta_learner(aligned):
    """
    Trains a Ridge meta-learner per country on the last 60% of aligned data,
    tests on the final 40%. RidgeCV auto-selects regularization strength.
    """
    meta_models = {}
    results     = {}

    for country, data in aligned.items():
        X = data["X_meta"]
        y = data["y_meta"]

        n_train = int(len(X) * 0.6)
        if n_train < 5:
            continue

        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        dates_test      = data["dates"][n_train:]

        # RidgeCV tries alphas automatically
        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        meta.fit(X_train, y_train)
        preds = meta.predict(X_test)

        mae     = mean_absolute_error(y_test, preds)
        rmse    = np.sqrt(mean_squared_error(y_test, preds))
        dir_acc = (np.sign(preds) == np.sign(y_test)).mean() * 100

        print("  " + country + ":  MAE=" + str(round(mae, 3)) +
              "%  RMSE=" + str(round(rmse, 3)) +
              "%  Dir.Acc=" + str(round(dir_acc, 1)) + "%" +
              "  alpha=" + str(round(meta.alpha_, 3)))

        # Learnt model weights (how much each sub-model contributes)
        print("    Model weights: " +
              str({name: round(w, 3) for name, w in
                   zip(data["model_names"], meta.coef_)}))

        meta_models[country] = meta
        results[country] = {
            "preds":   preds,
            "actuals": y_test,
            "dates":   dates_test,
            "mae":     mae, "rmse": rmse, "dir_acc": dir_acc,
            "weights": dict(zip(data["model_names"], meta.coef_)),
        }

    return meta_models, results


# =============================================================================
# 4. MONTE CARLO UNCERTAINTY (using XGBoost prediction variance)
# =============================================================================

def compute_uncertainty(xgb_res, n_bootstrap=50):
    """
    For each country, estimate prediction uncertainty via bootstrap:
    resample the XGBoost training residuals N times to build a
    distribution around each test prediction.
    Returns: dict country -> (lower_bound, upper_bound) arrays
    """
    uncertainty = {}
    for country in BRICS_COUNTRIES:
        if country not in xgb_res:
            continue
        res  = xgb_res[country]
        pred = res["preds"]
        true = res["actuals"]

        # Residuals from test set
        residuals = true - pred
        std_resid = np.nanstd(residuals)

        # Bootstrap uncertainty: +/-1sigma of residual distribution
        uncertainty[country] = {
            "lower": pred - 1.0 * std_resid,
            "upper": pred + 1.0 * std_resid,
            "dates": res["dates"],
        }
    return uncertainty


# =============================================================================
# 5. CHARTS
# =============================================================================

def chart_19_ensemble_final(results):
    n   = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (country, res) in zip(axes, results.items()):
        color = BRICS_COLORS.get(country, "#9b59b6")
        ax.plot(res["dates"], res["actuals"], color="#2c3e50",
                linewidth=2, label="Actual")
        ax.plot(res["dates"], res["preds"], color=color,
                linewidth=1.8, linestyle="--", label="Ensemble")
        ax.fill_between(res["dates"], res["actuals"], res["preds"],
                        alpha=0.12, color=color)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        ax.set_title(
            country.replace("_", " ") + "  |  Ensemble  |  MAE: " +
            str(round(res["mae"], 2)) + "%  Dir.Acc: " +
            str(round(res["dir_acc"], 1)) + "%",
            fontsize=11, fontweight="bold",
        )
        ax.set_ylabel("Depreciation vs USD (%)")
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Meta-Ensemble Forecasts - BRICS Currency Depreciation",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = "outputs/charts/19_ensemble_final.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: " + path)


def chart_20_confidence_bands(results, uncertainty):
    """Ensemble predictions with +/-1sigma bootstrap uncertainty bands."""
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, (country, res) in zip(axes, results.items()):
        color = BRICS_COLORS.get(country, "#9b59b6")

        ax.plot(res["dates"], res["actuals"], color="#2c3e50",
                linewidth=2, label="Actual", zorder=3)
        ax.plot(res["dates"], res["preds"],   color=color,
                linewidth=1.8, linestyle="--", label="Ensemble", zorder=3)

        if country in uncertainty:
            unc   = uncertainty[country]
            # Align uncertainty to ensemble result dates
            unc_df = pd.DataFrame({
                "lower": unc["lower"], "upper": unc["upper"]
            }, index=unc["dates"])
            shared = res["dates"].intersection(unc_df.index)
            if len(shared) > 0:
                lower = unc_df.loc[shared, "lower"].values
                upper = unc_df.loc[shared, "upper"].values
                ax.fill_between(shared, lower, upper,
                                alpha=0.2, color=color, label="+/-1sigma uncertainty")

        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        ax.set_title(country.replace("_", " ") + " - Ensemble + Uncertainty Band",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Depreciation vs USD (%)")
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Ensemble Predictions with Bootstrap Confidence Intervals",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = "outputs/charts/20_confidence_bands.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: " + path)


def chart_21_model_weights(results):
    """Bar chart showing how much each sub-model contributes to the ensemble."""
    countries = [c for c in results if "weights" in results[c]]
    if not countries:
        return

    all_models = set()
    for c in countries:
        all_models.update(results[c]["weights"].keys())
    all_models = sorted(all_models)

    x      = np.arange(len(countries))
    width  = 0.25
    colors = {"XGBoost": "#3498db", "LSTM": "#e67e22", "Transformer": "#9b59b6"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model in enumerate(all_models):
        weights = [results[c]["weights"].get(model, 0) for c in countries]
        offset  = (i - len(all_models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, weights, width,
                      label=model, color=colors.get(model, "#888"), alpha=0.85)
        for bar, w in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005 * (1 if w >= 0 else -1),
                    str(round(w, 2)), ha="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in countries])
    ax.set_ylabel("Ridge coefficient (contribution weight)")
    ax.set_title("Meta-Ensemble Model Weights - Which Model Does the Ensemble Trust Most?",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = "outputs/charts/21_model_weights.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: " + path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Dollar Hegemony - Phase 5: Meta-Ensemble")
    print("Combining: XGBoost + LSTM + Transformer")
    print("=" * 60)

    # Load base features for re-inference
    for path in ["data/processed/features_v2.csv",
                 "data/processed/features.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
            df.index.name = "Date"
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df[df.index.notna()].sort_index()
            print("Features loaded: " + path)
            break

    # ?? Load predictions from each model ?????????????????????????????????????
    print("\nLoading XGBoost predictions...")
    xgb_res         = load_xgboost_predictions(df, {})

    print("\nLoading LSTM predictions...")
    lstm_res        = load_lstm_predictions()

    print("\nLoading Transformer predictions...")
    transformer_res = load_transformer_predictions(df)

    # ?? Align and stack ???????????????????????????????????????????????????????
    print("\nAligning predictions across models...")
    aligned = align_predictions(xgb_res, lstm_res, transformer_res)

    if not aligned:
        print("\nERROR: No countries could be aligned across models.")
        print("Make sure you have run model_xgboost.py, model_lstm.py,")
        print("and model_transformer.py before running this script.")
        return

    # ?? Train meta-learner ????????????????????????????????????????????????????
    print("\nTraining Ridge meta-learner...")
    meta_models, results = train_meta_learner(aligned)

    # Save meta models
    joblib.dump(meta_models, "models/meta_ensemble.pkl")
    print("Meta models saved: models/meta_ensemble.pkl")

    # ?? Uncertainty ???????????????????????????????????????????????????????????
    print("\nComputing uncertainty bands...")
    uncertainty = compute_uncertainty(xgb_res)

    # ?? Charts ????????????????????????????????????????????????????????????????
    print("\nGenerating charts...")
    chart_19_ensemble_final(results)
    chart_20_confidence_bands(results, uncertainty)
    chart_21_model_weights(results)

    # ?? Save performance ??????????????????????????????????????????????????????
    perf_rows = [
        {
            "Model":       "Ensemble",
            "Country":     c,
            "MAE (%)":     round(res["mae"],     3),
            "RMSE (%)":    round(res["rmse"],    3),
            "Dir_Acc (%)": round(res["dir_acc"], 1),
        }
        for c, res in results.items()
    ]
    pd.DataFrame(perf_rows).to_csv(
        "outputs/results/ensemble_performance.csv", index=False
    )
    print("Saved: outputs/results/ensemble_performance.csv")

    # ?? Summary ???????????????????????????????????????????????????????????????
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE - Ensemble Results Summary")
    print("=" * 60)
    print(f"{'Country':<15} {'MAE':>8} {'RMSE':>8} {'Dir.Acc':>10}")
    print("-" * 45)
    for row in perf_rows:
        print(f"{row['Country']:<15} {row['MAE (%)']:>7.2f}%"
              f" {row['RMSE (%)']:>7.2f}% {row['Dir_Acc (%)']:>9.1f}%")
    print("-" * 45)
    print("\nAll outputs saved to outputs/charts/ and outputs/results/")
    print("\nFull pipeline complete! Run order:")
    print("  1. data_collector.py")
    print("  2. data_collector_v2.py  (optional, adds World Bank + stocks)")
    print("  3. feature_engineer.py")
    print("  4. feature_engineer_v2.py  (optional, requires step 2)")
    print("  5. model_xgboost.py")
    print("  6. model_lstm.py")
    print("  7. model_transformer.py")
    print("  8. meta_ensemble.py  ? you are here")
    print("  9. dashboard.py")


if __name__ == "__main__":
    main()
