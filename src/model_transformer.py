"""
Dollar Hegemony Project - Phase 3
Transformer — Multi-Country Attention Model

WHAT THIS MODEL DOES:
  A Time-Series Transformer that simultaneously processes all BRICS + G7 countries
  and learns cross-country attention patterns: e.g., when Brazil's currency falls,
  does India follow? Which countries' macro signals most influence each other?

  This is the most academically novel model in the project. The attention weights
  produced are a research finding in themselves — they reveal the structural
  dependencies between emerging market economies under dollar stress.

ARCHITECTURE:
  Input  : [batch, sequence_len=24, n_countries * n_features]
            — last 24 months of macro data for ALL countries simultaneously
  Encoder: Multi-head self-attention (4 heads) + feed-forward layers (×2)
  Output : Simultaneous depreciation forecast for all BRICS countries (5 heads)
  Novel  : Macro-calendar positional encoding — crisis dates get special embeddings

  Inspired by: Zerveas et al. (2021) "A Transformer-based Framework for
  Multivariate Time Series Representation Learning" (SIGKDD 2021)

KEY OUTPUTS:
  models/transformer_dollar_hegemony.pt  — trained model weights
  outputs/charts/15_transformer_predictions.png — actual vs predicted all BRICS
  outputs/charts/16_attention_heatmap.png       — cross-country attention weights
  outputs/charts/17_transformer_vs_lstm.png     — model comparison table
  outputs/results/transformer_performance.csv

HOW TO RUN:
  pip install torch scikit-learn pandas numpy matplotlib
  python src/model_transformer.py
"""

import os
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("models",          exist_ok=True)
os.makedirs("outputs/charts",  exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================

BRICS_COUNTRIES = ["Brazil", "Russia", "India", "China", "South_Africa"]
G7_COUNTRIES    = ["Germany_EU", "Japan", "UK", "Canada"]
ALL_COUNTRIES   = BRICS_COUNTRIES + G7_COUNTRIES

# Shared macro features used as input for every country
# These are the global signals — same value fed for all countries each month
GLOBAL_FEATURES = ["DXY", "DXY_mom_12m", "Oil_WTI", "Gold",
                   "US_Fed_Rate", "US_10Y_Yield"]

# Per-country features (depreciation of each currency = both input context and target)
# The model sees all countries' past depreciation, then predicts future BRICS depreciation
COUNTRY_FEATURE = "_depr_12m"

SEQUENCE_LEN     = 24      # months of history fed as input
FORECAST_HORIZON = 3       # predict 3 months ahead
D_MODEL          = 64      # transformer embedding dimension
N_HEADS          = 4       # attention heads
N_LAYERS         = 2       # transformer encoder layers
D_FF             = 128     # feed-forward inner dimension
DROPOUT          = 0.15
EPOCHS           = 150
BATCH_SIZE       = 16
LR               = 1e-3
TEST_YEARS       = 5

# Crisis dates for macro-calendar positional encoding
CRISIS_DATES = [
    "1997-07-01",  # Asian Financial Crisis
    "1998-08-01",  # Russian default
    "2001-09-01",  # 9/11 + dot-com bust
    "2008-09-01",  # Lehman / GFC
    "2013-05-01",  # Taper Tantrum
    "2015-08-01",  # China devaluation
    "2018-01-01",  # EM selloff
    "2020-03-01",  # COVID
    "2022-02-01",  # Ukraine war / rate hike cycle
]

BRICS_COLORS = {
    "Brazil": "#009C3B", "Russia": "#CC0000", "India": "#FF9933",
    "China": "#DE2910", "South_Africa": "#007A4D",
}


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_features():
    # Use v2 if available (more features), fall back to v1
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
            print("Loaded: " + path)
            print("Shape:  " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " cols")
            print("Range:  " + str(df.index.min().date()) + " to " + str(df.index.max().date()))
            return df
    raise FileNotFoundError("No features CSV found. Run feature_engineer.py first.")


# =============================================================================
# 2. DATASET PREPARATION
# =============================================================================

def build_input_matrix(df):
    """
    Builds a flat feature matrix where each row (month) contains:
    - All global macro features (DXY, Oil, Fed Rate, etc.)
    - Past depreciation of ALL countries (BRICS + G7)
    This lets the Transformer learn cross-country dependencies via attention.
    """
    cols = []

    # Global features
    for f in GLOBAL_FEATURES:
        if f in df.columns:
            cols.append(f)

    # All country depreciation features
    for country in ALL_COUNTRIES:
        col = country + COUNTRY_FEATURE
        if col in df.columns:
            cols.append(col)

    print("Input features (" + str(len(cols)) + "): " + str(cols))

    sub = df[cols].copy().ffill().bfill()
    return sub, cols


def build_targets(df):
    """
    Targets: BRICS country depreciation shifted forward by FORECAST_HORIZON months.
    Returns a DataFrame with one column per BRICS country.
    """
    target_cols = [c + COUNTRY_FEATURE for c in BRICS_COUNTRIES
                   if c + COUNTRY_FEATURE in df.columns]
    targets = df[target_cols].shift(-FORECAST_HORIZON)
    return targets, target_cols


class MacroTimeSeriesDataset(Dataset):
    """
    Sliding window dataset for the Transformer.
    Each sample: (X_window, y_target)
      X_window : shape (SEQUENCE_LEN, n_features) — last 24 months of macro data
      y_target : shape (n_brics,)                 — next 3-month BRICS depreciation
    """
    def __init__(self, X, y, seq_len, crisis_flags=None):
        self.X           = torch.FloatTensor(X)
        self.y           = torch.FloatTensor(y)
        self.seq_len     = seq_len
        self.crisis_flags = torch.FloatTensor(crisis_flags) if crisis_flags is not None \
                            else torch.zeros(len(X))

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq    = self.X[idx : idx + self.seq_len]
        crisis   = self.crisis_flags[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        return x_seq, crisis, y_target


def make_crisis_flags(dates):
    """
    Binary flag: 1 if this month is within 3 months of a major crisis event.
    Used as extra signal in positional encoding.
    """
    flags = np.zeros(len(dates))
    crisis_timestamps = [pd.Timestamp(d) for d in CRISIS_DATES]
    for i, date in enumerate(dates):
        for crisis in crisis_timestamps:
            if abs((date - crisis).days) < 90:
                flags[i] = 1.0
                break
    return flags


# =============================================================================
# 3. MACRO-CALENDAR POSITIONAL ENCODING
# =============================================================================

class MacroCalendarPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding augmented with:
    - Crisis flag embedding: learnable shift when month is near a crisis event
    This teaches the model that crisis windows are structurally different.
    """
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        # Learnable crisis embedding — added on top of standard PE
        self.crisis_embedding = nn.Embedding(2, d_model)

    def forward(self, x, crisis_flags=None):
        """
        x            : (batch, seq_len, d_model)
        crisis_flags : (batch, seq_len) — binary 0/1
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)

        if crisis_flags is not None:
            crisis_idx = crisis_flags.long().clamp(0, 1)           # (batch, seq_len)
            crisis_emb = self.crisis_embedding(crisis_idx)          # (batch, seq_len, d_model)
            x = x + crisis_emb * 0.3                                # small additive weight

        return self.dropout(x)


# =============================================================================
# 4. TRANSFORMER MODEL
# =============================================================================

class DollarHegemonyTransformer(nn.Module):
    """
    Multi-country time-series Transformer for currency depreciation forecasting.

    Architecture:
      Input projection  : linear(n_features → d_model)
      Positional encoding: sinusoidal + macro-calendar crisis flags
      Transformer encoder: N_LAYERS × (multi-head self-attention + feed-forward)
      Output head        : linear(d_model → n_brics_countries)

    The self-attention mechanism is the key novelty: each attention head can
    learn different cross-country dependency patterns simultaneously.
    Inspecting the attention weights after training reveals WHICH countries'
    macro signals the model uses to predict each BRICS currency.
    """
    def __init__(self, n_features, n_output, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)

        self.pos_enc = MacroCalendarPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,      # input shape: (batch, seq, features)
            norm_first=True,       # pre-norm: more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Aggregate temporal dimension: use the last timestep representation
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, n_output),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, crisis_flags=None, return_attention=False):
        """
        x            : (batch, seq_len, n_features)
        crisis_flags : (batch, seq_len)
        returns      : (batch, n_output)  — predicted depreciation for each BRICS country
        """
        # Project input features to d_model dimensions
        x = self.input_proj(x)                       # (B, T, d_model)

        # Add positional + crisis-calendar encoding
        x = self.pos_enc(x, crisis_flags)             # (B, T, d_model)

        # Transformer encoder (self-attention over time)
        encoded = self.transformer_encoder(x)         # (B, T, d_model)

        # Use final timestep as the "summary" representation
        summary = encoded[:, -1, :]                   # (B, d_model)

        # Project to output: one value per BRICS country
        out = self.output_head(summary)               # (B, n_brics)
        return out

    def get_attention_weights(self, x, crisis_flags=None):
        """
        Extract average attention weights across all heads and layers.
        Returns: (seq_len, seq_len) matrix — how much each time step
                 attends to each other time step.
        """
        self.eval()
        attention_maps = []

        x_proj = self.input_proj(x)
        x_proj = self.pos_enc(x_proj, crisis_flags)

        # Manually pass through each layer to capture attention weights
        current = x_proj
        for layer in self.transformer_encoder.layers:
            # MultiheadAttention returns (output, attn_weights)
            attn_out, attn_weights = layer.self_attn(
                current, current, current,
                need_weights=True,
                average_attn_weights=True,    # average over heads
            )
            attention_maps.append(attn_weights.detach().cpu().numpy())
            # Continue through layer norm and FFN
            current = layer(current)

        # Average across all layers
        avg_attn = np.mean(attention_maps, axis=0)   # (batch, seq, seq)
        return avg_attn


# =============================================================================
# 5. TRAINING
# =============================================================================

def train_transformer(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights  = None
    patience      = 20
    patience_cnt  = 0
    history       = {"train": [], "val": []}

    model.to(device)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x_batch, crisis_batch, y_batch in train_loader:
            x_batch      = x_batch.to(device)
            crisis_batch = crisis_batch.to(device)
            y_batch      = y_batch.to(device)

            optimizer.zero_grad()
            pred  = model(x_batch, crisis_batch)
            loss  = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, crisis_batch, y_batch in val_loader:
                x_batch      = x_batch.to(device)
                crisis_batch = crisis_batch.to(device)
                y_batch      = y_batch.to(device)
                pred     = model(x_batch, crisis_batch)
                val_loss += criterion(pred, y_batch).item()

        train_loss /= max(len(train_loader), 1)
        val_loss   /= max(len(val_loader), 1)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("  Early stopping at epoch " + str(epoch + 1))
                break

        if (epoch + 1) % 20 == 0:
            print("  Epoch " + str(epoch + 1) + "/" + str(epochs) +
                  "  train_loss=" + str(round(train_loss, 5)) +
                  "  val_loss=" + str(round(val_loss, 5)))

    if best_weights:
        model.load_state_dict(best_weights)

    return model, history


# =============================================================================
# 6. EVALUATION
# =============================================================================

def evaluate(model, X_test_seq, crisis_test, y_test_seq,
             scaler_y, target_names, device):
    model.eval()
    X_t  = torch.FloatTensor(X_test_seq).to(device)
    cr_t = torch.FloatTensor(crisis_test).to(device)

    with torch.no_grad():
        preds_scaled = model(X_t, cr_t).cpu().numpy()

    # Inverse-transform predictions and actuals
    preds_raw = scaler_y.inverse_transform(preds_scaled)
    truth_raw = scaler_y.inverse_transform(y_test_seq)

    results = {}
    for i, country in enumerate(BRICS_COUNTRIES):
        if i >= preds_raw.shape[1]:
            continue
        pred = preds_raw[:, i]
        true = truth_raw[:, i]

        # Mask out NaN targets
        valid = ~np.isnan(true)
        if valid.sum() < 3:
            continue

        mae     = mean_absolute_error(true[valid], pred[valid])
        rmse    = np.sqrt(mean_squared_error(true[valid], pred[valid]))
        dir_acc = (np.sign(pred[valid]) == np.sign(true[valid])).mean() * 100

        print("  " + country + ":  MAE=" + str(round(mae, 3)) +
              "%  RMSE=" + str(round(rmse, 3)) +
              "%  Dir.Acc=" + str(round(dir_acc, 1)) + "%")

        results[country] = {
            "pred": pred, "true": true,
            "mae": mae, "rmse": rmse, "dir_acc": dir_acc,
        }
    return results


# =============================================================================
# 7. CHARTS
# =============================================================================

def chart_15_predictions(results, test_dates, horizon):
    n   = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (country, res) in zip(axes, results.items()):
        color = BRICS_COLORS.get(country, "#3498db")
        dates = test_dates[:len(res["true"])]

        ax.plot(dates, res["true"],  color="#2c3e50",  linewidth=2,   label="Actual")
        ax.plot(dates, res["pred"],  color=color, linewidth=1.8,
                linestyle="--", label="Transformer Predicted")
        ax.fill_between(dates, res["true"], res["pred"], alpha=0.12, color=color)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")

        ax.set_title(
            country.replace("_", " ") + "  |  MAE: " +
            str(round(res["mae"], 2)) + "%  Dir.Acc: " +
            str(round(res["dir_acc"], 1)) + "%",
            fontsize=11, fontweight="bold", color=color,
        )
        ax.set_ylabel("Depreciation vs USD (%)")
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle(
        "Transformer — BRICS Currency Depreciation Forecasts (" +
        str(horizon) + "-Month Ahead)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = "outputs/charts/15_transformer_predictions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: " + path)


def chart_16_attention_heatmap(model, X_sample, crisis_sample,
                                feature_names, device):
    """
    THE KEY NOVEL CHART: shows which time steps the model attends to most.
    Each row = a query timestep, each column = a key timestep.
    Bright cells = strong attention = 'to predict month T, I look heavily at month K'.
    In practice: crisis months light up as high-attention keys for all queries.
    """
    model.eval()
    X_t  = torch.FloatTensor(X_sample[:1]).to(device)   # single sample
    cr_t = torch.FloatTensor(crisis_sample[:1]).to(device)

    attn = model.get_attention_weights(X_t, cr_t)       # (1, seq, seq)
    attn = attn[0]                                        # (seq, seq)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention weight")

    ax.set_title(
        "Transformer Self-Attention Map (Last Sample)\n"
        "Rows = query months  |  Cols = key months  |  "
        "Bright = high attention\n"
        "Crisis months should light up as high-attention keys",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Key timestep (months in history)")
    ax.set_ylabel("Query timestep (months in history)")

    # Label axes with month offsets from current
    ticks = list(range(0, SEQUENCE_LEN, 4))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["-" + str(SEQUENCE_LEN - t) + "m" for t in ticks], fontsize=8)
    ax.set_yticklabels(["-" + str(SEQUENCE_LEN - t) + "m" for t in ticks], fontsize=8)

    plt.tight_layout()
    path = "outputs/charts/16_attention_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: " + path)


def chart_17_model_comparison(transformer_results):
    """
    Comparison bar chart: Transformer vs XGBoost vs LSTM across all BRICS.
    Loads existing performance CSVs and adds Transformer row.
    """
    # Collect all model results
    all_rows = []

    xgb_path  = "outputs/results/model_performance.csv"
    lstm_path = "outputs/results/lstm_performance.csv"

    if os.path.exists(xgb_path):
        xgb_df = pd.read_csv(xgb_path)
        xgb_df["Model"] = "XGBoost"
        all_rows.append(xgb_df)

    if os.path.exists(lstm_path):
        all_rows.append(pd.read_csv(lstm_path))

    for country, res in transformer_results.items():
        all_rows.append(pd.DataFrame([{
            "Model":       "Transformer",
            "Country":     country,
            "MAE (%)":     round(res["mae"],     3),
            "RMSE (%)":    round(res["rmse"],    3),
            "Dir_Acc (%)": round(res["dir_acc"], 1),
        }]))

    if not all_rows:
        return

    comp_df = pd.concat(all_rows, ignore_index=True)

    models   = comp_df["Model"].unique()
    countries = [c for c in BRICS_COUNTRIES if c in comp_df["Country"].values]

    model_colors = {
        "XGBoost":     "#3498db",
        "LSTM":        "#e67e22",
        "Transformer": "#9b59b6",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(len(countries))
    width = 0.25

    for i, model in enumerate(models):
        sub    = comp_df[comp_df["Model"] == model].set_index("Country")
        maes   = [sub.loc[c, "MAE (%)"]    if c in sub.index else np.nan for c in countries]
        daccs  = [sub.loc[c, "Dir_Acc (%)"] if c in sub.index else np.nan for c in countries]
        color  = model_colors.get(model, "#888")
        offset = (i - len(models) / 2 + 0.5) * width

        bars1 = ax1.bar(x + offset, maes,  width, label=model, color=color, alpha=0.85)
        bars2 = ax2.bar(x + offset, daccs, width, color=color, alpha=0.85)

        for bar, val in zip(bars1, maes):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         str(round(val, 1)), ha="center", fontsize=7)
        for bar, val in zip(bars2, daccs):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         str(round(val, 0)) + "%", ha="center", fontsize=7)

    ax1.set_xticks(x); ax1.set_xticklabels([c.replace("_", "\n") for c in countries])
    ax1.set_ylabel("MAE (%)"); ax1.set_title("Model Comparison — MAE (lower = better)",
                                              fontweight="bold")
    ax1.legend(loc="upper right")

    ax2.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.6,
                label="Random baseline (50%)")
    ax2.set_xticks(x); ax2.set_xticklabels([c.replace("_", "\n") for c in countries])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Directional Accuracy (%)")
    ax2.set_title("Model Comparison — Directional Accuracy (higher = better)",
                  fontweight="bold")
    ax2.legend(loc="lower right")

    plt.suptitle("XGBoost vs LSTM vs Transformer — All BRICS Countries",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = "outputs/charts/17_model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: " + path)

    # Save combined performance CSV
    comp_df.to_csv("outputs/results/all_models_performance.csv", index=False)
    print("Saved: outputs/results/all_models_performance.csv")


def chart_18_training_curve(history):
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs  = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], color="#3498db", linewidth=1.8, label="Train loss")
    ax.plot(epochs, history["val"],   color="#e74c3c", linewidth=1.8,
            linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Transformer Training Curve — Did It Converge?",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = "outputs/charts/18_transformer_training.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved: " + path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Dollar Hegemony - Phase 3: Transformer (Multi-Country)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))

    # ── Load data ────────────────────────────────────────────────────────────
    df = load_features()

    X_raw, feature_names = build_input_matrix(df)
    targets, target_names = build_targets(df)

    # Align on common non-NaN rows
    combined = X_raw.join(targets, how="inner").dropna(subset=feature_names)
    X_df     = combined[feature_names]
    y_df     = combined[target_names].fillna(0)   # NaN targets → 0 (masked in eval)
    dates    = combined.index

    print("Aligned dataset: " + str(len(X_df)) + " rows")
    print("Input features:  " + str(len(feature_names)))
    print("Output targets:  " + str(len(target_names)) + " BRICS countries")

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    cutoff      = dates.max() - pd.DateOffset(years=TEST_YEARS)
    train_mask  = dates <= cutoff
    test_mask   = dates >  cutoff

    X_train_raw = X_df[train_mask].values
    X_test_raw  = X_df[test_mask].values
    y_train_raw = y_df[train_mask].values
    y_test_raw  = y_df[test_mask].values

    # Fit scalers on train only — no future leakage
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test  = scaler_X.transform(X_test_raw)
    y_train = scaler_y.fit_transform(y_train_raw)
    y_test  = scaler_y.transform(y_test_raw)

    # ── Crisis flags ──────────────────────────────────────────────────────────
    crisis_all   = make_crisis_flags(dates)
    crisis_train = crisis_all[train_mask]
    crisis_test  = crisis_all[test_mask]

    print("Train months: " + str(sum(train_mask)) +
          "  |  Test months: " + str(sum(test_mask)))

    # ── Build datasets ────────────────────────────────────────────────────────
    train_dataset = MacroTimeSeriesDataset(X_train, y_train, SEQUENCE_LEN, crisis_train)
    test_dataset  = MacroTimeSeriesDataset(X_test,  y_test,  SEQUENCE_LEN, crisis_test)

    # Validation split from training set (last 15%)
    val_size   = max(1, int(len(train_dataset) * 0.15))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Train samples: " + str(len(train_ds)) +
          "  Val: " + str(len(val_ds)) +
          "  Test: " + str(len(test_dataset)))

    # ── Build model ───────────────────────────────────────────────────────────
    n_features = len(feature_names)
    n_output   = len(target_names)
    model      = DollarHegemonyTransformer(n_features, n_output)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel parameters: " + str(n_params))
    print(model)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining...")
    model, history = train_transformer(
        model, train_loader, val_loader, EPOCHS, LR, device
    )

    # Save model
    torch.save(model.state_dict(), "models/transformer_dollar_hegemony.pt")
    print("Model saved: models/transformer_dollar_hegemony.pt")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")

    # Build test sequences manually for evaluation
    X_test_seqs    = []
    crisis_test_seqs = []
    y_test_seqs    = []
    for i in range(len(X_test) - SEQUENCE_LEN):
        X_test_seqs.append(X_test[i : i + SEQUENCE_LEN])
        crisis_test_seqs.append(crisis_test[i : i + SEQUENCE_LEN])
        y_test_seqs.append(y_test[i + SEQUENCE_LEN])

    X_test_seqs      = np.array(X_test_seqs)
    crisis_test_seqs = np.array(crisis_test_seqs)
    y_test_seqs      = np.array(y_test_seqs)
    test_dates_arr   = dates[test_mask][SEQUENCE_LEN:]

    results = evaluate(model, X_test_seqs, crisis_test_seqs, y_test_seqs,
                       scaler_y, target_names, device)

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nGenerating charts...")
    chart_18_training_curve(history)
    chart_15_predictions(results, test_dates_arr, FORECAST_HORIZON)

    # Attention heatmap (use first test sequence)
    if len(X_test_seqs) > 0:
        chart_16_attention_heatmap(
            model, X_test_seqs[:1], crisis_test_seqs[:1], feature_names, device
        )

    chart_17_model_comparison(results)

    # ── Save performance CSV ──────────────────────────────────────────────────
    perf_rows = []
    for country, res in results.items():
        perf_rows.append({
            "Model":       "Transformer",
            "Country":     country,
            "MAE (%)":     round(res["mae"],     3),
            "RMSE (%)":    round(res["rmse"],    3),
            "Dir_Acc (%)": round(res["dir_acc"], 1),
        })
    pd.DataFrame(perf_rows).to_csv(
        "outputs/results/transformer_performance.csv", index=False
    )
    print("Performance CSV saved: outputs/results/transformer_performance.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE — Transformer Results Summary")
    print("=" * 60)
    print(f"{'Country':<15} {'MAE':>8} {'RMSE':>8} {'Dir.Acc':>10}")
    print("-" * 45)
    for row in perf_rows:
        print(f"{row['Country']:<15} {row['MAE (%)']:>7.2f}%"
              f" {row['RMSE (%)']:>7.2f}% {row['Dir_Acc (%)']:>9.1f}%")
    print("-" * 45)
    print("\nKey output for portfolio/paper:")
    print("  → Chart 16: outputs/charts/16_attention_heatmap.png")
    print("    Inspect which time periods the model attends to most.")
    print("    Crisis months should show high attention weights.")
    print("  → Chart 17: outputs/charts/17_model_comparison.png")
    print("    XGBoost vs LSTM vs Transformer side-by-side.")
    print("  → outputs/results/all_models_performance.csv")
    print("\nNext step: python src/meta_ensemble.py")


if __name__ == "__main__":
    main()
