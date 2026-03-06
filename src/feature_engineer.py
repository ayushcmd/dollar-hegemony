"""
Dollar Hegemony Project - Phase 1 (Part 3)
Feature Engineer v2 — World Bank + Stock Index Features

PURPOSE:
  Extends features.csv with engineered signals from the new data layers
  collected by data_collector_v2.py (World Bank fundamentals + BRICS stocks).
  Output: data/processed/features_v2.csv — drop-in upgrade for all models.

NEW FEATURES BUILT:
  Fundamentals (from World Bank annual → monthly forward-filled):
    - {country}_gdp_yoy       : GDP growth rate (already annual %, kept as-is)
    - {country}_debt_ratio    : Debt-to-GDP ratio level
    - {country}_debt_chg_1y   : 1-year change in debt ratio (rising debt = stress)
    - {country}_cpi_wb        : World Bank CPI inflation (complements US FRED CPI)
    - {country}_trade_bal     : Trade balance (% of GDP)
    - {country}_current_acct  : Current account balance (% of GDP)
    - {country}_ext_vuln      : External Vulnerability Index = -(trade_bal + current_acct)
                                Higher = more exposed to dollar squeeze

  Stock indices (BRICS monthly prices + returns):
    - {country}_stock_ret_12m : 12-month stock return (already in v2 CSV)
    - {country}_stock_vol_6m  : 6-month rolling volatility of monthly returns
                                (captures risk-off sentiment before FX moves)
    - {country}_stock_drawdown: Rolling 12-month drawdown from peak
                                (early warning of capital flight)

  Cross-country signals:
    - BRICS_gdp_avg           : Average BRICS GDP growth (global EM health)
    - BRICS_debt_avg          : Average BRICS debt-to-GDP
    - BRICS_stock_avg_ret     : Average BRICS stock 12m return (risk appetite)
    - DXY_vs_BRICS_gdp        : DXY momentum / BRICS GDP growth
                                (dollar strength relative to EM fundamentals)

HOW TO RUN:
  Run data_collector_v2.py first, then:
  python src/feature_engineer_v2.py

OUTPUT:
  data/processed/features_v2.csv   ← use this in model_xgboost.py and model_lstm.py
  outputs/charts/11_*.png           ← new EDA charts
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

BRICS = ["Brazil", "Russia", "India", "China", "South_Africa"]
G7    = ["Germany_EU", "Japan", "UK", "Canada"]

BRICS_COLORS = {
    "Brazil": "#009C3B", "Russia": "#CC0000", "India": "#FF9933",
    "China": "#DE2910", "South_Africa": "#007A4D",
}

WB_INDICATORS = ["GDP_growth", "Debt_to_GDP", "CPI_inflation",
                 "Trade_balance", "Current_acct"]

STOCK_COUNTRIES = ["Brazil", "India", "China", "South_Africa", "Russia"]


# =============================================================================
# 1. LOAD BASE FEATURES (from feature_engineer.py)
# =============================================================================

def load_base_features():
    path = "data/processed/features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "features.csv not found. Run feature_engineer.py first."
        )
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df.index.notna()].sort_index()
    print("Base features loaded: " + str(df.shape[0]) + " rows, " +
          str(df.shape[1]) + " cols")
    return df


# =============================================================================
# 2. LOAD v2 MASTER (has World Bank + stock data)
# =============================================================================

def load_v2_master():
    path = "data/processed/master_monthly_v2.csv"
    if not os.path.exists(path):
        print("WARNING: master_monthly_v2.csv not found.")
        print("         Run data_collector_v2.py first.")
        print("         Proceeding with base features only.\n")
        return pd.DataFrame()

    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)
    df.index.name = "Date"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df.index.notna()].sort_index()
    print("v2 master loaded:     " + str(df.shape[0]) + " rows, " +
          str(df.shape[1]) + " cols")
    return df


# =============================================================================
# 3. ENGINEER FUNDAMENTAL FEATURES
# =============================================================================

def build_fundamental_features(v2, feat):
    """
    World Bank data is annual, forward-filled to monthly in data_collector_v2.
    We build rate-of-change and composite vulnerability signals on top.
    """
    new = pd.DataFrame(index=feat.index)
    countries_with_wb = []

    for country in BRICS + G7:
        gdp_col   = country + "_GDP_growth"
        debt_col  = country + "_Debt_to_GDP"
        cpi_col   = country + "_CPI_inflation"
        trade_col = country + "_Trade_balance"
        curr_col  = country + "_Current_acct"

        has_any = False

        if gdp_col in v2.columns:
            new[country + "_gdp_yoy"] = v2[gdp_col].reindex(feat.index, method="ffill")
            has_any = True

        if debt_col in v2.columns:
            debt = v2[debt_col].reindex(feat.index, method="ffill")
            new[country + "_debt_ratio"]  = debt
            # 12-month change in debt ratio: rising = more vulnerable
            new[country + "_debt_chg_1y"] = debt.diff(12)

        if cpi_col in v2.columns:
            new[country + "_cpi_wb"] = v2[cpi_col].reindex(feat.index, method="ffill")

        if trade_col in v2.columns:
            new[country + "_trade_bal"] = v2[trade_col].reindex(feat.index, method="ffill")

        if curr_col in v2.columns:
            new[country + "_current_acct"] = v2[curr_col].reindex(feat.index, method="ffill")

        # External Vulnerability Index: negative current account + negative trade = twin deficit
        if trade_col in v2.columns and curr_col in v2.columns:
            trade = v2[trade_col].reindex(feat.index, method="ffill")
            curr  = v2[curr_col].reindex(feat.index, method="ffill")
            # Higher value = more externally vulnerable to dollar squeeze
            new[country + "_ext_vuln"] = -(trade.fillna(0) + curr.fillna(0))
            has_any = True

        if has_any and country in BRICS:
            countries_with_wb.append(country)

    # Cross-country BRICS aggregates
    gdp_cols  = [c + "_gdp_yoy"  for c in BRICS if c + "_gdp_yoy"  in new.columns]
    debt_cols = [c + "_debt_ratio" for c in BRICS if c + "_debt_ratio" in new.columns]

    if gdp_cols:
        new["BRICS_gdp_avg"]  = new[gdp_cols].mean(axis=1)
    if debt_cols:
        new["BRICS_debt_avg"] = new[debt_cols].mean(axis=1)

    # DXY vs BRICS GDP: dollar strengthening while EM growth slows = stress signal
    if "DXY_mom_12m" in feat.columns and "BRICS_gdp_avg" in new.columns:
        # Avoid divide-by-zero
        safe_gdp = new["BRICS_gdp_avg"].replace(0, np.nan)
        new["DXY_vs_BRICS_gdp"] = feat["DXY_mom_12m"] / safe_gdp

    n_cols = len([c for c in new.columns if new[c].notna().sum() > 10])
    print("Fundamental features built: " + str(n_cols) + " non-empty columns")
    print("Countries with World Bank data: " + str(countries_with_wb))
    return new


# =============================================================================
# 4. ENGINEER STOCK FEATURES
# =============================================================================

def build_stock_features(v2, feat):
    """
    Stock market signals are leading indicators for currency stress.
    A falling stock market precedes capital flight and FX depreciation.
    """
    new = pd.DataFrame(index=feat.index)
    countries_with_stocks = []

    for country in STOCK_COUNTRIES:
        price_col  = country + "_Stock"
        ret_col    = country + "_Stock_return_12m"

        price_series = None

        if price_col in v2.columns:
            price_series = v2[price_col].reindex(feat.index, method="ffill")
        elif price_col in feat.columns:
            price_series = feat[price_col]

        if price_series is not None and price_series.notna().sum() > 12:
            # Monthly returns
            monthly_ret = price_series.pct_change() * 100

            # 6-month rolling volatility (std of monthly returns)
            new[country + "_stock_vol_6m"] = (
                monthly_ret.rolling(6).std()
            )

            # 12-month drawdown from rolling peak
            rolling_peak = price_series.rolling(12, min_periods=1).max()
            drawdown     = (price_series - rolling_peak) / rolling_peak * 100
            new[country + "_stock_drawdown"] = drawdown

            countries_with_stocks.append(country)

        # 12m return already computed in data_collector_v2 — just carry forward
        if ret_col in v2.columns:
            new[country + "_stock_ret_12m"] = v2[ret_col].reindex(feat.index, method="ffill")

    # Average BRICS stock return (risk appetite signal)
    ret_cols = [c + "_stock_ret_12m" for c in BRICS if c + "_stock_ret_12m" in new.columns]
    if ret_cols:
        new["BRICS_stock_avg_ret"] = new[ret_cols].mean(axis=1)

    n_cols = len([c for c in new.columns if new[c].notna().sum() > 10])
    print("Stock features built:  " + str(n_cols) + " non-empty columns")
    print("Countries with stocks: " + str(countries_with_stocks))
    return new


# =============================================================================
# 5. MERGE EVERYTHING
# =============================================================================

def build_features_v2(feat, fund_feat, stock_feat):
    """Merge base features with fundamental and stock features."""
    merged = feat.copy()

    for new_feat in [fund_feat, stock_feat]:
        if new_feat.empty:
            continue
        new_cols = [c for c in new_feat.columns if c not in merged.columns]
        if new_cols:
            merged = merged.join(new_feat[new_cols], how="left")

    merged = merged.sort_index()
    print("\nFinal features_v2 shape: " +
          str(merged.shape[0]) + " rows x " + str(merged.shape[1]) + " cols")
    print("New cols added vs features.csv: " +
          str(merged.shape[1] - feat.shape[1]))
    return merged


# =============================================================================
# 6. EDA CHARTS
# =============================================================================

def chart_11_gdp_vs_depreciation(feat_v2):
    """Scatter: GDP growth vs currency depreciation for each BRICS country."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    fig.suptitle("BRICS: GDP Growth vs Currency Depreciation (Annual, 2000-2024)",
                 fontsize=13, fontweight="bold")

    for ax, country in zip(axes, BRICS):
        gdp_col  = country + "_gdp_yoy"
        depr_col = country + "_depr_12m"
        color    = BRICS_COLORS.get(country, "#888")

        if gdp_col not in feat_v2.columns or depr_col not in feat_v2.columns:
            ax.set_title(country + "\n(no data)", fontsize=9)
            ax.axis("off")
            continue

        tmp = feat_v2[[gdp_col, depr_col]].dropna()
        if len(tmp) < 5:
            ax.set_title(country + "\n(insufficient data)", fontsize=9)
            ax.axis("off")
            continue

        ax.scatter(tmp[gdp_col], tmp[depr_col],
                   color=color, alpha=0.55, s=18, edgecolors="white", linewidths=0.3)

        # Regression line
        z = np.polyfit(tmp[gdp_col], tmp[depr_col], 1)
        p = np.poly1d(z)
        xs = np.linspace(tmp[gdp_col].min(), tmp[gdp_col].max(), 60)
        ax.plot(xs, p(xs), color=color, linewidth=1.8, linestyle="--")

        ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
        ax.axvline(0, color="grey", linewidth=0.7, linestyle=":")
        ax.set_xlabel("GDP Growth (%)", fontsize=8)
        ax.set_ylabel("Depreciation vs USD (%)", fontsize=8)
        ax.set_title(country.replace("_", " "), fontsize=10,
                     fontweight="bold", color=color)

    plt.tight_layout()
    path = "outputs/charts/11_gdp_vs_depreciation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 11 saved: " + path)


def chart_12_external_vulnerability(feat_v2):
    """Line chart: External Vulnerability Index over time for BRICS."""
    vuln_cols = [c + "_ext_vuln" for c in BRICS if c + "_ext_vuln" in feat_v2.columns]
    if not vuln_cols:
        print("No External Vulnerability data available — skipping chart 12")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    for col in vuln_cols:
        country = col.replace("_ext_vuln", "")
        series  = feat_v2[col].dropna()
        if len(series) < 5:
            continue
        ax.plot(series.index, series.values,
                label=country.replace("_", " "),
                color=BRICS_COLORS.get(country, "#888"),
                linewidth=1.8, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("External Vulnerability Index — BRICS Nations\n"
                 "(Higher = More Exposed to Dollar Squeeze | = -(Trade Balance + Current Account))",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("External Vulnerability (%  of GDP)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    path = "outputs/charts/12_external_vulnerability.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print("Chart 12 saved: " + path)


def chart_13_stock_vs_fx(feat_v2):
    """Dual-axis: stock drawdown vs FX depreciation — leading indicator check."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()
    plot_countries = [c for c in ["Brazil", "India", "China", "South_Africa"]
                      if c + "_stock_drawdown" in feat_v2.columns
                      and c + "_depr_12m" in feat_v2.columns]

    for ax, country in zip(axes, plot_countries):
        color    = BRICS_COLORS.get(country, "#888")
        dd_col   = country + "_stock_drawdown"
        depr_col = country + "_depr_12m"

        tmp = feat_v2[[dd_col, depr_col]].dropna()
        if len(tmp) < 12:
            ax.set_title(country + " — insufficient data")
            continue

        ax2 = ax.twinx()
        ax.fill_between(tmp.index, tmp[dd_col], 0,
                        color=color, alpha=0.25, label="Stock Drawdown (%)")
        ax.plot(tmp.index, tmp[dd_col], color=color, linewidth=1.2, alpha=0.7)
        ax2.plot(tmp.index, tmp[depr_col], color="#2c3e50",
                 linewidth=1.8, linestyle="--", label="FX Depreciation (%)")

        ax.set_ylabel("Stock Drawdown (%)", color=color, fontsize=9)
        ax2.set_ylabel("FX Depreciation (%)", color="#2c3e50", fontsize=9)
        ax.set_title(country.replace("_", " ") + " — Stocks Lead FX?",
                     fontsize=11, fontweight="bold")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    # Hide unused subplots
    for i in range(len(plot_countries), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Stock Market Drawdowns vs FX Depreciation — Leading Indicator Test",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = "outputs/charts/13_stock_vs_fx.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 13 saved: " + path)


def chart_14_feature_completeness(feat_v2):
    """Heatmap showing data availability across features and time."""
    # Sample every 6 months to keep chart readable
    sampled = feat_v2.resample("6ME").last()

    # Only show columns with at least 10% data
    coverage = sampled.notna().mean()
    good_cols = coverage[coverage > 0.10].index.tolist()

    if len(good_cols) < 3:
        print("Not enough columns for completeness heatmap — skipping chart 14")
        return

    # Cap at 40 columns for readability
    show_cols = good_cols[:40]
    mask_df   = sampled[show_cols].notna().astype(int)

    fig, ax = plt.subplots(figsize=(16, max(6, len(show_cols) * 0.3)))
    sns.heatmap(mask_df.T, cmap="YlGn", cbar=False,
                linewidths=0.3, linecolor="#ddd", ax=ax,
                xticklabels=[str(d.year) for d in mask_df.index])
    ax.set_title("Data Availability Heatmap — features_v2.csv\n"
                 "(Green = data present | sampled every 6 months)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("Year")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    path = "outputs/charts/14_feature_completeness.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print("Chart 14 saved: " + path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Dollar Hegemony - Phase 1 Part 3: Feature Engineer v2")
    print("New: World Bank Fundamentals + Stock Market Features")
    print("=" * 60)

    # Load inputs
    feat = load_base_features()
    v2   = load_v2_master()

    if v2.empty:
        print("\nNo v2 data available — saving base features as features_v2.csv")
        feat.to_csv("data/processed/features_v2.csv")
        return

    # Build new feature blocks
    print("\nBuilding fundamental features (World Bank)...")
    fund_feat = build_fundamental_features(v2, feat)

    print("\nBuilding stock market features...")
    stock_feat = build_stock_features(v2, feat)

    # Merge all
    print("\nMerging all feature blocks...")
    feat_v2 = build_features_v2(feat, fund_feat, stock_feat)

    # Save
    feat_v2.to_csv("data/processed/features_v2.csv")
    print("SAVED: data/processed/features_v2.csv")

    # EDA Charts
    print("\nGenerating v2 EDA charts...")
    chart_11_gdp_vs_depreciation(feat_v2)
    chart_12_external_vulnerability(feat_v2)
    chart_13_stock_vs_fx(feat_v2)
    chart_14_feature_completeness(feat_v2)

    # Summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEER v2 COMPLETE")
    print("=" * 60)
    print("Output:  data/processed/features_v2.csv")
    print("Rows:    " + str(len(feat_v2)))
    print("Columns: " + str(len(feat_v2.columns)))
    print("")

    # Column breakdown
    categories = {
        "Base (Phase 1)":        [c for c in feat_v2.columns if c in feat.columns],
        "Fundamentals (WB)":     [c for c in feat_v2.columns if any(
                                   ind in c for ind in ["gdp", "debt", "cpi_wb",
                                                         "trade_bal", "current_acct",
                                                         "ext_vuln", "BRICS_gdp",
                                                         "BRICS_debt"])],
        "Stock market":          [c for c in feat_v2.columns if any(
                                   s in c for s in ["stock_ret", "stock_vol",
                                                      "stock_drawdown", "BRICS_stock"])],
        "Cross-country signals": [c for c in feat_v2.columns if
                                   "DXY_vs_BRICS" in c],
    }
    for cat, cols in categories.items():
        print(f"  {cat:<28} {len(cols):>3} columns")

    print("\nTo use in models, change FEATURES_PATH to:")
    print("  data/processed/features_v2.csv")
    print("\nNext step: python src/model_transformer.py")


if __name__ == "__main__":
    main()
