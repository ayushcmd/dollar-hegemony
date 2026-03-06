"""
Dollar Hegemony Project - Data Collector v2
World Bank + BRICS Stock Indices
Fixed: memory overflow bug in World Bank merge
"""

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")
os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

START_DATE = "2000-01-01"
END_DATE   = "2024-12-31"

WB_COUNTRIES = {
    "Brazil":       "BRA",
    "Russia":       "RUS",
    "India":        "IND",
    "China":        "CHN",
    "South_Africa": "ZAF",
    "Germany_EU":   "DEU",
    "Japan":        "JPN",
    "UK":           "GBR",
    "Canada":       "CAN",
}

WB_INDICATORS = {
    "GDP_growth":    "NY.GDP.MKTP.KD.ZG",
    "Debt_to_GDP":   "GC.DOD.TOTL.GD.ZS",
    "CPI_inflation": "FP.CPI.TOTL.ZG",
    "Trade_balance": "NE.RSB.GNFS.ZS",
    "Current_acct":  "BN.CAB.XOKA.GD.ZS",
}

BRICS_STOCK_TICKERS = {
    "Brazil_Stock": "^BVSP",
    "India_Stock":  "^NSEI",
    "China_Stock":  "000001.SS",
}


# =============================================================================
# WORLD BANK - FIXED VERSION
# =============================================================================

def download_world_bank():
    try:
        import wbgapi as wb
    except ImportError:
        print("WARNING: wbgapi not installed. Run: pip install wbgapi")
        return pd.DataFrame()

    print("Downloading World Bank fundamentals...")

    country_codes = list(WB_COUNTRIES.values())
    code_to_name  = {v: k for k, v in WB_COUNTRIES.items()}
    years         = list(range(2000, 2025))

    all_series = []

    for ind_name, ind_code in WB_INDICATORS.items():
        print("  Fetching: " + ind_name)
        try:
            # Fetch one indicator at a time — safer than bulk fetch
            raw = wb.data.DataFrame(ind_code, economy=country_codes, time=years)

            # wbgapi returns rows=economy, cols=YR2000 etc  OR  rows=time, cols=economy
            # Normalize: we want rows=year (int), cols=country_code
            if raw.index.name == "economy" or (hasattr(raw.index, 'name') and raw.index.name == "economy"):
                # rows are countries — transpose
                raw = raw.T
            elif "economy" in str(raw.index.names):
                raw = raw.reset_index()
                raw = raw.set_index("economy").T

            # Clean up index — convert YR2000 -> 2000
            def parse_year(val):
                try:
                    s = str(val).replace("YR", "").strip()
                    return int(s)
                except Exception:
                    return None

            raw.index = [parse_year(i) for i in raw.index]
            raw = raw[~pd.isnull(raw.index)]
            raw.index = raw.index.astype(int)

            # Keep only years we want
            raw = raw[raw.index.isin(years)]

            # Rename country codes to names
            raw = raw.rename(columns=code_to_name)

            # Keep only columns we know
            known_cols = [c for c in raw.columns if c in WB_COUNTRIES]
            raw = raw[known_cols]

            # Add indicator suffix
            raw = raw.rename(columns={c: c + "_" + ind_name for c in raw.columns})

            raw.index.name = "Year"
            print("    OK - " + str(len(raw)) + " years, " + str(len(raw.columns)) + " columns")
            all_series.append(raw)

        except Exception as e:
            print("    ERROR: " + str(e))
            continue

    if not all_series:
        print("  No World Bank data downloaded.")
        return pd.DataFrame()

    # Use pd.concat instead of join — avoids index explosion bug
    wb_df = pd.concat(all_series, axis=1)
    wb_df.index.name = "Year"

    # Remove completely empty rows
    wb_df = wb_df.dropna(how="all")

    wb_df.to_csv("data/raw/worldbank_fundamentals.csv")
    print("SAVED: data/raw/worldbank_fundamentals.csv - Shape: " + str(wb_df.shape))
    return wb_df


def worldbank_to_monthly(wb_df):
    if wb_df.empty:
        return pd.DataFrame()

    dates   = pd.date_range(START_DATE, END_DATE, freq="ME")
    monthly = pd.DataFrame(index=dates)
    monthly.index.name = "Date"

    for col in wb_df.columns:
        series = wb_df[col].dropna()
        vals = {}
        for date in dates:
            year = date.year
            if year in series.index:
                vals[date] = series[year]
            else:
                vals[date] = np.nan
        monthly[col] = pd.Series(vals)

    monthly = monthly.ffill()
    return monthly


# =============================================================================
# BRICS STOCKS
# =============================================================================

def download_single_stock(ticker, name):
    try:
        raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            print("  WARNING: No data for " + name)
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].iloc[:, 0]
        else:
            close = raw["Close"]
        close.index = pd.to_datetime(close.index)
        close.name = name
        print("  OK - " + name + ": " + str(len(close)) + " rows")
        return close
    except Exception as e:
        print("  ERROR - " + name + ": " + str(e))
        return None


def download_brics_stocks():
    print("Downloading BRICS stock indices...")
    frames = []
    for name, ticker in BRICS_STOCK_TICKERS.items():
        series = download_single_stock(ticker, name)
        if series is not None:
            frames.append(series)

    if not frames:
        print("  No stock data downloaded.")
        return pd.DataFrame()

    stocks = pd.concat(frames, axis=1)
    stocks.index = pd.to_datetime(stocks.index)
    stocks.index.name = "Date"
    stocks_monthly = stocks.resample("ME").last()

    for col in list(stocks_monthly.columns):
        stocks_monthly[col + "_return_12m"] = stocks_monthly[col].pct_change(12) * 100

    stocks_monthly.to_csv("data/raw/brics_stocks.csv")
    print("SAVED: data/raw/brics_stocks.csv - Shape: " + str(stocks_monthly.shape))
    return stocks_monthly


# =============================================================================
# MERGE
# =============================================================================

def merge_all():
    print("\nMerging all data into master_monthly_v2.csv...")

    master_path = "data/processed/master_monthly.csv"
    if not os.path.exists(master_path):
        print("ERROR: master_monthly.csv not found. Run data_collector.py first.")
        return

    master = pd.read_csv(master_path)
    date_col = master.columns[0]
    master[date_col] = pd.to_datetime(master[date_col], errors="coerce")
    master = master.set_index(date_col)
    master.index.name = "Date"
    print("  Loaded master_monthly.csv: " + str(master.shape))

    # World Bank
    wb_annual  = download_world_bank()
    wb_monthly = worldbank_to_monthly(wb_annual)

    # BRICS Stocks
    stocks = download_brics_stocks()

    # Merge using pd.concat — avoids memory explosion
    frames = [master]
    if not wb_monthly.empty:
        frames.append(wb_monthly)
        print("  World Bank monthly added: " + str(wb_monthly.shape))
    if not stocks.empty:
        frames.append(stocks)
        print("  BRICS stocks added: " + str(stocks.shape))

    merged = pd.concat(frames, axis=1)
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]

    merged.to_csv("data/processed/master_monthly_v2.csv")
    print("\nSAVED: data/processed/master_monthly_v2.csv")
    print("Total rows: " + str(len(merged)))
    print("Total columns: " + str(len(merged.columns)))

    new_cols = [c for c in merged.columns if c not in master.columns]
    print("\nNew columns added (" + str(len(new_cols)) + "):")
    for col in new_cols[:20]:
        non_null = merged[col].notna().sum()
        print("  " + col + ": " + str(non_null) + " values")
    if len(new_cols) > 20:
        print("  ... and " + str(len(new_cols) - 20) + " more")


def main():
    print("=" * 55)
    print("Dollar Hegemony - Data Collector v2")
    print("World Bank GDP, Debt, CPI + BRICS Stocks")
    print("=" * 55)

    merge_all()

    print("\n" + "=" * 55)
    print("DATA COLLECTION v2 COMPLETE!")
    print("Next: python src/feature_engineer_v2.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
