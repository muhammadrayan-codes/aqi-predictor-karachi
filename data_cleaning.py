"""
Data Cleaning - AQI Predictor (Karachi)
----------------------------------------
Cleans raw feature data loaded from the local Parquet feature store before training.

Steps:
1. Remove duplicates
2. Remove invalid AQI values (out of 0-500 US AQI scale)
3. Remove invalid pollutant readings
4. Impute missing values using median (not 0 — 0 is a valid reading)
5. Remove statistical outliers using IQR method

Usage:
    from data_cleaning import clean_data
    df_clean = clean_data(df)
"""

import pandas as pd
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# US AQI valid range
AQI_MIN, AQI_MAX = 0, 500

# Physical valid ranges for pollutants
POLLUTANT_RANGES = {
    "pm10":             (0, 1000),   # µg/m³
    "pm2_5":            (0, 500),    # µg/m³
    "carbon_monoxide":  (0, 50000),  # µg/m³
    "nitrogen_dioxide": (0, 1000),   # µg/m³
    "sulphur_dioxide":  (0, 1000),   # µg/m³
    "ozone":            (0, 500),    # µg/m³
    "dust":             (0, 5000),   # µg/m³
    "uv_index":         (0, 20),     # index
}

WEATHER_RANGES = {
    "temperature":     (-10, 55),    # °C — Karachi range
    "humidity":        (0, 100),     # %
    "wind_speed":      (0, 150),     # km/h
    "wind_direction":  (0, 360),     # degrees
    "precipitation":   (0, 200),     # mm
    "pressure":        (900, 1100),  # hPa
}

TARGET_COLS  = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
FEATURE_COLS = list(POLLUTANT_RANGES.keys()) + list(WEATHER_RANGES.keys())


# ── Cleaning Steps ────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["unix_time", "city"])
    after = len(df)
    if before - after > 0:
        print(f"   🗑  Removed {before - after} duplicate rows")
    return df


def remove_invalid_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where current AQI or any target is out of valid range."""
    before = len(df)

    # Current AQI
    df = df[(df["aqi"] >= AQI_MIN) & (df["aqi"] <= AQI_MAX)]

    # Target AQI values — keep rows where targets are in valid range or zero
    # (zero means target was padded, filter those out too)
    for target in TARGET_COLS:
        if target in df.columns:
            df = df[(df[target] >= AQI_MIN) & (df[target] <= AQI_MAX)]

    after = len(df)
    if before - after > 0:
        print(f"   🗑  Removed {before - after} rows with invalid AQI values")
    return df


def remove_invalid_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where pollutant or weather readings are physically impossible."""
    before = len(df)

    for col, (lo, hi) in {**POLLUTANT_RANGES, **WEATHER_RANGES}.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    after = len(df)
    if before - after > 0:
        print(f"   🗑  Removed {before - after} rows with invalid pollutant/weather readings")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using column median (never 0)."""
    missing_before = df[FEATURE_COLS].isnull().sum().sum()

    if missing_before == 0:
        print("   ✅ No missing values found")
        return df

    for col in FEATURE_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    print(f"   🔧 Imputed {missing_before} missing values using column medians")
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Remove statistical outliers using IQR method.
    Uses multiplier=3.0 (instead of standard 1.5) to be conservative —
    AQI data has natural spikes during pollution events that are valid.
    """
    before = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    after = len(df)
    if before - after > 0:
        print(f"   🗑  Removed {before - after} outlier rows (IQR x{multiplier})")
    return df


def sort_and_reset(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and reset index."""
    if "unix_time" in df.columns:
        df = df.sort_values("unix_time").reset_index(drop=True)
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Main Clean Function ───────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline. Call this after loading from the feature store,
    before passing data to the training pipeline.
    """
    print(f"\n{'─'*55}")
    print(f"  Data Cleaning")
    print(f"{'─'*55}")
    print(f"   📊 Input:  {len(df)} rows, {len(df.columns)} columns")

    df = remove_duplicates(df)
    df = remove_invalid_aqi(df)
    df = remove_invalid_pollutants(df)
    df = impute_missing(df)
    df = remove_outliers_iqr(df, cols=list(POLLUTANT_RANGES.keys()) + ["aqi"])
    df = sort_and_reset(df)

    print(f"   📊 Output: {len(df)} rows, {len(df.columns)} columns")
    print(f"{'─'*55}\n")

    return df


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run standalone to inspect cleaning results without training.
    Fetches from the local Parquet feature store and prints a summary.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    FEATURES_PATH = os.path.join("data", "features.parquet")

    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Feature file not found at {FEATURES_PATH}. Please run the feature or backfill pipeline first!")
    else:
        print(f"📥 Loading data from local features store ({FEATURES_PATH})...")
        df = pd.read_parquet(FEATURES_PATH)

        df_clean = clean_data(df)

        print("\n📈 Summary after cleaning:")
        print(df_clean[["aqi", "pm2_5", "temperature", "target_aqi_24h"]].describe().round(2))