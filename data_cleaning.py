"""
Data Cleaning - AQI Predictor (Karachi)
----------------------------------------
Cleans raw feature data loaded from the local Parquet feature store before training.

Steps:
1. Remove duplicates
2. Remove invalid AQI values (out of 0-500 US AQI scale)
3. Remove invalid pollutant/weather readings (only checks columns that exist)
4. Impute missing values using median
5. Remove statistical outliers using IQR method

Usage:
    from data_cleaning import clean_data
    df_clean = clean_data(df)
"""

import pandas as pd
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

AQI_MIN, AQI_MAX = 0, 500

# Physical valid ranges for pollutants
POLLUTANT_RANGES = {
    "pm10":                  (0, 1000),   # µg/m³
    "pm2_5":                 (0, 500),    # µg/m³
    "carbon_monoxide":       (0, 50000),  # µg/m³
    "nitrogen_dioxide":      (0, 1000),   # µg/m³
    "sulphur_dioxide":       (0, 1000),   # µg/m³
    "ozone":                 (0, 500),    # µg/m³
    "dust":                  (0, 5000),   # µg/m³
    "aerosol_optical_depth": (0, 5),      # dimensionless AOD
    "uv_index":              (0, 20),     # index
}

# Physical valid ranges for weather — only columns kept by feature_pipeline
# NOTE: apparent_temperature, dewpoint, wind_gusts, vapour_pressure_deficit are
# dropped upstream; do NOT list them here or KeyError will occur.
WEATHER_RANGES = {
    "temperature":           (-10, 55),   # °C — Karachi range
    "humidity":              (5, 100),    # % — avoid erroneous 0% readings
    "wind_speed":            (0, 150),    # km/h
    "wind_direction":        (0, 360),    # degrees
    "precipitation":         (0, 200),    # mm
    "pressure":              (900, 1100), # hPa — zero-mask already applied upstream
    "boundary_layer_height": (0, 5000),   # metres
}

TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

# Columns checked for NaN imputation — only core measurable fields.
# Derived / engineered columns are excluded here.
IMPUTE_COLS = list(POLLUTANT_RANGES.keys()) + list(WEATHER_RANGES.keys())


# ── Cleaning Steps ────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    if "unix_time" in df.columns and "city" in df.columns:
        df = df.drop_duplicates(subset=["unix_time", "city"])
    elif "timestamp" in df.columns:
        df = df.drop_duplicates(subset=["timestamp"])
    after = len(df)
    if before - after > 0:
        print(f"   [DEL] Removed {before - after} duplicate rows")
    return df


def remove_invalid_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where current AQI or any target is out of valid range."""
    before = len(df)

    df = df[(df["aqi"] >= AQI_MIN) & (df["aqi"] <= AQI_MAX)]

    for target in TARGET_COLS:
        if target in df.columns:
            # Keep rows where target is in valid range (filter out zero-padded rows too)
            df = df[(df[target] > AQI_MIN) & (df[target] <= AQI_MAX)]

    after = len(df)
    if before - after > 0:
        print(f"   [DEL] Removed {before - after} rows with invalid AQI/target values")
    return df


def remove_invalid_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where pollutant or weather readings are physically impossible.
    Only checks columns that actually exist in the dataframe — safe to call
    even after feature_pipeline has dropped collinear columns."""
    before = len(df)

    all_ranges = {**POLLUTANT_RANGES, **WEATHER_RANGES}
    for col, (lo, hi) in all_ranges.items():
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    after = len(df)
    if before - after > 0:
        print(f"   [DEL] Removed {before - after} rows with physically impossible readings")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using column median (never 0)."""
    # Only check columns that exist in this dataframe
    check_cols = [c for c in IMPUTE_COLS if c in df.columns]
    missing_before = df[check_cols].isnull().sum().sum()

    if missing_before == 0:
        print("   [OK] No missing values found in core feature columns")
        return df

    for col in check_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"   [FIX] Imputed {missing_before} missing values using column medians")
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Remove statistical outliers using IQR method.
    multiplier=3.0 (conservative) preserves valid AQI spike events.
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
        print(f"   [DEL] Removed {before - after} outlier rows (IQR x{multiplier})")
    return df


def sort_and_reset(df: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and reset index."""
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

    Note: Zero-masking of corrupt weather values is handled upstream in
    feature_pipeline.compute_features() so pressure/temperature/humidity
    are already valid here.
    """
    print(f"\n{'-'*55}")
    print(f"  Data Cleaning")
    print(f"{'-'*55}")
    print(f"   Input:  {len(df)} rows, {len(df.columns)} columns")

    df = remove_duplicates(df)
    df = remove_invalid_aqi(df)
    df = remove_invalid_readings(df)
    df = impute_missing(df)
    df = remove_outliers_iqr(df, cols=list(POLLUTANT_RANGES.keys()) + ["aqi"])
    df = sort_and_reset(df)

    print(f"   Output: {len(df)} rows, {len(df.columns)} columns")
    print(f"{'-'*55}\n")

    return df


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    FEATURES_PATH = os.path.join("data", "features.parquet")

    if not os.path.exists(FEATURES_PATH):
        print(f"Feature file not found at {FEATURES_PATH}. Run backfill_pipeline.py first.")
    else:
        print(f"Loading data from {FEATURES_PATH}...")
        df = pd.read_parquet(FEATURES_PATH)
        df_clean = clean_data(df)
        print("\nSummary after cleaning:")
        show_cols = [c for c in ["aqi", "pm2_5", "temperature", "target_aqi_24h"] if c in df_clean.columns]
        print(df_clean[show_cols].describe().round(2))