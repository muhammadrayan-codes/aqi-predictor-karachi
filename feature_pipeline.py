"""
Feature Pipeline - AQI Predictor (Karachi)
------------------------------------------
1. Fetches raw air quality + weather data from Open-Meteo API
2. Computes features (time-based, derived, lag, dispersion)
3. Saves features to local data/features.parquet

Run manually:   python feature_pipeline.py
Run backfill:   python backfill_pipeline.py
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
LATITUDE  = float(os.getenv("LATITUDE", 24.8607))
LONGITUDE = float(os.getenv("LONGITUDE", 67.0011))
CITY      = os.getenv("CITY", "karachi")

DATA_DIR      = "data"
FEATURES_PATH = os.path.join(DATA_DIR, "features.parquet")

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL     = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL     = "https://archive-api.open-meteo.com/v1/archive"

# Summer/hot months for Karachi (March - October)
SUMMER_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10]


# ── Fetch Data ────────────────────────────────────────────────────────────────

def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch pollutant data from Open-Meteo Air Quality API."""
    params = {
        "latitude":  LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone",
            "dust", "ammonia", "aerosol_optical_depth",
            "us_aqi", "uv_index",
        ],
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "Asia/Karachi",
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.rename(columns={"time": "timestamp", "us_aqi": "aqi"}, inplace=True)
    return df


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch weather data from Open-Meteo.
    Uses archive API for past dates, forecast API for recent/current dates.
    """
    today = datetime.now(timezone.utc).date()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    params_base = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "hourly": [
            "temperature_2m", "relative_humidity_2m",
            "dewpoint_2m", "apparent_temperature",
            "precipitation", "rain", "snowfall", "weathercode",
            "cloudcover", "visibility",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "shortwave_radiation", "direct_radiation", "diffuse_radiation",
            "et0_fao_evapotranspiration",
            "soil_temperature_0cm", "soil_moisture_0_to_7cm",
            "surface_pressure",
            "boundary_layer_height", "vapour_pressure_deficit",
        ],
        "timezone":   "Asia/Karachi",
    }

    if end_dt < today:
        params = {**params_base, "start_date": start_date, "end_date": end_date}
        resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["hourly"])
    elif start_dt >= today:
        params = {**params_base, "start_date": start_date, "end_date": end_date}
        resp = requests.get(WEATHER_URL, params=params, timeout=30)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["hourly"])
    else:
        yesterday = today - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        today_str = today.strftime("%Y-%m-%d")

        # Fetch archive for past
        params_archive = {**params_base, "start_date": start_date, "end_date": yesterday_str}
        resp_arc = requests.get(ARCHIVE_URL, params=params_archive, timeout=30)
        resp_arc.raise_for_status()
        df_arc = pd.DataFrame(resp_arc.json()["hourly"])

        # Fetch forecast for recent/future
        params_forecast = {**params_base, "start_date": today_str, "end_date": end_date}
        resp_fore = requests.get(WEATHER_URL, params=params_forecast, timeout=30)
        resp_fore.raise_for_status()
        df_fore = pd.DataFrame(resp_fore.json()["hourly"])

        df = pd.concat([df_arc, df_fore], ignore_index=True)

    df["time"] = pd.to_datetime(df["time"])
    df.rename(columns={
        "time":                       "timestamp",
        "temperature_2m":             "temperature",
        "relative_humidity_2m":       "humidity",
        "dewpoint_2m":                "dewpoint",
        "apparent_temperature":       "apparent_temperature",
        "precipitation":              "precipitation",
        "rain":                       "rain",
        "snowfall":                   "snowfall",
        "weathercode":                "weathercode",
        "cloudcover":                 "cloudcover",
        "visibility":                 "visibility",
        "wind_speed_10m":             "wind_speed",
        "wind_direction_10m":         "wind_direction",
        "wind_gusts_10m":             "wind_gusts",
        "shortwave_radiation":        "solar_radiation",
        "direct_radiation":           "direct_radiation",
        "diffuse_radiation":          "diffuse_radiation",
        "et0_fao_evapotranspiration": "et0_fao_evapotranspiration",
        "soil_temperature_0cm":       "soil_temperature_0cm",
        "soil_moisture_0_to_7cm":     "soil_moisture_0_to_7cm",
        "surface_pressure":           "pressure",
        "boundary_layer_height":      "boundary_layer_height",
        "vapour_pressure_deficit":    "vapour_pressure_deficit",
    }, inplace=True)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_features(aq_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi"] = df["aqi"].fillna(0).astype(int)

    # ── STEP 1 (MUST BE FIRST): Zero-mask corrupt weather values ──────────────
    # Open-Meteo fills missing readings with 0.0 for physical variables where
    # 0 is impossible (e.g. pressure=0 hPa). If not fixed here, data_cleaning
    # will drop ~73% of rows because 0 fails the (900, 1100) hPa range check.
    CORRUPT_COLS = ["temperature", "pressure", "humidity", "apparent_temperature", "dewpoint"]
    for col in CORRUPT_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0.0, np.nan)
    # Linear interpolation naturally reconstructs the weather curve over time
    for col in CORRUPT_COLS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear").ffill().bfill()

    # ── STEP 2: Time features ──────────────────────────────────────────────────
    df["hour"]        = df["timestamp"].dt.hour
    df["day"]         = df["timestamp"].dt.day
    df["month"]       = df["timestamp"].dt.month   # kept — needed by save_features()
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding (replaces raw hour/day/month for tree models)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── STEP 3: AQI rolling features ──────────────────────────────────────────
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    # ── STEP 4: Derived features (on clean weather values) ────────────────────
    # Solar 3h rolling mean — photochemical ozone proxy
    if "solar_radiation" in df.columns:
        df["solar_rolling_3h"] = df["solar_radiation"].rolling(window=3, min_periods=1).mean()

    # PM2.5/PM10 ratio — combustion vs dust particle composition
    if "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 0.1)

    # Boundary Layer Height — controls pollution vertical mixing
    if "boundary_layer_height" in df.columns:
        df["blh_change_rate"] = df["boundary_layer_height"].diff().fillna(0.0)
        df["blh_rolling_3h"]  = df["boundary_layer_height"].rolling(window=3, min_periods=1).mean()

    # Vapour Pressure Deficit 3h rolling — drying power of air
    if "vapour_pressure_deficit" in df.columns:
        df["vpd_rolling_3h"] = df["vapour_pressure_deficit"].rolling(window=3, min_periods=1).mean()

    # ── STEP 5: Create targets (shift BEFORE dropping columns) ────────────────
    df["target_aqi_24h"] = df["aqi"].shift(-24).fillna(0.0).astype(float)
    df["target_aqi_48h"] = df["aqi"].shift(-48).fillna(0.0).astype(float)
    df["target_aqi_72h"] = df["aqi"].shift(-72).fillna(0.0).astype(float)

    # Drop last 72 rows — future targets are not yet real
    df = df.iloc[:-72].copy()

    # ── STEP 6: Drop dead-weight / redundant / collinear columns ─────────────
    COLUMNS_TO_DROP = [
        # 100% null from Open-Meteo
        "rain", "snowfall", "weathercode", "visibility",
        "direct_radiation", "diffuse_radiation", "et0_fao_evapotranspiration",
        "soil_temperature_0cm", "soil_moisture_0_to_7cm",
        # Zero-variance constant (Karachi has negligible ammonia readings)
        "ammonia",
        # Extreme collinearity with temperature
        "apparent_temperature", "dewpoint",
        # Unstable ratio features (zero-denominator traps)
        "wind_gusts", "wind_gust_ratio", "wind_blh_ratio", "solar_blh_ratio",
        # Redundant short-window rollings
        "humidity_deficit", "vapour_pressure_deficit",
        "aqi_rolling_3h", "pm25_rolling_3h", "heat_stress", "blh_rolling_3h",
        # Raw linear time axes (cyclic sin/cos kept instead)
        "unix_time", "hour", "day", "day_of_week",
        # NOTE: "month" is deliberately NOT dropped here — save_features() needs it
    ]
    df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], inplace=True, errors="ignore")

    # ── STEP 7: AQI lag features + dispersion index (momentum signals) ────────
    # IMPORTANT: these are left as NaN for the initial 72 rows.
    # The training pipeline calls dropna() on these columns before training,
    # which is the correct, leak-free way to handle them.
    if "aqi" in df.columns:
        df["aqi_lag_24h"] = df["aqi"].shift(24)
        df["aqi_lag_48h"] = df["aqi"].shift(48)
        df["aqi_lag_72h"] = df["aqi"].shift(72)
    if "wind_speed" in df.columns and "boundary_layer_height" in df.columns:
        df["dispersion_index"] = df["wind_speed"] * df["boundary_layer_height"]

    # ── STEP 8: Final fill — non-lag cols only; lag cols stay NaN ────────────
    LAG_COLS = {"aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h"}
    non_lag_cols = [c for c in df.columns if c not in LAG_COLS]
    df[non_lag_cols] = df[non_lag_cols].ffill().fillna(0)

    # ── STEP 9: Metadata ──────────────────────────────────────────────────────
    df["city"]      = CITY
    df["unix_time"] = df["timestamp"].astype(np.int64) // 10**9

    return df


# ── Save to Parquet ───────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)

    # Filter to summer months and recent 90-day window
    df = df[df["timestamp"].dt.month.isin(SUMMER_MONTHS)]
    df = df[df["timestamp"] >= cutoff]

    if os.path.exists(FEATURES_PATH):
        existing = pd.read_parquet(FEATURES_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["unix_time", "city"])
        combined = combined.sort_values("unix_time").reset_index(drop=True)

        # Re-apply window to combined dataset
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined[combined["timestamp"].dt.month.isin(SUMMER_MONTHS)]
        combined = combined[combined["timestamp"] >= cutoff]

        combined.to_parquet(FEATURES_PATH, index=False)
        print(f"[OK] Updated {FEATURES_PATH} -> {len(combined)} total rows (last 90 days of summer)")
    else:
        df.to_parquet(FEATURES_PATH, index=False)
        print(f"[OK] Created {FEATURES_PATH} -> {len(df)} rows (last 90 days of summer)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(start_date: str, end_date: str):
    print(f"\n{'='*55}")
    print(f"  AQI Feature Pipeline -- {start_date} to {end_date}")
    print(f"{'='*55}\n")

    print("[AQ]  Fetching air quality data...")
    aq_df = fetch_air_quality(start_date, end_date)
    print(f"   -> {len(aq_df)} hourly air quality records")

    print("[WX]  Fetching weather data...")
    weather_df = fetch_weather(start_date, end_date)
    print(f"   -> {len(weather_df)} hourly weather records")

    print("[FE]  Computing features...")
    features_df = compute_features(aq_df, weather_df)
    print(f"   -> {len(features_df)} feature rows, {len(features_df.columns)} columns")

    save_features(features_df)


if __name__ == "__main__":
    # Normal hourly run — last 5 days
    end_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    run_pipeline(start_date, end_date)