"""
Feature Pipeline - AQI Predictor (Karachi)
------------------------------------------
1. Fetches raw air quality + weather data from Open-Meteo API
2. Computes features (time-based + derived)
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

DATA_DIR     = "data"
FEATURES_PATH = os.path.join(DATA_DIR, "features.parquet")

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL     = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL     = "https://archive-api.open-meteo.com/v1/archive"


# ── Fetch Data ────────────────────────────────────────────────────────────────

def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude":  LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "us_aqi", "dust", "uv_index"
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
    today  = datetime.now(timezone.utc).date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    url    = ARCHIVE_URL if end_dt < today else WEATHER_URL

    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "precipitation", "surface_pressure"
        ],
        "start_date": start_date,
        "end_date":   end_date,
        "timezone":   "Asia/Karachi",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.rename(columns={
        "time":                 "timestamp",
        "temperature_2m":       "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m":       "wind_speed",
        "wind_direction_10m":   "wind_direction",
        "surface_pressure":     "pressure",
    }, inplace=True)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def compute_features(aq_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi"] = df["aqi"].fillna(0).astype(int)

    # Time-based features
    df["hour"]        = df["timestamp"].dt.hour
    df["day"]         = df["timestamp"].dt.day
    df["month"]       = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Derived features
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    df["aqi_rolling_3h"]  = df["aqi"].rolling(window=3,  min_periods=1).mean()
    df["aqi_rolling_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    df["pm25_rolling_3h"] = df["pm2_5"].rolling(window=3, min_periods=1).mean()

    # 3 targets: Day 1, Day 2, Day 3
    df["target_aqi_24h"] = df["aqi"].shift(-24).fillna(0.0).astype(float)
    df["target_aqi_48h"] = df["aqi"].shift(-48).fillna(0.0).astype(float)
    df["target_aqi_72h"] = df["aqi"].shift(-72).fillna(0.0).astype(float)

    # Drop last 72 rows — no valid targets yet
    df = df.iloc[:-72].copy()

    df = df.ffill().fillna(0)
    df["city"]      = CITY
    df["unix_time"] = df["timestamp"].astype(np.int64) // 10**9

    return df


# ── Save to Parquet ───────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(FEATURES_PATH):
        existing = pd.read_parquet(FEATURES_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["unix_time", "city"])
        combined = combined.sort_values("unix_time").reset_index(drop=True)
        combined.to_parquet(FEATURES_PATH, index=False)
        print(f"✅ Updated {FEATURES_PATH} → {len(combined)} total rows")
    else:
        df.to_parquet(FEATURES_PATH, index=False)
        print(f"✅ Created {FEATURES_PATH} → {len(df)} rows")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(start_date: str, end_date: str):
    print(f"\n{'='*55}")
    print(f"  AQI Feature Pipeline — {start_date} to {end_date}")
    print(f"{'='*55}\n")

    print("📡 Fetching air quality data...")
    aq_df = fetch_air_quality(start_date, end_date)
    print(f"   → {len(aq_df)} hourly air quality records")

    print("🌤  Fetching weather data...")
    weather_df = fetch_weather(start_date, end_date)
    print(f"   → {len(weather_df)} hourly weather records")

    print("⚙️  Computing features...")
    features_df = compute_features(aq_df, weather_df)
    print(f"   → {len(features_df)} feature rows, {len(features_df.columns)} columns")

    save_features(features_df)


if __name__ == "__main__":
    # Normal hourly run — last 5 days
    end_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    run_pipeline(start_date, end_date)