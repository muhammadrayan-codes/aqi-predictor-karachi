"""
Backfill Pipeline - AQI Predictor (Karachi)
--------------------------------------------
Fetches the last 90 days of SUMMER data (May-Sep) and saves to
data/features.parquet.  Run once to seed the feature store, then
let feature_pipeline.py keep it fresh on an hourly schedule.

Run: python backfill_pipeline.py
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from feature_pipeline import fetch_air_quality, fetch_weather, compute_features, save_features

load_dotenv()

# Summer/hot months for Karachi (March-October)
SUMMER_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10]


def run_backfill():
    # Date window: last 90 days
    end_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print(f"  AQI Backfill Pipeline -- {start_date} to {end_date}")
    print(f"  (90-day window | summer months {SUMMER_MONTHS})")
    print(f"{'='*55}\n")

    print("[AQ] Fetching historical air quality data...")
    aq_df = fetch_air_quality(start_date, end_date)
    print(f"   -> {len(aq_df)} hourly air quality records")

    print("[WX] Fetching historical weather data...")
    weather_df = fetch_weather(start_date, end_date)
    print(f"   -> {len(weather_df)} hourly weather records")

    print("[FE] Computing features...")
    features_df = compute_features(aq_df, weather_df)
    print(f"   -> {len(features_df)} feature rows before season filter")

    # Keep only summer months (derived from timestamp column)
    features_df = features_df[features_df["timestamp"].dt.month.isin(SUMMER_MONTHS)]
    print(f"   -> {len(features_df)} summer rows after season filter")

    save_features(features_df)


if __name__ == "__main__":
    run_backfill()