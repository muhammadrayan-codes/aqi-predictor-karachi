"""
Backfill Pipeline - AQI Predictor (Karachi)
--------------------------------------------
Fetches historical data from 2022-01-01 to 4 days ago
and saves to data/features.parquet

Run: python backfill_pipeline.py
"""

import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from feature_pipeline import fetch_air_quality, fetch_weather, compute_features, save_features

load_dotenv()


def run_backfill(start_date: str, end_date: str):
    print(f"\n{'='*55}")
    print(f"  AQI Backfill Pipeline — {start_date} to {end_date}")
    print(f"{'='*55}\n")

    print("📡 Fetching historical air quality data...")
    aq_df = fetch_air_quality(start_date, end_date)
    print(f"   → {len(aq_df)} hourly air quality records")

    print("🌤  Fetching historical weather data...")
    weather_df = fetch_weather(start_date, end_date)
    print(f"   → {len(weather_df)} hourly weather records")

    print("⚙️  Computing features...")
    features_df = compute_features(aq_df, weather_df)
    print(f"   → {len(features_df)} feature rows, {len(features_df.columns)} columns")

    save_features(features_df)


if __name__ == "__main__":
    end_date   = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
    start_date = "2022-01-01"
    run_backfill(start_date, end_date)