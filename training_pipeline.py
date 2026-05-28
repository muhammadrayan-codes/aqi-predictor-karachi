"""
Training Pipeline - AQI Predictor (Karachi)
--------------------------------------------
1. Reads features from data/features.parquet
2. Cleans data
3. Trains Ridge, Random Forest, XGBoost for each of 3 targets
4. Logs metrics and models to MLflow (DagsHub)
5. Saves best model per target locally

Run: python training_pipeline.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from xgboost                 import XGBRegressor

from data_cleaning import clean_data

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH = os.path.join("data", "features.parquet")
MODELS_DIR    = "models"

MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

TARGETS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

FEATURE_COLS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust", "uv_index",
    "temperature", "humidity", "wind_speed", "wind_direction",
    "precipitation", "pressure",
    "hour", "day", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_change_rate", "aqi_rolling_3h", "aqi_rolling_24h", "pm25_rolling_3h",
    "aqi",
]


# ── MLflow Setup ──────────────────────────────────────────────────────────────

def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("aqi_karachi_forecasting")
    print(f"✅ MLflow connected → {MLFLOW_TRACKING_URI}")


# ── Load Features ─────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    print(f"📥 Loading features from {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"   → {len(df)} rows loaded")
    return df


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"   {name:20s} → RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ── Train for one target ──────────────────────────────────────────────────────

def train_for_target(df: pd.DataFrame, target: str):
    print(f"\n{'─'*55}")
    print(f"  Target: {target}")
    print(f"{'─'*55}")

    X = df[FEATURE_COLS].copy()
    y = df[target].copy()

    # Filter out zero-padded targets
    mask = y > 0
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    results  = []
    trained  = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{target}_{name.replace(' ', '_')}"):
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            metrics = evaluate(name, y_test, y_pred)
            results.append(metrics)
            trained[name] = model

            # Log to MLflow
            mlflow.log_param("model",   name)
            mlflow.log_param("target",  target)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test",  len(X_test))
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae",  metrics["mae"])
            mlflow.log_metric("r2",   metrics["r2"])
            mlflow.sklearn.log_model(model, artifact_path="model")

    # Pick best by RMSE
    best        = min(results, key=lambda x: x["rmse"])
    best_model  = trained[best["model"]]
    print(f"\n   🏆 Best: {best['model']} (RMSE={best['rmse']:.2f}, R²={best['r2']:.4f})")

    return best_model, best["model"], best


# ── Save Best Model ───────────────────────────────────────────────────────────

def save_best_model(model, model_name: str, target: str, metrics: dict):
    model_dir = os.path.join(MODELS_DIR, target)
    os.makedirs(model_dir, exist_ok=True)

    model_path   = os.path.join(model_dir, "model.pkl")
    metrics_path = os.path.join(model_dir, "metrics.json")

    joblib.dump(model, model_path)

    meta = {"model_name": model_name, "target": target, **metrics,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    with open(metrics_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   💾 Saved → {model_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training():
    print(f"\n{'='*55}")
    print(f"  AQI Training Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    setup_mlflow()

    df = load_features()
    df = clean_data(df)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"   → {len(df)} rows ready for training")

    for target in TARGETS:
        best_model, best_name, best_metrics = train_for_target(df, target)
        save_best_model(best_model, best_name, target, best_metrics)

    print(f"\n{'='*55}")
    print("  ✅ Training complete! Check MLflow UI on DagsHub.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_training()