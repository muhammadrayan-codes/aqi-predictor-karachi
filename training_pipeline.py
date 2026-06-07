"""
Training Pipeline - AQI Predictor (Karachi)
--------------------------------------------
1. Reads features from data/features.parquet
2. Cleans data (zero-mask already applied by feature_pipeline)
3. Wraps estimators in AQIDeltaRegressor (predicts AQI delta to handle
   temporal mean-shifts and reduce overfitting)
4. Trains regularised Ridge, Random Forest, XGBoost for each of 3 targets
5. Logs three metric tiers to MLflow (DagsHub):
      - Per-fold metrics  (nested runs)
      - Avg fold metrics  (fold-average — biased downward on small datasets)
      - Pooled OOF metrics (all test predictions pooled — unbiased)
      - Final-fold metrics (fold 5 only — max training data, best signal)
6. Saves best model per target locally

Run: python training_pipeline.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Fix Windows cp1252 UnicodeEncodeError from MLflow emoji output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import mlflow.sklearn

from sklearn.base            import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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

# Columns that must exist in the feature matrix.
# Any column missing from the dataframe is silently skipped.
FEATURE_COLS = [
    # Core pollutants
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust", "aerosol_optical_depth", "uv_index",
    # Core weather (zero-masked and interpolated by feature_pipeline)
    "temperature", "humidity", "wind_speed", "wind_direction",
    "precipitation", "pressure", "boundary_layer_height",
    # Lag momentum features (key for multi-step forecasting)
    "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    # Dispersion index (wind_speed * boundary_layer_height)
    "dispersion_index",
    # Cyclical time encodings + weekend flag
    "hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend",
    # Rolling / rate features
    "aqi_change_rate", "aqi_rolling_24h",
    # Derived quality metrics
    "pm_ratio", "blh_change_rate", "vpd_rolling_3h", "solar_rolling_3h",
    # Current AQI — used by AQIDeltaRegressor at prediction time
    "aqi",
]


# ── AQI Delta Regressor ───────────────────────────────────────────────────────

class AQIDeltaRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper that trains on the *delta* between the
    target AQI and the current AQI, then reconstructs the absolute forecast
    at prediction time.

    Why delta?
    - AQI has a temporal mean-shift: summer 2024 baseline differs from
      summer 2025.  Tree models trained on raw targets overfit to the
      absolute level seen in training and fail on shifted test folds.
    - Predicting change (delta) makes the learning problem stationary.

    Fit:  y_delta = target_aqi - current_aqi
          base_estimator.fit(X, y_delta)

    Predict: aqi_pred = current_aqi + base_estimator.predict(X)
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y, aqi_current=None):
        """
        Parameters
        ----------
        X : array-like, feature matrix (must contain 'aqi' column or pass aqi_current)
        y : array-like, raw target AQI values
        aqi_current : array-like, optional override for current AQI column
        """
        self.estimator_ = clone(self.base_estimator)

        if aqi_current is None:
            if isinstance(X, pd.DataFrame) and "aqi" in X.columns:
                aqi_current = X["aqi"].values
            else:
                raise ValueError("AQIDeltaRegressor needs 'aqi' in X or aqi_current kwarg")

        y_delta = np.asarray(y) - np.asarray(aqi_current)
        self.estimator_.fit(X, y_delta)
        return self

    def predict(self, X, aqi_current=None):
        if aqi_current is None:
            if isinstance(X, pd.DataFrame) and "aqi" in X.columns:
                aqi_current = X["aqi"].values
            else:
                raise ValueError("AQIDeltaRegressor needs 'aqi' in X or aqi_current kwarg")

        delta_pred = self.estimator_.predict(X)
        return np.asarray(aqi_current) + delta_pred


# ── MLflow Setup ──────────────────────────────────────────────────────────────

def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME or ""
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD or ""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("aqi_karachi_forecasting")
    print(f"MLflow connected -> {MLFLOW_TRACKING_URI}")


# ── Load Features ─────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    print(f"Loading features from {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"   -> {len(df)} rows loaded")
    return df


# ── Temporal CV split ─────────────────────────────────────────────────────────

def temporal_split(X, n_splits=5):
    """
    Leak-free TimeSeriesSplit with gap=72.
    gap=72 ensures no overlap between training boundary and the 72h forecast
    window, preventing future targets from bleeding into training folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=72)
    return list(tscv.split(X))


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def print_metrics(label: str, m: dict):
    print(f"   {label:28s} -> RMSE: {m['rmse']:7.2f} | MAE: {m['mae']:6.2f} | R2: {m['r2']:7.4f}")


# ── Train for one target ──────────────────────────────────────────────────────

def train_for_target(df: pd.DataFrame, target: str):
    print(f"\n{'-'*60}")
    print(f"  Target: {target}")
    print(f"{'-'*60}")

    # ── Build feature matrix X ────────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLS if c in df.columns and c != target]
    excluded = set(TARGETS) | {"timestamp", "city", "unix_time", "month"}
    extra_numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in excluded and c not in available_features and c != target
    ]
    feature_cols = available_features + extra_numeric

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Keep only rows where target is valid (> 0)
    mask = y > 0
    X, y = X[mask], y[mask]

    if len(X) < 10:
        print(f"   [SKIP] Not enough rows ({len(X)}) for target {target}")
        return None, None, None

    splits = temporal_split(X, n_splits=5)
    print(f"   Features: {len(feature_cols)} | Rows: {len(X)} | Folds: {len(splits)}")

    # ── Model definitions (regularised to prevent overfitting) ────────────────
    #
    # AQIDeltaRegressor wraps each base estimator so training and inference
    # operate on the AQI delta rather than raw absolute values.
    #
    # Hyperparameter rationale
    # -------------------------
    # XGBoost:      max_depth=2  -> very shallow trees limit memorisation
    #               learning_rate=0.03 -> slow, careful learning
    #               reg_alpha/lambda -> L1+L2 shrinkage on leaf weights
    #               min_child_weight=10 -> require 10 samples to split
    # Random Forest: max_depth=4, min_samples_leaf=10 -> regularises similarly
    # Ridge:         alpha=50  -> strong L2 penalty on a normalised dataset
    base_models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=50.0)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=5.0,
            reg_lambda=20.0,
            min_child_weight=10,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    # Wrap each base model in AQIDeltaRegressor
    models = {name: AQIDeltaRegressor(base) for name, base in base_models.items()}

    # ── Cross-validation loop ─────────────────────────────────────────────────
    if mlflow.active_run():
        mlflow.end_run()

    results  = []
    trained  = {}

    for name, model in models.items():
        fold_rmses, fold_maes, fold_r2s = [], [], []

        # Accumulators for pooled OOF metrics
        oof_y_true_all = []
        oof_y_pred_all = []

        # Store last fold predictions for final-fold metrics
        final_fold_y_true = None
        final_fold_y_pred = None

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(X_train) < 5 or len(X_test) < 2:
                continue  # skip degenerate folds

            with mlflow.start_run(
                run_name=f"{target}__{name.replace(' ', '_')}__fold{fold}",
                nested=True,
            ):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fold_m = compute_metrics(y_test, y_pred)
                fold_rmses.append(fold_m["rmse"])
                fold_maes.append(fold_m["mae"])
                fold_r2s.append(fold_m["r2"])

                # Accumulate OOF predictions
                oof_y_true_all.extend(y_test.tolist())
                oof_y_pred_all.extend(y_pred.tolist())

                # Track final fold separately
                final_fold_y_true = y_test
                final_fold_y_pred = y_pred

                mlflow.log_param("model",  name)
                mlflow.log_param("target", target)
                mlflow.log_param("fold",   fold)
                mlflow.log_metric("rmse",  fold_m["rmse"])
                mlflow.log_metric("mae",   fold_m["mae"])
                mlflow.log_metric("r2",    fold_m["r2"])
                mlflow.sklearn.log_model(model, artifact_path="model")

        if not fold_rmses:
            continue

        # ── Three metric tiers ────────────────────────────────────────────────

        # 1. Fold-average (simple mean of per-fold scores)
        avg_m = {
            "rmse": float(np.mean(fold_rmses)),
            "mae":  float(np.mean(fold_maes)),
            "r2":   float(np.mean(fold_r2s)),
        }

        # 2. Pooled OOF (all test predictions concatenated — unbiased)
        pooled_m = compute_metrics(
            np.array(oof_y_true_all),
            np.array(oof_y_pred_all),
        )

        # 3. Final-fold only (maximum training data — best forward signal)
        final_m = compute_metrics(final_fold_y_true, final_fold_y_pred)

        print(f"\n   [{name}]")
        print_metrics("avg-fold   RMSE/MAE/R2", avg_m)
        print_metrics("pooled-OOF RMSE/MAE/R2", pooled_m)
        print_metrics("final-fold RMSE/MAE/R2", final_m)

        # Log all three tiers to the parent MLflow run
        if mlflow.active_run():
            tag = name.replace(" ", "_")
            for prefix, m in [("avg", avg_m), ("pooled", pooled_m), ("final_fold", final_m)]:
                mlflow.log_metric(f"{tag}_{prefix}_rmse", m["rmse"])
                mlflow.log_metric(f"{tag}_{prefix}_mae",  m["mae"])
                mlflow.log_metric(f"{tag}_{prefix}_r2",   m["r2"])

        # Primary ranking metric = pooled OOF RMSE (most reliable signal)
        results.append({
            "model":    name,
            "rmse":     pooled_m["rmse"],
            "mae":      pooled_m["mae"],
            "r2":       pooled_m["r2"],
            # Also store final-fold R2 for reporting
            "final_r2": final_m["r2"],
        })
        trained[name] = model

    if not results:
        print(f"   [WARN] No valid results for {target}")
        return None, None, None

    best    = min(results, key=lambda x: x["rmse"])
    best_r2 = max(results, key=lambda x: x["r2"])
    print(f"\n   [BEST-POOLED-RMSE] {best['model']} "
          f"(RMSE={best['rmse']:.2f}, R2={best['r2']:.4f}, final_R2={best['final_r2']:.4f})")
    print(f"   [BEST-POOLED-R2  ] {best_r2['model']} "
          f"(R2={best_r2['r2']:.4f}, RMSE={best_r2['rmse']:.2f})")
    return trained[best["model"]], best["model"], best


# ── Save Best Model ───────────────────────────────────────────────────────────

def save_best_model(model, model_name: str, target: str, metrics: dict):
    model_dir = os.path.join(MODELS_DIR, target)
    os.makedirs(model_dir, exist_ok=True)

    model_path   = os.path.join(model_dir, "model.pkl")
    metrics_path = os.path.join(model_dir, "metrics.json")

    joblib.dump(model, model_path)

    meta = {
        "model_name": model_name,
        "target":     target,
        **{k: v for k, v in metrics.items() if k != "model"},
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metric_basis": "pooled_oof",
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"   [SAVE] {model_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training(pretrain: bool = False):
    print(f"\n{'='*60}")
    print(f"  AQI Training Pipeline -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    setup_mlflow()

    df = load_features()

    # ── Temporal window filter (applied before cleaning) ──────────────────────
    if pretrain:
        # Use the full back‑filled 90‑day window (no additional cutoff).
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
    else:
        # Incremental training on the most recent 24 hours only.
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] >= cutoff]
    elif "unix_time" in df.columns:
        df = df[df["unix_time"] >= int(cutoff.timestamp())]

    # Summer months only — March through October
    if "month" in df.columns:
        df = df[df["month"].isin([3, 4, 5, 6, 7, 8, 9, 10])]
    elif "timestamp" in df.columns:
        df = df[df["timestamp"].dt.month.isin([3, 4, 5, 6, 7, 8, 9, 10])]

    print(f"   -> {len(df)} rows after temporal window filter")

    # ── Data cleaning ─────────────────────────────────────────────────────────
    df = clean_data(df)
    print(f"\n{'='*60}")
    print(f"  AQI Training Pipeline -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    setup_mlflow()

    df = load_features()

    # ── Temporal window filter (applied before cleaning) ──────────────────────
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] >= cutoff]
    elif "unix_time" in df.columns:
        df = df[df["unix_time"] >= int(cutoff.timestamp())]

    # Summer months only — March through October
    if "month" in df.columns:
        df = df[df["month"].isin([3, 4, 5, 6, 7, 8, 9, 10])]
    elif "timestamp" in df.columns:
        df = df[df["timestamp"].dt.month.isin([3, 4, 5, 6, 7, 8, 9, 10])]

    print(f"   -> {len(df)} rows after temporal window filter")

    # ── Data cleaning ─────────────────────────────────────────────────────────
    df = clean_data(df)

    # ── Drop rows where lag features are NaN ──────────────────────────────────
    lag_cols = [c for c in ["aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h"] if c in df.columns]
    if lag_cols:
        before = len(df)
        df = df.dropna(subset=lag_cols)
        print(f"   -> Dropped {before - len(df)} rows with missing lag features")

    print(f"   -> {len(df)} rows ready for training\n")

    if len(df) < 100:
        print("[ERROR] Not enough data to train. Run backfill_pipeline.py first.")
        return

    # ── Per-target training ───────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"aqi_training_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        for target in TARGETS:
            best_model, best_name, best_metrics = train_for_target(df, target)
            if best_model is not None:
                save_best_model(best_model, best_name, target, best_metrics)
                # Log and register the best model for this target in the Model Registry
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path=f"best_model_{target}",
                    registered_model_name=f"aqi_karachi_{target}"
                )

    print(f"\n{'='*60}")
    print("  [OK] Training complete! Check MLflow UI on DagsHub.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_training()