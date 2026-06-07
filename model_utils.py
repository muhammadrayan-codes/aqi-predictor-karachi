"""
model_utils.py – Lightweight model utilities for the Streamlit dashboard.

This module is intentionally free of mlflow / pyarrow imports so that
Streamlit Community Cloud (Python 3.14) can import it without hitting
the pyarrow binary-wheel build error.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone


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
