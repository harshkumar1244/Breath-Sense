"""
aqi_model.py — Core utilities for training and using a next-24h AQI forecaster.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

# ----------------------
# Data handling & features
# ----------------------

EXPECTED_TIME_COLS = ["datetime", "timestamp", "date", "time"]
EXPECTED_TARGET_COLS = ["AQI", "aqi", "PM2.5_AQI", "pm25_aqi"]

@dataclass
class TrainConfig:
    target_col: str = "AQI"
    dt_col: str = "datetime"
    freq: str = "H"
    lag_hours: int = 72                
    roll_windows: Tuple[int, ...] = (6, 12, 24, 48)
    n_splits: int = 5                  
    random_state: int = 42
    test_size_hours: int = 24*7        
    xgb_params: dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = dict(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=self.random_state,
                tree_method="hist",
            )

def smart_guess_columns(df: pd.DataFrame) -> Tuple[str, str]:
    dt_col = None
    for c in df.columns:
        if c.lower() in EXPECTED_TIME_COLS:
            dt_col = c
            break
    if dt_col is None:
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                dt_col = c
                break
    if dt_col is None:
        dt_col = df.columns[0]

    target_col = None
    for c in df.columns:
        if c.lower() in [t.lower() for t in EXPECTED_TARGET_COLS]:
            target_col = c
            break
    if target_col is None:
        for c in df.columns:
            if "aqi" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
                target_col = c
                break
    if target_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("No numeric column found to use as target.")
        target_col = num_cols[-1]

    return dt_col, target_col

def prepare_timeseries(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    dt_col, target_col = cfg.dt_col, cfg.target_col
    df = df.copy()
    if not np.issubdtype(df[dt_col].dtype, np.datetime64):
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
    df = df.dropna(subset=[dt_col]).sort_values(dt_col)
    
    # Handle duplicates by averaging
    if df[dt_col].duplicated().any():
        df = df.groupby(dt_col).mean().reset_index()

    df = df.set_index(dt_col).asfreq(cfg.freq)
    
    # Fill target
    df[target_col] = df[target_col].interpolate(limit=4).ffill().bfill()
    
    # Fill other columns (weather, etc) with forward fill
    df = df.ffill().bfill()

    # Calendar features
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["year"] = idx.year
    df["day"] = idx.day
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Lag features
    for h in range(1, cfg.lag_hours + 1):
        df[f"lag_{h}"] = df[target_col].shift(h)

    # Rolling stats
    for w in cfg.roll_windows:
        df[f"rollmean_{w}"] = df[target_col].rolling(w).mean()
        df[f"rollstd_{w}"] = df[target_col].rolling(w).std()

    df = df.dropna()
    return df

def split_train_test(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test = df.iloc[-cfg.test_size_hours:].copy()
    train = df.iloc[:-cfg.test_size_hours].copy()
    return train, test

def fit_xgb(train: pd.DataFrame, cfg: TrainConfig) -> Tuple[XGBRegressor, list, float]:
    y = train[cfg.target_col].values
    X = train.drop(columns=[cfg.target_col]).values

    model = XGBRegressor(**cfg.xgb_params)
    model.fit(X, y)
    
    return model, [], 0.0

def iterative_forecast_24h(model: XGBRegressor, df_full: pd.DataFrame, cfg: TrainConfig) -> pd.Series:
    """
    Generate 24 hourly forecasts.
    FIXED: Strictly selects only the features the model was trained on to prevent shape mismatches.
    """
    hist = df_full.copy()
    target = cfg.target_col

    # 1. Retrieve the exact feature names the model expects
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
    else:
        # Fallback for older XGBoost versions
        model_features = model.get_booster().feature_names

    future_times = pd.date_range(start=hist.index[-1] + pd.Timedelta(hours=1), periods=24, freq=cfg.freq)
    forecasts = []

    for t in future_times:
        row = {}
        
        # 2. Calculate Time-based Features
        row["hour"] = t.hour
        row["dayofweek"] = t.dayofweek
        row["month"] = t.month
        row["year"] = t.year
        row["day"] = t.day
        row["is_weekend"] = int(t.dayofweek >= 5)

        # 3. Calculate Lags (from updated history)
        for h in range(1, cfg.lag_hours + 1):
            row[f"lag_{h}"] = hist[target].iloc[-h]

        # 4. Calculate Rolling Stats (from updated history)
        for w in cfg.roll_windows:
            row[f"rollmean_{w}"] = hist[target].iloc[-w:].mean()
            row[f"rollstd_{w}"] = hist[target].iloc[-w:].std()

        # 5. Fill Missing Exogenous Features (Weather, etc.)
        # If the model needs a feature (e.g. 'PM10') that isn't a calc or lag,
        # grab it from the last row of history.
        for feat in model_features:
            if feat not in row:
                if feat in hist.columns:
                    row[feat] = hist[feat].iloc[-1]
                else:
                    # If model expects a column not in CSV, fill with 0 safely
                    row[feat] = 0.0

        # 6. Construct DataFrame and SELECT ONLY MODEL FEATURES
        # This prevents the "expected 32, got 49" error
        X_row = pd.DataFrame([row], index=[t])
        
        # Ensure all columns exist
        for col in model_features:
            if col not in X_row.columns:
                X_row[col] = 0.0
                
        # Reorder columns to match model strictly
        X_row = X_row[model_features]

        # Predict
        y_hat = float(model.predict(X_row.values)[0])
        forecasts.append(y_hat)

        # Append prediction to history so next lag calculation is correct
        next_row = hist.iloc[[-1]].copy()
        next_row.index = [t]
        next_row[target] = y_hat
        hist = pd.concat([hist, next_row])

    return pd.Series(forecasts, index=future_times, name="predicted_AQI")

def save_model(model: XGBRegressor, cfg: TrainConfig, path: str = "aqi_model.pkl"):
    payload = dict(model=model, cfg=cfg)
    joblib.dump(payload, path)
    return path

def load_model(path: str = "aqi_model.pkl"):
    payload = joblib.load(path)
    return payload["model"], payload["cfg"]

def train_from_csv(csv_path: str, cfg: Optional[TrainConfig] = None) -> dict:
    if cfg is None:
        cfg = TrainConfig()
    raw = pd.read_csv(csv_path)
    dt_col, target_col = smart_guess_columns(raw)
    cfg.dt_col = dt_col
    cfg.target_col = target_col

    prepared = prepare_timeseries(raw[[dt_col, target_col]].copy(), cfg)
    train, test = split_train_test(prepared, cfg)
    model, _, _ = fit_xgb(train, cfg)

    forecast_24h = iterative_forecast_24h(model, prepared, cfg)
    
    return dict(
        cfg=cfg,
        model=model,
        forecast_24h=forecast_24h,
    )