import pandas as pd
import numpy as np
from holidays import UnitedKingdom

def add_time_index_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["quarter"] = out[date_col].dt.quarter
    # cyclical encoding (optional; handy for tree models too)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out

def add_lagged_features(df: pd.DataFrame, target: str = "y", lags=(1, 2, 3, 12)) -> pd.DataFrame:
    out = df.copy()
    for l in lags:
        out[f"lag_{l}"] = out[target].shift(l)
    return out

def add_rolling_features(df: pd.DataFrame, target: str = "y") -> pd.DataFrame:
    out = df.copy()
    # Rolling means / stds over recent windows
    out["roll3_mean"] = out[target].shift(1).rolling(3).mean()
    out["roll6_mean"] = out[target].shift(1).rolling(6).mean()
    out["roll12_mean"] = out[target].shift(1).rolling(12).mean()
    out["roll3_std"]  = out[target].shift(1).rolling(3).std()
    out["roll6_std"]  = out[target].shift(1).rolling(6).std()
    out["roll12_std"] = out[target].shift(1).rolling(12).std()
    # momentum (last month vs same month last year)
    out["mom_1"] = out[target] / out[target].shift(1) - 1
    out["mom_yoy"] = out[target] / out[target].shift(12) - 1
    return out

def add_uk_holidays(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    # Monthly data: flag if any UK public holiday exists in that month
    years = range(out[date_col].dt.year.min(), out[date_col].dt.year.max() + 1)
    uk_h = UnitedKingdom()
    flags = []
    for d in out[date_col]:
        month_holiday = any(h.year == d.year and h.month == d.month for h in uk_h)
        flags.append(int(month_holiday))
    out["has_public_holiday"] = flags
    return out

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_time_index_features(out, "Date")
    out = add_lagged_features(out, "y")
    out = add_rolling_features(out, "y")
    out = add_uk_holidays(out, "Date")
    # drop rows with NaNs created by lags/rolls
    out = out.dropna().reset_index(drop=True)
    return out
