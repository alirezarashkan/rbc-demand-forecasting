import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Load RBC demand data from CSV. Expects columns: Date, Demand_rbc."""
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def train_test_split_series(series: pd.Series, train_ratio: float = 0.8):
    """Time-ordered split (no shuffle)."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    split_idx = int(round(len(series) * train_ratio))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    if len(test) == 0:
        raise ValueError("Test set is empty. Reduce train_ratio or check data length.")
    return train, test

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)
