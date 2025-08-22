import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def plot_forecast(df, date_col="Date", y_col="y", preds: dict | None = None, title="Forecast vs Actual"):
    plt.figure(figsize=(10,5))
    plt.plot(df[date_col], df[y_col], label="Actual")
    if preds:
        for name, series in preds.items():
            plt.plot(df[date_col], series, label=name)
    plt.title(title); plt.xlabel("Date"); plt.ylabel(y_col)
    plt.legend(); plt.tight_layout(); plt.show()

def results_table(**named_scores):
    """Return a DataFrame with method names and metrics you pass in."""
    rows = []
    for name, scores in named_scores.items():
        row = {"model": name}
        row.update(scores)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAPE")
