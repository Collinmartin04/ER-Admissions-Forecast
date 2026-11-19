import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
PRED_DIR = DATA_DIR / "predictions"

# --- Load Data ---
@st.cache_data
def load_core():
    ts = pd.read_csv(DATA_DIR / "ae_timeseries_clean.csv", parse_dates=["Date"])
    ts = ts.sort_values("Date")
    return ts

@st.cache_data
def load_predictions():
    dfs = {}
    dfs["naive"]   = pd.read_csv(PRED_DIR / "valid_naive12.csv", parse_dates=["Date"])
    dfs["sarimax"] = pd.read_csv(PRED_DIR / "valid_sarimax.csv", parse_dates=["Date"])
    dfs["prophet"] = pd.read_csv(PRED_DIR / "valid_prophet.csv", parse_dates=["Date"])
    dfs["xgb"]     = pd.read_csv(PRED_DIR / "valid_xgb.csv", parse_dates=["Date"])
    leaderboard = pd.read_csv(PRED_DIR / "leaderboard.csv")
    return dfs, leaderboard

def plot_series(ts, preds, model_name, months=48):
    col_map = {
        "Naïve-12": "y_hat_naive12",
        "SARIMAX": "y_hat_sarimax",
        "Prophet": "y_hat_prophet",
        "XGB (tuned)": "y_hat_xgb"
    }
    col = col_map[model_name]

    # Merge actuals with chosen predictions
    plot_df = ts.merge(preds["naive"][["Date","y_hat_naive12"]], on="Date", how="left")
    plot_df = plot_df.merge(preds["sarimax"][["Date","y_hat_sarimax"]], on="Date", how="left")
    plot_df = plot_df.merge(preds["prophet"][["Date","y_hat_prophet"]], on="Date", how="left")
    plot_df = plot_df.merge(preds["xgb"][["Date","y_hat_xgb"]], on="Date", how="left")

    tail = plot_df.tail(months)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["y"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail[col], mode="lines", name=model_name))
    fig.update_layout(
        title=f"ER Attendances — {model_name} vs Actual (Last {months} months)",
        xaxis_title="Date", yaxis_title="Attendances", hovermode="x unified", height=420
    )
    return fig

# --- Main UI ---
st.set_page_config(
    page_title="Emergency Department Demand Forecasting",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Emergency Department Demand Forecasting")

st.caption(
    "NHS England | Time Series Analysis & Machine Learning"
)

st.markdown(
    "**Forecasting emergency department attendances using 15 years of NHS data. "
    "Achieved a 43% reduction in forecasting error compared with a seasonal baseline.**"
)

st.info("""
**Models:**
- **Naïve** → assumes this month will look the same as the same month last year.  
- **SARIMAX** → a classic stats model that learns seasonal patterns.  
- **Prophet** → a forecasting tool from Facebook, good for trends/holidays.  
- **XGB** → a machine learning model using many past features.

**Metrics:**
- **MAPE** → average % error (lower is better).  
- **RMSE** → size of mistakes in number of attendances (lower is better).
""")

ts = load_core()
preds, leaderboard = load_predictions()

# Sidebar
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["SARIMAX", "XGB (tuned)", "Naïve-12", "Prophet"],
    index=0
)
months = st.sidebar.slider("Months to display", min_value=24, max_value=120, value=48, step=12)


# --- KPIs (robust lookup) ---
# Map the sidebar labels to exact leaderboard "Model" values
name_map = {
    "SARIMAX": "SARIMAX",
    "XGB (tuned)": "XGB (tuned)",
    "Naïve-12": "Seasonal-Naive",
    "Prophet": "Prophet",
}

target_name = name_map.get(model_choice, model_choice)

row_match = leaderboard.loc[leaderboard["Model"] == target_name]

if row_match.empty:
    # Fallback: try case-insensitive match
    row_match = leaderboard.loc[
        leaderboard["Model"].str.lower().str.strip() == target_name.lower().strip()
    ]

if row_match.empty:
    # Still nothing — warn and show placeholders so the app doesn't crash
    st.warning(
        f"Couldn't find metrics for '{model_choice}' in leaderboard. "
        f"Available models: {', '.join(leaderboard['Model'].unique())}"
    )
    kpi_mape = float("nan")
    kpi_rmse = float("nan")
else:
    row = row_match.iloc[0]
    kpi_mape = row["MAPE"]
    kpi_rmse = row["RMSE"]

c1, c2, c3 = st.columns(3)
c1.metric("Model", model_choice)
c2.metric("MAPE", f"{kpi_mape:.2f}%" if pd.notna(kpi_mape) else "N/A")
c3.metric("RMSE", f"{kpi_rmse:,.0f}" if pd.notna(kpi_rmse) else "N/A")

# Plot
st.plotly_chart(plot_series(ts, preds, model_choice, months), use_container_width=True)

# Leaderboard
st.subheader("Leaderboard (Validation Window)")
st.dataframe(
    leaderboard.style.format({"MAPE":"{:.2f}%", "RMSE":"{:,.0f}"}),
    use_container_width=True
)
