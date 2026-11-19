# 🏥 Emergency Room Demand Forecasting

### NHS England | Time Series Analysis & Machine Learning

Forecasting emergency department attendances using 15 years
of NHS England data. Compared SARIMAX, Prophet, and XGBoost
against a seasonal baseline, achieving a 43% reduction
in forecasting error.

## 📌 Project Overview  
Hospitals often face **unpredictable demand** in Emergency Rooms (ER), leading to under- or over-staffing. This project builds forecasting models on **NHS England A&E monthly attendance data** to predict demand and support better operational planning.  

## 🎯 Objectives  
- Forecast **monthly ER attendances** 1–3 months ahead.  
- Benchmark against a **seasonal-naïve baseline** (last year’s same month).  
- Improve accuracy to reduce staffing risk and waiting times.  

## 🛠️ Approach  
1. **Data Collection & Cleaning**  
   - NHS England A&E attendances (monthly time series).  
   - Cleaned CSV → standardized dates, removed anomalies.  

2. **Feature Engineering**  
   - Lag features (1, 2, 3, 12 months).  
   - Rolling averages (3, 6, 12 months).  
   - Seasonality encodings (month, quarter, sin/cos).  
   - UK public holidays (binary feature).  

3. **Models Evaluated**  
   - **Seasonal-Naïve** (baseline).  
   - **SARIMAX** (selected via AIC grid search).  
   - **Prophet** (with UK holidays).  
   - **XGBoost Regressor** (with engineered features).  

4. **Validation**  
   - Last 12 months held out as test set.  
   - Metrics: **MAPE** (Mean Absolute Percentage Error), **RMSE**.  
   - Rolling backtests for robustness.  

## 📊 Results  

| Model           | MAPE   | RMSE    |
|-----------------|--------|---------|
| Seasonal-Naïve  | 3.10%  | 86,447  |
| **SARIMAX**     | **1.77%** | **50,287** |
| Prophet         | 3.68%  | 95,537  |
| XGB (tuned)     | 2.25%  | 57,489  |

➡️ **SARIMAX reduced error by ~43% vs baseline**, making it the best performer.  

## 🖥️ Streamlit App  
Try the interactive dashboard here:  
👉 [Live Demo on Streamlit Cloud](https://er-admissions-forecast-vbdttw7nayjqnqxjdjfbxu.streamlit.app/)  

Features:  
- Compare actual vs forecast across models.  
- KPI cards (MAPE, RMSE).  
- Leaderboard of models.  
- Download predictions (CSV).  

## 📂 Repo Structure  

er-forecast/
├─ data/
│ ├─ raw/ # Original NHS CSV
│ ├─ processed/ # Cleaned & feature-engineered data
│ │ └─ predictions/ # Model outputs & leaderboard
├─ notebooks/ # Jupyter notebooks for each step
├─ src/
│ ├─ data/ # Data loaders
│ ├─ features/ # Feature builders
│ └─ models/ # Evaluation utilities
├─ app.py # Streamlit dashboard
├─ README.md # Project documentation
├─ requirements.txt # Python dependencies
└─ .gitignore

## 📌 Key Learnings  
- **SARIMAX outperformed ML (XGB) and Prophet** due to strong seasonal signal.  
- Prophet underperformed — shows that no single method always wins.  
- Feature engineering + backtesting gave robustness to forecasts.  
- Streamlit dashboards make technical results accessible to non-technical users.  

## 🚀 How to Run Locally  

```bash
git clone https://github.com/yourusername/er-forecast.git
cd er-forecast
pip install -r requirements.txt
streamlit run app.py
