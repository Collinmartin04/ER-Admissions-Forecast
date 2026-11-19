# 🏥 Emergency Department Demand Forecasting

### NHS England | Time Series Analysis & Machine Learning

Developed an end-to-end forecasting project using 15 years of NHS England emergency department attendance data. Compared SARIMAX, Prophet, and XGBoost against a seasonal-naïve baseline, achieving a **43% reduction in forecasting error (MAPE: 3.10% to 1.77%)**.

### 🚀 Live Interactive Dashboard

[View the Forecasting Dashboard](https://er-admissions-forecast-vbdttw7nayjqnqxjdjfbxu.streamlit.app/)

## 📌 Project Overview

Emergency departments face fluctuating patient demand, making staffing, resource allocation, and capacity planning challenging.

This project analyzes historical **NHS England A&E attendance data** to identify trends and seasonal patterns, develop time series forecasting models, and evaluate their ability to predict monthly patient demand.

The goal is to demonstrate how data-driven forecasting can support more informed healthcare operational planning.

## 🎯 Objectives

- Forecast **monthly emergency department attendances** 1–3 months ahead.
- Compare statistical and machine learning models, including **SARIMAX, Prophet, and XGBoost**.
- Benchmark against a **seasonal-naïve baseline**, which predicts demand using the same month from the previous year.
- Evaluate forecasting accuracy using **MAPE and RMSE**.
- Explore how accurate forecasts could support **staffing, resource allocation, and capacity planning**.

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
