# RBC Demand Forecasting (ARIMA vs SARIMA)

This repository contains a small, reproducible time-series forecasting project for monthly Red Blood Cell (RBC) demand.

## What this project does
- Uses an **80/20 time-ordered split** (first 80% for training, last 20% for testing)
- Estimates ARIMA and SARIMA parameters via **AIC-based order selection** (`auto_arima`)
- Fits the selected models and forecasts the test horizon
- Compares performance using **RMSE** and **MAPE**
- Plots ARIMA vs SARIMA forecasts against the real demand

## Files
- `main.py` runs the full pipeline
- `model.py` contains:
  - `estimate_parameters()` for AIC-based order selection (ARIMA and SARIMA)
  - fit + forecast functions
- `utils.py` contains data loading, splitting, and metrics
- `plotting.py` draws a comparison plot
- `rbc_data.csv` is a sample dataset in the expected format

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Notes
- SARIMA uses seasonal period **m=12** (monthly seasonality).
- You can replace `rbc_data.csv` with your own data (same column names).
