# RBC Demand Forecasting using ARIMA and SARIMA

> Healthcare Time-Series Forecasting | ARIMA | SARIMA | Statistical Modeling | Out-of-Sample Validation

This project focuses on forecasting monthly Red Blood Cell (RBC) demand using classical time-series modeling techniques. The study was developed within a graduate-level Revenue and Demand Management context, with application to healthcare demand forecasting.

The objective is to model and compare non-seasonal ARIMA and seasonal SARIMA approaches under a structured statistical framework, emphasizing model selection, parameter estimation, and out-of-sample performance evaluation.

---

## Project Overview

Healthcare demand forecasting requires models capable of capturing both trend and seasonal structures. Monthly RBC consumption data exhibit clear temporal dynamics and potential annual seasonality. 

This project implements a reproducible forecasting pipeline including:

- Time-ordered 80/20 train-test split
- AIC-based parameter estimation
- ARIMA modeling for non-seasonal structure
- SARIMA modeling for seasonal (m=12) structure
- Quantitative performance comparison on unseen data

The implementation follows a modular design with separate components for parameter estimation, model fitting, forecasting, and evaluation.

---

## Objective

- Forecast monthly RBC demand
- Compare ARIMA vs SARIMA performance
- Evaluate predictive accuracy using out-of-sample validation
- Investigate the impact of seasonal structure on forecasting quality

---

## Methodology

### Data Preparation
- Monthly time-series data (2014–2018)
- Chronological 80% training / 20% testing split

### Model Selection
- Stationarity induced via differencing (d parameter)
- Order selection using Akaike Information Criterion (AIC)
- Automated parameter estimation using `auto_arima`
- Seasonal period set to m = 12 for SARIMA

### Model Evaluation
Performance is evaluated on the test set using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

This out-of-sample validation ensures that the models are assessed based on generalization performance rather than in-sample fit.

---

## Key Insight

Results show that incorporating seasonal components via SARIMA improves the model’s ability to capture intra-year fluctuations compared to a purely non-seasonal ARIMA model. The comparison highlights the importance of structural modeling in healthcare time-series forecasting.

---

## Technical Stack

- Python
- pandas
- statsmodels
- pmdarima
- matplotlib

---

## Reproducibility

The project is fully reproducible. All scripts are modular and executable via:

```bash
pip install -r requirements.txt
python main.py
