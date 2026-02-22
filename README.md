# RBC Demand Forecasting using ARIMA and SARIMA

> Healthcare Time-Series Forecasting | ARIMA | SARIMA | Statistical Modeling | Out-of-Sample Validation

This project was developed as part of a graduate coursework in Revenue and Demand Management, focusing on healthcare demand forecasting using classical time-series modeling techniques.

---

## Project Overview

The objective of this project is to forecast monthly Red Blood Cell (RBC) demand and compare the performance of ARIMA and Seasonal ARIMA (SARIMA) models using a structured statistical workflow.

The implementation emphasizes reproducibility, modular design, and clear separation between modeling, diagnostics, and evaluation components.

---

## Objective

- Forecast monthly RBC demand
- Compare ARIMA and SARIMA performance
- Evaluate models using out-of-sample validation

---

## Methodology

- Train/Test split (final year as test set)
- Stationarity check using Augmented Dickey-Fuller (ADF) test
- ARIMA order selection based on Akaike Information Criterion (AIC)
- Seasonal ARIMA modeling using `auto_arima`
- Performance evaluation using:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)

---

## Repository Structure
