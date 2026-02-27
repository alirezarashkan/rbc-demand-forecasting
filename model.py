import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import rmse, mape

def estimate_parameters(train: pd.Series, seasonal: bool = False, m: int = 12,
                        max_p: int = 10, max_q: int = 10, max_d: int = 10,
                        max_P: int = 10, max_Q: int = 10, max_D: int = 10,
                        stepwise: bool = True, trace: bool = False):
    """
    Parameter estimation (order selection) using AIC via auto_arima.

    Returns a dict with:
      - order: (p,d,q)
      - seasonal_order: (P,D,Q,m) or None
      - aic: model AIC
      - fitted_auto_arima: the auto_arima object (optional use)
    """
    model = auto_arima(
        train.values,
        seasonal=seasonal,
        m=m if seasonal else 1,
        stepwise=stepwise,
        trace=trace,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
        max_p=max_p,
        max_q=max_q,
        max_d=max_d,
        max_P=max_P,
        max_Q=max_Q,
        max_D=max_D,
    )
    return {
        "order": model.order,
        "seasonal_order": model.seasonal_order if seasonal else None,
        "aic": float(model.aic()),
        "fitted_auto_arima": model,
    }

def fit_and_forecast_arima(train: pd.Series, test: pd.Series, order):
    """Fit ARIMA with selected order and forecast over test horizon."""
    fit = ARIMA(train, order=order).fit()
    fc = fit.forecast(steps=len(test))
    return pd.Series(fc, index=test.index, name="ARIMA_Forecast")

def fit_and_forecast_sarima(train: pd.Series, test: pd.Series, order, seasonal_order):
    """Fit SARIMA (SARIMAX) with selected orders and forecast over test horizon."""
    fit = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False)
    fc = fit.forecast(steps=len(test))
    return pd.Series(fc, index=test.index, name="SARIMA_Forecast")

def run_models(train: pd.Series, test: pd.Series, m: int = 12) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    1) Estimate parameters for ARIMA and SARIMA (AIC-based)
    2) Fit with selected parameters
    3) Forecast and evaluate on test set
    """
    # ---- ARIMA parameter estimation ----
    arima_sel = estimate_parameters(train, seasonal=False)
    arima_order = arima_sel["order"]
    arima_aic = arima_sel["aic"]

    # ---- SARIMA parameter estimation ----
    sarima_sel = estimate_parameters(train, seasonal=True, m=m)
    sarima_order = sarima_sel["order"]
    sarima_seasonal = sarima_sel["seasonal_order"]
    sarima_aic = sarima_sel["aic"]

    # ---- Fit + Forecast ----
    arima_fc = fit_and_forecast_arima(train, test, arima_order)
    sarima_fc = fit_and_forecast_sarima(train, test, sarima_order, sarima_seasonal)

    # ---- Metrics ----
    results = pd.DataFrame({
        "Model": ["ARIMA", "SARIMA"],
        "Order": [str(arima_order), str(sarima_order)],
        "Seasonal_Order": ["-", str(sarima_seasonal)],
        "AIC (Train)": [arima_aic, sarima_aic],
        "RMSE (Test)": [rmse(test, arima_fc), rmse(test, sarima_fc)],
        "MAPE % (Test)": [mape(test, arima_fc), mape(test, sarima_fc)],
    })

    return results, arima_fc, sarima_fc
