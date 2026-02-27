import matplotlib.pyplot as plt
import pandas as pd

def plot_forecasts(train: pd.Series, test: pd.Series, arima_fc: pd.Series, sarima_fc: pd.Series, title: str = "Forecast Comparison"):
    plt.figure(figsize=(12, 4))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test")
    plt.plot(arima_fc.index, arima_fc.values, label="ARIMA Forecast")
    plt.plot(sarima_fc.index, sarima_fc.values, label="SARIMA Forecast")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("RBC Demand")
    plt.legend()
    plt.tight_layout()
    plt.show()
