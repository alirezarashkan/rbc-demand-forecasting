from utils import load_data, train_test_split_series
from model import run_models
from plotting import plot_forecasts

def main():
    df = load_data("rbc_data.csv")
    series = df["Demand_rbc"]

    train, test = train_test_split_series(series, train_ratio=0.8)

    results, arima_fc, sarima_fc = run_models(train, test, m=12)

    print("\n=== ARIMA vs SARIMA (80/20 Split) ===\n")
    print(results.to_string(index=False))

    plot_forecasts(train, test, arima_fc, sarima_fc, title="ARIMA vs SARIMA Forecast (80/20 Split)")

if __name__ == "__main__":
    main()
