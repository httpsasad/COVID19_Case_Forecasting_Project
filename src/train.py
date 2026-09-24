import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

from .config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, PREDICTIONS_DIR
from .models import build_models
from .evaluate import regression_metrics

TARGET = "confirmed"
FEATURE = "days_since_start"

def train_and_evaluate():
    path = PROCESSED_DIR / "covid19_global_processed.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.download_data` and `python -m src.prepare_data` first.")

    df = pd.read_csv(path, parse_dates=["date"]).dropna(subset=[FEATURE, TARGET])
    window_days = min(180, len(df))
    work = df.tail(window_days).copy()

    X = work[[FEATURE]].values
    y = work[TARGET].values.astype(float)

    # Chronological Split: 80% train, 20% test
    split = max(5, int(len(work) * 0.80))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # TimeSeriesSplit Walk-Forward Cross Validation (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)

    results = []
    fitted = {}
    forecast_bounds = {}

    for bundle in build_models():
        # 1. TimeSeriesSplit CV
        cv_maes, cv_rmses = [], []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_tr, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_tr, y_cv_val = y_train[train_idx], y_train[val_idx]
            bundle.model.fit(X_cv_tr, y_cv_tr)
            val_pred = bundle.model.predict(X_cv_val)
            m = regression_metrics(y_cv_val, val_pred)
            cv_maes.append(m["MAE"])
            cv_rmses.append(m["RMSE"])

        # 2. Final Fit on full train set & test evaluation
        bundle.model.fit(X_train, y_train)
        pred_test = bundle.model.predict(X_test)
        metrics = regression_metrics(y_test, pred_test)
        
        metrics["Model"] = bundle.name
        metrics["CV_MAE_Mean"] = np.mean(cv_maes)
        metrics["CV_RMSE_Mean"] = np.mean(cv_rmses)
        
        # Calculate residual std dev for 95% prediction intervals (1.96 * std)
        residuals = y_train - bundle.model.predict(X_train)
        sigma = np.std(residuals)
        forecast_bounds[bundle.name] = sigma

        results.append(metrics)
        fitted[bundle.name] = bundle.model
        joblib.dump(bundle.model, MODELS_DIR / f"{bundle.name.lower().replace(' ', '_')}.joblib")

    metrics_df = pd.DataFrame(results)[["Model", "MAE", "MSE", "RMSE", "CV_MAE_Mean", "CV_RMSE_Mean"]]
    metrics_df.to_csv(PREDICTIONS_DIR / "metrics.csv", index=False)

    # 3. 14-Day Forward Forecast with 95% Confidence Intervals
    future_horizon = 14
    max_day = int(df[FEATURE].max())
    future_days = np.arange(max_day + 1, max_day + future_horizon + 1).reshape(-1, 1)
    
    forecast = pd.DataFrame({"days_since_start": future_days.ravel()})
    last_date = df["date"].max()
    forecast["date"] = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_horizon)

    for name, model in fitted.items():
        preds = model.predict(future_days)
        forecast[name] = preds
        
        # Expand uncertainty over horizon: sigma * sqrt(h)
        horizon_steps = np.sqrt(np.arange(1, future_horizon + 1))
        sigma = forecast_bounds[name]
        margin = 1.96 * sigma * horizon_steps
        
        forecast[f"{name}_lower"] = np.maximum(0, preds - margin)
        forecast[f"{name}_upper"] = preds + margin

    forecast.to_csv(PREDICTIONS_DIR / "forecast.csv", index=False)

    # 4. Generate Static Figures
    # Forecast chart
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"].tail(60), df[TARGET].tail(60), label="Actual Confirmed", color="#1f77b4", linewidth=2.5)
    for name in fitted:
        plt.plot(forecast["date"], forecast[name], marker="o", label=f"{name} (Forecast)")
    plt.axvline(last_date, linestyle="--", color="gray", alpha=0.7, label="Forecast Start")
    plt.title("COVID-19 Global Confirmed Cases — 14-Day Machine Learning Forecast")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Confirmed Cases")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "forecast.png", dpi=180)
    plt.close()

    # Test Window actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(work["date"].iloc[split:], y_test, marker="o", label="Actual", color="black", linewidth=2)
    for name, model in fitted.items():
        plt.plot(work["date"].iloc[split:], model.predict(X_test), marker="x", label=name)
    plt.title("Test Window Evaluation: Actual vs Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Confirmed Cases")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close()

    print(metrics_df.to_string(index=False))
    print(f"\nSaved forecast & 95% confidence bounds to {PREDICTIONS_DIR / 'forecast.csv'}")

if __name__ == "__main__":
    train_and_evaluate()

