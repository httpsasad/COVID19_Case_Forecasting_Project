import pandas as pd
from src.evaluate import regression_metrics
from src.prepare_data import enrich_features
from src.models import build_models

def test_metrics():
    out = regression_metrics([1, 2, 3], [1, 2, 4])
    assert set(["MAE", "MSE", "RMSE"]).issubset(out)
    assert out["MAE"] > 0

def test_processed_schema():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "confirmed": [10, 20, 35],
        "deaths": [1, 2, 3],
        "recovered": [5, 10, 15],
    })
    enriched = enrich_features(df)
    expected_cols = {"date", "confirmed", "deaths", "recovered", "days_since_start", "lag_1", "lag_7", "confirmed_7d_avg"}
    assert expected_cols <= set(enriched.columns)

def test_build_models():
    models = build_models()
    assert len(models) >= 5
    model_names = [m.name for m in models]
    assert "Random Forest" in model_names
    assert "Gradient Boosting" in model_names
