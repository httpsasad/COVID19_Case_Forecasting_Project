# Architecture

## Data layer
`src.download_data` downloads source CSVs. `src.prepare_data` normalizes and aggregates them.

## Modeling layer
`src.models` defines the baseline models. `src.train` performs chronological evaluation and forecasting.

## Evaluation layer
`src.evaluate` centralizes regression metrics.

## Presentation layer
`app/streamlit_app.py` provides interactive analytics and forecast visualization.

## Next production upgrades
- TimeSeriesSplit / walk-forward validation
- MLflow tracking
- Docker
- CI/CD
- data-quality checks
- scheduled retraining
- model monitoring
- prediction intervals
