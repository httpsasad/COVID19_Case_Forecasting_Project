# COVID-19 Case Forecasting & Intelligent Analytics

An end-to-end Machine Learning project for visualizing historical COVID-19 trends and forecasting short-term global confirmed cases using multiple regression models.

## Project goals

- Build a reproducible COVID-19 data pipeline.
- Aggregate country/province time-series data into global daily totals.
- Engineer a time feature (`days_since_start`) and optional lag/rolling features.
- Compare:
  - Support Vector Regression (SVR)
  - Polynomial Regression
  - Bayesian Ridge Polynomial Regression
- Evaluate models with MAE, MSE and RMSE.
- Generate a 10-day forecast.
- Visualize actual vs predicted values.
- Provide an interactive Streamlit dashboard.
- Keep the repository GitHub-ready with modular source code and tests.

## Data

The primary dataset is the archived Johns Hopkins CSSE COVID-19 global time-series data. JHU CSSE archived the repository on March 10, 2023 after ending its global collection/reporting. The repository contains global confirmed, deaths and recovered time-series files.

Official source:
https://github.com/CSSEGISandData/COVID-19

For a currently maintained source, WHO provides downloadable COVID-19 statistical releases. This project intentionally uses JHU CSSE for reproducibility with the historical reference project.

## Architecture

```text
                    ┌──────────────────────┐
                    │ JHU CSSE CSV files   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Downloader       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Cleaning & Aggregation│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
               SVR       Polynomial      Bayesian
                           Regression       Ridge
                 └─────────────┼─────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Evaluation & Forecast│
                    └──────────┬───────────┘
                               ▼
              ┌────────────────┴────────────────┐
              ▼                                 ▼
       Static figures                    Streamlit dashboard
```

## Quick start

### 1. Create environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Download data

```bash
python -m src.download_data
```

The downloader retrieves the three public JHU CSSE global time-series CSVs:

- confirmed
- deaths
- recovered

If the source is unavailable, manually place the CSVs under `data/raw/`.

### 4. Build the dataset

```bash
python -m src.prepare_data
```

### 5. Train and forecast

```bash
python -m src.train
```

### 6. Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

### 7. Run tests

```bash
pytest -q
```

## Expected outputs

After training:

```text
outputs/
├── figures/
│   ├── global_cases.png
│   ├── model_comparison.png
│   └── forecast.png
└── predictions/
    ├── metrics.csv
    └── forecast.csv
```

## Modeling notes

The original reference concept uses a recent time window and treats cumulative confirmed cases as a function of time. This implementation preserves that idea while adding a cleaner train/test workflow and reusable modules.

Important: cumulative-case forecasting is a simplified educational modeling problem. It does not model epidemiological mechanisms, interventions, variants, testing changes, reporting delays, vaccination, seasonality, or susceptible/infected/recovered dynamics. Forecasts should therefore be presented as model outputs, not medical or public-health predictions.

## Suggested GitHub presentation

Use these repository sections:

1. Problem statement
2. Dataset and provenance
3. Exploratory data analysis
4. Data preprocessing
5. Feature engineering
6. Model development
7. Model evaluation
8. Forecasting
9. Dashboard
10. Limitations
11. Reproducibility
12. Future improvements

## Portfolio description

> Built an end-to-end COVID-19 analytics and forecasting platform using Python and scikit-learn. Processed global time-series data, engineered temporal features, compared SVR, Polynomial Regression and Bayesian Ridge models, evaluated MAE/MSE/RMSE, generated short-horizon forecasts, and exposed results through an interactive Streamlit dashboard.

## Data attribution

Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE), archived COVID-19 repository.

Use the source's own citation guidance when publishing academic or formal work.
