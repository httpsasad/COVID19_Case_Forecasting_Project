# Advanced A2Z Roadmap — COVID-19 Forecasting Project

## Phase 0 — Product definition
- Define the problem: short-horizon forecasting of reported cumulative confirmed cases.
- Define users: ML portfolio reviewer, recruiter, data scientist, learner.
- Define success criteria: reproducibility, clean architecture, meaningful evaluation, explainable visuals.

## Phase 1 — Data engineering
- Download source CSVs.
- Validate file existence and schema.
- Parse date columns.
- Aggregate province/state rows into global totals.
- Handle duplicate dates.
- Check missing values and monotonicity.
- Save a normalized processed dataset.

## Phase 2 — EDA
Create:
- cumulative confirmed cases
- cumulative deaths
- cumulative recoveries
- daily new cases
- daily new deaths
- 7-day rolling average
- growth-rate charts
- top-country snapshots

Questions:
- Where are reporting gaps?
- Are there negative corrections?
- Does the target remain monotonic?
- Which time window is appropriate?

## Phase 3 — Feature engineering
Baseline:
- `days_since_start`

Enhanced experiments:
- lag 1 / lag 7
- rolling mean 7 / 14
- rolling standard deviation
- day-of-week
- log-transformed target
- growth rate

Keep the baseline model separate from enhanced experiments to avoid mixing results.

## Phase 4 — Modeling
Baseline models:
1. SVR with polynomial kernel
2. PolynomialFeatures + LinearRegression
3. PolynomialFeatures + BayesianRidge

Advanced experiments:
- Ridge/Lasso/ElasticNet
- Random Forest
- Gradient Boosting
- HistGradientBoosting
- XGBoost/LightGBM if desired
- time-series-specific models such as Prophet/ARIMA as separate experiments

## Phase 5 — Validation
Do not use random train/test splitting as the only validation method for a temporal forecasting problem.

Portfolio-grade approach:
- chronological holdout
- rolling/expanding-window validation
- walk-forward evaluation
- fixed forecast horizon

Track:
- MAE
- MSE
- RMSE
- MAPE only when target is safely non-zero
- forecast bias

## Phase 6 — Hyperparameter tuning
SVR:
- C
- gamma
- epsilon
- polynomial degree

Bayesian Ridge:
- alpha parameters
- lambda parameters
- tolerance

Use `TimeSeriesSplit` for tuning where appropriate.

## Phase 7 — Explainability
- Plot actual vs prediction.
- Plot residuals.
- Show forecast intervals if using a model that supports them.
- Explain how polynomial degree changes extrapolation.
- Document uncertainty and data limitations.

## Phase 8 — Dashboard
Streamlit pages:
- Overview
- Global Trends
- Model Comparison
- Forecast
- Data Quality
- About / Methodology

Controls:
- forecast horizon
- recent-window size
- model selector
- smoothing window
- date range

## Phase 9 — Engineering quality
- Type hints
- docstrings
- logging
- configuration file
- deterministic random seeds
- unit tests
- integration test
- requirements lock/constraints
- clean error messages

## Phase 10 — MLOps upgrade
Optional:
- MLflow experiment tracking
- model registry
- Docker
- GitHub Actions
- scheduled data refresh
- artifact versioning
- model monitoring
- data-drift checks

## Phase 11 — Deployment
Possible stack:
- Streamlit Community Cloud
- Render
- Railway
- Azure App Service
- AWS

Recommended deployment flow:

```text
GitHub → CI tests → build → deploy → dashboard
                         ↓
                    model artifacts
```

## Phase 12 — Portfolio polish
Add:
- architecture diagram
- screenshots/GIF
- model comparison table
- sample forecast chart
- limitations section
- data-source attribution
- reproducible commands
- project demo link
- concise resume bullet

## Phase 13 — Interview readiness
Be ready to explain:
- Why this target?
- Why cumulative rather than daily cases?
- Why chronological validation?
- Why SVR?
- Why polynomial regression?
- Why Bayesian Ridge?
- Why can polynomial extrapolation become unstable?
- What causes reporting artifacts?
- How would you improve the model?
- How would you deploy and monitor it?
- What would change if you had country-level covariates?

## Phase 14 — Advanced epidemiological version
For a serious research extension, move beyond pure time regression:
- population
- vaccination
- mobility
- policy interventions
- variant prevalence
- hospitalization
- testing volume
- weather/seasonality
- country-specific effects

Then compare ML forecasting with epidemiological/time-series baselines.

## Final portfolio maturity

### Level 1 — Notebook
EDA + 3 regression models + forecast.

### Level 2 — Reproducible project
Modular Python package + tests + README.

### Level 3 — Interactive product
Streamlit dashboard + model artifacts.

### Level 4 — Production ML
Time-series CV + tracking + Docker + CI/CD.

### Level 5 — Research-grade
Epidemiological features + uncertainty + robust backtesting + model monitoring.
