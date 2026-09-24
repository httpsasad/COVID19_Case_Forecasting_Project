import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import PROCESSED_DIR, PREDICTIONS_DIR

# ---------------------------------------------------------
# Page Configuration (Clean Native Theme)
# ---------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 AI Analytics & Forecasting Platform",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data
global_path = PROCESSED_DIR / "covid19_global_processed.csv"
country_path = PROCESSED_DIR / "covid19_country_processed.csv"
metrics_path = PREDICTIONS_DIR / "metrics.csv"
forecast_path = PREDICTIONS_DIR / "forecast.csv"

if not global_path.exists():
    st.error("Processed dataset not found. Please run: python -m src.download_data && python -m src.prepare_data")
    st.stop()

df_global = pd.read_csv(global_path, parse_dates=["date"])
df_country = pd.read_csv(country_path, parse_dates=["date"]) if country_path.exists() else None

# Sidebar Filters
st.sidebar.title("🦠 Control Center")

location_type = st.sidebar.radio("Analysis Scope", ["Global", "Country Level"])

if location_type == "Country Level" and df_country is not None:
    countries = sorted(df_country["Country/Region"].unique())
    default_country = "US" if "US" in countries else countries[0]
    selected_country = st.sidebar.selectbox("Select Country/Region", countries, index=countries.index(default_country) if default_country in countries else 0)
    df_analysis = df_country[df_country["Country/Region"] == selected_country].copy()
    
    df_analysis["new_confirmed"] = df_analysis["confirmed"].diff().fillna(0).clip(lower=0)
    df_analysis["new_deaths"] = df_analysis["deaths"].diff().fillna(0).clip(lower=0)
    df_analysis["confirmed_7d_avg"] = df_analysis["new_confirmed"].rolling(7, min_periods=1).mean()
else:
    selected_country = "Global"
    df_analysis = df_global.copy()

st.sidebar.divider()
st.sidebar.subheader("🔮 Forecast Settings")
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=14, value=14)

# Header Section
st.title("🦠 COVID-19 Case Forecasting & Intelligent Analytics")
st.caption(f"Historical analytics, country breakdown, and short-horizon AI forecasts for **{selected_country}**")

# ---------------------------------------------------------
# KPI Summary Metrics
# ---------------------------------------------------------
latest = df_analysis.iloc[-1]

total_conf = int(latest["confirmed"])
total_deaths = int(latest["deaths"])
total_rec = int(latest["recovered"]) if "recovered" in latest and not np.isnan(latest["recovered"]) else 0
active_cases = max(0, total_conf - total_deaths - total_rec)
mortality_rate = (total_deaths / total_conf * 100) if total_conf > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Latest Confirmed", f"{total_conf:,}", delta=f"+{int(latest.get('new_confirmed', 0)):,}")
c2.metric("Latest Deaths", f"{total_deaths:,}", delta=f"+{int(latest.get('new_deaths', 0)):,}")
c3.metric("Estimated Active", f"{active_cases:,}")
c4.metric("Mortality Rate", f"{mortality_rate:.2f}%")
c5.metric("7-Day New Cases Avg", f"{int(latest.get('confirmed_7d_avg', 0)):,}")

st.divider()

# ---------------------------------------------------------
# Tabs Section
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", 
    "Trends", 
    "Model Comparison", 
    "14-Day AI Forecast", 
    "Country Deep-Dive"
])

# ---------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------
with tab1:
    st.subheader(f"Cumulative Trend ({selected_country})")
    fig_overview = px.line(
        df_analysis, 
        x="date", 
        y="confirmed", 
        title=f"Cumulative Confirmed Cases over Time ({selected_country})"
    )
    st.plotly_chart(fig_overview, width="stretch")

    st.subheader("Data Explorer (Recent 20 Days)")
    display_cols = ["date", "confirmed", "deaths", "new_confirmed", "confirmed_7d_avg"]
    avail_cols = [c for c in display_cols if c in df_analysis.columns]
    st.dataframe(df_analysis[avail_cols].tail(20).sort_values("date", ascending=False), width="stretch")

# ---------------------------------------------------------
# TAB 2: Trends
# ---------------------------------------------------------
with tab2:
    st.subheader("Daily Case Dynamics")
    fig_daily = px.bar(
        df_analysis, 
        x="date", 
        y="new_confirmed", 
        title="Daily New Confirmed Cases with 7-Day Moving Average"
    )
    fig_daily.add_trace(
        go.Scatter(
            x=df_analysis["date"], 
            y=df_analysis["confirmed_7d_avg"], 
            mode="lines", 
            name="7-Day Avg",
            line=dict(color="red", width=2)
        )
    )
    st.plotly_chart(fig_daily, width="stretch")

    if df_country is not None:
        st.subheader("Top 10 Most Affected Countries")
        latest_by_country = df_country.groupby("Country/Region")["confirmed"].max().reset_index()
        top10 = latest_by_country.sort_values("confirmed", ascending=False).head(10)
        
        fig_top10 = px.bar(
            top10, 
            x="confirmed", 
            y="Country/Region", 
            orientation="h",
            title="Top 10 Global Total Confirmed Cases"
        )
        fig_top10.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top10, width="stretch")

# ---------------------------------------------------------
# TAB 3: Model Comparison
# ---------------------------------------------------------
with tab3:
    st.subheader("Machine Learning Model Evaluation & TimeSeriesSplit CV")
    st.write("Models are evaluated using **TimeSeriesSplit (5 Folds)** walk-forward validation and holdout test set metrics.")

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df.sort_values("MAE"), width="stretch")
        
        fig_mae = px.bar(
            metrics_df, 
            x="Model", 
            y=["MAE", "CV_MAE_Mean"], 
            barmode="group",
            title="Model Test MAE vs TimeSeriesSplit CV MAE Mean"
        )
        st.plotly_chart(fig_mae, width="stretch")
    else:
        st.info("Run `python -m src.train` first.")

# ---------------------------------------------------------
# TAB 4: 14-Day AI Forecast
# ---------------------------------------------------------
with tab4:
    st.subheader("14-Day Forward Case Forecast with 95% Confidence Intervals")
    
    if forecast_path.exists():
        forecast_df = pd.read_csv(forecast_path, parse_dates=["date"]).head(horizon)
        
        all_models = [c for c in forecast_df.columns if not c.endswith("_lower") and not c.endswith("_upper") and c not in ["date", "days_since_start"]]
        selected_model = st.selectbox("Select Model for Detailed View", all_models, index=0)
        
        fig_fc = go.Figure()
        
        # Historical
        recent_history = df_global.tail(40)
        fig_fc.add_trace(go.Scatter(
            x=recent_history["date"],
            y=recent_history["confirmed"],
            mode="lines+markers",
            name="Actual Confirmed",
            line=dict(color="#1f77b4", width=2.5)
        ))
        
        # Forecast
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df[selected_model],
            mode="lines+markers",
            name=f"{selected_model} Forecast",
            line=dict(color="#ff7f0e", width=2.5, dash="dash")
        ))
        
        # 95% Confidence Band
        if f"{selected_model}_upper" in forecast_df.columns:
            fig_fc.add_trace(go.Scatter(
                x=forecast_df["date"].tolist() + forecast_df["date"].tolist()[::-1],
                y=forecast_df[f"{selected_model}_upper"].tolist() + forecast_df[f"{selected_model}_lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(255, 127, 14, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% Confidence Interval"
            ))
            
        fig_fc.update_layout(
            title=f"14-Day Forecast ({selected_model}) with 95% Confidence Interval",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_fc, width="stretch")
        
        # Forecast Table & CSV Download
        st.markdown("### Forecast Data Table")
        cols_to_show = ["date", selected_model, f"{selected_model}_lower", f"{selected_model}_upper"]
        avail_cols = [c for c in cols_to_show if c in forecast_df.columns]
        
        st.dataframe(forecast_df[avail_cols], width="stretch")
        
        csv_data = forecast_df[avail_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_data,
            file_name=f"forecast_{selected_model.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Run `python -m src.train` first.")

# ---------------------------------------------------------
# TAB 5: Country Deep-Dive
# ---------------------------------------------------------
with tab5:
    st.subheader("Multi-Country Comparison")
    if df_country is not None:
        countries_list = sorted(df_country["Country/Region"].unique())
        selected_countries = st.multiselect("Select Countries to Compare", countries_list, default=["US", "India", "Brazil", "United Kingdom"])
        
        if selected_countries:
            filtered_c = df_country[df_country["Country/Region"].isin(selected_countries)]
            
            fig_compare = px.line(
                filtered_c, 
                x="date", 
                y="confirmed", 
                color="Country/Region",
                title="Cross-Country Cumulative Confirmed Cases"
            )
            st.plotly_chart(fig_compare, width="stretch")
    else:
        st.info("Country dataset unavailable.")
