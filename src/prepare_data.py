import pandas as pd
import numpy as np
from .config import RAW_DIR, PROCESSED_DIR

META_COLS = ["Province/State", "Country/Region", "Lat", "Long"]

def load_and_aggregate(path):
    """Aggregate raw JHU CSSE time-series CSV into global daily totals."""
    df = pd.read_csv(path)
    date_cols = [c for c in df.columns if c not in META_COLS]
    for c in date_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    series = df[date_cols].sum(axis=0)
    out = series.rename("value").reset_index()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"], format="%m/%d/%y")
    return out

def load_by_country(path, value_name):
    """Melt raw JHU CSSE time-series CSV by Country/Region."""
    df = pd.read_csv(path)
    date_cols = [c for c in df.columns if c not in META_COLS]
    for c in date_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    
    country_df = df.groupby("Country/Region")[date_cols].sum().reset_index()
    melted = country_df.melt(id_vars=["Country/Region"], var_name="date", value_name=value_name)
    melted["date"] = pd.to_datetime(melted["date"], format="%m/%d/%y")
    return melted

def enrich_features(df, target_col="confirmed"):
    """Engineer temporal features: lags, rolling averages, growth rates."""
    df = df.sort_values("date").reset_index(drop=True)
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    
    # New daily cases
    df["new_confirmed"] = df["confirmed"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df["deaths"].diff().fillna(0).clip(lower=0)
    df["new_recovered"] = df["recovered"].diff().fillna(0).clip(lower=0) if "recovered" in df.columns else 0

    # Moving averages
    df["confirmed_7d_avg"] = df["new_confirmed"].rolling(7, min_periods=1).mean()
    df["deaths_7d_avg"] = df["new_deaths"].rolling(7, min_periods=1).mean()
    df["confirmed_14d_avg"] = df["new_confirmed"].rolling(14, min_periods=1).mean()

    # Lag features
    df["lag_1"] = df[target_col].shift(1).bfill()
    df["lag_7"] = df[target_col].shift(7).bfill()
    df["lag_14"] = df[target_col].shift(14).bfill()

    # Growth rate
    df["daily_growth_rate"] = (df["new_confirmed"] / (df[target_col].shift(1) + 1)).fillna(0).clip(lower=0, upper=1)
    
    return df

def build_dataset():
    # 1. Global Aggregation
    confirmed = load_and_aggregate(RAW_DIR / "confirmed_global.csv")
    deaths = load_and_aggregate(RAW_DIR / "deaths_global.csv")
    recovered = load_and_aggregate(RAW_DIR / "recovered_global.csv")

    global_df = confirmed.rename(columns={"value": "confirmed"}).merge(
        deaths.rename(columns={"value": "deaths"}), on="date", how="outer"
    ).merge(
        recovered.rename(columns={"value": "recovered"}), on="date", how="outer"
    ).sort_values("date")

    for col in ["confirmed", "deaths", "recovered"]:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce").fillna(0)

    global_df = enrich_features(global_df)
    global_df.to_csv(PROCESSED_DIR / "covid19_global_processed.csv", index=False)

    # 2. Country-Level Aggregation
    c_conf = load_by_country(RAW_DIR / "confirmed_global.csv", "confirmed")
    c_death = load_by_country(RAW_DIR / "deaths_global.csv", "deaths")
    c_rec = load_by_country(RAW_DIR / "recovered_global.csv", "recovered")

    country_df = c_conf.merge(c_death, on=["Country/Region", "date"], how="outer").merge(
        c_rec, on=["Country/Region", "date"], how="outer"
    ).sort_values(["Country/Region", "date"])

    for col in ["confirmed", "deaths", "recovered"]:
        country_df[col] = pd.to_numeric(country_df[col], errors="coerce").fillna(0)

    country_df.to_csv(PROCESSED_DIR / "covid19_country_processed.csv", index=False)

    return global_df, country_df

if __name__ == "__main__":
    build_dataset()
    print("Processed global and country-level datasets created.")

