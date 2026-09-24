import pandas as pd
import matplotlib.pyplot as plt
from .config import PROCESSED_DIR, FIGURES_DIR

def main():
    df = pd.read_csv(PROCESSED_DIR / "covid19_global_processed.csv", parse_dates=["date"])
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["confirmed"], label="Confirmed")
    plt.plot(df["date"], df["deaths"], label="Deaths")
    plt.plot(df["date"], df["recovered"], label="Recovered")
    plt.title("Global COVID-19 Cumulative Trends")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "global_cases.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    main()
