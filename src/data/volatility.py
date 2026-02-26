import pandas as pd
import yfinance as yf
from src.config import load_config, PROJECT_ROOT


def fetch_volatility_data(config=None):
    if config is None:
        config = load_config()

    tickers = config["volatility"]["tickers"]
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]

    print(f"Downloading volatility data: {tickers}")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    prices = raw["Close"]
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    # Rename columns for clarity
    prices.columns = ["VIX", "VIX3M"]

    out_path = PROJECT_ROOT / "data" / "interim" / "volatility.parquet"
    prices.to_parquet(out_path)
    print(f"Saved volatility data: {prices.shape} to {out_path}")
    return prices


if __name__ == "__main__":
    fetch_volatility_data()
