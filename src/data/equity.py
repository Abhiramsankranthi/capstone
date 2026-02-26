import pandas as pd
import yfinance as yf
from src.config import load_config, PROJECT_ROOT


def fetch_equity_prices(config=None):
    if config is None:
        config = load_config()

    benchmark = config["equity"]["benchmark"]
    etfs = config["equity"]["sector_etfs"]
    tickers = [benchmark] + etfs
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]

    print(f"Downloading equity prices for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Extract Close prices (yfinance returns MultiIndex columns)
    prices = raw["Close"]
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    out_path = PROJECT_ROOT / "data" / "interim" / "equity_prices.parquet"
    prices.to_parquet(out_path)
    print(f"Saved equity prices: {prices.shape} to {out_path}")
    return prices


if __name__ == "__main__":
    fetch_equity_prices()
