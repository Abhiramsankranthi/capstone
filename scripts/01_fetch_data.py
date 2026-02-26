"""Fetch all raw data from yfinance, FRED, and process news sentiment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.equity import fetch_equity_prices
from src.data.macro import fetch_macro_data
from src.data.volatility import fetch_volatility_data
from src.data.sentiment import load_and_filter_articles, run_finbert_inference, aggregate_daily_sentiment


def main():
    config = load_config()

    print("\n=== Fetching Equity Prices ===")
    fetch_equity_prices(config)

    print("\n=== Fetching Macro Indicators ===")
    fetch_macro_data(config)

    print("\n=== Fetching Volatility Data ===")
    fetch_volatility_data(config)

    print("\n=== Processing News Sentiment ===")
    articles = load_and_filter_articles(config)
    scored = run_finbert_inference(articles, config)
    aggregate_daily_sentiment(scored)

    print("\n=== Data Fetch Complete ===")


if __name__ == "__main__":
    main()
