import pandas as pd
import numpy as np
import pandas_ta as ta
from src.config import load_config, PROJECT_ROOT


def compute_technical_features(config=None):
    if config is None:
        config = load_config()

    prices = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "equity_prices.parquet")
    benchmark = config["equity"]["benchmark"]
    close = prices[benchmark]

    feat = pd.DataFrame(index=prices.index)

    # Log returns
    feat["sp500_log_return"] = np.log(close / close.shift(1))

    # Rolling realized volatility (annualized)
    for w in config["features"]["rolling_windows"]:
        feat[f"sp500_realized_vol_{w}d"] = (
            feat["sp500_log_return"].rolling(w).std() * np.sqrt(252)
        )

    # RSI
    feat["sp500_rsi"] = ta.rsi(close, length=config["features"]["rsi_period"])

    # MACD
    macd_df = ta.macd(
        close,
        fast=config["features"]["macd_fast"],
        slow=config["features"]["macd_slow"],
        signal=config["features"]["macd_signal"],
    )
    feat["sp500_macd"] = macd_df.iloc[:, 0]
    feat["sp500_macd_signal"] = macd_df.iloc[:, 1]
    feat["sp500_macd_hist"] = macd_df.iloc[:, 2]

    # Bollinger Band width
    bb = ta.bbands(close, length=config["features"]["bollinger_period"])
    upper = bb.iloc[:, 0]  # BBU
    middle = bb.iloc[:, 1]  # BBM
    lower = bb.iloc[:, 2]  # BBL
    feat["sp500_bb_width"] = (upper - lower) / middle

    # Also compute log returns for sector ETFs
    for etf in config["equity"]["sector_etfs"]:
        if etf in prices.columns:
            feat[f"{etf}_log_return"] = np.log(prices[etf] / prices[etf].shift(1))

    print(f"Technical features: {feat.shape}")
    return feat


if __name__ == "__main__":
    compute_technical_features()
