import pandas as pd
from src.config import load_config, PROJECT_ROOT


def compute_vol_features(config=None):
    if config is None:
        config = load_config()

    vol = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "volatility.parquet")
    window = config["features"]["vix_percentile_window"]

    feat = pd.DataFrame(index=vol.index)
    feat["VIX"] = vol["VIX"]

    # 252-day percentile rank
    feat["VIX_percentile"] = vol["VIX"].rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # VIX/VIX3M term structure ratio (>1 means backwardation / stress)
    feat["VIX_term_ratio"] = vol["VIX"] / vol["VIX3M"]

    print(f"Volatility features: {feat.shape}")
    return feat


if __name__ == "__main__":
    compute_vol_features()
