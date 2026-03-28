import pandas as pd
import numpy as np
from src.config import load_config, PROJECT_ROOT
from src.features.technical import compute_technical_features
from src.features.macro_features import compute_macro_features
from src.features.vol_features import compute_vol_features


def merge_all_features(config=None):
    if config is None:
        config = load_config()

    # Compute features from each source
    tech = compute_technical_features(config)
    macro = compute_macro_features(config)
    vol_feat = compute_vol_features(config)

    # Load sentiment if available
    sent_path = PROJECT_ROOT / "data" / "interim" / "sentiment_daily.parquet"
    if sent_path.exists():
        sentiment = pd.read_parquet(sent_path)
    else:
        print("Warning: sentiment data not found, skipping")
        sentiment = None

    # Trading day index from technical features (derived from equity prices)
    trading_days = tech.index

    # Align macro to trading days via forward-fill
    macro_aligned = macro.reindex(trading_days, method="ffill")

    # Volatility features are already on trading days
    vol_aligned = vol_feat.reindex(trading_days)

    # Merge
    merged = tech.join(macro_aligned, how="left", rsuffix="_macro")
    merged = merged.join(vol_aligned, how="left", rsuffix="_vol")

    if sentiment is not None:
        sent_aligned = sentiment.reindex(trading_days)
        # Count column: fill missing with 0 (no articles that day)
        if "sent_article_count" in sent_aligned.columns:
            sent_aligned["sent_article_count"] = sent_aligned["sent_article_count"].fillna(0)
        # All other sentiment columns: forward-fill up to 5 days
        ffill_cols = [c for c in sent_aligned.columns if c != "sent_article_count"]
        sent_aligned[ffill_cols] = sent_aligned[ffill_cols].ffill(limit=5)
        merged = merged.join(sent_aligned, how="left")

    # Forward return targets
    for h in config["targets"]["forward_returns"]:
        if h == 1:
            merged[f"fwd_return_{h}d"] = merged["sp500_log_return"].shift(-1)
        else:
            merged[f"fwd_return_{h}d"] = (
                merged["sp500_log_return"]
                .rolling(h).sum()
                .shift(-h)
            )

    out_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    merged.to_parquet(out_path)
    print(f"Merged dataset: {merged.shape} saved to {out_path}")
    print(f"Date range: {merged.index.min()} to {merged.index.max()}")
    print(f"Columns: {list(merged.columns)}")
    return merged


if __name__ == "__main__":
    merge_all_features()
