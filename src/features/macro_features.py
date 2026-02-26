import pandas as pd
from src.config import load_config, PROJECT_ROOT


def compute_macro_features(config=None):
    if config is None:
        config = load_config()

    macro = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "macro_indicators.parquet")

    feat = macro.copy()

    # Add first differences for each series
    # For forward-filled daily data, diff() is 0 on non-reporting days
    # and shows the change on reporting days. We replace 0s with NaN
    # and forward-fill so the last known change persists.
    for col in macro.columns:
        d = macro[col].diff()
        d = d.replace(0, pd.NA)
        feat[f"{col}_diff"] = d.ffill()

    print(f"Macro features: {feat.shape}")
    return feat


if __name__ == "__main__":
    compute_macro_features()
