import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.config import load_config, PROJECT_ROOT

# Known crisis periods
CRISIS_PERIODS = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2011 Euro Debt": ("2011-08-01", "2011-10-31"),
    "2015 China Slowdown": ("2015-08-01", "2015-09-30"),
    "2018 Vol-mageddon": ("2018-02-01", "2018-02-28"),
    "2018 Q4 Selloff": ("2018-10-01", "2018-12-31"),
    "2020 COVID": ("2020-02-15", "2020-03-31"),
    "2022 Rate Hikes": ("2022-01-01", "2022-10-31"),
}


def validate_regimes(features_df, regime_labels, config=None):
    if config is None:
        config = load_config()

    df = features_df.loc[regime_labels.index].copy()
    df["regime"] = regime_labels

    # 1. Check crisis alignment
    print("=" * 60)
    print("REGIME VALIDATION: Crisis Period Alignment")
    print("=" * 60)

    for name, (start, end) in CRISIS_PERIODS.items():
        mask = (df.index >= start) & (df.index <= end)
        if mask.sum() == 0:
            print(f"  {name}: No data in range")
            continue
        crisis_regimes = df.loc[mask, "regime"].value_counts(normalize=True) * 100
        bear_pct = crisis_regimes.get("Bear/Crisis", 0) + crisis_regimes.get("Extreme", 0)
        print(f"  {name}: Bear/Crisis={bear_pct:.1f}%, distribution: {crisis_regimes.to_dict()}")

    # 2. VIX by regime (if available)
    if "VIX" in df.columns:
        print(f"\nMean VIX by regime:")
        vix_by_regime = df.groupby("regime")["VIX"].mean()
        print(vix_by_regime.to_string())

    # 3. Generate visualization
    _plot_regime_chart(df, config)

    return df


def _plot_regime_chart(df, config):
    benchmark = config["equity"]["benchmark"]
    prices = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "equity_prices.parquet")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    # Color map for regimes
    colors = {
        "Bull": "#2ecc71",
        "Normal": "#f39c12",
        "Bear/Crisis": "#e74c3c",
        "Extreme": "#8e44ad",
    }

    # Panel 1: S&P 500 price with regime shading
    ax1 = axes[0]
    price = prices[benchmark].reindex(df.index)
    ax1.plot(df.index, price, color="black", linewidth=0.8)
    ax1.set_ylabel("S&P 500 Price")
    ax1.set_title("Market Regime Detection — HMM Results")

    # Shade by regime
    regimes = df["regime"]
    for regime_name, color in colors.items():
        mask = regimes == regime_name
        if mask.any():
            ax1.fill_between(df.index, price.min(), price.max(),
                             where=mask, alpha=0.2, color=color, label=regime_name)
    ax1.legend(loc="upper left")

    # Panel 2: VIX
    ax2 = axes[1]
    if "VIX" in df.columns:
        ax2.plot(df.index, df["VIX"], color="purple", linewidth=0.8)
        ax2.axhline(y=20, color="gray", linestyle="--", alpha=0.5)
        ax2.axhline(y=30, color="red", linestyle="--", alpha=0.5)
    ax2.set_ylabel("VIX")

    # Panel 3: Regime labels as colored bars
    ax3 = axes[2]
    for i, (regime_name, color) in enumerate(colors.items()):
        mask = regimes == regime_name
        if mask.any():
            ax3.fill_between(df.index, 0, 1, where=mask, color=color, alpha=0.7)
    ax3.set_ylabel("Regime")
    ax3.set_yticks([])
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out_path = PROJECT_ROOT / "data" / "processed" / "regime_chart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nRegime chart saved to {out_path}")


if __name__ == "__main__":
    from src.models.hmm import fit_hmm_with_bic_selection
    results = fit_hmm_with_bic_selection()
    features_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    validate_regimes(features_df, results["regime_labels"])
