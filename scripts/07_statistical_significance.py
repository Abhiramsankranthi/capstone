"""
07_statistical_significance.py
===============================
Statistical significance testing for all model results:
  1. Binomial test on directional accuracy (vs 50% null)
  2. Sharpe ratio confidence intervals (Ledoit-Wolf / Lo adjustment)
  3. Paired comparison: regime-conditioned vs agnostic (DeLong-style)
  4. Summary table with significance flags

Place this in: scripts/07_statistical_significance.py

Usage:
    source .venv/bin/activate
    python scripts/07_statistical_significance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

try:
    from src.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. BINOMIAL TEST FOR DIRECTIONAL ACCURACY
# =============================================================================
def binomial_test_directional_accuracy(metrics_df):
    """
    Test H0: directional accuracy = 50% (random)
    H1: directional accuracy > 50%
    Uses exact binomial test (one-sided).
    """
    print("=" * 70)
    print("TEST 1: BINOMIAL TEST — DIRECTIONAL ACCURACY vs 50%")
    print("=" * 70)

    results = []
    for _, row in metrics_df.iterrows():
        n = int(row["n_test"])
        k = int(round(row["dir_acc"] * n))  # number of correct predictions

        # One-sided binomial test: P(X >= k) under H0: p=0.5
        p_value = stats.binomtest(k, n, 0.5, alternative="greater").pvalue

        # 95% confidence interval for true accuracy (Clopper-Pearson exact)
        ci_low, ci_high = proportion_confint(k, n, alpha=0.05, method="beta")

        sig_flag = ""
        if p_value < 0.001:
            sig_flag = "***"
        elif p_value < 0.01:
            sig_flag = "**"
        elif p_value < 0.05:
            sig_flag = "*"
        else:
            sig_flag = "ns"

        results.append({
            "model": row["model"],
            "dir_acc": round(row["dir_acc"], 4),
            "n_correct": k,
            "n_test": n,
            "p_value_binom": round(p_value, 6),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "significance": sig_flag,
        })

        print(f"  {row['model']:<50s} DA={row['dir_acc']:.4f}  "
              f"p={p_value:.6f}  95%CI=[{ci_low:.4f}, {ci_high:.4f}]  {sig_flag}")

    return pd.DataFrame(results)


# =============================================================================
# 2. SHARPE RATIO CONFIDENCE INTERVALS
# =============================================================================
def sharpe_confidence_intervals(metrics_df):
    """
    Compute approximate confidence intervals for Sharpe ratios.

    Under standard assumptions, the standard error of the Sharpe ratio
    (annualized) is approximately:
        SE(SR) = sqrt((1 + 0.5 * SR^2) / n)

    This follows from Lo (2002) "The Statistics of Sharpe Ratios".
    We use n = number of test observations.
    """
    print("\n" + "=" * 70)
    print("TEST 2: SHARPE RATIO CONFIDENCE INTERVALS (Lo 2002)")
    print("=" * 70)

    results = []
    for _, row in metrics_df.iterrows():
        sr = row["sharpe"]
        n = int(row["n_test"])

        # Standard error of Sharpe ratio (Lo 2002 approximation)
        se_sr = np.sqrt((1 + 0.5 * sr ** 2) / n)

        ci_low = sr - 1.96 * se_sr
        ci_high = sr + 1.96 * se_sr

        # Test H0: SR = 0 (two-sided)
        z_stat = sr / se_sr
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        sig_flag = ""
        if p_value < 0.001:
            sig_flag = "***"
        elif p_value < 0.01:
            sig_flag = "**"
        elif p_value < 0.05:
            sig_flag = "*"
        else:
            sig_flag = "ns"

        results.append({
            "model": row["model"],
            "sharpe": round(sr, 4),
            "se_sharpe": round(se_sr, 4),
            "z_stat": round(z_stat, 4),
            "p_value_sharpe": round(p_value, 6),
            "sr_ci_95_low": round(ci_low, 4),
            "sr_ci_95_high": round(ci_high, 4),
            "significance": sig_flag,
        })

        print(f"  {row['model']:<50s} SR={sr:+.4f}  "
              f"SE={se_sr:.4f}  z={z_stat:+.4f}  p={p_value:.6f}  "
              f"95%CI=[{ci_low:+.4f}, {ci_high:+.4f}]  {sig_flag}")

    return pd.DataFrame(results)


# =============================================================================
# 3. PAIRED COMPARISON: REGIME-CONDITIONED vs AGNOSTIC
# =============================================================================
def paired_comparison(metrics_df):
    """
    For each model type × target, compare regime-conditioned vs agnostic
    using a z-test on the difference in directional accuracy proportions.

    H0: DA_conditioned = DA_agnostic
    H1: DA_conditioned > DA_agnostic (one-sided)
    """
    print("\n" + "=" * 70)
    print("TEST 3: PAIRED COMPARISON — REGIME-CONDITIONED vs AGNOSTIC")
    print("=" * 70)

    results = []

    # Find all agnostic models
    agnostic_models = metrics_df[metrics_df["model"].str.contains("agnostic")]

    for _, ag_row in agnostic_models.iterrows():
        ag_name = ag_row["model"]
        cond_name = ag_name.replace("agnostic", "conditioned")

        cond_row = metrics_df[metrics_df["model"] == cond_name]
        if cond_row.empty:
            continue
        cond_row = cond_row.iloc[0]

        # Two-proportion z-test
        n1 = int(ag_row["n_test"])
        n2 = int(cond_row["n_test"])
        p1 = ag_row["dir_acc"]
        p2 = cond_row["dir_acc"]
        k1 = int(round(p1 * n1))
        k2 = int(round(p2 * n2))

        # Pooled proportion
        p_pool = (k1 + k2) / (n1 + n2)
        se_diff = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        if se_diff == 0:
            z_stat = 0
            p_value = 1.0
        else:
            z_stat = (p2 - p1) / se_diff
            p_value = 1 - stats.norm.cdf(z_stat)  # one-sided: conditioned > agnostic

        diff_da = p2 - p1
        diff_sharpe = cond_row["sharpe"] - ag_row["sharpe"]

        sig_flag = ""
        if p_value < 0.001:
            sig_flag = "***"
        elif p_value < 0.01:
            sig_flag = "**"
        elif p_value < 0.05:
            sig_flag = "*"
        else:
            sig_flag = "ns"

        # Extract model type and target for readability
        parts = ag_name.split("_")
        model_type = parts[0]
        target = "_".join(parts[1:3])

        results.append({
            "model_type": model_type,
            "target": target,
            "da_agnostic": round(p1, 4),
            "da_conditioned": round(p2, 4),
            "da_diff": round(diff_da, 4),
            "z_stat": round(z_stat, 4),
            "p_value_paired": round(p_value, 6),
            "sharpe_agnostic": round(ag_row["sharpe"], 4),
            "sharpe_conditioned": round(cond_row["sharpe"], 4),
            "sharpe_diff": round(diff_sharpe, 4),
            "significance": sig_flag,
        })

        print(f"\n  {model_type} / {target}:")
        print(f"    Agnostic:     DA={p1:.4f}  SR={ag_row['sharpe']:+.4f}")
        print(f"    Conditioned:  DA={p2:.4f}  SR={cond_row['sharpe']:+.4f}")
        print(f"    Δ DA={diff_da:+.4f}  Δ SR={diff_sharpe:+.4f}  "
              f"z={z_stat:+.4f}  p={p_value:.6f}  {sig_flag}")

    return pd.DataFrame(results)


# =============================================================================
# 4. VISUALIZATION
# =============================================================================
def plot_significance_summary(binom_df, sharpe_df, metrics_df):
    """Create a combined significance dashboard."""

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: Directional Accuracy with CIs ──────────────────────────────
    ax1 = axes[0]
    models = binom_df["model"].values
    y_pos = np.arange(len(models))
    da_vals = binom_df["dir_acc"].values
    ci_low = binom_df["ci_95_low"].values
    ci_high = binom_df["ci_95_high"].values
    errors = np.array([da_vals - ci_low, ci_high - da_vals])

    colors = ["#2ecc71" if p < 0.05 else "#e74c3c" for p in binom_df["p_value_binom"]]

    ax1.barh(y_pos, da_vals, xerr=errors, color=colors, alpha=0.7,
             capsize=3, edgecolor="black", linewidth=0.5)
    ax1.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Random (50%)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m.replace("_regime_", "\n") for m in models], fontsize=7)
    ax1.set_xlabel("Directional Accuracy")
    ax1.set_title("Directional Accuracy with 95% CI\n(Green = significant at p<0.05)")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0.46, 0.56)

    # ── Right: Sharpe Ratios with CIs ────────────────────────────────────
    ax2 = axes[1]
    sr_vals = sharpe_df["sharpe"].values
    sr_ci_low = sharpe_df["sr_ci_95_low"].values
    sr_ci_high = sharpe_df["sr_ci_95_high"].values
    sr_errors = np.array([sr_vals - sr_ci_low, sr_ci_high - sr_vals])

    colors_sr = ["#2ecc71" if p < 0.05 else "#e74c3c" for p in sharpe_df["p_value_sharpe"]]

    ax2.barh(y_pos, sr_vals, xerr=sr_errors, color=colors_sr, alpha=0.7,
             capsize=3, edgecolor="black", linewidth=0.5)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Sharpe")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m.replace("_regime_", "\n") for m in models], fontsize=7)
    ax2.set_xlabel("Annualized Sharpe Ratio")
    ax2.set_title("Sharpe Ratio with 95% CI (Lo 2002)\n(Green = significant at p<0.05)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / "statistical_significance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_paired_comparison(paired_df):
    """Bar chart showing DA improvement from regime conditioning."""
    if paired_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f"{row['model_type']}\n{row['target']}" for _, row in paired_df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, paired_df["da_agnostic"], width, label="Agnostic",
           color="steelblue", alpha=0.8)
    ax.bar(x + width/2, paired_df["da_conditioned"], width, label="Regime-Conditioned",
           color="coral", alpha=0.8)

    # Add significance stars
    for i, (_, row) in enumerate(paired_df.iterrows()):
        if row["significance"] != "ns":
            max_da = max(row["da_agnostic"], row["da_conditioned"])
            ax.text(i, max_da + 0.002, row["significance"],
                    ha="center", fontsize=12, fontweight="bold")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Model / Target")
    ax.set_ylabel("Directional Accuracy")
    ax.set_title("Regime-Conditioned vs Agnostic: Directional Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0.48, 0.55)

    plt.tight_layout()
    path = OUTPUT_DIR / "paired_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTING — ALL MODEL RESULTS")
    print("=" * 70)

    # Load metrics
    metrics_path = DATA_DIR / "model_metrics.csv"
    metrics_df = pd.read_csv(metrics_path)
    print(f"\nLoaded {len(metrics_df)} model results from {metrics_path}\n")

    # ── Test 1: Binomial test on directional accuracy ─────────────────────
    binom_df = binomial_test_directional_accuracy(metrics_df)

    # ── Test 2: Sharpe ratio confidence intervals ─────────────────────────
    sharpe_df = sharpe_confidence_intervals(metrics_df)

    # ── Test 3: Paired comparison ─────────────────────────────────────────
    paired_df = paired_comparison(metrics_df)

    # ── Save all results ──────────────────────────────────────────────────
    # Merge binomial and sharpe results into one comprehensive table
    combined = binom_df.merge(
        sharpe_df[["model", "sharpe", "se_sharpe", "z_stat",
                    "p_value_sharpe", "sr_ci_95_low", "sr_ci_95_high"]],
        on="model",
        suffixes=("_da", "_sr"),
    )

    # Rename for clarity
    combined = combined.rename(columns={"significance": "da_significance"})
    combined["sr_significance"] = sharpe_df["significance"].values

    combined_path = OUTPUT_DIR / "significance_tests.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined significance table saved: {combined_path}")

    paired_path = OUTPUT_DIR / "paired_comparison.csv"
    paired_df.to_csv(paired_path, index=False)
    print(f"Paired comparison saved: {paired_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_significance_summary(binom_df, sharpe_df, metrics_df)
    plot_paired_comparison(paired_df)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_sig_da = (binom_df["p_value_binom"] < 0.05).sum()
    n_sig_sr = (sharpe_df["p_value_sharpe"] < 0.05).sum()
    n_sig_paired = (paired_df["p_value_paired"] < 0.05).sum() if not paired_df.empty else 0

    print(f"Models with significant DA (p<0.05):     {n_sig_da}/{len(binom_df)}")
    print(f"Models with significant Sharpe (p<0.05):  {n_sig_sr}/{len(sharpe_df)}")
    print(f"Paired tests where conditioning helps (p<0.05): {n_sig_paired}/{len(paired_df)}")

    print(f"\nOutputs:")
    print(f"  - {combined_path}")
    print(f"  - {paired_path}")
    print(f"  - {OUTPUT_DIR / 'statistical_significance.png'}")
    print(f"  - {OUTPUT_DIR / 'paired_comparison.png'}")
    print(f"\n=== Statistical Significance Testing Complete ===")


if __name__ == "__main__":
    main()