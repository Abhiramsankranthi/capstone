"""
06_hmm_refinement.py
====================
HMM Refinement: BIC sweep (3–5 states), Markov Switching comparison,
crisis validation, and statistical significance testing.

Place this in: scripts/06_hmm_refinement.py

Usage:
    source .venv/bin/activate
    python scripts/06_hmm_refinement.py
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
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy import stats
from sklearn.preprocessing import StandardScaler

# ── Adjust this to match your project root ──────────────────────────────────
try:
    from src.config import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_features():
    """Load the master feature dataset."""
    path = DATA_DIR / "features.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded features: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df


# =============================================================================
# 2. BIC SWEEP — GAUSSIAN HMM (3, 4, 5 STATES)
# =============================================================================
def run_bic_sweep(df, state_range=(3, 4, 5), n_fits=10, random_state=42):
    """
    Fit Gaussian HMM for each state count, select best by BIC.

    We fit multiple random initializations per state count and keep
    the best (highest log-likelihood) to avoid local optima.

    Returns:
        bic_results: DataFrame with columns [n_states, log_likelihood, bic, aic]
        best_model: the fitted HMM model with lowest BIC
        best_n: optimal number of states
    """
    # Select features for HMM: S&P 500 log returns + realized vol
    hmm_features = ['sp500_log_return', 'sp500_realized_vol_5d', 'sp500_realized_vol_21d']

    # Verify all columns exist
    missing = [c for c in hmm_features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing HMM columns: {missing}. Available: {df.columns.tolist()}")

    print(f"\nHMM features: {hmm_features}")

    X_raw = df[hmm_features].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    n_samples = X.shape[0]
    n_features = X.shape[1]

    results = []
    models = {}

    for n_states in state_range:
        print(f"\n--- Fitting HMM with {n_states} states ({n_fits} random inits) ---")
        best_ll = -np.inf
        best_model_for_n = None

        for i in range(n_fits):
            try:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=200,
                    tol=1e-4,
                    random_state=random_state + i,
                )
                model.fit(X)
                ll = model.score(X)

                if ll > best_ll:
                    best_ll = ll
                    best_model_for_n = model
            except Exception as e:
                print(f"  Init {i} failed: {e}")
                continue

        if best_model_for_n is None:
            print(f"  WARNING: All fits failed for n_states={n_states}")
            continue

        # Compute BIC and AIC
        # Number of free parameters for Gaussian HMM with full covariance:
        # Transition matrix: n_states * (n_states - 1)
        # Start probs: n_states - 1
        # Means: n_states * n_features
        # Covariances (full): n_states * n_features * (n_features + 1) / 2
        n_params = (
            n_states * (n_states - 1)
            + (n_states - 1)
            + n_states * n_features
            + n_states * n_features * (n_features + 1) // 2
        )

        bic = -2 * best_ll + n_params * np.log(n_samples)
        aic = -2 * best_ll + 2 * n_params

        results.append({
            "n_states": n_states,
            "log_likelihood": round(best_ll, 2),
            "n_params": n_params,
            "bic": round(bic, 2),
            "aic": round(aic, 2),
        })
        models[n_states] = best_model_for_n

        print(f"  Best LL: {best_ll:.2f} | BIC: {bic:.2f} | AIC: {aic:.2f}")

    bic_df = pd.DataFrame(results)
    best_n = int(bic_df.loc[bic_df["bic"].idxmin(), "n_states"])

    print(f"\n{'='*50}")
    print(f"BIC SWEEP RESULTS")
    print(f"{'='*50}")
    print(bic_df.to_string(index=False))
    print(f"\nOptimal state count (by BIC): {best_n}")

    return bic_df, models[best_n], best_n, X_raw, X, scaler


# =============================================================================
# 3. EXTRACT AND LABEL REGIMES
# =============================================================================
def label_regimes(model, X_scaled, index, df):
    """
    Decode hidden states and assign interpretable labels
    based on mean return and mean volatility of each state.
    """
    states = model.predict(X_scaled)
    regime_series = pd.Series(states, index=index, name="regime")

    # Compute regime statistics
    regime_stats = []
    for s in range(model.n_components):
        mask = states == s
        regime_stats.append({
            "state": s,
            "pct_days": round(mask.mean() * 100, 1),
            "mean_return": round(df.loc[index, "sp500_log_return"].values[mask].mean() * 100, 4),
            "mean_vix": round(df.loc[index, "VIX"].values[mask].mean(), 1)
                if "VIX" in df.columns else 0,
        })

    stats_df = pd.DataFrame(regime_stats).sort_values("mean_return", ascending=False)

    # Assign labels based on ordering
    label_map = {}
    sorted_states = stats_df["state"].tolist()

    if len(sorted_states) == 3:
        labels = ["Bull", "Normal", "Bear/Crisis"]
    elif len(sorted_states) == 4:
        labels = ["Bull", "Normal", "Bear/Crisis", "Extreme Vol"]
    elif len(sorted_states) == 5:
        labels = ["Strong Bull", "Bull", "Normal", "Bear", "Extreme Crisis"]
    else:
        labels = [f"Regime_{i}" for i in range(len(sorted_states))]

    for i, state in enumerate(sorted_states):
        label_map[state] = labels[i]

    stats_df["label"] = stats_df["state"].map(label_map)

    print(f"\n{'='*50}")
    print("REGIME STATISTICS")
    print(f"{'='*50}")
    print(stats_df[["state", "label", "pct_days", "mean_return", "mean_vix"]].to_string(index=False))

    return regime_series, label_map, stats_df


# =============================================================================
# 4. CRISIS VALIDATION
# =============================================================================
def validate_against_crises(regime_series, label_map, df):
    """
    Check what percentage of known crisis periods are classified
    as Bear/Crisis or Extreme regimes.
    """
    crisis_periods = {
        "2008 GFC": ("2008-09-01", "2009-03-31"),
        "2020 COVID": ("2020-02-15", "2020-04-30"),
        "2022 Rate Hikes": ("2022-01-01", "2022-10-31"),
    }

    # Identify which state labels correspond to "bad" regimes
    bad_labels = [v for v in label_map.values()
                  if any(word in v.lower() for word in ["bear", "crisis", "extreme"])]

    bad_states = [k for k, v in label_map.items() if v in bad_labels]

    print(f"\n{'='*50}")
    print("CRISIS VALIDATION")
    print(f"{'='*50}")
    print(f"Bad regime labels: {bad_labels}")
    print(f"Bad regime state IDs: {bad_states}\n")

    validation_results = []

    for name, (start, end) in crisis_periods.items():
        mask = (regime_series.index >= start) & (regime_series.index <= end)
        crisis_regimes = regime_series[mask]

        if len(crisis_regimes) == 0:
            print(f"  {name}: No data in range")
            continue

        crisis_as_bad = crisis_regimes.isin(bad_states).mean() * 100

        validation_results.append({
            "crisis": name,
            "start": start,
            "end": end,
            "n_days": len(crisis_regimes),
            "pct_classified_crisis": round(crisis_as_bad, 1),
        })

        print(f"  {name}: {crisis_as_bad:.1f}% classified as Bear/Crisis/Extreme "
              f"({len(crisis_regimes)} trading days)")

    return pd.DataFrame(validation_results)


# =============================================================================
# 5. MARKOV SWITCHING REGRESSION COMPARISON
# =============================================================================
def run_markov_switching(df, n_regimes=4):
    """
    Fit a Markov Switching Dynamic Regression as an alternative to HMM.
    Uses log returns as the dependent variable.
    """
    print(f"\n{'='*50}")
    print(f"MARKOV SWITCHING REGRESSION ({n_regimes} regimes)")
    print(f"{'='*50}")

    # Prepare the data
    if "sp500_log_return" not in df.columns:
        print("  ERROR: 'sp500_log_return' column not found. Skipping Markov Switching.")
        return None, None

    y = df["sp500_log_return"].dropna()

    try:
        ms_model = MarkovRegression(
            y,
            k_regimes=n_regimes,
            trend="c",          # constant (mean) per regime
            switching_variance=True,  # regime-dependent variance
        )
        ms_result = ms_model.fit(maxiter=200, em_iter=100)

        print(ms_result.summary())

        # Extract smoothed probabilities and most likely regimes
        smoothed_probs = ms_result.smoothed_marginal_probabilities
        ms_regimes = smoothed_probs.values.argmax(axis=1)
        ms_regime_series = pd.Series(ms_regimes, index=y.index, name="ms_regime")

        return ms_result, ms_regime_series

    except Exception as e:
        print(f"  Markov Switching failed: {e}")
        print("  Trying with 3 regimes as fallback...")
        try:
            ms_model = MarkovRegression(
                y,
                k_regimes=3,
                trend="c",
                switching_variance=True,
            )
            ms_result = ms_model.fit(maxiter=200, em_iter=100)
            print(ms_result.summary())

            smoothed_probs = ms_result.smoothed_marginal_probabilities
            ms_regimes = smoothed_probs.values.argmax(axis=1)
            ms_regime_series = pd.Series(ms_regimes, index=y.index, name="ms_regime")

            return ms_result, ms_regime_series

        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return None, None


# =============================================================================
# 6. COMPARE HMM vs MARKOV SWITCHING
# =============================================================================
def compare_hmm_ms(hmm_regimes, ms_regimes, label_map, df):
    """
    Compare regime assignments between HMM and Markov Switching.
    """
    if ms_regimes is None:
        print("\nMarkov Switching comparison skipped (model did not converge).")
        return None

    print(f"\n{'='*50}")
    print("HMM vs MARKOV SWITCHING COMPARISON")
    print(f"{'='*50}")

    # Align the two series
    common_idx = hmm_regimes.index.intersection(ms_regimes.index)
    hmm_aligned = hmm_regimes.loc[common_idx]
    ms_aligned = ms_regimes.loc[common_idx]

    # Build a contingency table
    ct = pd.crosstab(
        hmm_aligned.map(label_map).rename("HMM Regime"),
        ms_aligned.rename("MS Regime"),
        margins=True,
    )
    print("\nContingency Table (HMM rows × MS columns):")
    print(ct)

    # Compute agreement rate
    # Since labels may not match, we use the approach:
    # for each HMM regime, find which MS regime overlaps most
    agreement = 0
    for hmm_state in hmm_aligned.unique():
        hmm_mask = hmm_aligned == hmm_state
        ms_in_hmm = ms_aligned[hmm_mask]
        most_common_ms = ms_in_hmm.mode().iloc[0] if len(ms_in_hmm) > 0 else -1
        agreement += (ms_in_hmm == most_common_ms).sum()

    agreement_rate = agreement / len(common_idx) * 100
    print(f"\nBest-case agreement rate: {agreement_rate:.1f}%")
    print("(Each HMM regime mapped to its most-overlapping MS regime)")

    # Crisis-period comparison
    crisis_periods = {
        "2008 GFC": ("2008-09-01", "2009-03-31"),
        "2020 COVID": ("2020-02-15", "2020-04-30"),
    }

    print("\nCrisis period regime comparison:")
    for name, (start, end) in crisis_periods.items():
        mask = (common_idx >= start) & (common_idx <= end)
        if mask.sum() == 0:
            continue
        hmm_crisis = hmm_aligned[mask]
        ms_crisis = ms_aligned[mask]
        print(f"\n  {name}:")
        print(f"    HMM regime distribution: {hmm_crisis.map(label_map).value_counts().to_dict()}")
        print(f"    MS regime distribution:  {ms_crisis.value_counts().to_dict()}")

    return agreement_rate


# =============================================================================
# 7. PLOTS
# =============================================================================
def plot_bic_comparison(bic_df):
    """Bar chart of BIC and AIC across state counts."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(bic_df))
    width = 0.35

    ax.bar(x - width/2, bic_df["bic"], width, label="BIC", color="steelblue")
    ax.bar(x + width/2, bic_df["aic"], width, label="AIC", color="coral")

    ax.set_xlabel("Number of States")
    ax.set_ylabel("Information Criterion")
    ax.set_title("HMM Model Selection: BIC vs AIC")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in bic_df["n_states"]])
    ax.legend()

    # Mark the BIC-optimal
    best_idx = bic_df["bic"].idxmin()
    ax.annotate("← BIC optimal",
                xy=(best_idx - width/2, bic_df.loc[best_idx, "bic"]),
                fontsize=10, color="steelblue", fontweight="bold",
                xytext=(10, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="steelblue"))

    plt.tight_layout()
    path = OUTPUT_DIR / "hmm_bic_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_regime_timeline(regime_series, label_map, df):
    """Timeline plot showing regime labels overlaid on price."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # Top: price with regime coloring
    ax1 = axes[0]
    if "SPY_close" in df.columns:
        price_col = "SPY_close"
    elif "spy_close" in df.columns:
        price_col = "spy_close"
    else:
        # Find any close price column
        price_cols = [c for c in df.columns if "close" in c.lower() and "vix" not in c.lower()]
        price_col = price_cols[0] if price_cols else None

    if price_col:
        ax1.plot(df.index, df[price_col], color="black", linewidth=0.5, alpha=0.7)
        ax1.set_ylabel("Price")
        ax1.set_title("Market Regimes (HMM) with Price Overlay")

        # Color background by regime
        colors = {"Bull": "#2ecc71", "Normal": "#3498db", "Bear/Crisis": "#e74c3c",
                  "Extreme Vol": "#9b59b6", "Strong Bull": "#27ae60",
                  "Bear": "#c0392b", "Extreme Crisis": "#8e44ad"}

        for state, label in label_map.items():
            mask = regime_series == state
            segments = mask.astype(int).diff().fillna(0)
            starts = regime_series.index[segments == 1]
            ends = regime_series.index[segments == -1]

            # Handle edge cases
            if mask.iloc[0]:
                starts = starts.insert(0, regime_series.index[0])
            if mask.iloc[-1]:
                ends = ends.append(pd.DatetimeIndex([regime_series.index[-1]]))

            color = colors.get(label, "#95a5a6")
            for s, e in zip(starts, ends):
                ax1.axvspan(s, e, alpha=0.2, color=color, label=label)

        # Deduplicate legend
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=9)

    # Bottom: regime state as categorical
    ax2 = axes[1]
    regime_labeled = regime_series.map(label_map)
    unique_labels = regime_labeled.unique()
    label_to_num = {l: i for i, l in enumerate(sorted(unique_labels))}
    ax2.scatter(regime_series.index, regime_labeled.map(label_to_num),
                c=regime_series, cmap="Set1", s=1, alpha=0.5)
    ax2.set_yticks(list(label_to_num.values()))
    ax2.set_yticklabels(list(label_to_num.keys()))
    ax2.set_ylabel("Regime")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    path = OUTPUT_DIR / "regime_timeline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# 8. MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("HMM REFINEMENT — BIC Sweep + Markov Switching Comparison")
    print("=" * 60)

    # Load data
    df = load_features()

    # ── Step 1: BIC sweep ──────────────────────────────────────────────────
    bic_df, best_model, best_n, X_raw, X_scaled, scaler = run_bic_sweep(
        df, state_range=(3, 4, 5), n_fits=10
    )

    # Save BIC results
    bic_path = OUTPUT_DIR / "hmm_bic_sweep.csv"
    bic_df.to_csv(bic_path, index=False)
    print(f"\nBIC sweep saved: {bic_path}")

    # Plot BIC comparison
    plot_bic_comparison(bic_df)

    # ── Step 2: Label regimes for the BIC-optimal model ────────────────────
    regime_series, label_map, stats_df = label_regimes(
        best_model, X_scaled, X_raw.index, df
    )

    # Save regime stats
    stats_path = OUTPUT_DIR / "regime_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Regime stats saved: {stats_path}")

    # ── Step 3: Crisis validation ──────────────────────────────────────────
    crisis_df = validate_against_crises(regime_series, label_map, df)
    if not crisis_df.empty:
        crisis_path = OUTPUT_DIR / "crisis_validation.csv"
        crisis_df.to_csv(crisis_path, index=False)
        print(f"Crisis validation saved: {crisis_path}")

    # ── Step 4: Markov Switching comparison ────────────────────────────────
    ms_result, ms_regimes = run_markov_switching(df, n_regimes=best_n)
    agreement = compare_hmm_ms(regime_series, ms_regimes, label_map, df)

    # ── Step 5: Save final regime labels ───────────────────────────────────
    regime_out = regime_series.to_frame("regime")
    regime_out["regime_label"] = regime_series.map(label_map)
    regime_path = OUTPUT_DIR / "regime_labels.parquet"
    regime_out.to_parquet(regime_path)
    print(f"\nFinal regime labels saved: {regime_path}")

    # ── Step 6: Plot timeline ──────────────────────────────────────────────
    plot_regime_timeline(regime_series, label_map, df)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("REFINEMENT SUMMARY")
    print(f"{'='*60}")
    print(f"BIC-optimal states: {best_n}")
    print(f"Regimes: {label_map}")
    if agreement is not None:
        print(f"HMM–MS agreement: {agreement:.1f}%")
    print(f"\nOutputs in: {OUTPUT_DIR}")
    print(f"  - hmm_bic_sweep.csv")
    print(f"  - hmm_bic_sweep.png")
    print(f"  - regime_stats.csv")
    print(f"  - crisis_validation.csv")
    print(f"  - regime_labels.parquet")
    print(f"  - regime_timeline.png")
    print(f"\n=== HMM Refinement Complete ===")


if __name__ == "__main__":
    main()
