"""
Markov Switching Model comparison vs HMM regime labels.
Fits a statsmodels Markov Switching Regression and compares regime assignments.

Usage:
    source .venv/bin/activate
    python scripts/09_markov_switching.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from sklearn.metrics import adjusted_rand_score

from src.config import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "data" / "processed"
REGIME_COLORS = {"Bull": "#2ecc71", "Normal": "#3498db", "Bear/Crisis": "#e74c3c", "Extreme": "#9b59b6"}


def load_data():
    features = pd.read_parquet(OUT_DIR / "features.parquet")
    hmm_labels = pd.read_parquet(OUT_DIR / "regime_labels.parquet")["regime"]
    returns = features["sp500_log_return"].dropna()
    vol = features["sp500_realized_vol_21d"].dropna()
    return returns, vol, hmm_labels


def fit_markov_switching(returns: pd.Series, k_regimes: int = 4) -> pd.DataFrame:
    """Fit Markov Switching AR(1) model on log returns."""
    print(f"  Fitting Markov Switching ({k_regimes} regimes) on {len(returns)} observations...")
    model = MarkovAutoregression(
        returns,
        k_regimes=k_regimes,
        order=1,
        switching_ar=False,
        switching_variance=True,
    )
    result = model.fit(
        search_reps=10,
        search_iter=50,
        disp=False,
        em_iter=100,
        maxiter=500,
    )
    return result


def label_ms_regimes(result, returns: pd.Series) -> pd.Series:
    """Label MS regimes by volatility level (low→Bull, high→Bear)."""
    smoothed = result.smoothed_marginal_probabilities
    ms_regime_idx = smoothed.values.argmax(axis=1)
    regime_series = pd.Series(ms_regime_idx, index=returns.index[1:])  # AR(1) loses 1 obs

    # Map indices by variance (ascending → Bull/Normal/Bear/Extreme)
    variances = {}
    for k in range(result.k_regimes):
        mask = regime_series == k
        if mask.sum() > 0:
            variances[k] = returns[1:][mask].std()

    sorted_by_var = sorted(variances, key=variances.get)
    labels_map_names = ["Bull", "Normal", "Bear/Crisis", "Extreme"]
    label_map = {orig: labels_map_names[i] for i, orig in enumerate(sorted_by_var)}
    return regime_series.map(label_map)


def compare_regimes(hmm: pd.Series, ms: pd.Series) -> dict:
    """Overlap and agreement metrics between HMM and MS regime assignments."""
    common = hmm.index.intersection(ms.index)
    h = hmm.loc[common]
    m = ms.loc[common]

    # Convert to numeric for ARI
    label_set = list(set(h.unique()) | set(m.unique()))
    label_enc = {l: i for i, l in enumerate(label_set)}
    h_enc = h.map(label_enc)
    m_enc = m.map(label_enc)

    ari = adjusted_rand_score(h_enc.values, m_enc.values)

    # Per-regime overlap (how often MS agrees with HMM)
    overlap = {}
    for regime in ["Bull", "Normal", "Bear/Crisis", "Extreme"]:
        hmm_mask = h == regime
        ms_mask = m == regime
        if hmm_mask.sum() == 0:
            continue
        agreement = (hmm_mask & ms_mask).sum() / hmm_mask.sum()
        overlap[regime] = round(float(agreement), 3)

    return {"ari": round(ari, 4), "overlap": overlap, "n_common": len(common)}


def plot_comparison(hmm: pd.Series, ms: pd.Series, returns: pd.Series, out_path: Path):
    """Side-by-side regime timeline comparison."""
    common = hmm.index.intersection(ms.index).intersection(returns.index)
    cum_ret = np.exp(returns.loc[common].cumsum())

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [2, 1, 1]})

    # Price
    ax = axes[0]
    ax.plot(cum_ret.index, cum_ret.values, color="black", linewidth=0.7)
    ax.set_ylabel("S&P 500 (normalized)")
    ax.set_title("HMM vs Markov Switching Regime Comparison")
    ax.grid(alpha=0.2)

    def shade_regimes(ax, regime_series, alpha=0.4):
        prev, span_start = None, None
        for dt in regime_series.index:
            r = regime_series.get(dt)
            if r != prev:
                if prev is not None and span_start is not None:
                    ax.axvspan(span_start, dt, alpha=alpha,
                               color=REGIME_COLORS.get(prev, "gray"), linewidth=0)
                span_start = dt
                prev = r
        if prev is not None and span_start is not None:
            ax.axvspan(span_start, regime_series.index[-1], alpha=alpha,
                       color=REGIME_COLORS.get(prev, "gray"), linewidth=0)

    # HMM
    ax2 = axes[1]
    ax2.plot(cum_ret.index, cum_ret.values, color="black", linewidth=0.5, alpha=0.6)
    shade_regimes(ax2, hmm.loc[common])
    ax2.set_ylabel("HMM Regimes")
    patches = [mpatches.Patch(color=c, alpha=0.6, label=r) for r, c in REGIME_COLORS.items()]
    ax2.legend(handles=patches, loc="upper left", fontsize=7, ncol=4)
    ax2.grid(alpha=0.2)

    # Markov Switching
    ax3 = axes[2]
    ax3.plot(cum_ret.index, cum_ret.values, color="black", linewidth=0.5, alpha=0.6)
    shade_regimes(ax3, ms.loc[common])
    ax3.set_ylabel("Markov Switching")
    ax3.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_transition_matrices(hmm_result, ms_result, out_path: Path):
    """Plot transition probability matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    regime_names = ["Bull", "Normal", "Bear/Crisis", "Extreme"]

    # HMM transition matrix (from hmmlearn)
    try:
        from hmmlearn import hmm as hmmlearn_hmm
        hmm_model_path = PROJECT_ROOT / "data" / "processed" / "hmm_model.pkl"
        if hmm_model_path.exists():
            import pickle
            with open(hmm_model_path, "rb") as f:
                hmm_model = pickle.load(f)
            trans_mat = hmm_model.transmat_
        else:
            trans_mat = None
    except Exception:
        trans_mat = None

    # MS transition matrix from statsmodels
    ms_trans = ms_result.transition_matrix() if hasattr(ms_result, "transition_matrix") else None

    for ax, (mat, title) in zip(axes, [
        (trans_mat, "HMM Transition Matrix"),
        (ms_trans, "Markov Switching Transition Matrix"),
    ]):
        if mat is None:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")
            ax.set_title(title)
            continue
        k = mat.shape[0]
        labels = regime_names[:k]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("To"); ax.set_ylabel("From")
        plt.colorbar(im, ax=ax)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if mat[i,j] > 0.6 else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("\n=== MARKOV SWITCHING COMPARISON ===")
    returns, vol, hmm_labels = load_data()

    print("Fitting Markov Switching model (4 regimes)...")
    try:
        ms_result = fit_markov_switching(returns, k_regimes=4)
        print(f"  Log-likelihood: {ms_result.llf:.2f}")
        print(f"  AIC: {ms_result.aic:.2f}  BIC: {ms_result.bic:.2f}")
    except Exception as e:
        print(f"  4-regime fit failed ({e}), trying 3 regimes...")
        ms_result = fit_markov_switching(returns, k_regimes=3)

    ms_labels = label_ms_regimes(ms_result, returns)

    # Compare
    print("\nComparing HMM vs Markov Switching regimes...")
    stats = compare_regimes(hmm_labels, ms_labels)
    print(f"  Adjusted Rand Index: {stats['ari']}")
    print(f"  Per-regime overlap: {stats['overlap']}")
    print(f"  Common observations: {stats['n_common']}")

    # Save labels
    ms_df = ms_labels.to_frame("ms_regime")
    ms_df.to_parquet(OUT_DIR / "ms_regime_labels.parquet")
    print(f"  Saved: {OUT_DIR / 'ms_regime_labels.parquet'}")

    # Comparison stats CSV
    pd.DataFrame([{
        "model": "MarkovSwitching_AR1",
        "ari_vs_hmm": stats["ari"],
        "n_common": stats["n_common"],
        **{f"overlap_{k}": v for k, v in stats["overlap"].items()},
        "aic": round(ms_result.aic, 2),
        "bic": round(ms_result.bic, 2),
        "loglik": round(ms_result.llf, 2),
    }]).to_csv(OUT_DIR / "ms_comparison.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'ms_comparison.csv'}")

    # Regime distribution comparison
    print("\nRegime distribution comparison:")
    hmm_dist = hmm_labels.value_counts(normalize=True).round(3)
    ms_dist = ms_labels.value_counts(normalize=True).round(3)
    dist_df = pd.DataFrame({"HMM": hmm_dist, "Markov Switching": ms_dist}).fillna(0)
    print(dist_df.to_string())
    dist_df.to_csv(OUT_DIR / "regime_distribution_comparison.csv")

    # Plots
    print("\nGenerating plots...")
    plot_comparison(hmm_labels, ms_labels, returns, OUT_DIR / "ms_regime_comparison.png")

    # Regime distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(dist_df))
    w = 0.35
    ax.bar(x - w/2, dist_df["HMM"].values, w, label="HMM", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, dist_df["Markov Switching"].values, w, label="Markov Switching", color="tomato", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(dist_df.index)
    ax.set_ylabel("Fraction of Time")
    ax.set_title("Regime Distribution: HMM vs Markov Switching")
    ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout()
    dist_path = OUT_DIR / "ms_distribution_comparison.png"
    fig.savefig(dist_path, dpi=130); plt.close(fig)
    print(f"  Saved: {dist_path}")

    print("\n=== MARKOV SWITCHING COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()
