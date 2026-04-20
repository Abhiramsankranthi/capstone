"""
SHAP stability & regime interpretability analysis for LightGBM models.
Outputs: per-regime SHAP summary plots, stability heatmap, shap_values CSV.

Usage:
    source .venv/bin/activate
    python scripts/08_shap_analysis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lightgbm as lgb
import shap
from scipy.stats import spearmanr

from src.config import load_config, PROJECT_ROOT

REGIMES = ["Bull", "Normal", "Bear/Crisis", "Extreme"]
REGIME_COLORS = {"Bull": "#2ecc71", "Normal": "#3498db", "Bear/Crisis": "#e74c3c", "Extreme": "#9b59b6"}
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    features_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    regime_labels = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "regime_labels.parquet")["regime"]
    return features_df, regime_labels


def get_feature_cols(features_df):
    drop = {"fwd_return_1d", "fwd_return_5d"}
    return [c for c in features_df.columns if c not in drop]


# ── Model ─────────────────────────────────────────────────────────────────────
def train_lgb(X_train, y_train):
    params = dict(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        verbosity=-1,
        n_jobs=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


# ── SHAP helpers ──────────────────────────────────────────────────────────────
def compute_shap(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X, check_additivity=False)
    return pd.DataFrame(sv, index=X.index, columns=X.columns)


def mean_abs_shap(shap_df):
    return shap_df.abs().mean().sort_values(ascending=False)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_shap_summary(shap_df, X, title, out_path, top_n=20):
    """Horizontal bar of mean |SHAP| with a beeswarm-style dot overlay."""
    imp = mean_abs_shap(shap_df).head(top_n)
    feats = imp.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.5]})

    # Left: bar
    ax = axes[0]
    ax.barh(feats[::-1], imp.values[::-1], color="steelblue", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{title}\nMean |SHAP|", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # Right: dot strip (feature value → color, SHAP → x-axis)
    ax2 = axes[1]
    for i, feat in enumerate(feats[::-1]):
        sv = shap_df[feat].values
        fv = X[feat].values
        # normalize feature values 0-1 for color
        fv_norm = (fv - np.nanpercentile(fv, 5)) / (
            np.nanpercentile(fv, 95) - np.nanpercentile(fv, 5) + 1e-9
        )
        fv_norm = np.clip(fv_norm, 0, 1)
        sc = ax2.scatter(sv, np.full_like(sv, i) + np.random.uniform(-0.2, 0.2, len(sv)),
                         c=fv_norm, cmap="coolwarm", alpha=0.3, s=6, linewidths=0)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(feats[::-1], fontsize=8)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP value (impact on prediction)")
    ax2.set_title("Feature value (blue=low, red=high)", fontsize=10)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_regime_comparison(regime_shap: dict, top_n=15, out_path=None):
    """Side-by-side top-N |SHAP| bars per regime."""
    n_regimes = len(regime_shap)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 6), sharey=False)
    if n_regimes == 1:
        axes = [axes]

    all_top = set()
    for imp in regime_shap.values():
        all_top.update(imp.head(top_n).index.tolist())

    for ax, (regime, imp) in zip(axes, regime_shap.items()):
        top = imp.head(top_n)
        color = REGIME_COLORS.get(regime, "gray")
        ax.barh(top.index[::-1], top.values[::-1], color=color, alpha=0.85)
        ax.set_title(regime, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Mean |SHAP|")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Feature Importance by Regime (SHAP)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_stability_heatmap(regime_shap: dict, overall_imp: pd.Series, top_n=20, out_path=None):
    """Heatmap of mean |SHAP| rank across regimes — shows stability."""
    top_feats = overall_imp.head(top_n).index.tolist()

    # Build rank matrix
    rank_data = {}
    for regime, imp in regime_shap.items():
        ranked = imp.rank(ascending=False)
        rank_data[regime] = ranked.reindex(top_feats).fillna(top_n + 5)

    rank_df = pd.DataFrame(rank_data, index=top_feats)

    # Spearman correlation between regime rank vectors
    regime_names = list(regime_shap.keys())
    corr_matrix = np.ones((len(regime_names), len(regime_names)))
    for i, r1 in enumerate(regime_names):
        for j, r2 in enumerate(regime_names):
            if i != j:
                rho, _ = spearmanr(rank_df[r1].values, rank_df[r2].values)
                corr_matrix[i, j] = rho

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: rank heatmap
    ax = axes[0]
    im = ax.imshow(rank_df.values.T, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(regime_names)))
    ax.set_yticklabels(regime_names)
    ax.set_title("Feature Rank by Regime\n(darker = higher importance)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Rank")

    # Annotate ranks
    for i in range(len(top_feats)):
        for j in range(len(regime_names)):
            ax.text(i, j, f"{rank_df.iloc[i, j]:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if rank_df.iloc[i, j] > top_n / 2 else "black")

    # Right: Spearman correlation
    ax2 = axes[1]
    im2 = ax2.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(regime_names)))
    ax2.set_xticklabels(regime_names, rotation=30, ha="right")
    ax2.set_yticks(range(len(regime_names)))
    ax2.set_yticklabels(regime_names)
    ax2.set_title("SHAP Rank Stability\n(Spearman ρ between regimes)", fontsize=11)
    plt.colorbar(im2, ax=ax2, label="Spearman ρ")
    for i in range(len(regime_names)):
        for j in range(len(regime_names)):
            ax2.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    print("Loading data...")
    features_df, regime_labels = load_data()
    feat_cols = get_feature_cols(features_df)

    for target in ["fwd_return_1d", "fwd_return_5d"]:
        print(f"\n{'='*60}")
        print(f"SHAP ANALYSIS — {target}")
        print(f"{'='*60}")

        # Drop columns that are entirely NaN, impute the rest with median
        work_df = features_df[feat_cols + [target]].copy()
        all_nan_cols = [c for c in feat_cols if work_df[c].isna().all()]
        if all_nan_cols:
            print(f"  Dropping all-NaN columns: {all_nan_cols}")
        use_feat_cols = [c for c in feat_cols if c not in all_nan_cols]
        work_df = work_df[use_feat_cols + [target]].dropna(subset=[target])
        work_df[use_feat_cols] = work_df[use_feat_cols].fillna(work_df[use_feat_cols].median())

        regime_aligned = regime_labels.reindex(work_df.index).dropna()
        df = work_df.loc[regime_aligned.index]

        # Time-based 70/30 split
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        regime_test = regime_aligned.iloc[split_idx:]

        X_train = train_df[use_feat_cols]
        y_train = train_df[target]
        X_test = test_df[use_feat_cols]
        y_test = test_df[target]

        print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

        # ── Overall model ──────────────────────────────────────────────────
        print("  Training overall LightGBM...")
        model = train_lgb(X_train, y_train)

        print("  Computing SHAP values (overall)...")
        shap_df = compute_shap(model, X_test)
        overall_imp = mean_abs_shap(shap_df)

        # Save SHAP values
        shap_path = OUT_DIR / f"shap_values_{target}.parquet"
        shap_df.to_parquet(shap_path)
        print(f"  Saved: {shap_path}")

        # Save mean |SHAP| summary
        summary_path = OUT_DIR / f"shap_summary_{target}.csv"
        overall_imp.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"}).to_csv(
            summary_path, index=False
        )
        print(f"  Saved: {summary_path}")

        # Overall summary plot
        plot_shap_summary(
            shap_df, X_test,
            title=f"Overall — {target}",
            out_path=OUT_DIR / f"shap_summary_{target}.png",
        )

        # ── Per-regime SHAP ────────────────────────────────────────────────
        regime_shap = {}
        for regime in REGIMES:
            idx = regime_test[regime_test == regime].index.intersection(X_test.index)
            if len(idx) < 30:
                print(f"  Skipping {regime} (only {len(idx)} samples)")
                continue
            print(f"  Computing SHAP for {regime} ({len(idx)} samples)...")
            sv_regime = compute_shap(model, X_test.loc[idx])
            regime_shap[regime] = mean_abs_shap(sv_regime)

            plot_shap_summary(
                sv_regime, X_test.loc[idx],
                title=f"{regime} Regime — {target}",
                out_path=OUT_DIR / f"shap_{regime.replace('/', '_')}_{target}.png",
            )

        # ── Regime comparison ──────────────────────────────────────────────
        if regime_shap:
            plot_regime_comparison(
                regime_shap, top_n=15,
                out_path=OUT_DIR / f"shap_regime_comparison_{target}.png",
            )
            plot_stability_heatmap(
                regime_shap, overall_imp, top_n=20,
                out_path=OUT_DIR / f"shap_stability_{target}.png",
            )

            # Print stability summary
            print(f"\n  --- SHAP Rank Stability ({target}) ---")
            regime_names = list(regime_shap.keys())
            for i in range(len(regime_names)):
                for j in range(i + 1, len(regime_names)):
                    r1, r2 = regime_names[i], regime_names[j]
                    top20_feats = overall_imp.head(20).index
                    rho, _ = spearmanr(
                        regime_shap[r1].reindex(top20_feats).fillna(0),
                        regime_shap[r2].reindex(top20_feats).fillna(0),
                    )
                    print(f"  {r1} vs {r2}: ρ={rho:.3f}")

    print("\n=== SHAP Analysis Complete ===")


if __name__ == "__main__":
    main()
