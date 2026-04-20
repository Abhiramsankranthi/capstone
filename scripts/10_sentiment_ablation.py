"""
Sentiment ablation study + predictive correlation analysis.
Measures the incremental value of FinBERT sentiment features.

Usage:
    source .venv/bin/activate
    python scripts/10_sentiment_ablation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr, spearmanr

from src.config import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "data" / "processed"

SENTIMENT_COLS = ["sent_mean", "sent_max_neg", "sent_article_count", "sent_weighted", "sent_momentum"]
TARGETS = ["fwd_return_1d", "fwd_return_5d"]


def load_data():
    df = pd.read_parquet(OUT_DIR / "features.parquet")
    regime = pd.read_parquet(OUT_DIR / "regime_labels.parquet")["regime"]
    return df, regime


def get_feature_sets(df: pd.DataFrame) -> tuple[list, list]:
    all_feats = [c for c in df.columns if c not in TARGETS]
    sent_available = [c for c in SENTIMENT_COLS if c in all_feats and df[c].notna().sum() > 100]
    no_sent = [c for c in all_feats if c not in SENTIMENT_COLS]
    with_sent = no_sent + sent_available
    return no_sent, with_sent, sent_available


def impute(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    d = df[cols].copy()
    # drop all-NaN columns
    d = d.drop(columns=[c for c in d.columns if d[c].isna().all()])
    d = d.fillna(d.median())
    return d


def train_lgb(X_train, y_train) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        verbosity=-1, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def walk_forward_metrics(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Expanding-window walk-forward evaluation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_preds, all_actual = [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if len(y_tr) < 100:
            continue
        model = train_lgb(X_tr, y_tr)
        preds = model.predict(X_te)
        all_preds.extend(preds)
        all_actual.extend(y_te.values)

    pred_s = np.array(all_preds)
    act_s = np.array(all_actual)
    valid = ~(np.isnan(pred_s) | np.isnan(act_s))
    pred_s, act_s = pred_s[valid], act_s[valid]
    if len(pred_s) < 20:
        return {}

    rmse = np.sqrt(np.mean((pred_s - act_s) ** 2))
    dir_acc = np.mean(np.sign(pred_s) == np.sign(act_s))
    strat_ret = np.sign(pred_s) * act_s
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    ic = float(spearmanr(pred_s, act_s).statistic)

    return {"rmse": round(rmse, 6), "dir_acc": round(dir_acc, 4),
            "sharpe": round(sharpe, 4), "ic": round(ic, 4), "n": len(pred_s)}


# ── Predictive correlation ────────────────────────────────────────────────────

def predictive_correlation(df: pd.DataFrame, sent_cols: list, targets: list) -> pd.DataFrame:
    rows = []
    for feat in sent_cols:
        for target in targets:
            sub = df[[feat, target]].dropna()
            if len(sub) < 30:
                continue
            pr, pp = pearsonr(sub[feat], sub[target])
            sr, sp = spearmanr(sub[feat], sub[target])
            # Lagged correlations (sentiment leads returns by 1, 5, 10 days)
            lags = {}
            for lag in [1, 5, 10]:
                s_lagged = sub[feat].shift(lag)
                sub2 = pd.DataFrame({"feat": s_lagged, "target": sub[target]}).dropna()
                if len(sub2) > 30:
                    lags[f"pearson_lag{lag}d"] = round(pearsonr(sub2["feat"], sub2["target"])[0], 4)
                else:
                    lags[f"pearson_lag{lag}d"] = np.nan
            rows.append({
                "sentiment_feature": feat, "target": target,
                "pearson_r": round(pr, 4), "pearson_p": round(pp, 4),
                "spearman_r": round(sr, 4), "spearman_p": round(sp, 4),
                "n_obs": len(sub),
                **lags,
            })
    return pd.DataFrame(rows)


def plot_ablation(ablation_df: pd.DataFrame, out_path: Path):
    targets = ablation_df["target"].unique()
    metrics = ["sharpe", "dir_acc", "rmse", "ic"]
    labels_map = {"sharpe": "Sharpe Ratio", "dir_acc": "Directional Acc",
                  "rmse": "RMSE (lower=better)", "ic": "Information Coef"}

    fig, axes = plt.subplots(len(metrics), len(targets),
                             figsize=(5 * len(targets), 3.5 * len(metrics)))
    if len(targets) == 1:
        axes = axes.reshape(-1, 1)

    colors = {"Without Sentiment": "#e74c3c", "With Sentiment": "#2ecc71"}

    for col_i, target in enumerate(targets):
        sub = ablation_df[ablation_df["target"] == target]
        for row_i, metric in enumerate(metrics):
            ax = axes[row_i][col_i]
            vals = {r["label"]: r.get(metric, 0) for _, r in sub.iterrows()}
            bars = ax.bar(list(vals.keys()), list(vals.values()),
                          color=[colors.get(k, "steelblue") for k in vals],
                          alpha=0.85, width=0.5)
            ax.set_title(f"{labels_map[metric]} — {target}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            # Improvement label
            if "Without Sentiment" in vals and "With Sentiment" in vals:
                diff = vals["With Sentiment"] - vals["Without Sentiment"]
                sign = "+" if diff > 0 else ""
                better = (diff > 0 and metric != "rmse") or (diff < 0 and metric == "rmse")
                ax.text(0.5, 0.92, f"Δ={sign}{diff:.4f}", transform=ax.transAxes,
                        ha="center", fontsize=9,
                        color="#27ae60" if better else "#c0392b",
                        fontweight="bold")

    plt.suptitle("Sentiment Ablation: With vs Without FinBERT Features", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_predictive_correlation(corr_df: pd.DataFrame, out_path: Path):
    targets = corr_df["target"].unique()
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        sub = corr_df[corr_df["target"] == target].copy()
        sub = sub.sort_values("spearman_r", key=abs, ascending=False)
        colors = ["#27ae60" if v > 0 else "#e74c3c" for v in sub["spearman_r"]]
        ax.barh(sub["sentiment_feature"], sub["spearman_r"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(f"Sentiment → {target}", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

        # Lag correlation overlay
        lag_cols = [c for c in sub.columns if "lag" in c]
        for lag_col in lag_cols[:1]:
            ax2 = ax.twinx()
            ax2.scatter(sub[lag_col], sub["sentiment_feature"],
                        color="navy", marker="D", s=40, alpha=0.7, label=lag_col)
            ax2.set_ylabel("1-day lagged Pearson r", fontsize=8)
            ax2.legend(loc="lower right", fontsize=7)

    plt.suptitle("Sentiment Features: Predictive Correlation with Forward Returns", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("\n=== SENTIMENT ABLATION + PREDICTIVE CORRELATION ===")
    df, regime = load_data()
    no_sent_feats, with_sent_feats, sent_cols = get_feature_sets(df)

    print(f"  Non-sentiment features: {len(no_sent_feats)}")
    print(f"  Sentiment features available: {sent_cols}")
    print(f"  Total with sentiment: {len(with_sent_feats)}")

    # ── Predictive correlation ────────────────────────────────────────────────
    print("\nComputing predictive correlations...")
    corr_df = predictive_correlation(df, sent_cols, TARGETS)
    corr_path = OUT_DIR / "sentiment_predictive_correlation.csv"
    corr_df.to_csv(corr_path, index=False)
    print(corr_df.to_string(index=False))
    print(f"  Saved: {corr_path}")

    plot_predictive_correlation(corr_df, OUT_DIR / "sentiment_predictive_correlation.png")

    # ── Ablation study ────────────────────────────────────────────────────────
    print("\nRunning ablation study (walk-forward LightGBM)...")
    ablation_rows = []

    # Only use rows where at least some sentiment data exists
    # (2012-2022 coverage) — use full dataset for fair comparison
    for target in TARGETS:
        print(f"\n  Target: {target}")
        work = df[no_sent_feats + [target]].dropna(subset=[target])
        X_no_sent = impute(work, no_sent_feats)
        y = work[target].reindex(X_no_sent.index)

        print("    Without sentiment...")
        m_no = walk_forward_metrics(X_no_sent, y)
        m_no["label"] = "Without Sentiment"
        m_no["target"] = target
        ablation_rows.append(m_no)
        print(f"      {m_no}")

        work_sent = df[with_sent_feats + [target]].dropna(subset=[target])
        X_with_sent = impute(work_sent, with_sent_feats)
        y_sent = work_sent[target].reindex(X_with_sent.index)

        print("    With sentiment...")
        m_with = walk_forward_metrics(X_with_sent, y_sent)
        m_with["label"] = "With Sentiment"
        m_with["target"] = target
        ablation_rows.append(m_with)
        print(f"      {m_with}")

    ablation_df = pd.DataFrame(ablation_rows)
    abl_path = OUT_DIR / "sentiment_ablation.csv"
    ablation_df.to_csv(abl_path, index=False)
    print(f"\nSaved: {abl_path}")

    # Summary delta
    print("\n  === SENTIMENT VALUE SUMMARY ===")
    for target in TARGETS:
        sub = ablation_df[ablation_df["target"] == target]
        if len(sub) < 2:
            continue
        no_row = sub[sub["label"] == "Without Sentiment"].iloc[0]
        with_row = sub[sub["label"] == "With Sentiment"].iloc[0]
        print(f"\n  {target}:")
        for m in ["sharpe", "dir_acc", "rmse", "ic"]:
            delta = with_row.get(m, 0) - no_row.get(m, 0)
            better = (delta > 0 and m != "rmse") or (delta < 0 and m == "rmse")
            sign = "✓" if better else "✗"
            print(f"    {sign} {m}: {no_row.get(m,'?')} → {with_row.get(m,'?')} (Δ={delta:+.4f})")

    plot_ablation(ablation_df, OUT_DIR / "sentiment_ablation.png")
    print("\n=== SENTIMENT ABLATION COMPLETE ===")


if __name__ == "__main__":
    main()
