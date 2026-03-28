"""
Train all structured ML models and compare regime-agnostic vs regime-conditioned.
Outputs: metrics table, feature importance CSV, and importance plot.

Usage:
    source .venv/bin/activate
    python scripts/05_train_models.py [--quick]

Options:
    --quick   Use n_trials=10 for Optuna (faster, for smoke-testing)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless

from src.config import load_config, PROJECT_ROOT
from src.models.ridge_lasso import run_ridge_lasso
from src.models.lightgbm_model import run_lightgbm


def build_metrics_table(all_results: dict) -> pd.DataFrame:
    rows = []
    for key, val in all_results.items():
        if "metrics" not in val:
            continue
        m = val["metrics"]
        rows.append({
            "model": key,
            "rmse": round(m["rmse"], 6),
            "dir_acc": round(m["dir_acc"], 4),
            "sharpe": round(m["sharpe"], 4),
            "n_test": val.get("n_test", ""),
        })
    df = pd.DataFrame(rows).sort_values(["model"])
    return df


def plot_feature_importance(lgb_results: dict, out_dir: Path):
    """Save feature importance bar charts for LightGBM models."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, val in lgb_results.items():
        if "feature_importance" not in val or val["feature_importance"].empty:
            continue
        imp = val["feature_importance"]

        # If regime-conditioned, aggregate across regimes
        if "regime" in imp.columns:
            imp = imp.groupby("feature")["importance_gain"].mean().reset_index()
            imp = imp.sort_values("importance_gain", ascending=False)
        else:
            imp = imp.head(20)

        fig, ax = plt.subplots(figsize=(10, 6))
        top = imp.head(20)
        ax.barh(top["feature"][::-1], top["importance_gain"][::-1], color="steelblue")
        ax.set_xlabel("Mean Gain")
        ax.set_title(f"Feature Importance — {key}")
        plt.tight_layout()
        fname = out_dir / f"feature_importance_{key}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_prediction_vs_actual(results: dict, features_df: pd.DataFrame, out_dir: Path):
    """Rolling directional accuracy plot for each model × target."""
    out_dir.mkdir(parents=True, exist_ok=True)
    window = 252  # 1-year rolling

    for key, val in results.items():
        if "predictions" not in val:
            continue
        preds = val["predictions"]
        target = "fwd_return_1d" if "1d" in key else "fwd_return_5d"
        actual = features_df[target].reindex(preds.index)
        dir_correct = (np.sign(preds) == np.sign(actual)).astype(float)
        rolling_acc = dir_correct.rolling(window).mean()

        fig, ax = plt.subplots(figsize=(12, 4))
        rolling_acc.plot(ax=ax, color="royalblue")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (50%)")
        ax.set_title(f"Rolling {window}d Directional Accuracy — {key}")
        ax.set_ylabel("Directional Accuracy")
        ax.legend()
        plt.tight_layout()
        fname = out_dir / f"rolling_dir_acc_{key}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use n_trials=10 for fast testing")
    args = parser.parse_args()
    n_trials = 10 if args.quick else 50

    config = load_config()
    features_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    regime_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "regime_labels.parquet")
    regime_labels = regime_df["regime"]

    print(f"\nFeatures: {features_df.shape}  |  Regimes: {regime_labels.value_counts().to_dict()}")

    all_results = {}

    # ── Ridge / Lasso ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RIDGE / LASSO — REGIME-AGNOSTIC")
    print("="*60)
    rl_agnostic = run_ridge_lasso(features_df, regime_labels, regime_conditioned=False, config=config)
    all_results.update(rl_agnostic)

    print("\n" + "="*60)
    print("RIDGE / LASSO — REGIME-CONDITIONED")
    print("="*60)
    rl_conditioned = run_ridge_lasso(features_df, regime_labels, regime_conditioned=True, config=config)
    all_results.update(rl_conditioned)

    # ── LightGBM ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"LIGHTGBM — REGIME-AGNOSTIC (n_trials={n_trials})")
    print("="*60)
    lgb_agnostic = run_lightgbm(features_df, regime_labels, regime_conditioned=False,
                                 n_optuna_trials=n_trials, config=config)
    all_results.update(lgb_agnostic)

    print("\n" + "="*60)
    print(f"LIGHTGBM — REGIME-CONDITIONED (n_trials={n_trials})")
    print("="*60)
    lgb_conditioned = run_lightgbm(features_df, regime_labels, regime_conditioned=True,
                                    n_optuna_trials=n_trials, config=config)
    all_results.update(lgb_conditioned)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    metrics_df = build_metrics_table(all_results)
    print(metrics_df.to_string(index=False))

    out_dir = PROJECT_ROOT / "data" / "processed"
    metrics_path = out_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved: {metrics_path}")

    # ── Feature importance ─────────────────────────────────────────────────────
    lgb_results = {k: v for k, v in all_results.items() if k.startswith("lgb_")}
    charts_dir = PROJECT_ROOT / "data" / "processed"
    print("\nGenerating feature importance plots...")
    plot_feature_importance(lgb_results, charts_dir)

    print("\nGenerating directional accuracy plots...")
    plot_prediction_vs_actual(all_results, features_df, charts_dir)

    # Save feature importance tables
    for key, val in lgb_results.items():
        if "feature_importance" in val and not val["feature_importance"].empty:
            imp = val["feature_importance"]
            imp_path = out_dir / f"feature_importance_{key}.csv"
            imp.to_csv(imp_path, index=False)
            print(f"  Saved: {imp_path}")

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
