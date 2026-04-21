"""
Train 12 final models (Ridge, Lasso, LightGBM × 1d/5d × agnostic/conditioned)
on full historical data and persist to disk for live ensemble inference.

Must be run once before live trading starts, and re-run monthly.

Usage:
    source .venv/bin/activate
    python scripts/11_save_models.py
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT, load_config
from src.models.hmm import fit_hmm_with_bic_selection

PROCESSED  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROCESSED / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "sp500_log_return", "sp500_realized_vol_5d", "sp500_realized_vol_21d",
    "sp500_rsi", "sp500_macd", "sp500_macd_hist", "sp500_bb_width",
    "XLK_log_return", "XLF_log_return", "XLE_log_return", "XLV_log_return",
    "XLY_log_return", "XLP_log_return", "XLI_log_return", "XLB_log_return",
    "XLU_log_return", "CPIAUCSL_diff", "UNRATE_diff", "DFF_diff",
    "INDPRO_diff", "T10Y2Y_diff", "T10Y2Y", "VIX", "VIX_percentile",
    "VIX_term_ratio", "sent_mean", "sent_max_neg", "sent_weighted", "sent_momentum",
]

FAMILIES = {
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=0.001, max_iter=10000),
    "lgb":   lambda: lgb.LGBMRegressor(
        objective="regression", verbosity=-1, n_jobs=-1,
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    ),
}
TARGETS = ["fwd_return_1d", "fwd_return_5d"]
MODES   = ["regime_agnostic", "regime_conditioned"]


def _backtest_sharpe(model_key: str) -> float:
    """Look up backtest Sharpe for ensemble weighting."""
    bt_path = PROCESSED / "backtest_results.csv"
    if not bt_path.exists():
        return 0.0
    bt = pd.read_csv(bt_path)
    for suffix in ("__regime_sized", "__plain"):
        m = bt[bt["model"] == f"{model_key}{suffix}"]
        if len(m):
            return float(m["sharpe"].iloc[0])
    return 0.0


def save_hmm():
    features_df = pd.read_parquet(PROCESSED / "features.parquet")
    hmm_path = PROCESSED / "hmm_model.pkl"
    if hmm_path.exists():
        print(f"  HMM already saved: {hmm_path}")
        return
    config = load_config()
    result = fit_hmm_with_bic_selection(features_df=features_df, config=config)
    best_model = result["best_model"]
    input_cols = config["hmm"]["input_features"]
    X_raw = features_df[input_cols].dropna()
    hmm_bundle = {
        "model":       best_model,
        "input_cols":  input_cols,
        "scaler_mean": X_raw.mean().to_dict(),
        "scaler_std":  X_raw.std().to_dict(),
    }
    with open(hmm_path, "wb") as f:
        pickle.dump(hmm_bundle, f)
    print(f"  Saved HMM: {hmm_path}")


def train_one(family: str, target: str, mode: str,
              features_df: pd.DataFrame, regime_labels: pd.Series):
    use_cols = [c for c in FEATURE_COLS if c in features_df.columns]

    if mode == "regime_conditioned":
        regime_dummies = pd.get_dummies(
            regime_labels.reindex(features_df.index), prefix="regime"
        ).astype(float)
        all_cols    = use_cols + list(regime_dummies.columns)
        regime_cols = list(regime_dummies.columns)
        work = pd.concat([features_df[use_cols], regime_dummies,
                          features_df[[target]]], axis=1)
    else:
        all_cols    = use_cols
        regime_cols = []
        work = features_df[use_cols + [target]].copy()

    work  = work.dropna(subset=[target])
    X_raw = work[all_cols]
    y     = work[target].values
    valid = ~np.isnan(y)

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    if family in ("ridge", "lasso"):
        scaler = StandardScaler()
        X_fit  = scaler.fit_transform(X_imp)
    else:
        scaler = None
        X_fit  = X_imp

    model = FAMILIES[family]()
    model.fit(X_fit[valid], y[valid])

    model_key = f"{family}_{target}_{mode}"
    return {
        "model":       model,
        "imputer":     imputer,
        "scaler":      scaler,
        "feat_cols":   all_cols,
        "regime_cols": regime_cols,
        "family":      family,
        "target":      target,
        "mode":        mode,
        "model_key":   model_key,
        "sharpe":      _backtest_sharpe(model_key),
        "n_train":     int(valid.sum()),
    }


def main():
    print("\n=== SAVING 12 MODELS + HMM FOR LIVE ENSEMBLE ===")
    features_df   = pd.read_parquet(PROCESSED / "features.parquet")
    regime_labels = pd.read_parquet(PROCESSED / "regime_labels.parquet")["regime"]

    print("\n[HMM]")
    save_hmm()

    summary = []
    for family in FAMILIES:
        for target in TARGETS:
            for mode in MODES:
                bundle   = train_one(family, target, mode, features_df, regime_labels)
                out_path = MODELS_DIR / f"{bundle['model_key']}.pkl"
                joblib.dump(bundle, out_path)
                print(f"  saved {bundle['model_key']:48s}  "
                      f"sharpe={bundle['sharpe']:+.3f}  n={bundle['n_train']}")
                summary.append({
                    "model":   bundle["model_key"],
                    "sharpe":  bundle["sharpe"],
                    "n_train": bundle["n_train"],
                })

    pd.DataFrame(summary).to_csv(MODELS_DIR / "ensemble_manifest.csv", index=False)
    print(f"\n=== Saved {len(summary)} models + HMM. Ready for ensemble trading. ===")


if __name__ == "__main__":
    main()
