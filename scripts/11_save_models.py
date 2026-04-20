"""
Train final LightGBM + HMM models on full historical data and persist to disk.
Must be run once before live trading starts, and re-run monthly to refresh.

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
from sklearn.impute import SimpleImputer

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


def train_final_lgb(X: np.ndarray, y: np.ndarray, params: dict) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression", verbosity=-1, n_jobs=-1,
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        **{k: v for k, v in params.items()
           if k not in ("objective", "metric", "verbosity")},
    )
    model.fit(X, y)
    return model


def main():
    print("\n=== SAVING FINAL MODELS FOR LIVE TRADING ===")
    features_df = pd.read_parquet(PROCESSED / "features.parquet")
    regime_labels = pd.read_parquet(PROCESSED / "regime_labels.parquet")["regime"]

    # ── 1. HMM model ────────────────────────────────────────────────────────
    print("\nFitting HMM on full dataset...")
    hmm_path = PROCESSED / "hmm_model.pkl"
    if not hmm_path.exists():
        config = load_config()
        result = fit_hmm_with_bic_selection(features_df=features_df, config=config)
        best_model = result["best_model"]
        input_cols = config["hmm"]["input_features"]
        X_raw = features_df[input_cols].dropna()
        hmm_bundle = {
            "model":      best_model,
            "input_cols": input_cols,
            "scaler_mean": X_raw.mean().to_dict(),
            "scaler_std":  X_raw.std().to_dict(),
        }
        with open(hmm_path, "wb") as f:
            pickle.dump(hmm_bundle, f)
        print(f"  Saved HMM: {hmm_path}")
    else:
        print(f"  HMM already saved: {hmm_path}")

    # ── 2. LightGBM — 1d and 5d, regime-conditioned ─────────────────────────
    for target in ["fwd_return_1d", "fwd_return_5d"]:
        print(f"\nTraining LightGBM ({target})...")

        use_cols = [c for c in FEATURE_COLS if c in features_df.columns]
        regime_dummies = pd.get_dummies(
            regime_labels.reindex(features_df.index), prefix="regime"
        ).astype(float)
        all_cols = use_cols + list(regime_dummies.columns)

        work = pd.concat([features_df[use_cols], regime_dummies, features_df[[target]]], axis=1)
        work = work.dropna(subset=[target])

        imputer = SimpleImputer(strategy="median")
        X_raw  = work[all_cols]
        X_imp  = imputer.fit_transform(X_raw)
        y      = work[target].values

        valid = ~np.isnan(y)
        X_clean, y_clean = X_imp[valid], y[valid]

        print(f"  Training on {len(y_clean)} samples, {len(all_cols)} features...")
        model = train_final_lgb(X_clean, y_clean, {})

        bundle = {
            "model":    model,
            "imputer":  imputer,
            "feat_cols": all_cols,
            "regime_cols": list(regime_dummies.columns),
        }
        out_path = MODELS_DIR / f"lgb_{target}_regime_conditioned.pkl"
        joblib.dump(bundle, out_path)
        print(f"  Saved: {out_path}")

    print("\n=== MODELS SAVED — ready for live trading ===")


if __name__ == "__main__":
    main()
