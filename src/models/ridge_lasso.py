"""
Ridge and Lasso regression baselines with expanding-window walk-forward validation.
Supports regime-agnostic and regime-conditioned training variants.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from src.config import load_config, PROJECT_ROOT


FEATURE_COLS = [
    "sp500_log_return", "sp500_realized_vol_5d", "sp500_realized_vol_21d",
    "sp500_rsi", "sp500_macd", "sp500_macd_hist", "sp500_bb_width",
    "XLK_log_return", "XLF_log_return", "XLE_log_return", "XLV_log_return",
    "XLY_log_return", "XLP_log_return", "XLI_log_return", "XLB_log_return",
    "XLU_log_return", "CPIAUCSL_diff", "UNRATE_diff", "DFF_diff",
    "INDPRO_diff", "T10Y2Y_diff", "T10Y2Y", "VIX", "VIX_percentile",
    "VIX_term_ratio", "sent_mean", "sent_max_neg", "sent_weighted", "sent_momentum",
]

TARGETS = ["fwd_return_1d", "fwd_return_5d"]
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    # Sharpe: annualised signal-weighted return / std
    signal_returns = np.sign(y_pred) * y_true
    sharpe = (signal_returns.mean() / (signal_returns.std() + 1e-10)) * np.sqrt(252)
    return {"rmse": rmse, "dir_acc": dir_acc, "sharpe": sharpe}


def expanding_window_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_class,
    alpha: float,
    min_train_size: int = 500,
    step_size: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expanding-window walk-forward CV. No shuffling, no future leakage.
    Returns arrays of (dates, predictions).
    """
    n = len(X)
    all_pred_idx = []
    all_preds = []

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    for end in range(min_train_size, n, step_size):
        X_train = X.iloc[:end]
        y_train = y.iloc[:end]
        X_test = X.iloc[end: end + step_size]

        if X_test.empty:
            break

        # Fit imputer and scaler on training data only
        X_tr_imp = imputer.fit_transform(X_train)
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_te_imp = imputer.transform(X_test)
        X_te_scaled = scaler.transform(X_te_imp)

        # Handle NaN in target (drop those rows)
        valid_mask = ~np.isnan(y_train.values)
        if valid_mask.sum() < 50:
            continue

        model = model_class(alpha=alpha)
        model.fit(X_tr_scaled[valid_mask], y_train.values[valid_mask])

        preds = model.predict(X_te_scaled)
        all_pred_idx.extend(X_test.index.tolist())
        all_preds.extend(preds.tolist())

    return np.array(all_pred_idx), np.array(all_preds)


def _select_alpha(X: pd.DataFrame, y: pd.Series, model_class, min_train_size: int = 500) -> float:
    """Select best alpha via CV on first half of data only (to avoid lookahead)."""
    cutoff = len(X) // 2
    X_sel, y_sel = X.iloc[:cutoff], y.iloc[:cutoff]
    best_alpha, best_rmse = ALPHAS[0], np.inf
    for alpha in ALPHAS:
        _, preds = expanding_window_cv(X_sel, y_sel, model_class, alpha, min_train_size)
        if len(preds) == 0:
            continue
        valid = ~np.isnan(y_sel.values[-len(preds):])
        if valid.sum() < 10:
            continue
        rmse = np.sqrt(mean_squared_error(
            y_sel.values[-len(preds):][valid],
            preds[valid]
        ))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    return best_alpha


def run_ridge_lasso(
    features_df: pd.DataFrame = None,
    regime_labels: pd.Series = None,
    regime_conditioned: bool = False,
    config=None,
) -> dict:
    """
    Run Ridge and Lasso models (regime-agnostic or regime-conditioned).

    Parameters
    ----------
    features_df : merged features DataFrame (40 cols)
    regime_labels : Series with regime per date (from HMM)
    regime_conditioned : if True, train a separate model per regime
    config : project config dict

    Returns
    -------
    dict with keys: {model_name}_{target} → {predictions, metrics}
    """
    if config is None:
        config = load_config()
    if features_df is None:
        features_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    if regime_labels is None:
        regime_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "regime_labels.parquet")
        regime_labels = regime_df["regime"]

    # Use only features that exist in the dataframe
    feature_cols = [c for c in FEATURE_COLS if c in features_df.columns]

    results = {}
    mode_label = "regime_conditioned" if regime_conditioned else "regime_agnostic"

    for target in TARGETS:
        df = features_df[feature_cols + [target]].dropna(subset=feature_cols[:5])
        X = df[feature_cols]
        y = df[target]

        for model_name, model_class in [("ridge", Ridge), ("lasso", Lasso)]:
            key = f"{model_name}_{target}_{mode_label}"
            print(f"\n--- {key} ---")

            if not regime_conditioned:
                best_alpha = _select_alpha(X, y, model_class)
                print(f"  Selected alpha={best_alpha}")
                pred_idx, preds = expanding_window_cv(X, y, model_class, best_alpha)
                valid_y = y.loc[pred_idx].values
                valid_mask = ~np.isnan(valid_y) & ~np.isnan(preds)
                metrics = _compute_metrics(valid_y[valid_mask], preds[valid_mask])
                results[key] = {
                    "predictions": pd.Series(preds, index=pred_idx, name=key),
                    "metrics": metrics,
                    "alpha": best_alpha,
                    "n_test": valid_mask.sum(),
                }
                print(f"  RMSE={metrics['rmse']:.5f}  DirAcc={metrics['dir_acc']:.3f}  Sharpe={metrics['sharpe']:.3f}")

            else:
                # Per-regime models: train separately, evaluate combined
                regimes = regime_labels.reindex(X.index).dropna()
                all_preds = pd.Series(dtype=float, name=key)
                regime_metrics = {}

                for regime_name in regimes.unique():
                    idx_regime = regimes[regimes == regime_name].index
                    X_r = X.loc[idx_regime]
                    y_r = y.loc[idx_regime]

                    if len(X_r) < 100:
                        print(f"  Skipping regime '{regime_name}' (only {len(X_r)} obs)")
                        continue

                    # Use a fixed alpha (no nested CV per regime to avoid tiny data)
                    best_alpha = _select_alpha(X_r, y_r, model_class, min_train_size=100)
                    pred_idx_r, preds_r = expanding_window_cv(
                        X_r, y_r, model_class, best_alpha, min_train_size=100
                    )
                    if len(preds_r) == 0:
                        continue
                    s = pd.Series(preds_r, index=pred_idx_r, name=key)
                    all_preds = pd.concat([all_preds, s]).sort_index()

                    valid_y_r = y_r.loc[pred_idx_r].values
                    valid_mask_r = ~np.isnan(valid_y_r) & ~np.isnan(preds_r)
                    if valid_mask_r.sum() > 10:
                        rm = _compute_metrics(valid_y_r[valid_mask_r], preds_r[valid_mask_r])
                        regime_metrics[regime_name] = rm
                        print(f"  [{regime_name}] alpha={best_alpha}  RMSE={rm['rmse']:.5f}  DirAcc={rm['dir_acc']:.3f}")

                # Combined metrics across all regimes
                if len(all_preds) > 0:
                    valid_y_all = y.reindex(all_preds.index).values
                    valid_mask_all = ~np.isnan(valid_y_all) & ~np.isnan(all_preds.values)
                    combined = _compute_metrics(valid_y_all[valid_mask_all], all_preds.values[valid_mask_all])
                    results[key] = {
                        "predictions": all_preds,
                        "metrics": combined,
                        "regime_metrics": regime_metrics,
                        "n_test": valid_mask_all.sum(),
                    }
                    print(f"  Combined RMSE={combined['rmse']:.5f}  DirAcc={combined['dir_acc']:.3f}  Sharpe={combined['sharpe']:.3f}")

    return results


if __name__ == "__main__":
    print("=== Ridge/Lasso — Regime-Agnostic ===")
    r_agnostic = run_ridge_lasso(regime_conditioned=False)
    print("\n=== Ridge/Lasso — Regime-Conditioned ===")
    r_conditioned = run_ridge_lasso(regime_conditioned=True)
