"""
LightGBM with Optuna hyperparameter tuning.
Supports regime-agnostic and regime-conditioned training variants.
Feature importance via native LightGBM gain.
"""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from src.config import load_config, PROJECT_ROOT

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    signal_returns = np.sign(y_pred) * y_true
    sharpe = (signal_returns.mean() / (signal_returns.std() + 1e-10)) * np.sqrt(252)
    return {"rmse": rmse, "dir_acc": dir_acc, "sharpe": sharpe}


def _tune_lgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Optuna search over LightGBM hyperparameters."""

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def expanding_window_lgb(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
    n_optuna_trials: int = 50,
    min_train_size: int = 500,
    step_size: int = 63,  # re-tune quarterly
    val_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Expanding-window walk-forward with periodic Optuna re-tuning.
    Returns (pred_index, predictions, best_params_from_last_tune).
    """
    n = len(X)
    imputer = SimpleImputer(strategy="median")

    all_pred_idx = []
    all_preds = []
    last_params = {}
    tune_every = step_size * 4  # re-tune every ~1 year of steps

    for end in range(min_train_size, n, step_size):
        X_train_raw = X.iloc[:end]
        y_train = y.iloc[:end]
        X_test = X.iloc[end: end + step_size]

        if X_test.empty:
            break

        # Fit imputer on train
        X_tr_imp = imputer.fit_transform(X_train_raw)
        X_te_imp = imputer.transform(X_test)

        # Drop target NaNs
        valid_mask = ~np.isnan(y_train.values)
        X_tr_clean = X_tr_imp[valid_mask]
        y_tr_clean = y_train.values[valid_mask]

        if len(y_tr_clean) < 100:
            continue

        # Re-tune hyperparameters periodically
        should_tune = (end == min_train_size) or ((end - min_train_size) % tune_every == 0)
        if should_tune or not last_params:
            val_size = max(50, int(len(y_tr_clean) * val_fraction))
            X_val = X_tr_clean[-val_size:]
            y_val = y_tr_clean[-val_size:]
            X_fit = X_tr_clean[:-val_size]
            y_fit = y_tr_clean[:-val_size]
            if len(y_fit) >= 50:
                last_params = _tune_lgb(X_fit, y_fit, X_val, y_val, n_trials=n_optuna_trials)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            **last_params,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr_clean, y_tr_clean)

        preds = model.predict(X_te_imp)
        all_pred_idx.extend(X_test.index.tolist())
        all_preds.extend(preds.tolist())

    return np.array(all_pred_idx), np.array(all_preds), last_params


def get_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
    best_params: dict,
) -> pd.DataFrame:
    """
    Fit a single LightGBM on the full dataset and return gain-based feature importance.
    For inspection only — not used during walk-forward.
    """
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    valid_mask = ~np.isnan(y.values)

    params = {"objective": "regression", "verbosity": -1, **best_params}
    model = lgb.LGBMRegressor(**params)
    model.fit(X_imp[valid_mask], y.values[valid_mask])

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return importance


def run_lightgbm(
    features_df: pd.DataFrame = None,
    regime_labels: pd.Series = None,
    regime_conditioned: bool = False,
    n_optuna_trials: int = 50,
    config=None,
) -> dict:
    """
    Run LightGBM models (regime-agnostic or regime-conditioned).

    Returns
    -------
    dict keyed by '{target}_{mode}' → {predictions, metrics, feature_importance, params}
    """
    if config is None:
        config = load_config()
    if features_df is None:
        features_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    if regime_labels is None:
        regime_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "regime_labels.parquet")
        regime_labels = regime_df["regime"]

    feature_cols = [c for c in FEATURE_COLS if c in features_df.columns]
    mode_label = "regime_conditioned" if regime_conditioned else "regime_agnostic"
    results = {}

    for target in TARGETS:
        df = features_df[feature_cols + [target]].dropna(subset=feature_cols[:5])
        X = df[feature_cols]
        y = df[target]
        key = f"lgb_{target}_{mode_label}"
        print(f"\n--- {key} (n_trials={n_optuna_trials}) ---")

        if not regime_conditioned:
            pred_idx, preds, best_params = expanding_window_lgb(
                X, y, feature_cols, n_optuna_trials=n_optuna_trials
            )
            valid_y = y.loc[pred_idx].values
            valid_mask = ~np.isnan(valid_y) & ~np.isnan(preds)
            metrics = _compute_metrics(valid_y[valid_mask], preds[valid_mask])
            importance = get_feature_importance(X, y, feature_cols, best_params)
            results[key] = {
                "predictions": pd.Series(preds, index=pred_idx, name=key),
                "metrics": metrics,
                "feature_importance": importance,
                "params": best_params,
                "n_test": valid_mask.sum(),
            }
            print(f"  RMSE={metrics['rmse']:.5f}  DirAcc={metrics['dir_acc']:.3f}  Sharpe={metrics['sharpe']:.3f}")

        else:
            regimes = regime_labels.reindex(X.index).dropna()
            all_preds = pd.Series(dtype=float, name=key)
            regime_metrics = {}
            combined_importance = []

            for regime_name in sorted(regimes.unique()):
                idx_regime = regimes[regimes == regime_name].index
                X_r = X.loc[idx_regime]
                y_r = y.loc[idx_regime]

                if len(X_r) < 150:
                    print(f"  Skipping regime '{regime_name}' ({len(X_r)} obs)")
                    continue

                print(f"  Training on regime: {regime_name} ({len(X_r)} obs)")
                pred_idx_r, preds_r, best_params_r = expanding_window_lgb(
                    X_r, y_r, feature_cols,
                    n_optuna_trials=n_optuna_trials,
                    min_train_size=100,
                    step_size=21,
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
                    print(f"    RMSE={rm['rmse']:.5f}  DirAcc={rm['dir_acc']:.3f}  Sharpe={rm['sharpe']:.3f}")

                    imp = get_feature_importance(X_r, y_r, feature_cols, best_params_r)
                    imp["regime"] = regime_name
                    combined_importance.append(imp)

            if len(all_preds) > 0:
                valid_y_all = y.reindex(all_preds.index).values
                valid_mask_all = ~np.isnan(valid_y_all) & ~np.isnan(all_preds.values)
                combined = _compute_metrics(valid_y_all[valid_mask_all], all_preds.values[valid_mask_all])
                importance_df = pd.concat(combined_importance) if combined_importance else pd.DataFrame()
                results[key] = {
                    "predictions": all_preds,
                    "metrics": combined,
                    "regime_metrics": regime_metrics,
                    "feature_importance": importance_df,
                    "n_test": valid_mask_all.sum(),
                }
                print(f"  Combined RMSE={combined['rmse']:.5f}  DirAcc={combined['dir_acc']:.3f}  Sharpe={combined['sharpe']:.3f}")

    return results


if __name__ == "__main__":
    print("=== LightGBM — Regime-Agnostic ===")
    r_agnostic = run_lightgbm(regime_conditioned=False, n_optuna_trials=50)
    print("\n=== LightGBM — Regime-Conditioned ===")
    r_conditioned = run_lightgbm(regime_conditioned=True, n_optuna_trials=50)
