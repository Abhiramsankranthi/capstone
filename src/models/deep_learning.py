"""
Deep learning forecasting models (LSTM + Transformer) with expanding-window
walk-forward evaluation.

Supports:
- Regime-agnostic training
- Regime-conditioned training (regime one-hot features)
- Deterministic seeds
- Lightweight hyperparameter tuning on the first fold
- Learning-curve tracking for overfitting diagnostics
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import PROJECT_ROOT, load_config


FEATURE_COLS = [
    "sp500_log_return",
    "sp500_realized_vol_5d",
    "sp500_realized_vol_21d",
    "sp500_rsi",
    "sp500_macd",
    "sp500_macd_signal",
    "sp500_macd_hist",
    "sp500_bb_width",
    "XLK_log_return",
    "XLF_log_return",
    "XLE_log_return",
    "XLV_log_return",
    "XLY_log_return",
    "XLP_log_return",
    "XLI_log_return",
    "XLB_log_return",
    "XLU_log_return",
    "XLRE_log_return",
    "XLC_log_return",
    "CPIAUCSL_diff",
    "UNRATE_diff",
    "DFF_diff",
    "INDPRO_diff",
    "T10Y2Y",
    "T10Y2Y_diff",
    "VIX",
    "VIX_percentile",
    "VIX_term_ratio",
    "sent_mean",
    "sent_std",
    "sent_max_neg",
    "sent_article_count",
    "sent_weighted",
    "sent_momentum",
]

TARGETS = ["fwd_return_1d", "fwd_return_5d"]


DEFAULT_DL_CONFIG = {
    "seed": 42,
    "device": "cpu",
    "sequence_length": 30,
    "min_train_size": 756,
    "step_size": 126,
    "max_folds": None,
    "val_fraction": 0.15,
    "min_train_sequences": 200,
    "batch_size": 64,
    "max_epochs": 25,
    "tuning_trials": 4,
    "tuning_max_epochs": 10,
    "patience": 4,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "dropout": 0.2,
    "lstm_hidden_size": 64,
    "lstm_layers": 2,
    "transformer_d_model": 64,
    "transformer_heads": 4,
    "transformer_layers": 2,
    "models": ["lstm", "tft"],
}

_THREADS_CONFIGURED = False


@dataclass
class FitResult:
    model: nn.Module
    history: pd.DataFrame
    best_val_loss: float


def _set_seed(seed: int) -> None:
    global _THREADS_CONFIGURED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic seeding without forcing strict deterministic kernels, which
    # can become prohibitively slow on CPU LSTM ops.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Avoid OpenMP barrier stalls on some local CPU builds.
    if not _THREADS_CONFIGURED:
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # Thread-pool already initialized in this process.
                pass
        _THREADS_CONFIGURED = True


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    signal_returns = np.sign(y_pred) * y_true
    sharpe = (signal_returns.mean() / (signal_returns.std() + 1e-10)) * np.sqrt(252)
    return {"rmse": float(rmse), "dir_acc": float(dir_acc), "sharpe": float(sharpe)}


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TFTRegressor(nn.Module):
    """
    Lightweight Transformer-style regressor used as a practical TFT proxy.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_steps = x.size(1)
        h = self.input_proj(x) + self.pos_embedding[:, :n_steps, :]
        h = self.encoder(h)
        h_last = self.norm(h[:, -1, :])
        return self.head(h_last).squeeze(-1)


def _make_model(
    model_name: str,
    input_dim: int,
    seq_len: int,
    params: dict[str, Any],
) -> nn.Module:
    if model_name == "lstm":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_size=int(params["lstm_hidden_size"]),
            num_layers=int(params["lstm_layers"]),
            dropout=float(params["dropout"]),
        )
    if model_name == "tft":
        d_model = int(params["transformer_d_model"])
        nhead = int(params["transformer_heads"])
        if d_model % nhead != 0:
            d_model = nhead * max(1, d_model // nhead)
        return TFTRegressor(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=int(params["transformer_layers"]),
            dropout=float(params["dropout"]),
        )
    raise ValueError(f"Unsupported deep model name: {model_name}")


def _build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    start_idx: int,
    end_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build rolling sequence windows [t-seq_len+1 ... t] -> y[t].
    """
    x_seq = []
    y_seq = []
    raw_idx = []
    first_t = max(seq_len - 1, start_idx)
    for t in range(first_t, end_idx):
        y_t = y[t]
        if np.isnan(y_t):
            continue
        x_win = X[t - seq_len + 1 : t + 1]
        if np.isnan(x_win).any():
            continue
        x_seq.append(x_win)
        y_seq.append(y_t)
        raw_idx.append(t)
    if not x_seq:
        return (
            np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(raw_idx, dtype=np.int64),
    )


def _fit_single_model(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    model_name: str,
    seq_len: int,
    params: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    val_fraction: float,
    min_train_sequences: int,
    batch_size: int,
    device: str,
) -> FitResult | None:
    n = len(y_seq)
    if n < max(min_train_sequences, 64):
        return None

    n_val = max(32, int(n * val_fraction))
    n_train = n - n_val
    if n_train < min_train_sequences:
        return None

    X_train, X_val = X_seq[:n_train], X_seq[n_train:]
    y_train, y_val = y_seq[:n_train], y_seq[n_train:]

    _set_seed(seed)
    model = _make_model(model_name, input_dim=X_seq.shape[2], seq_len=seq_len, params=params)
    model = model.to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)),
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if np.isfinite(val_loss) and (val_loss + 1e-10) < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    return FitResult(model=model, history=history, best_val_loss=best_val)


@torch.no_grad()
def _predict_batches(
    model: nn.Module,
    X_seq: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if len(X_seq) == 0:
        return np.empty((0,), dtype=float)
    ds = TensorDataset(torch.from_numpy(X_seq))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False)
    preds = []
    model.eval()
    for (xb,) in loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(float)


def _sample_params(
    model_name: str,
    base_params: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    params = dict(base_params)
    params["learning_rate"] = float(rng.choice([5e-4, 1e-3, 2e-3]))
    params["weight_decay"] = float(rng.choice([0.0, 1e-6, 1e-5, 1e-4]))
    params["dropout"] = float(rng.choice([0.1, 0.2, 0.3]))

    if model_name == "lstm":
        params["lstm_hidden_size"] = int(rng.choice([32, 64, 96]))
        params["lstm_layers"] = int(rng.choice([1, 2]))
    elif model_name == "tft":
        params["transformer_d_model"] = int(rng.choice([32, 64, 96]))
        params["transformer_heads"] = int(rng.choice([2, 4, 8]))
        params["transformer_layers"] = int(rng.choice([1, 2, 3]))
    return params


def _tune_first_fold(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    model_name: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    n_trials = int(cfg["tuning_trials"])
    if n_trials <= 1:
        return dict(cfg)

    rng = np.random.default_rng(int(cfg["seed"]))
    best_params = dict(cfg)
    best_loss = float("inf")

    for trial_id in range(n_trials):
        trial_params = _sample_params(model_name, cfg, rng)
        fit = _fit_single_model(
            X_seq=X_train_seq,
            y_seq=y_train_seq,
            model_name=model_name,
            seq_len=int(cfg["sequence_length"]),
            params=trial_params,
            seed=int(cfg["seed"]) + trial_id,
            max_epochs=int(cfg["tuning_max_epochs"]),
            patience=max(2, int(cfg["patience"]) - 1),
            val_fraction=float(cfg["val_fraction"]),
            min_train_sequences=int(cfg["min_train_sequences"]),
            batch_size=int(cfg["batch_size"]),
            device=str(cfg["device"]),
        )
        if fit is None:
            continue
        if fit.best_val_loss < best_loss:
            best_loss = fit.best_val_loss
            best_params = trial_params

    return best_params


def _walk_forward_predict(
    X_df: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    cfg: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Expanding-window walk-forward predictions for one target/model pair.
    """
    seq_len = int(cfg["sequence_length"])
    min_train_size = int(cfg["min_train_size"])
    step_size = int(cfg["step_size"])
    max_folds = cfg.get("max_folds")
    if max_folds is not None:
        max_folds = int(max_folds)

    X_raw = X_df.astype(float)
    y_raw = y.astype(float)
    n = len(X_raw)

    if n < min_train_size + step_size:
        return (
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.DataFrame(),
            dict(cfg),
        )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    pred_dates: list[pd.Timestamp] = []
    pred_vals: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []
    tuned_params: dict[str, Any] | None = None

    fold_idx = 0
    y_vals_full = y_raw.values
    X_vals_full = X_raw.values
    dates = X_raw.index

    for end in range(min_train_size, n, step_size):
        if max_folds is not None and fold_idx >= max_folds:
            break

        test_end = min(end + step_size, n)
        if end <= seq_len:
            continue

        X_train_raw = X_vals_full[:end]
        X_until_test = X_vals_full[:test_end]
        y_until_test = y_vals_full[:test_end]

        X_train_imp = imputer.fit_transform(X_train_raw)
        scaler.fit(X_train_imp)
        X_scaled = scaler.transform(imputer.transform(X_until_test))

        X_train_seq, y_train_seq, _ = _build_sequences(
            X=X_scaled,
            y=y_until_test,
            seq_len=seq_len,
            start_idx=0,
            end_idx=end,
        )
        X_test_seq, y_test_seq, test_raw_idx = _build_sequences(
            X=X_scaled,
            y=y_until_test,
            seq_len=seq_len,
            start_idx=end,
            end_idx=test_end,
        )
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            continue

        if tuned_params is None:
            tuned_params = _tune_first_fold(
                X_train_seq=X_train_seq,
                y_train_seq=y_train_seq,
                model_name=model_name,
                cfg=cfg,
            )

        fit = _fit_single_model(
            X_seq=X_train_seq,
            y_seq=y_train_seq,
            model_name=model_name,
            seq_len=seq_len,
            params=tuned_params,
            seed=int(cfg["seed"]) + fold_idx,
            max_epochs=int(cfg["max_epochs"]),
            patience=int(cfg["patience"]),
            val_fraction=float(cfg["val_fraction"]),
            min_train_sequences=int(cfg["min_train_sequences"]),
            batch_size=int(cfg["batch_size"]),
            device=str(cfg["device"]),
        )
        if fit is None:
            continue

        fold_preds = _predict_batches(
            model=fit.model,
            X_seq=X_test_seq,
            batch_size=int(cfg["batch_size"]),
            device=str(cfg["device"]),
        )
        test_dates = dates[test_raw_idx]

        pred_dates.extend(test_dates.tolist())
        pred_vals.extend(fold_preds.tolist())

        fold_rows.append(
            {
                "fold": fold_idx,
                "train_end": str(dates[end - 1].date()),
                "test_start": str(test_dates.min().date()),
                "test_end": str(test_dates.max().date()),
                "n_train_seq": len(X_train_seq),
                "n_test_seq": len(X_test_seq),
                "best_val_loss": float(fit.best_val_loss),
            }
        )

        if not fit.history.empty:
            h = fit.history.copy()
            h["fold"] = fold_idx
            h["train_end"] = str(dates[end - 1].date())
            curve_rows.append(h)

        fold_idx += 1

    pred_series = pd.Series(pred_vals, index=pd.Index(pred_dates), dtype=float)
    pred_series = pred_series[~pred_series.index.duplicated(keep="last")].sort_index()
    fold_df = pd.DataFrame(fold_rows)
    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    if tuned_params is None:
        tuned_params = dict(cfg)
    return pred_series, fold_df, curve_df, tuned_params


def _get_deep_config(config: dict[str, Any] | None, quick: bool) -> dict[str, Any]:
    merged = dict(DEFAULT_DL_CONFIG)
    if config is not None and "deep_learning" in config:
        merged.update(config["deep_learning"] or {})

    if quick:
        # Strict smoke-test profile (fast E2E verification).
        merged["max_epochs"] = min(int(merged["max_epochs"]), 4)
        merged["tuning_trials"] = 1
        merged["tuning_max_epochs"] = min(int(merged["tuning_max_epochs"]), 2)
        merged["step_size"] = max(int(merged["step_size"]), 504)
        merged["max_folds"] = 2 if merged.get("max_folds") is None else min(int(merged["max_folds"]), 2)

    return merged


def run_deep_models(
    features_df: pd.DataFrame | None = None,
    regime_labels: pd.Series | None = None,
    regime_conditioned: bool = False,
    config: dict[str, Any] | None = None,
    quick: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Run LSTM/TFT models with walk-forward forecasting.

    Returns
    -------
    dict keyed as '{model}_{target}_{mode}' with:
      - predictions: pd.Series
      - metrics: dict
      - n_test: int
      - params: dict
      - fold_summary: pd.DataFrame
      - learning_curve: pd.DataFrame
      - regime_metrics (conditioned mode only)
    """
    if config is None:
        config = load_config()
    dl_cfg = _get_deep_config(config, quick=quick)
    _set_seed(int(dl_cfg["seed"]))

    if features_df is None:
        features_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
        if not features_path.exists():
            raise FileNotFoundError(f"Required file not found: {features_path}")
        features_df = pd.read_parquet(features_path)

    if regime_labels is None:
        regime_path = PROJECT_ROOT / "data" / "processed" / "regime_labels.parquet"
        if not regime_path.exists():
            raise FileNotFoundError(f"Required file not found: {regime_path}")
        regime_labels = pd.read_parquet(regime_path)["regime"]

    feature_cols = [c for c in FEATURE_COLS if c in features_df.columns]
    if not feature_cols:
        raise ValueError("No deep-model feature columns found in features dataframe.")

    models = dl_cfg.get("models", ["lstm", "tft"])
    mode_label = "regime_conditioned" if regime_conditioned else "regime_agnostic"
    results: dict[str, dict[str, Any]] = {}

    for target in TARGETS:
        if target not in features_df.columns:
            raise ValueError(f"Target column not found: {target}")

        base_df = features_df[feature_cols + [target]].copy()
        base_df = base_df.loc[~base_df[feature_cols].isna().all(axis=1)].sort_index()
        model_feature_cols = feature_cols.copy()

        if regime_conditioned:
            regimes = regime_labels.reindex(base_df.index)
            if regimes.isna().all():
                raise ValueError("Regime labels missing after merge (all NaN).")
            reg_dummies = pd.get_dummies(regimes.astype("Int64"), prefix="regime", dummy_na=False)
            reg_dummies = reg_dummies.astype(float)
            reg_cols = reg_dummies.columns.tolist()
            base_df = base_df.join(reg_dummies, how="left")
            base_df[reg_cols] = base_df[reg_cols].fillna(0.0)
            model_feature_cols.extend(reg_cols)

        X_df = base_df[model_feature_cols]
        y = base_df[target]

        for model_name in models:
            key = f"{model_name}_{target}_{mode_label}"
            print(f"\n--- {key} ---")

            preds, fold_df, curve_df, best_params = _walk_forward_predict(
                X_df=X_df,
                y=y,
                model_name=model_name,
                cfg=dl_cfg,
            )

            if preds.empty:
                print("  Skipped (insufficient sequence samples in walk-forward folds).")
                continue

            y_true = y.reindex(preds.index)
            valid_mask = (~y_true.isna()) & (~preds.isna())
            y_true_v = y_true[valid_mask].values
            y_pred_v = preds[valid_mask].values
            if len(y_true_v) == 0:
                print("  Skipped (no valid aligned predictions).")
                continue

            metrics = _compute_metrics(y_true_v, y_pred_v)
            result_payload: dict[str, Any] = {
                "predictions": preds,
                "metrics": metrics,
                "n_test": int(len(y_true_v)),
                "params": best_params,
                "fold_summary": fold_df,
                "learning_curve": curve_df,
            }

            if regime_conditioned:
                pred_regimes = regime_labels.reindex(preds.index)
                regime_metrics = {}
                for regime_val in sorted(pred_regimes.dropna().unique()):
                    idx = pred_regimes[pred_regimes == regime_val].index
                    yy = y.reindex(idx).dropna()
                    pp = preds.reindex(yy.index).dropna()
                    common_idx = yy.index.intersection(pp.index)
                    if len(common_idx) < 25:
                        continue
                    regime_metrics[int(regime_val)] = _compute_metrics(
                        yy.loc[common_idx].values,
                        pp.loc[common_idx].values,
                    )
                result_payload["regime_metrics"] = regime_metrics

            results[key] = result_payload
            print(
                f"  RMSE={metrics['rmse']:.5f}  "
                f"DirAcc={metrics['dir_acc']:.3f}  "
                f"Sharpe={metrics['sharpe']:.3f}  "
                f"n_test={len(y_true_v)}"
            )

    return results


if __name__ == "__main__":
    print("=== Deep Learning Models — Regime-Agnostic ===")
    run_deep_models(regime_conditioned=False, quick=True)
    print("\n=== Deep Learning Models — Regime-Conditioned ===")
    run_deep_models(regime_conditioned=True, quick=True)
