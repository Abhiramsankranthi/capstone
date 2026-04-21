"""
Ensemble inference across all saved models for live trading.

Loads every {family}_{target}_{mode}.pkl bundle from data/processed/models/.
For a given feature row, returns:
    - per-model predictions and votes
    - Sharpe-weighted ensemble prediction (negative-Sharpe models excluded)
    - directional agreement fraction
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "data" / "processed" / "models"


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _load_bundles() -> list[dict]:
    bundles = []
    for p in sorted(MODELS_DIR.glob("*.pkl")):
        if p.name in ("hmm_model.pkl",):
            continue
        try:
            b = joblib.load(p)
            if "model" in b and "feat_cols" in b:
                bundles.append(b)
        except Exception as e:
            print(f"  [ensemble] skip {p.name}: {e}")
    return bundles


def _infer(bundle: dict, features_today: pd.Series, regime: str) -> float:
    row = {}
    for col in bundle["feat_cols"]:
        if col in bundle.get("regime_cols", []):
            row[col] = 1.0 if col == f"regime_{regime}" else 0.0
        else:
            row[col] = features_today.get(col, np.nan)

    X = pd.DataFrame([row])[bundle["feat_cols"]]
    X_imp = bundle["imputer"].transform(X)
    if bundle.get("scaler") is not None:
        X_imp = bundle["scaler"].transform(X_imp)
    return float(bundle["model"].predict(X_imp)[0])


def ensemble_predict(features_today: pd.Series, regime: str,
                      threshold: float = 0.0005) -> dict:
    """
    Run all models, return weighted-average prediction + per-model breakdown.

    Returns
    -------
    dict with keys:
        pred        : Sharpe-weighted average predicted return
        signal      : "buy" / "sell" / "flat" (after agreement gate)
        agreement   : fraction of surviving models voting same direction as ensemble
        votes       : list of {model_key, sharpe, weight, pred, vote}
        n_models    : total models attempted
        n_active    : surviving models (sharpe > 0)
    """
    bundles = _load_bundles()
    if not bundles:
        return {"pred": 0.0, "signal": "flat", "agreement": 0.0,
                "votes": [], "n_models": 0, "n_active": 0}

    votes = []
    for b in bundles:
        try:
            p = _infer(b, features_today, regime)
        except Exception as e:
            p = 0.0
        votes.append({
            "model_key": b.get("model_key", "?"),
            "sharpe":    float(b.get("sharpe", 0.0)),
            "pred":      p,
            "vote":      "buy" if p > threshold else ("sell" if p < -threshold else "flat"),
        })

    # Weight = max(Sharpe, 0); negative-Sharpe models excluded
    active = [v for v in votes if v["sharpe"] > 0]
    total_w = sum(v["sharpe"] for v in active) or 1.0
    for v in votes:
        v["weight"] = v["sharpe"] / total_w if v["sharpe"] > 0 else 0.0

    ensemble_pred = sum(v["pred"] * v["weight"] for v in active)

    # Directional agreement among active models
    if active:
        ens_sign = _sign(ensemble_pred)
        agree = sum(1 for v in active if _sign(v["pred"]) == ens_sign and ens_sign != 0)
        agreement = agree / len(active)
    else:
        agreement = 0.0

    # Signal gate: require ≥60% agreement AND |pred| > threshold
    if abs(ensemble_pred) < threshold or agreement < 0.6:
        signal = "flat"
    else:
        signal = "buy" if ensemble_pred > 0 else "sell"

    return {
        "pred":      ensemble_pred,
        "signal":    signal,
        "agreement": agreement,
        "votes":     votes,
        "n_models":  len(votes),
        "n_active":  len(active),
    }
