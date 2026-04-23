"""
Daily live paper-trading script using Alpaca.
Fetches latest market data, detects current regime, predicts 1-day return,
sizes a SPY position, and submits the order via Alpaca paper API.

Prerequisites:
    pip install alpaca-py
    Add to .env:
        ALPACA_API_KEY=your_key
        ALPACA_SECRET_KEY=your_secret

Usage:
    source .venv/bin/activate
    python scripts/12_live_trading.py

Run after market close (4:30 PM ET). Sets up position for next day open.
Recommend scheduling with cron:
    30 16 * * 1-5  /path/to/.venv/bin/python /path/to/scripts/12_live_trading.py
"""
import sys, pickle, json
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer

from src.config import PROJECT_ROOT

load_dotenv()

PROCESSED   = PROJECT_ROOT / "data" / "processed"
MODELS_DIR  = PROCESSED / "models"
TRADE_LOG   = PROCESSED / "trade_log.csv"
SYMBOL_LONG  = "SPY"          # bullish exposure
SYMBOL_SHORT = "SH"           # ProShares Short S&P 500 — inverse ETF (avoids short-sell locate delay)
NOTIONAL    = 10_000          # dollars per trade (paper)
THRESHOLD   = 0.0005          # min predicted return to act
REGIME_SCALE = {
    "Bull": 1.0,
    "Normal": 0.8,
    "Bear/Crisis": 0.6,
    "Extreme": 0.4,
}
LOOKBACK_DAYS = 300           # days of history needed for features


# ── Feature computation (mirrors scripts/02_build_features.py) ───────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the subset of features needed for live inference."""
    out = pd.DataFrame(index=df.index)
    out["sp500_log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    out["sp500_realized_vol_5d"]  = out["sp500_log_return"].rolling(5).std() * np.sqrt(252)
    out["sp500_realized_vol_21d"] = out["sp500_log_return"].rolling(21).std() * np.sqrt(252)

    # RSI-14
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    out["sp500_rsi"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    out["sp500_macd"]      = ema12 - ema26
    out["sp500_macd_hist"] = out["sp500_macd"] - out["sp500_macd"].ewm(span=9).mean()

    # Bollinger band width
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    out["sp500_bb_width"] = (2 * std20) / (sma20 + 1e-10)

    return out


def fetch_latest_features() -> pd.DataFrame:
    """Download recent data and build features for inference."""
    print("  Fetching market data from yfinance...")

    tickers = ["^GSPC", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
               "XLI", "XLB", "XLU", "^VIX"]

    raw = yf.download(tickers, period=f"{LOOKBACK_DAYS}d",
                      auto_adjust=True, progress=False)["Close"]
    raw.columns = [c.replace("^", "") for c in raw.columns]

    features = compute_features(raw.rename(columns={"GSPC": "Close"})[["Close"]])
    features = features.rename(columns=lambda c: c)  # already prefixed

    # Sector returns
    for etf in ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]:
        if etf in raw.columns:
            features[f"{etf}_log_return"] = np.log(raw[etf] / raw[etf].shift(1))

    # VIX features
    if "VIX" in raw.columns:
        features["VIX"] = raw["VIX"]
        features["VIX_percentile"] = (
            raw["VIX"].expanding().rank() / raw["VIX"].expanding().count()
        )
        features["VIX_term_ratio"] = 1.0  # ^VIX3M not always available

    # Macro features — use last known values from features.parquet as fallback
    hist = pd.read_parquet(PROCESSED / "features.parquet")
    macro_cols = ["CPIAUCSL_diff", "UNRATE_diff", "DFF_diff", "INDPRO_diff",
                  "T10Y2Y_diff", "T10Y2Y"]
    for col in macro_cols:
        if col in hist.columns:
            last_val = hist[col].dropna().iloc[-1]
            features[col] = last_val  # static fill — refreshed monthly when models retrained

    # Sentiment — use last known value
    sent_cols = ["sent_mean", "sent_max_neg", "sent_weighted", "sent_momentum"]
    for col in sent_cols:
        if col in hist.columns:
            last_val = hist[col].dropna().iloc[-1] if hist[col].notna().any() else 0.0
            features[col] = last_val

    return features.dropna(how="all")


# ── Regime detection ─────────────────────────────────────────────────────────

def detect_regime(features_today: pd.Series) -> str:
    """Use saved HMM to predict today's regime; fall back to last known."""
    hmm_path = PROCESSED / "hmm_model.pkl"
    if not hmm_path.exists():
        # Fall back to last label in regime_labels.parquet
        regime_df = pd.read_parquet(PROCESSED / "regime_labels.parquet")
        return regime_df["regime"].dropna().iloc[-1]

    with open(hmm_path, "rb") as f:
        bundle = pickle.load(f)

    model      = bundle["model"]
    input_cols = bundle["input_cols"]
    mean       = pd.Series(bundle["scaler_mean"])
    std        = pd.Series(bundle["scaler_std"])

    vals = []
    for col in input_cols:
        v = features_today.get(col, mean.get(col, 0.0))
        m = mean.get(col, 0.0)
        s = std.get(col, 1.0)
        vals.append((v - m) / (s + 1e-10))

    X = np.array(vals).reshape(1, -1)
    state = model.predict(X)[0]

    # Re-derive label map the same way as hmm.py (sort by volatility)
    vol_col_idx = next(
        (i for i, c in enumerate(input_cols) if "vol" in c.lower()), 0
    )
    means = model.means_[:, vol_col_idx]
    order = np.argsort(means)
    labels = ["Bull", "Normal", "Bear/Crisis", "Extreme"][:model.n_components]
    label_map = {order[i]: labels[i] for i in range(len(order))}
    return label_map.get(state, "Normal")


# ── Prediction (ensemble across all saved models) ─────────────────────────────

def predict_return(features_today: pd.Series, regime: str) -> float:
    """Backwards-compat: return ensemble's weighted-average prediction."""
    return ensemble_forecast(features_today, regime)["pred"]


def ensemble_forecast(features_today: pd.Series, regime: str) -> dict:
    """Run full ensemble; returns pred + agreement + per-model votes."""
    from src.trading.ensemble import ensemble_predict
    return ensemble_predict(features_today, regime, threshold=THRESHOLD)


# ── Position sizing ───────────────────────────────────────────────────────────

def compute_target_position(pred: float, regime: str,
                            signal_override: str | None = None) -> tuple[str, float]:
    """
    Returns (side, notional_dollars).
    If signal_override is provided (from ensemble), it wins over threshold test.
    """
    scale = REGIME_SCALE.get(regime, 0.7)
    scaled_notional = NOTIONAL * scale

    if signal_override is not None:
        if signal_override in ("buy", "sell"):
            return signal_override, scaled_notional
        return "flat", 0.0

    if pred > THRESHOLD:
        return "buy", scaled_notional
    elif pred < -THRESHOLD:
        return "sell", scaled_notional
    else:
        return "flat", 0.0


# ── Trade logging ─────────────────────────────────────────────────────────────

def log_trade(record: dict):
    row = pd.DataFrame([record])
    if TRADE_LOG.exists():
        row.to_csv(TRADE_LOG, mode="a", header=False, index=False)
    else:
        row.to_csv(TRADE_LOG, index=False)
    print(f"  Logged to {TRADE_LOG}")


# ── Pipeline (callable from CLI + Streamlit button) ──────────────────────────

def run_pipeline(force: bool = False, submit_alpaca: bool = True,
                 verbose: bool = True) -> dict:
    """
    Run the full live-trading pipeline once.

    Returns a dict with keys:
        status   : 'ok' | 'skipped' | 'error'
        message  : human-readable summary
        record   : the trade_log row written
        ensemble : full ensemble output (votes + agreement)
    """
    _log = print if verbose else (lambda *a, **k: None)

    today = date.today().isoformat()
    _log(f"\n=== LIVE TRADING — {today} ===")

    if not force and TRADE_LOG.exists():
        try:
            existing = pd.read_csv(TRADE_LOG, engine="python", on_bad_lines="skip")
            if existing["date"].astype(str).str.startswith(today).any():
                msg = f"Already traded today ({today}). Use force=True to override."
                _log(f"  {msg}")
                return {"status": "skipped", "message": msg}
        except Exception:
            pass

    # 1. Features
    features_df    = fetch_latest_features()
    features_today = features_df.iloc[-1]

    # 2. Regime
    regime = detect_regime(features_today)
    _log(f"  Regime: {regime}")

    # 3. Ensemble
    ens  = ensemble_forecast(features_today, regime)
    pred = ens["pred"]
    _log(f"  Ensemble pred={pred:+.5f}  agreement={ens['agreement']:.0%}  "
         f"active={ens['n_active']}/{ens['n_models']}")

    # Persist consensus JSON for dashboard
    import json as _json
    (PROCESSED / "latest_consensus.json").write_text(_json.dumps({
        "date":      today,
        "regime":    regime,
        "pred":      ens["pred"],
        "signal":    ens["signal"],
        "agreement": ens["agreement"],
        "n_active":  ens["n_active"],
        "votes":     ens["votes"],
    }, indent=2, default=float))

    # 4. Position
    side, notional = compute_target_position(pred, regime,
                                             signal_override=ens["signal"])
    target_symbol = (SYMBOL_LONG if side == "buy"
                     else SYMBOL_SHORT if side == "sell" else None)

    # 5. Alpaca order (optional)
    order_result = {"status": "skipped", "order_id": None}
    if submit_alpaca:
        try:
            from src.trading.alpaca_client import (
                get_position, submit_order, close_position
            )

            other_symbol = (SYMBOL_SHORT if side == "buy"
                            else SYMBOL_LONG if side == "sell" else None)
            if other_symbol and get_position(other_symbol):
                close_position(other_symbol)
            if side == "flat":
                for sym in (SYMBOL_LONG, SYMBOL_SHORT):
                    if get_position(sym):
                        close_position(sym)

            if target_symbol:
                price = float(yf.Ticker(target_symbol).fast_info["last_price"])
                qty   = round(notional / price, 2)
                if qty > 0:
                    order_result = submit_order(
                        target_symbol, qty, "buy",
                        note=f"regime={regime} pred={pred:+.5f} signal={side}"
                    )
        except Exception as e:
            order_result = {"status": f"error: {e}", "order_id": None}

    # 6. Log
    record = {
        "date":          today,
        "regime":        regime,
        "pred_return":   round(pred, 6),
        "signal":        side,
        "symbol":        target_symbol or "",
        "notional":      notional,
        "order_id":      order_result.get("order_id"),
        "order_status":  order_result.get("status"),
        "vix":           float(features_today.get("VIX", np.nan)),
        "sp500_return":  float(features_today.get("sp500_log_return", np.nan)),
        "agreement":     round(ens["agreement"], 3),
        "n_active_models": ens["n_active"],
    }
    log_trade(record)

    _log(f"=== DONE: {side.upper()} -> {target_symbol or '-'} ===")
    return {
        "status":   "ok",
        "message":  f"{side.upper()} {target_symbol or ''} — pred {pred:+.3%}, agreement {ens['agreement']:.0%}",
        "record":   record,
        "ensemble": ens,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-submit", action="store_true", help="Skip Alpaca order submission")
    args = parser.parse_args()
    run_pipeline(force=args.force, submit_alpaca=not args.no_submit, verbose=True)


if __name__ == "__main__":
    main()
