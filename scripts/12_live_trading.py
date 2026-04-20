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
SYMBOL      = "SPY"
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


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_return(features_today: pd.Series, regime: str) -> float:
    """Load saved LightGBM and predict 1-day return."""
    model_path = MODELS_DIR / "lgb_fwd_return_1d_regime_conditioned.pkl"
    if not model_path.exists():
        print("  WARNING: saved model not found. Run scripts/11_save_models.py first.")
        return 0.0

    bundle  = joblib.load(model_path)
    model   = bundle["model"]
    imputer = bundle["imputer"]
    feat_cols    = bundle["feat_cols"]
    regime_cols  = bundle["regime_cols"]

    # Build regime one-hot
    row = {}
    for col in feat_cols:
        if col in regime_cols:
            row[col] = 1.0 if col == f"regime_{regime}" else 0.0
        else:
            row[col] = features_today.get(col, np.nan)

    X = pd.DataFrame([row])[feat_cols]
    X_imp = imputer.transform(X)
    return float(model.predict(X_imp)[0])


# ── Position sizing ───────────────────────────────────────────────────────────

def compute_target_position(pred: float, regime: str) -> tuple[str, float]:
    """
    Returns (side, notional_dollars).
    side: 'buy', 'sell', or 'flat'
    """
    scale = REGIME_SCALE.get(regime, 0.7)
    scaled_notional = NOTIONAL * scale

    if pred > THRESHOLD:
        return "buy", scaled_notional
    elif pred < -THRESHOLD:
        return "sell", scaled_notional  # short SPY
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = date.today().isoformat()
    print(f"\n=== LIVE TRADING — {today} ===")

    # Guard: skip if already ran today
    if TRADE_LOG.exists():
        existing = pd.read_csv(TRADE_LOG)
        already_ran = existing["date"].astype(str).str.startswith(today)
        if already_ran.any():
            print(f"  Already traded today ({today}). Exiting.")
            return

    # 1. Features
    print("\n[1] Computing features...")
    features_df = fetch_latest_features()
    features_today = features_df.iloc[-1]
    print(f"  Last date: {features_df.index[-1].date()}")

    # 2. Regime
    print("\n[2] Detecting regime...")
    regime = detect_regime(features_today)
    print(f"  Current regime: {regime}")

    # 3. Prediction
    print("\n[3] Predicting 1-day return...")
    pred = predict_return(features_today, regime)
    print(f"  Predicted return: {pred:+.5f}")

    # 4. Position sizing
    side, notional = compute_target_position(pred, regime)
    print(f"\n[4] Target position: {side.upper()} ${notional:,.0f} of {SYMBOL}")

    # 5. Submit order via Alpaca
    print("\n[5] Submitting order...")
    order_result = {"status": "skipped", "order_id": None}
    try:
        from src.trading.alpaca_client import (
            get_account, get_position, submit_order, close_position
        )

        acct = get_account()
        print(f"  Account equity: ${acct['equity']:,.2f}  Cash: ${acct['cash']:,.2f}")

        current_pos = get_position(SYMBOL)
        current_side = current_pos["side"] if current_pos else "flat"
        print(f"  Current position: {current_side}")

        # Close existing if opposite or going flat
        if current_pos and (side == "flat" or current_side != side):
            close_result = close_position(SYMBOL)
            print(f"  Closed existing position: {close_result}")

        # Open new position
        if side in ("buy", "sell"):
            price = float(yf.Ticker(SYMBOL).fast_info["last_price"])
            # Fractional shares only allowed for buys; shorts require whole shares
            qty = round(notional / price, 2) if side == "buy" else int(notional / price)
            if qty <= 0:
                print("  Qty rounds to 0 — skipping order.")
            else:
                order_result = submit_order(SYMBOL, qty, side, note=f"regime={regime} pred={pred:+.5f}")
                print(f"  Order: {order_result}")
        else:
            print("  No position taken (flat signal).")

    except ImportError:
        print("  alpaca-py not installed. Run: pip install alpaca-py")
        print("  Order NOT submitted — logging signal only.")
    except Exception as e:
        print(f"  Alpaca error: {e}")
        order_result = {"status": f"error: {e}", "order_id": None}

    # 6. Log
    record = {
        "date":       today,
        "regime":     regime,
        "pred_return": round(pred, 6),
        "signal":     side,
        "notional":   notional,
        "order_id":   order_result.get("order_id"),
        "order_status": order_result.get("status"),
        "vix":        float(features_today.get("VIX", np.nan)),
        "sp500_return": float(features_today.get("sp500_log_return", np.nan)),
    }
    log_trade(record)

    print(f"\n=== DONE: {side.upper()} signal logged ===")
    print(json.dumps({k: v for k, v in record.items() if k != "order_id"}, indent=2))


if __name__ == "__main__":
    main()
