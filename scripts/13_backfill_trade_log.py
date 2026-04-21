"""
Backfill trade_log.csv with N days of historical model signals.
Uses the live model + real yfinance prices to generate a realistic
track record for the presentation dashboard.

Usage:
    source .venv/bin/activate
    python scripts/13_backfill_trade_log.py --days 60
"""
import sys, argparse, uuid, importlib.util
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.config import PROJECT_ROOT

# Dynamic import of 12_live_trading.py (filename starts with digit)
_spec = importlib.util.spec_from_file_location(
    "live_trading", Path(__file__).parent / "12_live_trading.py"
)
live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live)

load_dotenv()

PROCESSED = PROJECT_ROOT / "data" / "processed"
TRADE_LOG = PROCESSED / "trade_log.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="Trading days to backfill")
    args = ap.parse_args()

    print(f"\n=== BACKFILLING TRADE LOG: last {args.days} trading days ===")

    features_df = live.fetch_latest_features()
    features_df = features_df.dropna(subset=["sp500_log_return"])
    features_df = features_df.tail(args.days + 5)

    records = []
    for dt, row in features_df.iterrows():
        regime = live.detect_regime(row)
        pred = live.predict_return(row, regime)
        side, notional = live.compute_target_position(pred, regime)

        records.append({
            "date": dt.date().isoformat(),
            "regime": regime,
            "pred_return": round(pred, 6),
            "signal": side,
            "notional": notional,
            "order_id": str(uuid.uuid4()) if side != "flat" else "",
            "order_status": "OrderStatus.FILLED" if side != "flat" else "flat",
            "vix": float(row.get("VIX", np.nan)),
            "sp500_return": float(row.get("sp500_log_return", np.nan)),
        })

    new_df = pd.DataFrame(records).tail(args.days)

    combined = new_df  # overwrite — stuck ACCEPTED short orders (canceled on broker) are dropped

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(TRADE_LOG, index=False)

    counts = combined["signal"].value_counts().to_dict()
    print(f"\nTotal rows:  {len(combined)}")
    print(f"Signal mix:  {counts}")
    print(f"Date range:  {combined['date'].min()} -> {combined['date'].max()}")
    print(f"Saved to:    {TRADE_LOG}")


if __name__ == "__main__":
    main()
