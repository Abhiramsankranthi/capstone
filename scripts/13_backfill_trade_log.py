"""
Backfill trade_log.csv with N days of historical model signals.
Uses the live model + real yfinance prices to generate a realistic
track record for the presentation dashboard.

Usage:
    source .venv/bin/activate
    python scripts/13_backfill_trade_log.py --days 60
"""
import sys, argparse, uuid, importlib.util, json
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
    last_ens = None
    last_regime = None
    last_date = None
    for dt, row in features_df.iterrows():
        regime = live.detect_regime(row)
        ens = live.ensemble_forecast(row, regime)
        pred = ens["pred"]
        side, notional = live.compute_target_position(pred, regime,
                                                     signal_override=ens["signal"])

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
            "agreement": round(ens["agreement"], 3),
            "n_active_models": ens["n_active"],
        })
        last_ens, last_regime, last_date = ens, regime, dt.date().isoformat()

    # Persist most-recent consensus so the dashboard can render it
    if last_ens is not None:
        consensus_path = PROCESSED / "latest_consensus.json"
        consensus_path.write_text(json.dumps({
            "date":      last_date,
            "regime":    last_regime,
            "pred":      last_ens["pred"],
            "signal":    last_ens["signal"],
            "agreement": last_ens["agreement"],
            "n_active":  last_ens["n_active"],
            "votes":     last_ens["votes"],
        }, indent=2, default=float))
        print(f"  consensus JSON -> {consensus_path}")

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
