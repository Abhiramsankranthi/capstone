"""
Alpaca paper-trading client wrapper.
Uses alpaca-py (pip install alpaca-py).
"""
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
PAPER      = os.getenv("ALPACA_PAPER", "true").lower() != "false"


def _client():
    from alpaca.trading.client import TradingClient
    return TradingClient(API_KEY, API_SECRET, paper=PAPER)


def get_account() -> dict:
    acct = _client().get_account()
    return {
        "equity":       float(acct.equity),
        "cash":         float(acct.cash),
        "buying_power": float(acct.buying_power),
        "pnl":          float(acct.equity) - float(acct.last_equity),
    }


def get_position(symbol: str = "SPY") -> dict | None:
    """Return current position dict or None if flat."""
    from alpaca.trading.client import TradingClient
    from alpaca.common.exceptions import APIError
    try:
        pos = _client().get_open_position(symbol)
        return {
            "symbol":      pos.symbol,
            "qty":         float(pos.qty),
            "side":        pos.side.value,
            "avg_entry":   float(pos.avg_entry_price),
            "market_value":float(pos.market_value),
            "unrealized_pl":float(pos.unrealized_pl),
        }
    except APIError:
        return None


def submit_order(symbol: str, qty: float, side: str, note: str = "") -> dict:
    """
    side: 'buy' or 'sell'
    qty: fractional shares supported on paper.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = _client().submit_order(req)
    return {"order_id": str(order.id), "status": str(order.status), "note": note}


def close_position(symbol: str = "SPY") -> dict | None:
    """Close entire position if one exists."""
    from alpaca.common.exceptions import APIError
    try:
        resp = _client().close_position(symbol)
        return {"status": "closed", "symbol": symbol}
    except APIError:
        return None


def list_recent_orders(limit: int = 10) -> pd.DataFrame:
    """Return recent orders as DataFrame for dashboard display."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
    orders = _client().get_orders(req)
    rows = []
    for o in orders:
        rows.append({
            "submitted":   pd.to_datetime(o.submitted_at).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
            "symbol":      o.symbol,
            "side":        o.side.value,
            "qty":         float(o.qty) if o.qty else 0.0,
            "filled_qty":  float(o.filled_qty) if o.filled_qty else 0.0,
            "avg_fill":    float(o.filled_avg_price) if o.filled_avg_price else None,
            "status":      str(o.status.value) if hasattr(o.status, "value") else str(o.status),
        })
    return pd.DataFrame(rows)


def get_portfolio_history(period: str = "1M") -> pd.DataFrame:
    """Return daily equity history as DataFrame."""
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    req = GetPortfolioHistoryRequest(period=period, timeframe="1D")
    hist = _client().get_portfolio_history(req)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hist.timestamp, unit="s", utc=True).tz_convert("America/New_York"),
        "equity":    hist.equity,
        "pnl":       hist.profit_loss,
        "pnl_pct":   hist.profit_loss_pct,
    })
    return df.set_index("timestamp")
