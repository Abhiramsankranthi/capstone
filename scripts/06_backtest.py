import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.config import PROJECT_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# Strategy functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_positions(preds, threshold=0.001):
    """Convert predictions to long/short positions."""
    return np.where(preds > threshold, 1,
           np.where(preds < -threshold, -1, 0))


def compute_returns(position, actual_returns):
    """Compute strategy returns (NO lookahead bias)."""
    return position.shift(1) * actual_returns


def apply_transaction_costs(returns, position, cost=0.0005):
    """Apply transaction costs based on turnover."""
    trade = np.abs(position - position.shift(1))
    return returns - cost * trade


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(returns):
    std = np.std(returns)
    if std == 0:
        return 0
    return np.mean(returns) / std * np.sqrt(252)


def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()


def information_coefficient(preds, actual):
    df = pd.DataFrame({"pred": preds, "actual": actual}).dropna()
    if df.empty:
        return np.nan
    return df["pred"].corr(df["actual"])

def turnover(position):
    return np.mean(np.abs(position - position.shift(1)))


# ─────────────────────────────────────────────────────────────────────────────
# Backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(preds, actual, name, cost=0.0005, threshold=0.001):
    """Run full backtest for one model."""
    df = pd.DataFrame({
        "pred": preds,
        "actual": actual
    }).dropna()

    # positions
    position = pd.Series(generate_positions(df["pred"], threshold), index=df.index)
    

    # returns
    returns = compute_returns(position, df["actual"])

    # apply transaction costs
    net_returns = apply_transaction_costs(returns, position, cost)

    # metrics
    sharpe = sharpe_ratio(net_returns)
    mdd = max_drawdown(net_returns)
    ic = information_coefficient(df["pred"], df["actual"])
    to = turnover(position)

    return {
        "model": name,
        "sharpe": round(sharpe, 4),
        "max_dd": round(mdd, 4),
        "ic": round(ic, 4),
        "turnover": round(to, 4),
        "net_returns": net_returns  
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== BACKTESTING START ===")

    data_dir = PROJECT_ROOT / "data" / "processed"
    preds_dir = data_dir / "predictions"

    # Load actual returns
    actual_df = pd.read_parquet(data_dir / "actual_returns.parquet")

    results = []

    # Loop through prediction files
    for file in preds_dir.glob("*_preds.parquet"):
        name = file.stem.replace("_preds", "")
        preds = pd.read_parquet(file)["prediction"]

        # Match correct target
        if "1d" in name:
            actual = actual_df["fwd_return_1d"]
        else:
            actual = actual_df["fwd_return_5d"]

        print(f"\nRunning backtest: {name}")

        for cost in [0.0005, 0.001]:
          for threshold in [0, 0.001, 0.002]:
            res = run_backtest(
                preds,
                actual,
                f"{name}_c{cost}_t{threshold}",
                cost,
                threshold
            )
            results.append(res)

    # Convert to DataFrame
    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    print("\n=== RESULTS ===")
    print(results_df.to_string(index=False))

    # Save results
    out_path = data_dir / "backtest_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results: {out_path}")

    print("\n=== BACKTEST COMPLETE ===")


if __name__ == "__main__":
    main()