"""
Regime-aware portfolio backtest with full economic metrics.
Outputs: backtest_results.csv, per-model cumulative return plots, regime breakdown.

Usage:
    source .venv/bin/activate
    python scripts/06_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import PROJECT_ROOT

REGIME_COLORS = {"Bull": "#2ecc71", "Normal": "#3498db", "Bear/Crisis": "#e74c3c", "Extreme": "#9b59b6"}
OUT_DIR = PROJECT_ROOT / "data" / "processed"


# ── Position generation ───────────────────────────────────────────────────────

def generate_positions(preds: pd.Series, threshold: float = 0.001) -> pd.Series:
    return pd.Series(
        np.where(preds > threshold, 1, np.where(preds < -threshold, -1, 0)),
        index=preds.index,
    )


def regime_scaled_positions(preds: pd.Series, regime_labels: pd.Series,
                             threshold: float = 0.001) -> pd.Series:
    """Scale position size by regime risk: reduce size in Extreme, increase in Bull."""
    scale = regime_labels.map({"Bull": 1.0, "Normal": 0.8, "Bear/Crisis": 0.6, "Extreme": 0.4})
    scale = scale.reindex(preds.index).fillna(0.7)
    base = generate_positions(preds, threshold).astype(float)
    return (base * scale).clip(-1, 1)


# ── Metrics ───────────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series) -> float:
    std = returns.std()
    return (returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def sortino_ratio(returns: pd.Series) -> float:
    downside = returns[returns < 0].std()
    return (returns.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()


def calmar_ratio(returns: pd.Series) -> float:
    ann_return = returns.mean() * 252
    mdd = abs(max_drawdown(returns))
    return ann_return / mdd if mdd > 0 else 0.0


def information_coefficient(preds: pd.Series, actual: pd.Series) -> float:
    df = pd.DataFrame({"pred": preds, "actual": actual}).dropna()
    return df["pred"].corr(df["actual"]) if len(df) > 10 else np.nan


def hit_rate(preds: pd.Series, actual: pd.Series) -> float:
    df = pd.DataFrame({"pred": preds, "actual": actual}).dropna()
    return (np.sign(df["pred"]) == np.sign(df["actual"])).mean()


def turnover(position: pd.Series) -> float:
    return position.diff().abs().mean()


def compute_metrics(returns: pd.Series, preds: pd.Series, actual: pd.Series,
                    position: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) < 10:
        return {}
    return {
        "sharpe": round(sharpe_ratio(r), 4),
        "sortino": round(sortino_ratio(r), 4),
        "max_dd": round(max_drawdown(r), 4),
        "calmar": round(calmar_ratio(r), 4),
        "ann_return": round(r.mean() * 252, 4),
        "ic": round(information_coefficient(preds, actual), 4),
        "hit_rate": round(hit_rate(preds, actual), 4),
        "turnover": round(turnover(position), 4),
        "n": len(r),
    }


# ── Backtest core ─────────────────────────────────────────────────────────────

def run_backtest(preds: pd.Series, actual: pd.Series, name: str,
                 regime_labels: pd.Series | None = None,
                 cost: float = 0.0005, threshold: float = 0.001) -> dict:
    df = pd.DataFrame({"pred": preds, "actual": actual}).dropna()
    if len(df) < 20:
        return {"model": name}

    if regime_labels is not None:
        pos = regime_scaled_positions(df["pred"], regime_labels, threshold)
    else:
        pos = generate_positions(df["pred"], threshold)

    # No lookahead: position set at t, return realized at t+1
    strategy_returns = pos.shift(1) * df["actual"]
    tc = cost * pos.diff().abs()
    net_returns = (strategy_returns - tc).dropna()

    result = {"model": name}
    result.update(compute_metrics(net_returns, df["pred"], df["actual"], pos))
    result["net_returns"] = net_returns
    result["position"] = pos

    # Per-regime breakdown
    if regime_labels is not None:
        regime_breakdown = {}
        for regime in ["Bull", "Normal", "Bear/Crisis", "Extreme"]:
            idx = regime_labels[regime_labels == regime].index.intersection(net_returns.index)
            if len(idx) < 20:
                continue
            r_ret = net_returns.loc[idx]
            r_pred = df["pred"].loc[idx]
            r_act = df["actual"].loc[idx]
            r_pos = pos.loc[idx]
            regime_breakdown[regime] = compute_metrics(r_ret, r_pred, r_act, r_pos)
        result["regime_breakdown"] = regime_breakdown

    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cumulative(result: dict, bh_returns: pd.Series, regime_labels: pd.Series,
                    out_path: Path):
    net_returns = result.get("net_returns")
    if net_returns is None or len(net_returns) < 10:
        return

    cum_strat = (1 + net_returns).cumprod()
    bh_aligned = bh_returns.reindex(net_returns.index).dropna()
    cum_bh = (1 + bh_aligned).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                             gridspec_kw={"height_ratios": [3, 1.2, 1]})

    # Top: cumulative returns with regime shading
    ax = axes[0]
    reg = regime_labels.reindex(cum_strat.index)
    for regime, color in REGIME_COLORS.items():
        in_span = False
        for dt in cum_strat.index:
            r = reg.get(dt)
            if r == regime and not in_span:
                span_start = dt; in_span = True
            elif r != regime and in_span:
                ax.axvspan(span_start, dt, alpha=0.15, color=color, linewidth=0)
                in_span = False
        if in_span:
            ax.axvspan(span_start, cum_strat.index[-1], alpha=0.15, color=color, linewidth=0)

    ax.plot(cum_strat.index, cum_strat.values, color="royalblue", linewidth=1.2, label="Strategy")
    ax.plot(cum_bh.index, cum_bh.values, color="gray", linewidth=0.8, linestyle="--", label="Buy & Hold")
    ax.axhline(1, color="black", linewidth=0.5)
    patches = [mpatches.Patch(color=c, alpha=0.4, label=r) for r, c in REGIME_COLORS.items()]
    ax.legend(handles=[plt.Line2D([0],[0],color="royalblue",label="Strategy"),
                       plt.Line2D([0],[0],color="gray",linestyle="--",label="Buy & Hold")] + patches,
              fontsize=8, loc="upper left")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"Regime-Aware Backtest — {result['model']}", fontsize=11)
    ax.grid(alpha=0.2)

    # Middle: drawdown
    ax2 = axes[1]
    cum = (1 + net_returns).cumprod()
    dd = (cum / cum.cummax() - 1)
    ax2.fill_between(dd.index, dd.values, 0, color="tomato", alpha=0.6)
    ax2.set_ylabel("Drawdown")
    ax2.set_ylim(dd.min() * 1.1, 0.05)
    ax2.grid(alpha=0.2)

    # Bottom: rolling 63-day Sharpe
    ax3 = axes[2]
    roll_sharpe = net_returns.rolling(63).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0, raw=True
    )
    ax3.plot(roll_sharpe.index, roll_sharpe.values, color="steelblue", linewidth=0.8)
    ax3.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax3.set_ylabel("63d Sharpe")
    ax3.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_regime_breakdown(results: list, out_path: Path):
    """Heatmap of Sharpe per model × regime."""
    regimes = ["Bull", "Normal", "Bear/Crisis", "Extreme"]
    # Only include top models by overall Sharpe
    top_results = sorted(
        [r for r in results if "regime_breakdown" in r and "sharpe" in r],
        key=lambda x: x.get("sharpe", -99), reverse=True
    )[:12]

    if not top_results:
        return

    models = [r["model"] for r in top_results]
    matrix = np.full((len(models), len(regimes)), np.nan)
    for i, res in enumerate(top_results):
        rb = res.get("regime_breakdown", {})
        for j, reg in enumerate(regimes):
            if reg in rb and "sharpe" in rb[reg]:
                matrix[i, j] = rb[reg]["sharpe"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace("fwd_return_", "").replace("_", " ") for m in models], fontsize=8)
    ax.set_title("Annualised Sharpe by Model × Regime")
    plt.colorbar(im, ax=ax, label="Sharpe")
    for i in range(len(models)):
        for j in range(len(regimes)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== REGIME-AWARE PORTFOLIO BACKTEST ===")

    data_dir = OUT_DIR
    preds_dir = data_dir / "predictions"

    actual_df = pd.read_parquet(data_dir / "actual_returns.parquet")
    regime_labels = pd.read_parquet(data_dir / "regime_labels.parquet")["regime"]

    all_results = []

    for file in sorted(preds_dir.glob("*_preds.parquet")):
        name = file.stem.replace("_preds", "")
        preds = pd.read_parquet(file)["prediction"]
        target_col = "fwd_return_1d" if "1d" in name else "fwd_return_5d"
        actual = actual_df[target_col]

        print(f"\nBacktesting: {name}")

        # Regime-agnostic run
        res_plain = run_backtest(preds, actual, f"{name}__plain",
                                 regime_labels=None, cost=0.0005, threshold=0.001)
        all_results.append(res_plain)

        # Regime-aware position sizing
        res_regime = run_backtest(preds, actual, f"{name}__regime_sized",
                                  regime_labels=regime_labels, cost=0.0005, threshold=0.001)
        all_results.append(res_regime)

        # Plot cumulative return for regime-aware version
        plot_cumulative(
            res_regime,
            bh_returns=actual,
            regime_labels=regime_labels,
            out_path=data_dir / f"backtest_cumret_{name}.png",
        )

        metrics_keys = ["sharpe", "sortino", "max_dd", "calmar", "ann_return", "ic", "hit_rate", "turnover", "n"]
        for res in [res_plain, res_regime]:
            row = {k: res.get(k, "") for k in ["model"] + metrics_keys}
            print(f"  {res['model'][-30:]:<30}  "
                  f"Sharpe={row.get('sharpe',''):<7}  "
                  f"MaxDD={row.get('max_dd',''):<8}  "
                  f"IC={row.get('ic',''):<7}  "
                  f"HitRate={row.get('hit_rate','')}")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_cols = ["model", "sharpe", "sortino", "max_dd", "calmar",
                    "ann_return", "ic", "hit_rate", "turnover", "n"]
    summary_df = pd.DataFrame(
        [{k: r.get(k, np.nan) for k in summary_cols} for r in all_results]
    ).sort_values("sharpe", ascending=False)

    print("\n" + "="*80)
    print("BACKTEST SUMMARY (sorted by Sharpe)")
    print("="*80)
    print(summary_df.to_string(index=False))

    out_path = data_dir / "backtest_results.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Regime breakdown heatmap ───────────────────────────────────────────────
    print("\nGenerating regime breakdown heatmap...")
    plot_regime_breakdown(all_results, out_path=data_dir / "backtest_regime_heatmap.png")

    # ── Per-regime breakdown CSV ───────────────────────────────────────────────
    breakdown_rows = []
    for res in all_results:
        for regime, m in res.get("regime_breakdown", {}).items():
            row = {"model": res["model"], "regime": regime}
            row.update(m)
            breakdown_rows.append(row)
    if breakdown_rows:
        bd_df = pd.DataFrame(breakdown_rows).sort_values(["model", "regime"])
        bd_path = data_dir / "backtest_regime_breakdown.csv"
        bd_df.to_csv(bd_path, index=False)
        print(f"Saved: {bd_path}")

    print("\n=== BACKTEST COMPLETE ===")


if __name__ == "__main__":
    main()
