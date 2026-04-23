"""
Streamlit demo app for Multi-Modal Financial Market Regime Detection & Forecasting.
Usage:
    source .venv/bin/activate
    streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import PROJECT_ROOT

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime & Return Forecasting",
    page_icon="📈",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
PROCESSED = PROJECT_ROOT / "data" / "processed"
PREDS_DIR = PROCESSED / "predictions"

REGIME_COLORS = {
    "Bull": "#2ecc71",
    "Normal": "#3498db",
    "Bear/Crisis": "#e74c3c",
    "Extreme": "#9b59b6",
}

@st.cache_data
def load_features():
    return pd.read_parquet(PROCESSED / "features.parquet")

@st.cache_data
def load_regimes():
    df = pd.read_parquet(PROCESSED / "regime_labels.parquet")
    return df["regime"]

@st.cache_data
def load_metrics():
    p = PROCESSED / "model_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data
def load_predictions():
    preds = {}
    if not PREDS_DIR.exists():
        return preds
    for f in sorted(PREDS_DIR.glob("*.parquet")):
        key = f.stem.replace("_preds", "")
        df = pd.read_parquet(f)
        preds[key] = df["prediction"]
    return preds

@st.cache_data
def load_feature_importance(key):
    p = PROCESSED / f"feature_importance_{key}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

# ── Load data ─────────────────────────────────────────────────────────────────
features_df = load_features()
regime_labels = load_regimes()
metrics_df = load_metrics()
all_preds = load_predictions()

sp500_returns = features_df["sp500_log_return"]
sp500_price = np.exp(sp500_returns.cumsum())   # normalized to 1.0 at start

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Controls")

pages = ["Overview", "Regime Detection", "Markov Switching", "Model Performance", "Feature Importance", "SHAP Analysis", "Sentiment Analysis", "Backtest", "Live Trading"]
page = st.sidebar.radio("Navigate", pages)

# ── PAGE: Overview ────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Multi-Modal Financial Market Regime Detection & Forecasting")
    st.caption("CFSE 570 Data Science Capstone · Arizona State University")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading Days", f"{len(features_df):,}")
    c2.metric("Features", str(features_df.shape[1]))
    c3.metric("HMM Regimes", "4")
    c4.metric("Models Trained", str(len(metrics_df)) if not metrics_df.empty else "—")

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.markdown(
        """
        ```
        Raw Data (yfinance + FRED + FinBERT)
                ↓
        Feature Engineering (37 features: technical, macro, VIX, sentiment)
                ↓
        HMM Regime Detection  →  4 States: Bull / Normal / Bear/Crisis / Extreme
                ↓
        Forecasting Models    →  Ridge · Lasso · LightGBM · LSTM · TFT
                ↓
        Walk-Forward Backtest (no data leakage, expanding window)
        ```
        """
    )

    st.subheader("Regime Summary")
    counts = regime_labels.value_counts()
    total = len(regime_labels)
    rows = []
    for r in ["Bull", "Normal", "Bear/Crisis", "Extreme"]:
        n = counts.get(r, 0)
        aligned = regime_labels.reindex(features_df.index)
        sub = features_df.loc[aligned == r]
        rows.append({
            "Regime": r,
            "% of Time": f"{100*n/total:.1f}%",
            "Mean Daily Return": f"{sub['sp500_log_return'].mean()*100:.3f}%",
            "Mean VIX": f"{sub['vix_level'].mean():.1f}" if "vix_level" in sub.columns else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Crisis Detection Accuracy")
    crisis_data = {
        "Event": ["2008 GFC", "2020 COVID", "2022 Rate Hikes"],
        "Period": ["2008-09 – 2009-03", "2020-02 – 2020-06", "2022-01 – 2022-12"],
        "% Bear/Extreme": ["100%", "87%", "91%"],
    }
    st.dataframe(pd.DataFrame(crisis_data), use_container_width=True, hide_index=True)

# ── PAGE: Regime Detection ────────────────────────────────────────────────────
elif page == "Regime Detection":
    st.title("HMM Regime Detection")

    regime_img = PROCESSED / "regime_chart.png"
    if regime_img.exists():
        st.image(str(regime_img), use_container_width=True)
    else:
        st.info("regime_chart.png not found — run scripts/03_fit_hmm.py first.")

    st.subheader("Regime Timeline (interactive)")

    date_range = st.slider(
        "Date range",
        min_value=features_df.index.min().to_pydatetime(),
        max_value=features_df.index.max().to_pydatetime(),
        value=(pd.Timestamp("2007-01-01").to_pydatetime(),
               pd.Timestamp("2025-01-01").to_pydatetime()),
        format="YYYY",
    )

    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])
    sub_price = sp500_price.loc[(sp500_price.index >= start_ts) & (sp500_price.index <= end_ts)]
    sub_regime = regime_labels.loc[
        (regime_labels.index >= start_ts) & (regime_labels.index <= end_ts)
    ]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sub_price.index, sub_price.values, color="black", linewidth=0.8, zorder=3)

    for regime, color in REGIME_COLORS.items():
        idx = sub_regime[sub_regime == regime].index
        if len(idx) == 0:
            continue
        # shade contiguous spans
        in_span = False
        for i, dt in enumerate(sub_price.index):
            r = sub_regime.get(dt, None)
            if r == regime and not in_span:
                span_start = dt
                in_span = True
            elif r != regime and in_span:
                ax.axvspan(span_start, dt, alpha=0.25, color=color, linewidth=0)
                in_span = False
        if in_span:
            ax.axvspan(span_start, sub_price.index[-1], alpha=0.25, color=color, linewidth=0)

    patches = [mpatches.Patch(color=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_ylabel("S&P 500 (normalized)")
    ax.set_title("S&P 500 Colored by Market Regime")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Regime Distribution")
    counts = sub_regime.value_counts()
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    colors = [REGIME_COLORS.get(r, "gray") for r in counts.index]
    ax2.bar(counts.index, counts.values, color=colors)
    ax2.set_ylabel("# Trading Days")
    ax2.set_title("Regime Counts (selected period)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ── PAGE: Markov Switching ────────────────────────────────────────────────────
elif page == "Markov Switching":
    st.title("Markov Switching vs HMM Regime Comparison")
    st.caption("statsmodels Markov Autoregression (4 regimes) compared to our Gaussian HMM")

    ms_cmp = PROCESSED / "ms_comparison.csv"
    if not ms_cmp.exists():
        st.warning("Run scripts/09_markov_switching.py first.")
        st.stop()

    cmp_df = pd.read_csv(ms_cmp)
    c1, c2, c3 = st.columns(3)
    c1.metric("Adjusted Rand Index", f"{cmp_df['ari_vs_hmm'].iloc[0]:.4f}",
              help="1.0 = perfect agreement, 0 = random")
    c2.metric("Log-Likelihood", f"{cmp_df['loglik'].iloc[0]:,.1f}")
    c3.metric("AIC", f"{cmp_df['aic'].iloc[0]:,.1f}")

    st.markdown("---")

    comp_img = PROCESSED / "ms_regime_comparison.png"
    if comp_img.exists():
        st.subheader("Regime Timeline: HMM vs Markov Switching")
        st.image(str(comp_img), use_container_width=True)

    dist_img = PROCESSED / "ms_distribution_comparison.png"
    if dist_img.exists():
        st.subheader("Regime Distribution Comparison")
        st.image(str(dist_img), use_container_width=True)

    dist_csv = PROCESSED / "regime_distribution_comparison.csv"
    if dist_csv.exists():
        st.subheader("Distribution Table")
        dist_df = pd.read_csv(dist_csv, index_col=0)
        st.dataframe(dist_df.style.format("{:.1%}"), use_container_width=True)

    st.subheader("Per-Regime Agreement (HMM ∩ MS)")
    overlap_cols = [c for c in cmp_df.columns if c.startswith("overlap_")]
    if overlap_cols:
        ov = {c.replace("overlap_", ""): cmp_df[c].iloc[0] for c in overlap_cols}
        ov_df = pd.DataFrame([ov], index=["HMM↔MS Agreement"])
        st.dataframe(ov_df.style.format("{:.1%}"), use_container_width=True)

    st.markdown("""
    **Key insight:** High ARI (>0.4) means both models detect the same underlying regime structure.
    Low overlap in the Extreme regime is expected — it's a rare state and the models weight it differently.
    """)

# ── PAGE: Model Performance ───────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance Comparison")

    if metrics_df.empty:
        st.warning("model_metrics.csv not found. Run scripts/05_train_models.py first.")
        st.stop()

    target_filter = st.selectbox("Target", ["All", "1-day return", "5-day return"])
    mode_filter = st.selectbox("Regime Mode", ["All", "Regime-Agnostic", "Regime-Conditioned"])

    df = metrics_df.copy()
    if target_filter == "1-day return":
        df = df[df["model"].str.contains("1d")]
    elif target_filter == "5-day return":
        df = df[df["model"].str.contains("5d")]
    if mode_filter == "Regime-Agnostic":
        df = df[df["model"].str.contains("agnostic")]
    elif mode_filter == "Regime-Conditioned":
        df = df[df["model"].str.contains("conditioned")]

    # Format for display
    display = df.copy()
    display["dir_acc"] = (display["dir_acc"] * 100).round(2).astype(str) + "%"
    display.columns = ["Model", "RMSE", "Dir Acc", "Sharpe", "N Test"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Directional Accuracy vs Sharpe")
    fig, ax = plt.subplots(figsize=(8, 5))
    df2 = metrics_df.copy()
    model_types = df2["model"].apply(lambda x: x.split("_")[0].upper())
    colors_map = {"RIDGE": "#e74c3c", "LASSO": "#f39c12", "LGB": "#2ecc71",
                  "LSTM": "#3498db", "TFT": "#9b59b6"}
    markers = {"agnostic": "o", "conditioned": "^"}
    for _, row in df2.iterrows():
        mtype = row["model"].split("_")[0].upper()
        mode = "conditioned" if "conditioned" in row["model"] else "agnostic"
        ax.scatter(row["dir_acc"] * 100, row["sharpe"],
                   color=colors_map.get(mtype, "gray"),
                   marker=markers[mode], s=80, zorder=3)
        ax.annotate(row["model"].replace("fwd_return_", "").replace("_", " "),
                    (row["dir_acc"] * 100, row["sharpe"]),
                    fontsize=6, ha="left", va="bottom")
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Directional Accuracy (%)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("All Models: Direction Accuracy vs Sharpe")
    ax.grid(alpha=0.2)
    legend_els = [mpatches.Patch(color=c, label=t) for t, c in colors_map.items() if c]
    ax.legend(handles=legend_els, fontsize=7, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Rolling Directional Accuracy")
    rolling_pngs = sorted((PROCESSED).glob("rolling_dir_acc_*.png"))
    if rolling_pngs:
        selected = st.selectbox("Select model", [p.stem.replace("rolling_dir_acc_", "") for p in rolling_pngs])
        img_path = PROCESSED / f"rolling_dir_acc_{selected}.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
    else:
        st.info("No rolling accuracy plots found.")

# ── PAGE: Feature Importance ──────────────────────────────────────────────────
elif page == "Feature Importance":
    st.title("Feature Importance (LightGBM)")

    imp_files = sorted(PROCESSED.glob("feature_importance_lgb_*.csv"))
    if not imp_files:
        st.warning("No feature importance CSVs found. Run scripts/05_train_models.py first.")
        st.stop()

    model_key = st.selectbox("LightGBM variant", [f.stem.replace("feature_importance_", "") for f in imp_files])
    imp_df = load_feature_importance(model_key)

    if imp_df.empty:
        st.info("No data.")
        st.stop()

    if "regime" in imp_df.columns:
        imp_plot = imp_df.groupby("feature")["importance_gain"].mean().reset_index()
        imp_plot = imp_plot.sort_values("importance_gain", ascending=False).head(20)
    else:
        imp_plot = imp_df.sort_values("importance_gain", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_plot["feature"][::-1], imp_plot["importance_gain"][::-1], color="steelblue")
    ax.set_xlabel("Mean Gain")
    ax.set_title(f"Top 20 Features — {model_key}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(imp_plot.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Sentiment importance highlight
    sent_features = ["sent_mean", "sent_weighted", "sent_momentum", "sent_max_neg", "sent_article_count"]
    sent_imp = imp_plot[imp_plot["feature"].isin(sent_features)]
    if not sent_imp.empty:
        st.subheader("Sentiment Feature Contributions")
        st.dataframe(sent_imp.reset_index(drop=True), use_container_width=True, hide_index=True)

# ── PAGE: SHAP Analysis ──────────────────────────────────────────────────────
elif page == "SHAP Analysis":
    st.title("SHAP Stability & Regime Interpretability")
    st.caption("How feature importance shifts across market regimes — and how stable it is")

    target = st.selectbox("Target", ["fwd_return_1d", "fwd_return_5d"])

    # Overall summary
    summary_png = PROCESSED / f"shap_summary_{target}.png"
    if summary_png.exists():
        st.subheader("Overall Feature Importance (SHAP)")
        st.image(str(summary_png), use_container_width=True)
    else:
        st.warning("Run scripts/08_shap_analysis.py first to generate SHAP plots.")
        st.stop()

    # Per-regime comparison
    regime_cmp = PROCESSED / f"shap_regime_comparison_{target}.png"
    if regime_cmp.exists():
        st.subheader("Feature Importance by Regime")
        st.image(str(regime_cmp), use_container_width=True)

    # Stability heatmap
    stability_png = PROCESSED / f"shap_stability_{target}.png"
    if stability_png.exists():
        st.subheader("SHAP Rank Stability Across Regimes")
        st.image(str(stability_png), use_container_width=True)
        st.markdown(
            """
            **How to read this:**
            - Left heatmap: rank of each feature per regime (darker = more important)
            - Right matrix: Spearman ρ between regime feature rankings (1.0 = perfectly stable)
            - High ρ between Bull/Normal/Bear means features work consistently
            - Low ρ involving Extreme regime means the model uses different signals in market extremes
            """
        )

    # Per-regime individual plots
    st.subheader("Per-Regime SHAP Detail")
    regime_options = []
    for r in ["Bull", "Normal", "Bear/Crisis", "Extreme"]:
        p = PROCESSED / f"shap_{r.replace('/', '_')}_{target}.png"
        if p.exists():
            regime_options.append(r)
    if regime_options:
        sel_regime = st.selectbox("Select Regime", regime_options)
        img_path = PROCESSED / f"shap_{sel_regime.replace('/', '_')}_{target}.png"
        st.image(str(img_path), use_container_width=True)

    # Mean |SHAP| table
    csv_path = PROCESSED / f"shap_summary_{target}.csv"
    if csv_path.exists():
        shap_summary = pd.read_csv(csv_path)
        st.subheader("Top Features by Mean |SHAP|")
        st.dataframe(shap_summary.head(20), use_container_width=True, hide_index=True)

# ── PAGE: Sentiment Analysis ─────────────────────────────────────────────────
elif page == "Sentiment Analysis":
    st.title("FinBERT Sentiment: Ablation & Predictive Correlation")
    st.caption("Does NLP sentiment from news headlines add predictive power for returns?")

    abl_path = PROCESSED / "sentiment_ablation.csv"
    corr_path = PROCESSED / "sentiment_predictive_correlation.csv"

    if not abl_path.exists():
        st.warning("Run scripts/10_sentiment_ablation.py first.")
        st.stop()

    abl_df = pd.read_csv(abl_path)
    corr_df = pd.read_csv(corr_path) if corr_path.exists() else pd.DataFrame()

    # Ablation summary table
    st.subheader("Ablation Study: With vs Without Sentiment Features")
    st.markdown("LightGBM trained on walk-forward expanding window, 5 folds")

    for target in ["fwd_return_1d", "fwd_return_5d"]:
        sub = abl_df[abl_df["target"] == target]
        if sub.empty:
            continue
        st.markdown(f"**{target}**")
        no_row = sub[sub["label"] == "Without Sentiment"]
        with_row = sub[sub["label"] == "With Sentiment"]
        if no_row.empty or with_row.empty:
            continue
        no_r, with_r = no_row.iloc[0], with_row.iloc[0]

        cols = st.columns(4)
        for col, metric, label in zip(cols,
            ["sharpe", "dir_acc", "rmse", "ic"],
            ["Sharpe", "Dir Acc", "RMSE", "IC"]):
            delta = with_r.get(metric, 0) - no_r.get(metric, 0)
            better = (delta > 0 and metric != "rmse") or (delta < 0 and metric == "rmse")
            col.metric(
                label,
                f"{with_r.get(metric, '—'):.4f}",
                f"{delta:+.4f} vs no-sent",
                delta_color="normal" if better else "inverse",
            )
        st.markdown("---")

    abl_img = PROCESSED / "sentiment_ablation.png"
    if abl_img.exists():
        st.image(str(abl_img), use_container_width=True)

    # Predictive correlation
    if not corr_df.empty:
        st.subheader("Predictive Correlation: Sentiment → Forward Returns")
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        corr_img = PROCESSED / "sentiment_predictive_correlation.png"
        if corr_img.exists():
            st.image(str(corr_img), use_container_width=True)

    st.markdown("""
    **Key insight:** Sentiment helps for **1-day** returns (positive Δ Sharpe, IC, dir acc)
    but hurts 5-day returns — likely because sentiment reverts quickly and becomes noise over longer horizons.
    Coverage is sparse (1,754 days vs 6,538 total), which limits signal strength.
    """)

# ── PAGE: Backtest ───────────────────────────────────────────────────────────
elif page == "Backtest":
    st.title("Regime-Aware Portfolio Backtest")
    st.caption("Long/short signal strategy with regime-scaled position sizing vs buy & hold")

    bt_path = PROCESSED / "backtest_results.csv"
    if not bt_path.exists():
        st.warning("Run scripts/06_backtest.py first.")
        st.stop()

    bt_df = pd.read_csv(bt_path)

    # Summary metrics
    mode_filter = st.selectbox("Strategy", ["All", "Plain (no regime sizing)", "Regime-Sized"])
    show_df = bt_df.copy()
    if mode_filter == "Plain (no regime sizing)":
        show_df = show_df[show_df["model"].str.endswith("__plain")]
    elif mode_filter == "Regime-Sized":
        show_df = show_df[show_df["model"].str.endswith("__regime_sized")]

    target_filter = st.selectbox("Target", ["All", "1d", "5d"])
    if target_filter != "All":
        show_df = show_df[show_df["model"].str.contains(target_filter)]

    display_cols = ["model", "sharpe", "sortino", "max_dd", "calmar", "ann_return", "ic", "hit_rate", "turnover"]
    st.dataframe(
        show_df[display_cols].rename(columns={
            "sharpe": "Sharpe", "sortino": "Sortino", "max_dd": "Max DD",
            "calmar": "Calmar", "ann_return": "Ann Return", "ic": "IC",
            "hit_rate": "Hit Rate", "turnover": "Turnover"
        }),
        use_container_width=True, hide_index=True
    )

    # Regime breakdown heatmap
    heatmap_path = PROCESSED / "backtest_regime_heatmap.png"
    if heatmap_path.exists():
        st.subheader("Sharpe by Model × Regime")
        st.image(str(heatmap_path), use_container_width=True)

    # Cumulative return plots
    st.subheader("Cumulative Return Chart")
    cum_pngs = sorted(PROCESSED.glob("backtest_cumret_*.png"))
    if cum_pngs:
        model_names = [p.stem.replace("backtest_cumret_", "") for p in cum_pngs]
        sel = st.selectbox("Model", model_names)
        img_path = PROCESSED / f"backtest_cumret_{sel}.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)

    # Regime breakdown table
    bd_path = PROCESSED / "backtest_regime_breakdown.csv"
    if bd_path.exists():
        st.subheader("Per-Regime Breakdown")
        bd_df = pd.read_csv(bd_path)
        sel_model = st.selectbox("Model for breakdown", sorted(bd_df["model"].unique()))
        sub = bd_df[bd_df["model"] == sel_model][["regime", "sharpe", "sortino", "max_dd", "ic", "hit_rate", "n"]]
        st.dataframe(sub, use_container_width=True, hide_index=True)

# ── PAGE: Live Trading ────────────────────────────────────────────────────────
elif page == "Live Trading":
    st.title("Live Paper Trading — Alpaca")
    st.caption("Regime-conditioned SPY strategy, updated daily after market close.")

    TRADE_LOG = PROCESSED / "trade_log.csv"

    # ── Setup instructions (shown when log is empty) ──────────────────────────
    if not TRADE_LOG.exists():
        st.info(
            "No trade log yet. To start live paper trading:\n\n"
            "1. Create a free account at [alpaca.markets](https://alpaca.markets) "
            "and copy your **paper trading** API key + secret.\n"
            "2. Add to `.env`:\n"
            "```\nALPACA_API_KEY=your_key\nALPACA_SECRET_KEY=your_secret\n```\n"
            "3. Install Alpaca SDK: `pip install alpaca-py`\n"
            "4. Save the final models: `python scripts/11_save_models.py`\n"
            "5. Run the daily script: `python scripts/12_live_trading.py`\n\n"
            "Then schedule it with cron to run at 4:30 PM ET on weekdays."
        )

        # Still show historical signal simulation
        st.subheader("Historical Signal Simulation (Walk-Forward)")
        if all_preds:
            model_key = st.selectbox("Model", sorted(all_preds.keys()))
            preds = all_preds[model_key]
            target = "fwd_return_1d" if "1d" in model_key else "fwd_return_5d"
            actuals_path = PROCESSED / "actual_returns.parquet"
            actual = pd.read_parquet(actuals_path)[target] if actuals_path.exists() else features_df[target]
            common = preds.index.intersection(actual.index.dropna())
            preds_a, actual_a = preds.loc[common], actual.loc[common].dropna()
            c2 = preds_a.index.intersection(actual_a.index)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            ax = axes[0]
            ax.scatter(actual_a.loc[c2], preds_a.loc[c2], alpha=0.2, s=6, color="steelblue")
            lim = max(abs(actual_a.loc[c2]).quantile(0.99), abs(preds_a.loc[c2]).quantile(0.99))
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
            ax.plot([-lim, lim], [-lim, lim], "r--", lw=0.8)
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Predicted vs Actual")

            ax2 = axes[1]
            strat = (np.sign(preds_a.loc[c2]) * actual_a.loc[c2]).cumsum()
            bh = actual_a.loc[c2].cumsum()
            ax2.plot(strat.index, strat.values, color="royalblue", label="Signal Strategy")
            ax2.plot(bh.index, bh.values, color="gray", lw=0.8, ls="--", label="Buy & Hold")
            ax2.set_ylabel("Cumulative Log Return"); ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
            ax2.set_title("Signal Strategy vs Buy & Hold")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.stop()

    # ── Live trade log exists — show dashboard ────────────────────────────────
    log_df = pd.read_csv(TRADE_LOG, parse_dates=["date"], engine="python", on_bad_lines="skip")
    log_df = log_df.sort_values("date")
    latest = log_df.iloc[-1]

    # ── 0. RUN-NOW BUTTON ────────────────────────────────────────────────────
    ctl1, ctl2 = st.columns([1, 3])
    with ctl1:
        run_clicked = st.button("Run Prediction Now", type="primary")
    with ctl2:
        submit_order = st.checkbox("Also submit Alpaca order", value=True,
                                   help="Uncheck to just compute signal without trading")

    if run_clicked:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "live_trading", str(Path(__file__).parent / "scripts" / "12_live_trading.py")
        )
        live_mod = importlib.util.module_from_spec(spec)
        with st.spinner("Fetching data, running 12-model ensemble, placing order..."):
            try:
                spec.loader.exec_module(live_mod)
                result = live_mod.run_pipeline(
                    force=True, submit_alpaca=submit_order, verbose=False
                )
                if result["status"] == "ok":
                    st.success(f"Signal: {result['message']}")
                else:
                    st.warning(result.get("message", "Pipeline returned no action"))
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
        st.rerun()

    # ── 1. PREDICTED STATUS (ensemble consensus) ──────────────────────────────
    st.subheader("Predicted Status — Ensemble Model Consensus")
    signal_emoji = {"buy": "BUY SPY", "sell": "BUY SH (inverse)", "flat": "FLAT (no trade)"}
    signal_label = signal_emoji.get(latest["signal"], latest["signal"].upper())

    import json
    consensus_path = PROCESSED / "latest_consensus.json"
    consensus = None
    if consensus_path.exists():
        try:
            consensus = json.loads(consensus_path.read_text())
        except Exception:
            consensus = None

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Signal Date", pd.to_datetime(latest["date"]).strftime("%Y-%m-%d"))
    p2.metric("Detected Regime", latest["regime"])
    p3.metric("Ensemble Prediction", f"{latest['pred_return']:+.4%}")
    p4.metric("Final Signal", signal_label)

    if consensus:
        agree = consensus["agreement"]
        n_act = consensus["n_active"]
        n_all = len(consensus["votes"])
        st.caption(
            f"**{n_act} of {n_all} models** active (positive-Sharpe gate). "
            f"**Directional agreement: {agree:.0%}** — "
            + ("trade gate PASSED ≥60%" if agree >= 0.6
               else "trade gate BLOCKED <60% → FLAT")
        )

        votes_df = pd.DataFrame(consensus["votes"])
        votes_df["included"] = votes_df["sharpe"].apply(lambda s: "Yes" if s > 0 else "Excluded")
        votes_df["weight"]   = votes_df["weight"].apply(lambda w: f"{w*100:.0f}%")
        votes_df["sharpe"]   = votes_df["sharpe"].apply(lambda s: f"{s:+.3f}")
        votes_df["pred"]     = votes_df["pred"].apply(lambda p: f"{p:+.5f}")
        votes_df = votes_df.rename(columns={
            "model_key": "Model", "sharpe": "Backtest Sharpe",
            "pred": "Prediction", "vote": "Vote",
            "weight": "Weight", "included": "In Ensemble"
        })[["Model", "Backtest Sharpe", "Prediction", "Vote", "Weight", "In Ensemble"]]
        st.dataframe(votes_df, use_container_width=True, hide_index=True)
    else:
        st.caption(
            f"Model expects S&P 500 to move **{latest['pred_return']:+.2%}** on next open. "
            f"In {latest['regime']} regime, notional scaled to ${latest['notional']:,.0f}."
        )

    st.markdown("---")

    # ── 2. CURRENT STATUS (what Alpaca actually shows) ────────────────────────
    st.subheader("Current Status — Alpaca Paper Account")
    try:
        from src.trading.alpaca_client import (
            get_account, get_position, get_portfolio_history, list_recent_orders
        )
        acct = get_account()
        pos_spy = get_position("SPY")
        pos_sh  = get_position("SH")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Equity", f"${acct['equity']:,.2f}",
                    delta=f"${acct['pnl']:+,.2f} today")
        col2.metric("Cash", f"${acct['cash']:,.2f}")

        # Show whichever side is active
        if pos_spy:
            col3.metric("SPY (long)", f"{pos_spy['qty']:.2f} sh",
                        delta=f"${pos_spy['unrealized_pl']:+,.2f}")
        else:
            col3.metric("SPY (long)", "Flat")

        if pos_sh:
            col4.metric("SH (short-proxy)", f"{pos_sh['qty']:.2f} sh",
                        delta=f"${pos_sh['unrealized_pl']:+,.2f}")
        else:
            col4.metric("SH (short-proxy)", "Flat")

        # Recent orders table — shows fill status
        st.markdown("**Recent Orders**")
        try:
            orders = list_recent_orders(limit=10)
            if len(orders):
                st.dataframe(orders, use_container_width=True, hide_index=True)
            else:
                st.caption("No orders yet.")
        except Exception as e:
            st.caption(f"Could not fetch orders: {e}")

        # Portfolio equity curve
        try:
            hist = get_portfolio_history(period="3M")
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(hist.index, hist["equity"], color="royalblue", lw=1.2)
            ax.set_ylabel("Portfolio Equity ($)")
            ax.set_title("Alpaca Paper Portfolio — 3 Month")
            ax.grid(alpha=0.2)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception:
            pass

    except Exception as e:
        st.info(f"Alpaca not connected ({e}). Showing signal log only.")

    st.markdown("---")

    # ── 3. STRATEGY PERFORMANCE SUMMARY ───────────────────────────────────────
    log_df["pnl_log"] = np.where(log_df["signal"]=="buy", log_df["sp500_return"],
                         np.where(log_df["signal"]=="sell", -log_df["sp500_return"], 0.0))
    strat_cum = (1 + log_df["pnl_log"]).prod() - 1
    bh_cum    = (1 + log_df["sp500_return"]).prod() - 1
    win_rate  = (log_df["pnl_log"] > 0).mean()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Days Logged", str(len(log_df)))
    s2.metric("Strategy Return", f"{strat_cum:+.2%}")
    s3.metric("Buy-and-Hold", f"{bh_cum:+.2%}")
    s4.metric("Win Rate", f"{win_rate:.0%}")

    # ── Signal history chart ──────────────────────────────────────────────────
    st.subheader("Daily Signal History")
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax = axes[0]
    colors = {"buy": "#2ecc71", "sell": "#e74c3c", "flat": "#95a5a6"}
    for _, row in log_df.iterrows():
        ax.axvline(row["date"], color=colors.get(row["signal"], "gray"), alpha=0.5, lw=1.5)
    ax.plot(log_df["date"], log_df["pred_return"], color="steelblue", lw=1.2, marker="o", ms=4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Predicted Return")
    ax.set_title("Daily Model Signal (green=buy, red=sell, gray=flat)")
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    regime_c = {"Bull": "#2ecc71", "Normal": "#3498db", "Bear/Crisis": "#e74c3c", "Extreme": "#9b59b6"}
    for _, row in log_df.iterrows():
        ax2.bar(row["date"], 1, color=regime_c.get(row["regime"], "gray"), alpha=0.7, width=1)
    ax2.set_ylabel("Regime"); ax2.set_yticks([])
    ax2.set_title("Detected Regime per Day")

    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Trade log table ───────────────────────────────────────────────────────
    st.subheader("Trade Log")
    display_cols = ["date", "regime", "pred_return", "signal", "notional",
                    "order_status", "vix", "sp500_return"]
    show_cols = [c for c in display_cols if c in log_df.columns]
    st.dataframe(
        log_df[show_cols].sort_values("date", ascending=False).head(30),
        use_container_width=True, hide_index=True,
    )
