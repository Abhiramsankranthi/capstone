# Multi-Modal Financial Market Regime Detection & Asset Return Forecasting

A machine learning framework that detects latent market regimes (bull, bear, high-volatility) using Hidden Markov Models and forecasts short-term asset returns using regime-conditioned models. Combines price data, macroeconomic indicators, VIX volatility, and FinBERT news sentiment.

**Course**: CFSE 570 Data Science Capstone, Arizona State University

**Team**: Naveen Sathyanarayanan, Shravan Swaminathan, Renu Jakkampudi, Abhiram Sankranthi, Aditya Lokesh

---

## Project Overview

### Problem
Financial markets cycle through distinct regimes (bull markets, bear markets, high-volatility periods) that fundamentally change how assets behave. Traditional forecasting models ignore these regime shifts, leading to poor performance during market transitions.

### Approach
1. **Regime Detection**: Fit a Gaussian HMM on S&P 500 returns and realized volatility to identify latent market states
2. **Multi-Modal Features**: Combine technical indicators, macroeconomic data, volatility metrics, and NLP-derived news sentiment
3. **Regime-Conditioned Forecasting**: Train separate forecasting models per regime and compare against regime-agnostic baselines
4. **Validation**: Expanding-window walk-forward backtesting with no data leakage

### Results (Preliminary)
The HMM identifies 4 market regimes with clear economic interpretation:

| Regime | % of Time | Mean Daily Return | Mean VIX |
|--------|-----------|-------------------|----------|
| Bull | 40% | +0.056% | 14.3 |
| Normal | 28% | +0.022% | 18.4 |
| Bear/Crisis | 20% | -0.031% | 23.9 |
| Extreme | 12% | +0.019% | 35.4 |

Crisis detection accuracy:
- 2008 GFC: 100% classified as Bear/Extreme
- 2020 COVID: 87% classified as Bear/Extreme
- 2022 Rate Hikes: 91% classified as Bear/Extreme

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| [yfinance](https://pypi.org/project/yfinance/) | S&P 500 + 11 sector ETFs (adjusted close) | 2000-present |
| [FRED API](https://fred.stlouisfed.org/docs/api/) | CPI, unemployment, fed funds rate, industrial production, yield curve | 2000-present |
| [yfinance](https://pypi.org/project/yfinance/) | VIX and VIX3M volatility indices | 2000-present |
| [Kaggle News Dataset](https://www.kaggle.com/datasets/irakozekelly/financial-news-sentiment-dataset-20122022) | HuffPost articles filtered to BUSINESS & MONEY categories | 2012-2022 |
| [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | Sentiment scoring of news headlines | - |

## Features (37 total)

- **Technical** (8): log returns, 5/21-day realized volatility, RSI, MACD (line/signal/histogram), Bollinger Band width
- **Sector Returns** (11): log returns for XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC
- **Macro** (10): CPI, unemployment, fed funds rate, industrial production, yield curve spread (levels + first differences)
- **Volatility** (3): VIX level, 252-day percentile rank, VIX/VIX3M term structure ratio
- **Sentiment** (3): daily FinBERT mean score, max negative score, article count
- **Targets** (2): forward 1-day and 5-day log returns

## Setup

### Prerequisites
- Python 3.10+
- A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)
- The [Kaggle news dataset](https://www.kaggle.com/datasets/irakozekelly/financial-news-sentiment-dataset-20122022) downloaded as `data_YYYY.json` files in `data/raw/`

### Installation

```bash
# Clone the repo
git clone https://github.com/Abhiramsankranthi/capstone.git
cd capstone

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Set up FRED API key
echo "FRED_API_KEY=your_key_here" > .env
```

### Running the Pipeline

```bash
# Step 1: Fetch all data (equity, macro, volatility, sentiment)
# Note: FinBERT inference takes ~6 minutes on CPU
python3 scripts/01_fetch_data.py

# Step 2: Build features and merge into unified dataset
python3 scripts/02_build_features.py

# Step 3: Fit HMM and validate regimes
python3 scripts/03_fit_hmm.py
```

### Output Files

| File | Description |
|------|-------------|
| `data/interim/equity_prices.parquet` | Adjusted close prices for all tickers |
| `data/interim/macro_indicators.parquet` | Daily forward-filled FRED series |
| `data/interim/volatility.parquet` | VIX and VIX3M daily values |
| `data/interim/sentiment_daily.parquet` | Aggregated daily FinBERT sentiment |
| `data/processed/features.parquet` | Final merged dataset (6,538 days x 37 features) |
| `data/processed/regime_labels.parquet` | HMM regime labels per trading day |
| `data/processed/regime_chart.png` | S&P 500 price chart shaded by regime |

## Project Structure

```
capstone/
├── config.yaml              # All tickers, FRED series, feature params, model config
├── .env                     # FRED API key (gitignored)
├── requirements.txt
├── src/
│   ├── config.py            # Config and env loading
│   ├── data/
│   │   ├── equity.py        # yfinance equity price download
│   │   ├── macro.py         # FRED API macro indicators
│   │   ├── volatility.py    # VIX/VIX3M download
│   │   └── sentiment.py     # HuffPost filtering + FinBERT inference
│   ├── features/
│   │   ├── technical.py     # Log returns, RSI, MACD, Bollinger Bands
│   │   ├── macro_features.py# Levels + first differences
│   │   ├── vol_features.py  # VIX percentile rank, term structure ratio
│   │   └── merge.py         # Join all sources + add forward return targets
│   ├── models/
│   │   └── hmm.py           # Gaussian HMM fitting, BIC selection, regime labeling
│   └── validation/
│       └── regime_validation.py  # Crisis period checks + regime visualization
├── scripts/
│   ├── 01_fetch_data.py     # Run all data downloads
│   ├── 02_build_features.py # Run feature engineering + merge
│   └── 03_fit_hmm.py        # Run HMM fitting + validation
├── data/
│   ├── raw/                 # Kaggle news JSONs (gitignored)
│   ├── interim/             # Per-source parquet files (gitignored)
│   └── processed/           # Final merged dataset (gitignored)
└── notebooks/               # EDA and analysis notebooks
```

## Models

### Current: HMM Regime Detection
- Gaussian HMM (hmmlearn) with full covariance
- Input: S&P 500 log returns + 21-day realized volatility
- Model selection: 2/3/4 states compared via BIC (4 states selected)
- 10 random restarts per state count to avoid local optima
- Regimes labeled by volatility level (low vol = Bull, high vol = Bear/Crisis)

### Planned: Forecasting Models
- **Baselines**: Ridge regression, Lasso regression
- **Tree-based**: LightGBM
- **Deep learning**: LSTM, Temporal Fusion Transformer
- Each trained in regime-agnostic and regime-conditioned variants
- Expanding-window walk-forward backtesting
- Evaluation: directional accuracy, Sharpe ratio, max drawdown, RMSE, information coefficient
- Interpretability via SHAP analysis

## Timeline

| Phase | Dates | Deliverable |
|-------|-------|-------------|
| Data & Feature Engineering | Feb 12 - Mar 4 | Integrated dataset + EDA |
| **Status Update 1** | **Mar 5** | **HMM regimes validated** |
| Model Development | Mar 6 - Apr 1 | Walk-forward backtesting results |
| Status Update 2 | Apr 2 | Model comparison |
| Final Evaluation | Apr 3 - Apr 29 | SHAP analysis, portfolio backtesting |
| Final Presentation | Apr 30 | Report + presentation |
