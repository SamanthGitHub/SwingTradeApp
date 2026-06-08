# SwingTrade Pro

## Quick Start (one click)

The app is **self-contained** — no manual Python setup needed.

- **macOS:** double-click **`run.command`** in Finder, or in a terminal run `./run.sh`

On first launch it automatically:
1. picks a compatible Python (prefers 3.11–3.13; avoids the too-new 3.14),
2. creates an isolated virtual environment at `~/.swingtradeapp/venv` (kept **outside**
   the Google Drive folder so it doesn't sync/lock),
3. installs the pinned dependencies from `requirements.txt`,
4. opens the dashboard at **http://localhost:8501**.

Subsequent launches are instant (deps reinstall only if `requirements.txt` changes).

### Optional AI features
Forecasting / upgraded sentiment / event tagging / summarization / novelty are off by
default and degrade to heuristics. To enable the real Hugging Face models:
```bash
source ~/.swingtradeapp/venv/bin/activate
pip install -r requirements-ai.txt
```
Then turn them on under **Settings → AI Features** in the app.

### Manual run (use your own environment)
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app entry point is **`app.py`**.

---

# SwingTradeApp — Project Summary

Short summary
- Purpose: lightweight, modular swing-trading prototype that scans a broad US-equity universe using free data (yfinance), generates trend signals (RSI / SMA / momentum), sizes positions conservatively (Bayesian Kelly), and exposes a Streamlit dashboard for review and inspection.
- Key design goals: run offline without API keys, scale beyond a few tickers (no penny stocks), and provide evidence for signals via news-headline sentiment when available.

What I added
- Core package: `swingtradeapp/` — modular modules: `config`, `providers`, `universe`, `tickers`, `signals`, `nlp`, `risk`.
- Data provider: `YFinanceProvider` (yfinance-based) and a `ProviderFactory` for easy extension to other data sources.
- Universe: expanded ~350-ticker universe and screening filters (no penny stocks, minimum liquidity).
- Signals: `TrendSignalGenerator` (RSI, SMA alignment, reasons), optional `VolumeAnomalyDetector` (uses scikit-learn when available).
- Risk sizing: `BayesianKellySizer` returning conservative fractions and dollar allocations.
- Dashboard: `dashboard.py` — Streamlit prototype showing scan results, charts, sector breakdowns, CSV export, and per-ticker inspection.
- News sentiment helper: `check_news_sentiment.py` — fetches headlines via yfinance and scores them with `FinBERTSentimentAnalyzer` (lazy-loads transformers; falls back to neutral stubs when not installed).

Important files
- `main.py` — CLI-style scanner orchestrator.
- `dashboard.py` — Streamlit UI. Run with Streamlit to open in browser.
- `check_news_sentiment.py` — script to attach headline sentiment to top signals.

Quick start
1. Create or review `.env` for any API keys (not required for yfinance-only mode).
2. Install minimal dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Run a scan (sample 50 symbols):

```bash
SAMPLE_SIZE=50 python3 main.py
```

4. Run the Streamlit dashboard (default port 8502 if 8501 is occupied):

```bash
python3 -m streamlit run dashboard.py --server.port 8502
```

5. Run the news sentiment helper for top signals:

```bash
python3 check_news_sentiment.py
```

Notes on NLP / FinBERT
- `swingtradeapp/nlp.py` contains `FinBERTSentimentAnalyzer` which lazy-loads the `transformers` pipeline using model `ProsusAI/finbert` when `transformers` is installed. Without `transformers` installed the analyzer returns neutral stub responses so the project remains runnable.
- To enable real FinBERT sentiment scoring (recommended for evidence):

```bash
python3 -m pip install "transformers[torch]"
```

Performance & caveats
- yfinance is used for historicals and (where available) headline lists via `Ticker.get_news()` — coverage varies by ticker and region.
- FinBERT and Reddit (`praw`) are optional; their installation can be heavy (PyTorch backend). The app handles their absence gracefully.
- Some optional dependencies may conflict with broker APIs (e.g., `websockets` version). If you add Alpaca or similar brokers, pin dependencies accordingly.

Next steps (planned)
- Integrate the aggregated headline-sentiment summary into the Streamlit `dashboard.py` signals table and ticker inspection panel (work in-progress).
- Add caching and asynchronous fetching to speed up scans and reduce UI latency.

If you'd like, I can proceed to integrate the `check_news_sentiment.py` output directly into the dashboard (add a sentiment column and per-ticker evidence view). Reply `Yes, integrate` and I will implement it next.

---
Generated: June 3, 2026
# SwingTradeApp

A scaffold for a multi-modal quantitative trading system based on the provided architectural specification.

## Features

- Multi-provider data ingestion wrappers for equities, options, and fundamentals
- Universe filtering for liquidity, market cap, and spread constraints
- Pre-market and gap scanning logic
- Volume anomaly detection using Isolation Forest
- Regime-aware signal generation with trend and momentum filters
- NLP sentiment analysis stubs for FinBERT and Reddit
- Bayesian Kelly position sizing
- Alpaca execution bridge with bracket order support
- Vectorized backtesting engine skeleton

## Getting Started

1. Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Edit `swingtradeapp/config.py` to add API keys.
3. Run the entry point:

```bash
python main.py
```

## Detailed Run & Troubleshooting

- Ensure you are using the same Python interpreter that has the project's dependencies installed. If you see "No module named streamlit" or similar, install into the active interpreter.

- To run a quick sample scan (50 symbols):

```bash
cd "$(dirname "${BASH_SOURCE[0]}")"
SAMPLE_SIZE=50 python3 main.py
```

- To run the Streamlit dashboard (tries port 8501, fall back to 8502/8503):

```bash
# primary
python3 -m streamlit run dashboard.py --server.port 8501

# if 8501 unavailable
python3 -m streamlit run dashboard.py --server.port 8502

# alternative
python3 -m streamlit run dashboard.py --server.port 8503
```

- If Streamlit reports a port is unavailable, choose a different `--server.port` value or stop the process currently using the port (use `lsof -i :8501` on macOS to find the PID).

- To enable real FinBERT headline scoring (may require PyTorch):

```bash
python3 -m pip install "transformers[torch]"
```

- Run the news sentiment helper (prints per-ticker evidence):

```bash
python3 check_news_sentiment.py
```

- If the scanner stalls or you see cURL/cffi callback interrupts, try re-running `main.py` without parallel network-heavy tasks, or reduce `SAMPLE_SIZE` to isolate problematic tickers.

cd /Users/samanth_m/Library/CloudStorage/GoogleDrive-samanth473@gmail.com/My\ Drive/StockAI/SwingTradeApp && source .venv/bin/activate && python -m streamlit run dashboard_finviz.py --server.port 8503