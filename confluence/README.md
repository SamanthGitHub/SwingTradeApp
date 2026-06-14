# SwingTrade Pro

A free, self-contained swing-trading dashboard. It scans a broad universe of US stocks using
free data (Yahoo Finance), generates trend signals, sizes positions, and shows it all in a clean
browser dashboard.

**Free for everyone — no accounts, no API keys, no paid services.** It runs entirely on your own
computer.

## Run it (the only setup is Python)

The launcher builds its own isolated environment the first time and opens the app in your browser
at **http://localhost:8501**.

**1. Install Python** *(one-time, skip if you already have it)*
Download **Python 3.12** from <https://www.python.org/downloads/> and install it.
On **Windows**, tick **"Add Python to PATH"** during install.
*(Avoid 3.14 for now — some packages don't publish builds for it yet.)*

**2. Get the project**
On GitHub click **Code → Download ZIP** and unzip it (or `git clone` the repo).

**3. Start it**

| Your computer | How to start |
| --- | --- |
| **Windows** | Double-click **`run.bat`** |
| **macOS** | Double-click **`run.command`** (or run `./run.sh` in Terminal) |
| **Linux** | Run `./run.sh` in a terminal |

The first launch takes a minute or two to install dependencies; after that it starts instantly.
To stop the app, close the window or press **Ctrl+C**.

That's the whole thing. Everything below is optional.

---

## Optional extras

### AI features (forecasting, smarter sentiment, summaries)
These are **off by default** and the app falls back to fast built-in heuristics, so you never need
them. To enable the real models:

```bash
# macOS / Linux
~/.swingtradeapp/venv/bin/pip install -r requirements-ai.txt
```
```bat
REM Windows
"%USERPROFILE%\.swingtradeapp\venv\Scripts\pip" install -r requirements-ai.txt
```
Then turn them on under **Settings → AI Features** in the app.

### Live order placement (Alpaca)
The dashboard is fully usable without this — it just won't place real orders. To enable it, copy
`.env.example` to `.env` and add your (paper) Alpaca keys. Left blank, the app stays in
read-only / offline mode.

### Run it in your own Python environment
If you'd rather manage the environment yourself:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Troubleshooting

- **macOS — "run.command can't be opened" (unidentified developer):** right-click the file →
  **Open** → **Open**. You only need to confirm once.
- **macOS — double-click does nothing after a ZIP download:** the executable flag can be lost when
  unzipping. In Terminal, `cd` into the folder and run `chmod +x run.sh run.command` once.
- **Port 8501 already in use:** stop whatever is using it, or run
  `streamlit run app.py --server.port 8502`.
- **`yfinance` 401/429 errors:** usually transient throttling — the app retries automatically. If
  it persists, wait a minute and re-run, or clear the cache at `~/Library/Caches/py-yfinance`.

---

## How it works (short version)

- **Entry point:** `app.py` (a Streamlit app). Core logic lives in the `swingtradeapp/` package.
- **Data (all free, no keys):** Yahoo Finance via `yfinance` (quotes, history, screeners), the
  Nasdaq Trader symbol directory (universe), and Google News RSS (headlines).
- **What it does:** screens stocks, builds trend signals (RSI / moving-average / momentum), sizes
  positions (Bayesian Kelly), forecasts the next session, and surfaces pre-/after-market movers,
  options flow, ETFs, market events, watchlists, P&L and alerts.

Want to host it online for free instead of running locally? See **[DEPLOY.md](DEPLOY.md)**.
