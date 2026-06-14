# CLAUDE.md — context for AI sessions (read this first, don't re-read everything)

SwingTrade Pro: a Streamlit swing-trading dashboard. **Single entry point: `app.py`**
(~1900 lines). Core logic in the `swingtradeapp/` package (21 modules). No tests yet.

## Run / verify (ENVIRONMENT GOTCHAS — important)
- **Launch:** `./run.sh` (or double-click `run.command`). Builds an isolated venv at
  `~/.swingtradeapp/venv` — deliberately **OUTSIDE** this Google Drive folder (avoids
  sync churn / file locks). Opens http://localhost:8501.
- **Always verify code with the venv Python**, not system Python:
  `~/.swingtradeapp/venv/bin/python`
- Multiple system Pythons exist: `/usr/local/bin/python3` → **3.14** (too new: missing
  wheels, and PEP-649 lazy annotations *hide* bugs), anaconda **3.13**. Launcher prefers 3.13.
- **No matplotlib installed** → never use pandas `Styler.background_gradient`/`.bar`; use
  `.map(fn, subset=[...])` colorizers instead.
- Newer libs in venv: **pandas 3.0, numpy 2.4, streamlit 1.58**. On 3.13 annotations are
  eager — bare `typing.Optional` (no arg) raises at import.
- **Every `st.plotly_chart` that can appear twice in one run needs a unique `key=`**
  (else `StreamlitDuplicateElementId`). Pattern: `key=f"{key_prefix}_..._{symbol}"`.
- Quick checks:
  - `~/.swingtradeapp/venv/bin/python -c "import ast; ast.parse(open('app.py').read())"`
  - import all modules: `... -c "import importlib,pkgutil,swingtradeapp; [importlib.import_module(f'swingtradeapp.{m.name}') for m in pkgutil.iter_modules(swingtradeapp.__path__)]"`

## Architecture
- **app.py** `run_dashboard()` — sidebar nav pages: Screener · Pre-Market Movers · Live Movers ·
  After-Hours & IPOs · Whale Movements · Options Flow · Predictions · Auto Watchlist · ETF
  Screener · Market Events · Heat Map · Watchlists · Compare · P&L Tracker · Alerts · Settings.
  Cached singletons via `@st.cache_resource get_*`.
  (Live Movers = raw Yahoo predefined-screener passthrough via `get_raw_screen`/`fetch_raw_movers`
  + `RAW_SCREENS` map — deliberately NO signals/filters, just "who's moving". Predictions =
  next-session forecast via `predict_tomorrow` (PriceForecaster horizon=1, Chronos→heuristic
  fallback) + a 5-session drill-down cone. After-Hours & IPOs = post-market movers via
  `fetch_afterhours`→`PreMarketScanner.fetch_afterhours_movers` (Yahoo `postMarket*` fields; empty
  outside ~4–8pm ET) + `ipo_table` over the curated `IPOTracker.RECENT_IPOS`. Options Flow =
  single-ticker `analyze_options` + small cached `scan_options_flow`, both off `OptionsAnalyzer`.)
- **swingtradeapp/**: `config`, `tickers` (live Nasdaq-Trader universe + yf screen actives),
  `universe`, `providers` (yfinance only), `signals` (`TrendSignalGenerator.build_signal`,
  has `min_score` param), `risk` (`BayesianKellySizer`, OOS-calibrated, shrinkage),
  `backtest` (`VectorBacktestEngine`: cost model + `run_walk_forward` OOS), `execution`
  (Alpaca bracket), `fundamentals`, `macro_filters` (VIX/breadth/calendar; 2026 FOMC +
  programmatic Jobs/CPI), `watchlist`, `nlp` (sentiment + events + summary + novelty +
  news fetch), `forecast` (Chronos + heuristic MC fallback), `etf_screener`,
  `options_analysis` (IV rank/current-IV, put/call, unusual volume, Greeks, IV-crush — surfaced
  on Options Flow), `multi_timeframe`, `market_structure`,
  `ipo_premarket` (`IPOTracker.RECENT_IPOS` is a **curated, manually-maintained** list — refresh
  periodically like the Fed calendar), `retry`,
  `whale` (`WhaleDetector`: infers large-money footprints from daily OHLCV — relative-volume
  surge + $ traded + closing strength → 0–100 whale score; pure/no-Streamlit, app.py owns the
  cached universe loop `scan_whale_activity`),
  `ui` (presentation: `inject_theme` global CSS, `render_header` branded band, `render_nav`
  icon sidebar menu). `run_dashboard` calls these right after `set_page_config`.

## Key conventions
- **Optional AI (Hugging Face)**: lazy-load, **graceful heuristic fallback**, gated by
  Settings → AI Features toggles in `st.session_state`: `ai_forecast/ai_events/ai_summary/
  ai_novelty` (helper `_ai_on(key)`). Models in `requirements-ai.txt` (not installed by
  default). Loaders: `get_forecaster/get_event_classifier/get_summarizer/get_novelty`.
- **News**: `fetch_news_items` (Yahoo Finance + **Google News RSS**, deduped) for
  on-demand single-ticker & market views; `_fetch_headlines` (Yahoo-only, fast) for the
  bulk 100-symbol scan so it stays fast. Cached: `get_ticker_news`, `get_market_news`.
  Google News only in on-demand paths — never the bulk scan.
- **Recommendations**: `recommend_label(score, trend)` → Buy ≥0.65 / Watch ≥0.45 / Weak /
  Avoid (bearish). Colors via `_reco_color`; forecast via `_forecast_color`. Glossary:
  `_render_legend()` (an expander shown on label-heavy pages).
- **Percentages: 2 decimals everywhere** (`{:.2f}%` / `{:.2%}`).
- **UI/theme**: `swingtradeapp/ui.py` owns all styling. Theme is **mode-agnostic** (auto
  light/dark) — `.streamlit/config.toml` sets only `primaryColor`/`font`/`baseRadius`; CSS uses
  theme vars (`var(--secondary-background-color)`) / `@media (prefers-color-scheme)`, never a
  hardcoded background. Nav = `streamlit-option-menu` (in `requirements.txt`) with graceful
  fallback to `st.radio`. **`ui.NAV_ITEMS` labels must stay byte-identical to the `page == "…"`
  branches** in `run_dashboard` — they're the routing keys.
- **Universe**: `get_tradable_universe()` (Nasdaq Trader dirs, cached `.data/universe.json`,
  24h) + `get_screening_universe()` (today's most-actives first via `yf.screen`).
- **Backtest realism (done)**: costs (`slippage_bps`/`commission_bps` in config),
  out-of-sample walk-forward; Kelly priors calibrated once OOS in `calibrate_kelly_priors`
  (NOT per-symbol in-sample). Don't reintroduce in-sample calibration.
- **Market mood**: `compute_market_mood` (news tone + VIX + breadth + SPY trend) →
  `create_mood_gauge` on Market Events page.
- `.data/` holds caches/journal/portfolio_state — gitignored, regenerated at runtime.

## Data sources
yfinance (quotes/history/info/news/screeners), Nasdaq Trader symbol directory (universe),
Google News RSS (broad free news), Alpaca (execution; creds in `.env`, offline if blank).
All network calls wrapped with `swingtradeapp/retry.py` `@with_retry` (yfinance 401/429 are
transient — clear cache `~/Library/Caches/py-yfinance` if persistent).

## Git
Private repo **github.com/SamanthGitHub/SwingTradeApp** (branch `main`). `gh` CLI installed
at `~/.swingtradeapp/tools/gh_*/bin/gh` (on PATH via `~/.zshrc`; token in macOS keyring).
**Commit/push only when the user asks.** `.env`, `.data/`, `.venv/`, `__pycache__/`,
`.claude/` are gitignored. `.env.example` is the template.

## Status / next ideas (not done)
Survivorship-bias-free backtests (needs point-in-time data); live automation loop +
Alpaca position reconciliation; pytest suite; database (currently JSON files); broaden
Market Events page news to Google News too (Screener already uses it).
