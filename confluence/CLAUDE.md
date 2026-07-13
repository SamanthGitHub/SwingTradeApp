# CLAUDE.md — context for AI sessions (read this first, don't re-read everything)

SwingTrade Pro: a Streamlit swing-trading dashboard. **Three-layer layout** (post-split, July 2026):
- **`app.py`** — entry point: theme, nav, routing only (~1.8k lines incl. the 21 not-yet-migrated
  page branches). `run_dashboard()` tries `screens.PAGES` first, then falls back to its `elif` chain.
- **`services.py`** — the shared data/scan/chart layer (~2.6k lines): cached singletons, provider-aware
  fetchers, bulk prefetch, scan engines, chart builders, analyst-brief glue. Exposes everything
  (incl. `_underscore` helpers) via `__all__` for the `from services import *` star-import that
  `app.py` and `screens/*` use. **Never imports `app` or `screens`.**
- **`screens/`** — page renderers extracted one at a time: Screener, Morning Insights, YouTube,
  Signal Stack, Alpha Engine, Settings. Each is `def render(ctx)`; `Ctx` = (config, account_size,
  watchlist_mgr). **Deliberately not named `pages/`** (that would trigger Streamlit's native router).
- **`swingtradeapp/`** package: ~40 pure modules (below). **`tests/`**: offline pytest suite (71 tests).

## Run / verify (ENVIRONMENT GOTCHAS — important)
- **Launch:** `./run.sh` (or double-click `run.command`). Builds an isolated venv at
  `~/.swingtradeapp/venv` — deliberately **OUTSIDE** this Google Drive folder (avoids
  sync churn / file locks). Opens http://localhost:8501.
- **Always verify code with the venv Python**, not system Python:
  `~/.swingtradeapp/venv/bin/python`
- **Tests:** `~/.swingtradeapp/venv/bin/python -m pytest -q` — fully offline (synthetic OHLCV),
  must stay green. Dev deps: `requirements-dev.txt`.
- **Full-page smoke:** `streamlit.testing.v1.AppTest.from_file("app.py")` +
  `at.query_params["page"] = <nav label>` renders any page headlessly (see git history for a
  27-page loop script).
- **No matplotlib installed** → never use pandas `Styler.background_gradient`/`.bar`; use
  `.map(fn, subset=[...])` colorizers instead.
- Newer libs in venv: **pandas 3.0, numpy 2.4, streamlit 1.58**.
- **Every `st.plotly_chart` that can appear twice in one run needs a unique `key=`**
  (else `StreamlitDuplicateElementId`). Pattern: `key=f"{key_prefix}_..._{symbol}"`.

## Architecture notes
- **Routing contract:** `ui.NAV_ITEMS` labels must stay byte-identical to `screens.PAGES` keys /
  `page == "…"` branches — **enforced by `tests/test_nav_routing.py`**.
- **Speed path:** scans call `prefetch_histories(symbols, days)` (chunked multi-ticker
  `yf.download`, main thread, progress bar) → `swingtradeapp/pricestore.py` (thread-safe
  longest-span store, 1h TTL) → `fetch_symbol_history` becomes a memory hit. The Screener is
  two-pass: pass 1 CPU-only signal gate, pass 2 `ThreadPoolExecutor(8)` info/fundamentals/news
  for survivors only (workers never touch `st.*` or the ApiBudget).
- **Paid-tier guarantee:** Polygon calls go through `ApiBudget.try_acquire` (one locked
  check-and-record — no TOCTOU) in `providers.PolygonProvider._reserve`; bulk scans are
  Yahoo-only by design. Never let anything concurrent call Polygon.
- **Reliability:** all JSON stores write via `swingtradeapp/jsonstore.py`
  (`atomic_write_json` = same-dir temp + `os.replace`; `read_json` recovers corrupt files).
  Swallowed errors are recorded in `swingtradeapp/errlog.py` and surfaced in the data-status
  strip + Settings → **Data health** — use `errlog.record(...)`/`errlog.soft(...)` instead of
  bare `except Exception: pass`.
- **Clock:** `swingtradeapp/clock.py` `now_et()`/`market_phase()` is the canonical ET time —
  never `datetime.now()` for market logic or fetch windows.
- **Indicators (fixed July 2026):** `signals.py` has real Wilder smoothing (`wilder_smooth`,
  `compute_rsi_series`, proper ADX, real StochRSI %K/%D). `ml_signal.FEATURE_VERSION` gates
  saved models — bump it whenever feature semantics change (forces a clean retrain).
- **Analyst briefs (free, local):** `swingtradeapp/analyst.py` composes template-NLG theses from
  the app's own structured outputs (can't hallucinate numbers — tested); optional Ollama polish
  via `swingtradeapp/llm_local.py` (urllib-only, silent fallback). Surfaced on Screener
  drill-down, Signal Stack, Morning Insights; toggles in Settings.
- **Shared numerics:** `swingtradeapp/num.py` (`clip01`, `ema`) — don't re-add local copies.
- **swingtradeapp/ modules** (all pure/Streamlit-free unless noted): `config`, `tickers`
  (live Nasdaq-Trader universe; `MAJOR_US_STOCKS` is offline fallback, audited 7/2026),
  `universe`, `providers`, `ratelimit`, `retry`, `jsonstore`, `errlog`, `clock`, `num`,
  `pricestore`, `datalake`, `signals`, `risk` (`BayesianKellySizer`, OOS-calibrated),
  `backtest`, `patterns`, `setups`, `momentum_radar`, `whale`, `regime`, `market_structure`,
  `multi_timeframe`, `confluence`, `ml_signal`, `alpha_engine`, `alpha_factors`, `alpha_ml`,
  `alpha_validation`, `nlp`, `forecast`, `options_analysis`, `fundamentals`, `insiders`,
  `ipo_premarket` (curated `RECENT_IPOS` — refresh periodically), `etf_screener`,
  `macro_filters` (2026 FOMC + programmatic Jobs/CPI), `execution`, `watchlist`,
  `analysis_guide`, `analyst`, `llm_local`, `youtube`, `ui` (presentation; owns NAV_GROUPS).

## Key conventions
- **Optional AI (Hugging Face)**: lazy-load, **graceful heuristic fallback**, gated by
  Settings → AI Features toggles (`_ai_on(key)`); models in `requirements-ai.txt`.
- **News**: `fetch_news_items` (Yahoo + Google News RSS, deduped) for on-demand views;
  `_fetch_headlines` (Yahoo-only, fast) for bulk scans. Google News never in the bulk scan.
- **Recommendations**: `recommend_label(score, trend)` → Buy ≥0.65 / Watch ≥0.45 / Weak / Avoid.
- **Percentages: 2 decimals everywhere** (`{:.2f}%` / `{:.2%}`).
- **UI/theme**: `swingtradeapp/ui.py` owns all styling; mode-agnostic (auto light/dark) — never
  hardcode a background color.
- **Backtest realism (done)**: costs (`slippage_bps`/`commission_bps`), out-of-sample
  walk-forward; Kelly priors calibrated once OOS in `calibrate_kelly_priors`.
  Don't reintroduce in-sample calibration.
- `.data/` holds caches/journal/portfolio_state — gitignored, regenerated at runtime.

## Data sources
yfinance (quotes/history/info/news/screeners), Nasdaq Trader symbol directory (universe),
Google News RSS (broad free news), Polygon/Massive **free tier only** (5/min, budget-enforced,
single-symbol drill-downs), Alpaca (execution; creds in `.env`, offline if blank). Yahoo calls
wrapped with `@with_retry`; Polygon deliberately NOT retried (retries would fire uncounted calls).

## Git
Private repo **github.com/SamanthGitHub/SwingTradeApp**. **Commit/push only when the user asks.**
`.env`, `.data/`, `.venv/`, `__pycache__/`, `.claude/` are gitignored. `.env.example` is the template.

## Status / next ideas (not done)
Migrate the remaining 21 page branches into `screens/`; survivorship-bias-free backtests
(needs point-in-time data); live automation loop + Alpaca position reconciliation; database
(currently JSON files, now atomic); broaden Market Events news to Google News too.
