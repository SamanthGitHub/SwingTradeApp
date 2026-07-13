"""Shared data & logic layer for SwingTrade Pro.

Everything the page renderers consume lives here: cached singletons, provider-aware
fetchers, the bulk-prefetch fast path, scan engines, chart builders, analyst-brief glue
and small helpers. ``app.py`` (routing) and ``screens/*`` (page renderers) import this
module — never each other — so extraction of pages out of the old monolith stays acyclic.

Moved verbatim out of the pre-split ``app.py``; the trailing ``__all__`` keeps
``from services import *`` exporting the underscore-prefixed helpers the pages use.
"""


import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from swingtradeapp.backtest import VectorBacktestEngine
from swingtradeapp.config import TradingConfig
from swingtradeapp.etf_screener import ETFScreener
from swingtradeapp.execution import AlpacaExecutionBridge, BracketOrder
from swingtradeapp.fundamentals import FundamentalsExtractor
from swingtradeapp.ipo_premarket import IPOTracker
from swingtradeapp.macro_filters import MacroContext
from swingtradeapp.options_analysis import OptionsAnalyzer
from swingtradeapp.forecast import PriceForecaster, expected_return_pct, forecast_confirms
from swingtradeapp.nlp import (
    FinBERTSentimentAnalyzer,
    NewsEventClassifier,
    NewsNovelty,
    NewsSummarizer,
    TranscriptCleaner,
    aggregate_news_sentiment,
)
from swingtradeapp.providers import ProviderFactory, PolygonProvider
from swingtradeapp.ratelimit import ApiBudget, PROVIDERS
from swingtradeapp.retry import with_retry
from swingtradeapp.risk import BayesianKellySizer
from swingtradeapp.signals import TrendSignalGenerator
from swingtradeapp.tickers import get_raw_screen, get_screening_universe, get_tradable_universe
from swingtradeapp import ui
from swingtradeapp import youtube as yt
from swingtradeapp import alpha_engine, alpha_factors, alpha_ml, alpha_validation, datalake
from swingtradeapp import analyst as analyst_mod
from swingtradeapp import clock
from swingtradeapp import confluence as cf
from swingtradeapp import errlog
from swingtradeapp import jsonstore
from swingtradeapp import pricestore
from swingtradeapp.llm_local import OllamaClient
from swingtradeapp import patterns as patterns_mod, setups as setups_mod, regime as regime_mod
from swingtradeapp import analysis_guide as ag
from swingtradeapp import insiders as ins
from swingtradeapp.universe import PreMarketScanner, UniverseFilter
from swingtradeapp.watchlist import WatchlistManager
from swingtradeapp.whale import WhaleConfig, WhaleDetector
from swingtradeapp.momentum_radar import RallyConfig, RallyDetector
from swingtradeapp import ml_signal
from swingtradeapp.ml_signal import MLSignalModel

# ── Cached singletons ──────────────────────────────────────────────────────────

@st.cache_resource
def get_config():
    return TradingConfig.load_from_env()

@st.cache_resource
def get_signal_generator(config):
    return TrendSignalGenerator(config)

@st.cache_resource
def get_sizer(config):
    return BayesianKellySizer(config)

@st.cache_resource
def get_sentiment_analyzer(config):
    return FinBERTSentimentAnalyzer(config)

@st.cache_resource
def get_fundamentals():
    return FundamentalsExtractor()

@st.cache_resource
def get_watchlist_manager():
    return WatchlistManager()

@st.cache_resource
def get_macro_context():
    return MacroContext()

@st.cache_resource
def get_backtest_engine(config):
    return VectorBacktestEngine(config)


@st.cache_resource
def get_api_budget():
    """Shared free-tier usage ledger (hard-stops before any provider's limit)."""
    return ApiBudget()


def _polygon_key() -> str:
    """Polygon key: a key typed in Settings (session) wins, then config/.env, then env."""
    return str(st.session_state.get("polygon_api_key")
               or getattr(get_config(), "polygon_api_key", "")
               or os.environ.get("POLYGON_API_KEY", "")).strip()


def get_polygon_provider() -> PolygonProvider:
    """Budget-guarded Polygon provider (cheap to build; reads the key dynamically)."""
    return PolygonProvider(api_key=_polygon_key(), budget=get_api_budget())


def _provider_on(key: str) -> bool:
    """Whether a data provider is enabled. Defaults **ON** when a key is configured (you added the
    key to use it); an explicit Settings → Data & APIs toggle always overrides the default."""
    flag = f"use_{key}"
    if flag in st.session_state:
        return bool(st.session_state[flag])
    if key == "polygon":
        return bool(_polygon_key())
    return False

@st.cache_resource
def get_options_analyzer():
    return OptionsAnalyzer()

@st.cache_resource
def get_ipo_tracker():
    return IPOTracker()


# ── Optional AI models (loaded lazily, only when the Settings toggle is on) ──────

@st.cache_resource(show_spinner="Loading forecasting model…")
def get_forecaster():
    return PriceForecaster()

@st.cache_resource(show_spinner="Loading event classifier…")
def get_event_classifier():
    return NewsEventClassifier()

@st.cache_resource(show_spinner="Loading summarizer…")
def get_summarizer():
    return NewsSummarizer()

@st.cache_resource(show_spinner="Loading novelty embedder…")
def get_novelty():
    return NewsNovelty()

@st.cache_resource(show_spinner="Loading transcript cleaner…")
def get_transcript_cleaner():
    return TranscriptCleaner()


def _ai_on(key: str) -> bool:
    """Whether an AI feature toggle is enabled on the Settings page."""
    return bool(st.session_state.get(key, False))


# ── Data helpers ───────────────────────────────────────────────────────────────

# Wall-clock of the most recent *actual* market-data pull (set only on a real fetch, not a cache
# hit) + which source served it. Drives the "last live pull" stamp in the data-status strip so the
# user can see how fresh the data is. Plain module global (process-wide; no Streamlit cache warning).
_LAST_CAPTURE: Dict[str, object] = {"at": None, "source": None}


def _mark_capture(source: str) -> None:
    _LAST_CAPTURE["at"] = datetime.now()
    _LAST_CAPTURE["source"] = source


@with_retry()
def _yf_download(symbol: str, start, end) -> pd.DataFrame:
    # auto_adjust pinned: yfinance has flipped the default across versions, and the bulk
    # path (_yf_download_bulk) must return identical price series to this fallback.
    _mark_capture("Yahoo · free")
    return yf.download(symbol, start=start, end=end, progress=False, threads=False,
                       auto_adjust=True)


@with_retry(retries=2)
def _yf_info(symbol: str) -> Dict:
    t = yf.Ticker(symbol)
    return t.info if hasattr(t, "info") else {}


_HISTORY_FIELDS = ["Open", "High", "Low", "Close", "Volume"]

# Bulk-download chunk size. Large single calls get rate-limited by Yahoo and come back
# half-empty; ≤40 names per request stays reliable (see also _download_field's 25).
_BULK_CHUNK = 40
_BULK_CHUNK_PAUSE_S = 0.25


def fetch_symbol_history(symbol: str, days: int = 90) -> pd.DataFrame:
    """Daily bars for one symbol, served from the shared prefetch store when warm.

    Scan pages warm the store in bulk via ``prefetch_histories`` (chunked multi-ticker
    downloads), so per-symbol calls here are usually memory hits. A miss falls back to
    the retry-wrapped single-symbol download and feeds the store for the next screen.
    """
    cached = pricestore.get(symbol, days)
    if cached is not None:
        return cached
    end = _now_et().replace(tzinfo=None)
    start = end - timedelta(days=days)
    try:
        data = _yf_download(symbol, start, end)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):   # yfinance (Price, Ticker) shape
            data = data.droplevel(1, axis=1)
        cols = [c for c in _HISTORY_FIELDS if c in data.columns]
        out = data[cols].dropna()
        pricestore.put_many({symbol: out}, days)
        return out.copy()
    except Exception as exc:
        errlog.record("fetch_symbol_history", exc, note=symbol)
        return pd.DataFrame()


def _yf_download_bulk(symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
    """One multi-ticker Yahoo download → per-symbol OHLCV frames (the bulk fast path).

    Failed tickers come back as all-NaN column groups, not exceptions — they're dropped
    here and counted by the caller. A fully-empty response is retried once.
    """
    end = _now_et().replace(tzinfo=None)
    start = end - timedelta(days=days)
    raw = pd.DataFrame()
    for attempt in range(2):
        try:
            raw = yf.download(symbols, start=start, end=end, group_by="ticker",
                              auto_adjust=True, progress=False, threads=True)
        except Exception as exc:
            errlog.record("bulk_download", exc, note=f"{len(symbols)} symbols, attempt {attempt + 1}")
            raw = pd.DataFrame()
        if raw is not None and not raw.empty:
            break
        time.sleep(1.0)
    out: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for sym in symbols:
            if sym not in available:
                continue
            df = raw[sym]
            cols = [c for c in _HISTORY_FIELDS if c in df.columns]
            df = df[cols].dropna()
            if not df.empty:
                out[sym] = df
    elif len(symbols) == 1:                            # single-name chunk → flat columns
        cols = [c for c in _HISTORY_FIELDS if c in raw.columns]
        df = raw[cols].dropna()
        if not df.empty:
            out[symbols[0]] = df
    return out


def prefetch_histories(symbols: List[str], days: int, label: str = "prices") -> tuple:
    """Warm the price store for a scan: chunked multi-ticker downloads with a progress bar.

    Main-thread only (drives ``st.progress``). Yahoo-only by design — Polygon stays on the
    budgeted single-symbol path in ``fetch_history_ondemand``, so bulk scans can never touch
    the paid 5/min cap. Returns ``(fetched_ok, failed)`` over the symbols actually missing.
    """
    todo = pricestore.missing(symbols, days)
    if not todo:
        return (0, 0)
    chunks = [todo[i:i + _BULK_CHUNK] for i in range(0, len(todo), _BULK_CHUNK)]
    progress = st.progress(0.0, text=f"Downloading {label} for {len(todo)} symbols…")
    ok = 0
    try:
        for i, chunk in enumerate(chunks):
            frames = _yf_download_bulk(chunk, days)
            if frames:
                pricestore.put_many(frames, days)
                _mark_capture("Yahoo · free")
                ok += len(frames)
            progress.progress((i + 1) / len(chunks),
                              text=f"Downloading {label}… {min((i + 1) * _BULK_CHUNK, len(todo))}/{len(todo)}")
            if i + 1 < len(chunks):
                time.sleep(_BULK_CHUNK_PAUSE_S)
    finally:
        progress.empty()
    failed = len(todo) - ok
    if failed > max(2, int(len(todo) * 0.10)):
        st.caption(f"⚠ {failed} of {len(todo)} symbols returned no data (Yahoo throttling or "
                   "delisted names) — they're skipped in this scan.")
    return (ok, failed)


def fetch_history_ondemand(symbol: str, days: int = 90):
    """Single-symbol price history, provider-aware. Returns ``(df, source_label, note)``.

    Uses Polygon/Massive when it's enabled, keyed, and within its free-tier budget; otherwise (or
    when the per-minute budget is exhausted) falls back to free yfinance with a notice. **Only for
    on-demand single-symbol views** (drill-down charts, one-ticker lookups) — bulk scans must call
    ``fetch_symbol_history`` directly so they never hit Polygon's 5/min cap.
    """
    if _provider_on("polygon") and _polygon_key():
        allowed, reason = get_api_budget().check("polygon")
        if allowed:
            df = get_polygon_provider().fetch_daily_bars(symbol, days=days)
            if df is not None and not df.empty:
                _mark_capture("Polygon · Massive")
                return df, "Polygon · Massive", None
            return fetch_symbol_history(symbol, days=days), "Yahoo · free", None
        return (fetch_symbol_history(symbol, days=days), "Yahoo · free",
                f"Polygon {reason} reached — using free data.")
    return fetch_symbol_history(symbol, days=days), "Yahoo · free", None


def _now_et() -> datetime:
    """Current time in US/Eastern (market time) — canonical source is ``swingtradeapp.clock``."""
    return clock.now_et()


def render_data_status() -> None:
    """One-line strip shown at the top of every screen: the refresh time, whether any PAID
    (Polygon) data is in use, and when live market data was last actually pulled."""
    now = _now_et()
    polygon_live = _provider_on("polygon") and bool(_polygon_key())
    if polygon_live:
        stt = get_api_budget().status("polygon")
        used, lim = int(stt.get("used_minute", 0)), int(stt.get("per_minute", 5) or 5)
        icon = "🟢" if used < lim else "🔴"
        src = f"{icon} Yahoo · free **+ Polygon · Massive** on single-symbol ({used}/{lim} this min)"
    else:
        src = "🆓 Yahoo Finance · free (no paid data in use)"
    cap = _LAST_CAPTURE.get("at")
    cap_txt = (f" · live data last pulled **{cap:%I:%M:%S %p}** ({_LAST_CAPTURE.get('source')})"
               if isinstance(cap, datetime) else " · data loads on a screen's Scan / Re-scan")
    issues = errlog.count()
    issues_txt = (f"  ·  ⚠ **{issues} data issue{'s' if issues != 1 else ''}** this session "
                  "(details: Settings → Data health)") if issues else ""
    st.caption(f"🕒 As of **{now:%a %b %d, %Y · %I:%M %p}** ET  ·  Data: {src}{cap_txt}{issues_txt}")


def _persist_env(key: str, value: str) -> None:
    """Write/update a ``KEY=value`` line in the gitignored ``.env`` so a key entered in Settings
    survives restarts, and mirror it into the live process env."""
    try:
        p = Path(".env")
        lines = p.read_text().splitlines() if p.exists() else []
        out, found = [], False
        for ln in lines:
            if ln.strip().startswith(f"{key}="):
                out.append(f"{key}={value}")
                found = True
            else:
                out.append(ln)
        if not found:
            out.append(f"{key}={value}")
        jsonstore.atomic_write_text(p, "\n".join(out) + "\n")
        os.environ[key] = value
    except Exception as exc:
        errlog.record("persist_env", exc, note=key)


def _summarize_info(symbol: str, info: Dict) -> Dict:
    """Raw yfinance ``info`` payload → the compact dict the UI uses. Pure (no network)."""
    try:
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        prev = info.get("previousClose")
        change = (price - prev) / prev * 100.0 if price and prev else info.get("regularMarketChangePercent")
        return {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "price": price,
            "change_pct": change,
            "market_cap": info.get("marketCap"),
            "name": info.get("longName") or info.get("shortName") or symbol,
            "summary": info.get("longBusinessSummary") or "",
            "website": info.get("website") or "",
        }
    except Exception:
        return {"sector": "Unknown", "industry": "Unknown", "price": None,
                "change_pct": None, "market_cap": None,
                "name": symbol, "summary": "", "website": ""}


@st.cache_data(ttl=3600)
def fetch_symbol_info(symbol: str) -> Dict:
    try:
        return _summarize_info(symbol, _yf_info(symbol))
    except Exception:
        return _summarize_info(symbol, {})


@st.cache_data(ttl=600)
def get_screen_universe() -> List[str]:
    """Live listed universe, most-active names first (cached 10 min)."""
    return get_screening_universe()


# ── YouTube scanner helpers (all free / key-less; see swingtradeapp/youtube.py) ──

YT_PICKS_PATH = Path(".data/yt_picks.json")


@st.cache_data(ttl=86400)
def get_universe_set() -> set:
    """Full tradable-symbol set for validating tickers spoken in videos (cached 24h)."""
    try:
        return set(get_tradable_universe())
    except Exception:
        return set()


@st.cache_data(ttl=604800, show_spinner=False)
def get_yt_channel_id(handle: str) -> Optional[str]:
    """Resolve an @handle → UC… channel ID (cached a week — IDs never change)."""
    return yt.resolve_channel_id(handle)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yt_uploads(channel_id: str, channel_name: str, within_hours: float) -> List[yt.Upload]:
    """Recent uploads for one channel (cached 30 min)."""
    return yt.fetch_channel_uploads(channel_id, channel_name, within_hours)


@st.cache_data(ttl=86400, show_spinner=False)
def get_yt_transcript(video_id: str) -> Optional[List[yt.Segment]]:
    """Full transcript segments for a video (immutable — cached 24h). None if unavailable."""
    return yt.fetch_transcript_segments(video_id)


@st.cache_data(ttl=3600, show_spinner=False)
def yt_spy_return_since(from_date: str) -> Optional[float]:
    """SPY fractional return from ``from_date`` (YYYY-MM-DD) to now — the pick benchmark."""
    try:
        hist = fetch_symbol_history("SPY", days=400)
        if hist.empty:
            return None
        closes = hist["Close"]
        ref = closes[closes.index >= pd.Timestamp(from_date)]
        if ref.empty:
            return None
        start, end = float(ref.iloc[0]), float(closes.iloc[-1])
        return (end - start) / start if start else None
    except Exception:
        return None


def yt_current_price(symbol: str) -> Optional[float]:
    return fetch_symbol_info(symbol).get("price")


@st.cache_data(ttl=86400, show_spinner=False)
def yt_price_on(symbol: str, date_str: str) -> Optional[float]:
    """Close on (or the last trading day before) ``date_str`` — entry price for a dated call."""
    try:
        hist = fetch_symbol_history(symbol, days=400)
        if hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"]
        upto = closes[closes.index <= pd.Timestamp(date_str) + pd.Timedelta(days=1)]
        return float(upto.iloc[-1]) if not upto.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def yt_peak_since(symbol: str, date_str: str) -> Optional[float]:
    """Highest High since ``date_str`` — used to check whether a price target was reached."""
    try:
        hist = fetch_symbol_history(symbol, days=400)
        if hist.empty or "High" not in hist.columns:
            return None
        highs = hist["High"][hist.index >= pd.Timestamp(date_str)]
        return float(highs.max()) if not highs.empty else None
    except Exception:
        return None


def load_yt_store() -> Dict:
    return jsonstore.read_json(YT_PICKS_PATH, default={"picks": []})


def save_yt_store(store: Dict) -> None:
    jsonstore.atomic_write_json(YT_PICKS_PATH, store)


@st.cache_data(ttl=300)
def fetch_movers(top_n: int = 25, min_change_pct: float = 1.0) -> pd.DataFrame:
    """Live pre-market / session movers (cached 5 min)."""
    scanner = PreMarketScanner(get_config())
    return pd.DataFrame(scanner.fetch_movers(top_n=top_n, min_change_pct=min_change_pct))


# Yahoo predefined screeners surfaced on the raw "who's moving" page (no logic, passthrough).
RAW_SCREENS = {
    "Most Actives": "most_actives",
    "Day Gainers": "day_gainers",
    "Day Losers": "day_losers",
    "Small-Cap Gainers": "small_cap_gainers",
    "Aggressive Small Caps": "aggressive_small_caps",
    "Most Shorted": "most_shorted_stocks",
}


@st.cache_data(ttl=180)
def fetch_raw_movers(predefined: str, count: int = 50) -> pd.DataFrame:
    """Raw Yahoo screener feed as a tidy DataFrame — no filters, no signals (cached 3 min)."""
    quotes = get_raw_screen(predefined, count=count)
    rows = []
    for q in quotes:
        sym = q.get("symbol")
        if not sym:
            continue
        rows.append({
            "Symbol": sym,
            "Name": q.get("shortName") or q.get("longName") or sym,
            "Price": q.get("regularMarketPrice"),
            "Change %": q.get("regularMarketChangePercent"),
            "Pre-mkt %": q.get("preMarketChangePercent"),
            "Volume": q.get("regularMarketVolume"),
            "Avg Vol (3M)": q.get("averageDailyVolume3Month"),
            "Mkt Cap": q.get("marketCap"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_afterhours(top_n: int = 25, min_change_pct: float = 1.0) -> pd.DataFrame:
    """Post-market (after-hours) movers (cached 5 min). Empty outside the post-market session."""
    scanner = PreMarketScanner(get_config())
    return pd.DataFrame(scanner.fetch_afterhours_movers(top_n=top_n, min_change_pct=min_change_pct))


# ── Options analytics (single-ticker + small unusual-flow scan) ─────────────────

@st.cache_data(ttl=900, show_spinner=False)
def analyze_options(symbol: str) -> Dict:
    """Per-ticker options snapshot: IV rank/level, put-call ratio, unusual flow, earnings/IV-crush.

    Each field is best-effort and None-safe (live option chains are flaky). Cached 15 min.
    """
    oa = get_options_analyzer()
    current_iv = oa.fetch_current_iv(symbol)
    earn = oa.fetch_earnings_date(symbol)
    spot = fetch_symbol_info(symbol).get("price")
    return {
        "iv_rank": oa.fetch_iv_rank(symbol),
        "current_iv": current_iv,
        "pc_ratio": oa.fetch_put_call_ratio_symbol(symbol),
        "unusual": oa.detect_unusual_volume(symbol),
        "key_strikes": oa.analyze_key_strikes(symbol, spot=spot),
        "earnings_date": earn,
        "iv_crush": oa.estimate_iv_crush(symbol, current_iv) if current_iv else None,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_insider(symbol: str) -> Dict:
    """Real SEC Form-4 insider transactions for a ticker (free via yfinance, cached 1h).

    Returns a tidy transactions frame, the 6-month buy/sell summary, and a net-sentiment dict.
    """
    try:
        t = yf.Ticker(symbol)
        raw = t.insider_transactions
        purch = t.insider_purchases
    except Exception:
        raw, purch = None, None
    tidy = ins.tidy_transactions(raw)
    return {
        "tidy": tidy,
        "purchases": purch if isinstance(purch, pd.DataFrame) else pd.DataFrame(),
        "summary": ins.summarize(tidy),
    }


def _pc_sentiment(pc: Optional[float]) -> str:
    """Put/call ratio → sentiment label. <0.70 call-heavy (bullish), >1.00 put-heavy (bearish)."""
    if pc is None:
        return "—"
    if pc < 0.70:
        return "Bullish"
    if pc > 1.00:
        return "Bearish"
    return "Neutral"


@st.cache_data(ttl=1200, show_spinner=False)
def scan_options_flow(sample_size: int = 15) -> pd.DataFrame:
    """Small, slow scan of the most-actives for options sentiment + unusual flow (cached 20 min).

    Per symbol pulls the live nearest-expiry chain (put/call ratio) and flags unusual volume
    (volume > 10% of open interest). Kept intentionally small — each symbol is a network round-trip.
    """
    oa = get_options_analyzer()
    rows: List[Dict] = []
    for sym in get_screen_universe()[:sample_size]:
        try:
            pc = oa.fetch_put_call_ratio_symbol(sym)
            unusual = oa.detect_unusual_volume(sym)
        except Exception:
            continue
        if pc is None and unusual is None:
            continue
        n_unusual = 0
        signal = "—"
        if unusual:
            n_unusual = len(unusual.get("unusual_calls", [])) + len(unusual.get("unusual_puts", []))
            signal = "calls" if unusual.get("signal") == "calls" else "puts"
        rows.append({
            "Symbol": sym,
            "P/C Ratio": pc,
            "Sentiment": _pc_sentiment(pc),
            "Unusual": signal,
            "# Unusual": n_unusual,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["# Unusual", "P/C Ratio"], ascending=[False, True]).reset_index(drop=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def ipo_table() -> pd.DataFrame:
    """Performance of the curated recent-IPO list (cached 1h). IPO price ≈ earliest available."""
    tracker = get_ipo_tracker()
    rows: List[Dict] = []
    for sym, meta in tracker.RECENT_IPOS.items():
        perf = tracker.fetch_ipo_performance(sym)
        age = tracker.get_ipo_age(sym)
        if perf is None:
            continue
        rows.append({
            "Symbol": sym,
            "Name": meta["name"],
            "Days since IPO": age,
            "IPO price ≈": perf["ipo_price_approx"],
            "Current": perf["current_price"],
            "Gain %": perf["gain_pct"],
            "Phase": "early" if (age or 0) < 90 else "established",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Days since IPO").reset_index(drop=True)
    return df


def recommend_label(score: Optional[float], trend: str = "long") -> str:
    """Plain-English action from a signal score + direction.

    Buy ≥ 0.65 · Watch ≥ 0.45 · Weak below · Avoid for non-bullish/short signals.
    Shared by every table that surfaces a signal score.
    """
    if score is None:
        return "—"
    if trend == "short":
        return "Avoid (bearish)"
    if trend not in ("long", ""):
        return "Avoid (not bullish)"
    if score >= 0.65:
        return "Buy"
    if score >= 0.45:
        return "Watch"
    return "Weak"


def _recommendation(signal) -> str:
    """Map a Signal object into a recommendation label."""
    if signal is None:
        return "Avoid (no signal)"
    return recommend_label(signal.score, signal.signal_type)


def _reco_color(v: str) -> str:
    """Cell color for a recommendation label (used in dataframe stylers)."""
    if v == "Buy":
        return "color:#00c851;font-weight:bold"
    if v == "Watch":
        return "color:#ffbb33"
    if isinstance(v, str) and v.startswith("Avoid"):
        return "color:#ff4444"
    return "color:#888888"


def _ml_prob_color(v) -> str:
    """Cell color for the ML P(up): green ≥0.55, red ≤0.45, neutral between (dataframe styler)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f >= 0.55:
        return "color:#00c851;font-weight:bold"
    if f <= 0.45:
        return "color:#ff4444"
    return "color:#888888"


def _hl_pct(v) -> str:
    """Green/red text color for a signed numeric percentage (dataframe styler)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "color:#00c851" if f > 0 else ("color:#ff4444" if f < 0 else "")


def _limit_note(entry, current) -> str:
    """Caption clarifying the entry is a pullback *limit* (vs the live price), not a market buy."""
    try:
        entry, current = float(entry), float(current)
    except (TypeError, ValueError):
        return ""
    if not entry or not current or abs(entry - current) < 0.005 * current:
        return ""  # market entry (or ATR fallback) — nothing to clarify
    pct = abs(entry - current) / current * 100
    side = "below" if entry < current else "above"
    return (f"⏳ Entry is a **limit** {pct:.2f}% {side} the live ${current:.2f} — "
            f"it fills only if price reaches it.")


# ── Scan gating: screens run on a button (not on page open) + a recommended run-time ──────
# Best ET window to run each scanning screen, so you don't have to run them all every time.
RECOMMENDED_TIMES: Dict[str, str] = {
    "Screener": "After 9:45 AM ET (let the open settle) — or after the close to plan tomorrow",
    "Rally Radar": "After the close (completed daily bars) — or 9:45 AM+ once the open settles",
    "Setup Scanner": "After the close (completed daily bars) — or 9:45 AM+ once the open settles",
    "Backtest Lab": "Any time — it runs on completed historical daily bars",
    "Signal Stack": "After the close, or 9:45 AM+ once the open settles",
    "ETF Screener": "After the close (daily bars) — or anytime",
    "Pre-Market Movers": "8:00–9:15 AM ET (after the 8:30 econ data)",
    "Live Movers": "Intraday, 9:30 AM–4:00 PM ET",
    "After-Hours & IPOs": "4:00–8:00 PM ET (post-market session)",
    "Whale Movements": "After the close (uses completed daily bars)",
    "Options Flow": "Midday ~10:00 AM–3:30 PM ET (chains most liquid)",
    "Predictions": "After the close (next-session forecast)",
    "Auto Watchlist": "8:00–9:15 AM ET (pre-market movers) or just after the open",
    "Market Events": "Pre-open ~8:00 AM ET, or midday for a fresh read",
    "Heat Map": "Intraday or after the close",
    "YouTube": "Evening / after the close (creators post post-market)",
    "Alpha Engine": "After the close (uses completed daily bars) — or any time to inspect the book",
}


def scan_gate(key: str, recommended: str, clear=None) -> bool:
    """Gate a heavy scan behind a button. Shows the recommended run-time + a Scan/Re-scan button
    and returns whether to run. The screen does NOT scan on page open — only on click — and the
    gate is reset on navigation (see run_dashboard) so each visit asks again.
    """
    st.caption(f"🕒 **Best time to run:** {recommended}")
    flag = f"_scan_{key}"
    if st.session_state.get(flag):
        if st.button("↻ Re-scan (fresh data)", key=f"rescan_{key}"):
            if clear is not None:
                try:
                    clear()
                except Exception:
                    pass
            pricestore.clear()  # "fresh data" includes prices — the prefetch re-warms in seconds
        return True
    if st.button("▶ Scan now", key=f"scanbtn_{key}", type="primary"):
        st.session_state[flag] = True
        return True
    st.info("This screen doesn't run automatically — press **▶ Scan now** when you're ready.")
    return False


@st.cache_data(ttl=300, show_spinner=False)
def build_auto_watchlist(_config, top_n: int = 20, min_change_pct: float = 2.0,
                         with_forecast: bool = False) -> pd.DataFrame:
    """Auto-build a watchlist from pre-market BULLISH movers with entry/exit + indicators.

    For each bullish mover we generate a trend signal and surface the recommended entry,
    stop, target, risk:reward and the key momentum indicators. When ``with_forecast`` is
    set, a Chronos/heuristic forecast adds an expected-return % and a confirmation flag.
    """
    movers = fetch_movers(top_n=60, min_change_pct=min_change_pct)
    if movers.empty:
        return pd.DataFrame()
    bulls = movers[movers["change_pct"] > 0].head(top_n)
    gen = get_signal_generator(_config)
    forecaster = get_forecaster() if with_forecast else None
    rows = []
    for _, m in bulls.iterrows():
        sym = m["symbol"]
        hist = fetch_symbol_history(sym, days=120)
        if hist.empty or len(hist) < 26:
            continue
        closes = _to_series(hist, "Close")
        volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0] * len(closes)
        highs = _to_series(hist, "High") if "High" in hist.columns else closes
        lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
        sig = gen.build_signal(sym, closes, volumes, highs=highs, lows=lows)
        if sig is None:
            continue
        risk = sig.entry_price - sig.stop_price
        reward = sig.target_price - sig.entry_price
        rr = round(reward / risk, 2) if risk > 0 else 0.0
        row = {
            "Symbol": sym,
            "Pre-mkt %": float(m["change_pct"]),
            "Price": m.get("price") or sig.entry_price,
            "Recommendation": _recommendation(sig),
            "Score": sig.score,
            "RSI": sig.metadata.get("rsi"),
            "MACD hist": sig.metadata.get("macd_hist"),
            "Vol surge": sig.metadata.get("vol_surge"),
            "Entry": sig.entry_price,
            "Stop": sig.stop_price,
            "Target": sig.target_price,
            "R:R": rr,
        }
        if forecaster is not None:
            fc = forecaster.forecast(closes, horizon=5)
            row["Fcst ret %"] = expected_return_pct(fc)
            row["Forecast"] = forecast_confirms(sig, fc)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def predict_tomorrow(sample_size: int = 60) -> pd.DataFrame:
    """Next-session (tomorrow) probabilistic price forecast across the most-active universe.

    Runs the PriceForecaster at horizon=1 for each symbol: Chronos-Bolt when installed, else a
    Monte-Carlo random-walk heuristic (always available). Surfaces the median predicted close,
    the p10–p90 band, the implied next-day return %, and an uncertainty width. Sorted most-bullish
    first. Cached 30 min. This is a probabilistic model output — not financial advice.
    """
    forecaster = get_forecaster()
    universe = get_screen_universe()[:sample_size]
    rows: List[Dict] = []
    for sym in universe:
        hist = fetch_symbol_history(sym, days=200)
        if hist.empty or len(hist) < 40:
            continue
        fc = forecaster.forecast(_to_series(hist, "Close"), horizon=1)
        if fc is None or not fc.get("last_price"):
            continue
        last = float(fc["last_price"])
        p10, p50, p90 = float(fc["p10"][0]), float(fc["p50"][0]), float(fc["p90"][0])
        ret = (p50 - last) / last * 100.0 if last else 0.0
        band = (p90 - p10) / p50 * 100.0 if p50 else 0.0
        rows.append({
            "Symbol": sym,
            "Price": last,
            "Pred Close": p50,
            "Pred Return %": ret,
            "Direction": "Up" if p50 > last else ("Down" if p50 < last else "Flat"),
            "Low (p10)": p10,
            "High (p90)": p90,
            "Uncertainty %": band,
            "Model": fc["source"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Pred Return %", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def scan_whale_activity(sample_size: int = 120, min_rvol: float = 2.0,
                        min_dollar_vol_m: float = 50.0, min_price: float = 5.0) -> pd.DataFrame:
    """Scan the most-active universe for large-money ("whale") footprints (cached 10 min).

    For each symbol we pull recent daily OHLCV and ask the WhaleDetector whether the latest
    bar is an outsized volume event (relative volume ≥ ``min_rvol`` AND ≥ ``min_dollar_vol_m``
    $M traded), then tag it Heavy Buying / Heavy Selling / Accumulation / Distribution / Churn
    with a 0–100 whale score. Most-active names come first so the scan favors where the size is.
    """
    detector = WhaleDetector(WhaleConfig(min_rvol=min_rvol,
                                         min_dollar_vol=min_dollar_vol_m * 1e6))
    universe = get_screen_universe()[:sample_size]
    rows: List[Dict] = []
    for sym in universe:
        hist = fetch_symbol_history(sym, days=60)
        if hist.empty or len(hist) < 24 or "Volume" not in hist.columns:
            continue
        closes = _to_series(hist, "Close")
        if min_price and float(closes[-1]) < min_price:
            continue
        volumes = _to_series(hist, "Volume")
        highs = _to_series(hist, "High") if "High" in hist.columns else closes
        lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
        opens = _to_series(hist, "Open") if "Open" in hist.columns else closes
        res = detector.analyze(sym, opens, highs, lows, closes, volumes)
        if res is not None:
            rows.append(res)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("whale_score", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def scan_rally_radar(sample_size: int = 150, min_score: float = 45.0,
                     min_price: float = 5.0) -> pd.DataFrame:
    """Scan the most-active universe for *early* / pre-breakout bullish setups (cached 10 min).

    Unlike the Screener (which rewards already-established trends), the RallyDetector looks for
    momentum *igniting*: a volatility squeeze, volume starting to build, a MACD/RSI inflection,
    a 20-day-MA reclaim, and price pressing the top of its base — scored 0–100 with a stage label
    (Coiling → Igniting → Breaking out). Most-active names are scanned first.
    """
    detector = RallyDetector(RallyConfig())
    universe = get_screen_universe()[:sample_size]
    records: List[Dict] = []
    for sym in universe:
        hist = fetch_symbol_history(sym, days=160)
        if hist.empty or len(hist) < 60 or "Volume" not in hist.columns:
            continue
        closes = _to_series(hist, "Close")
        if min_price and float(closes[-1]) < min_price:
            continue
        records.append({
            "symbol": sym,
            "closes": closes,
            "volumes": _to_series(hist, "Volume"),
            "highs": _to_series(hist, "High") if "High" in hist.columns else closes,
            "lows": _to_series(hist, "Low") if "Low" in hist.columns else closes,
        })
    rows = detector.scan(records, min_score=min_score)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("rally_score", ascending=False).reset_index(drop=True)
    return df


# ── Setup Scanner / Backtest Lab / Market Regime helpers ─────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def get_regime_read():
    """Current market-regime verdict (Trade / Caution / Stand-aside) — cached 10 min.

    Combines SPY vs its 200-SMA (+ slope), market breadth and VIX via ``regime.assess_regime``.
    Used to gate the Setup Scanner and surfaced on the Market Regime page.
    """
    spy = fetch_symbol_history("SPY", days=400)
    closes = _to_series(spy, "Close") if not spy.empty else []
    macro = get_macro_context()
    try:
        vix = macro.fetch_vix()
    except Exception as exc:
        errlog.record("regime_vix", exc)
        vix = None
    breadth = None
    try:
        b = macro.fetch_market_breadth()
        breadth = b.get("bullish_breadth_pct") if b else None
    except Exception as exc:
        errlog.record("regime_breadth", exc)
        breadth = None
    return regime_mod.assess_regime(closes, vix=vix, breadth_pct=breadth)


@st.cache_data(ttl=600, show_spinner=False)
def scan_setups(sample_size: int = 150, min_price: float = 5.0) -> pd.DataFrame:
    """Scan the most-active universe for every named setup (cached 10 min).

    For each symbol runs ``setups.detect_all`` (VCP, 20-EMA pullback, double bottom, liquidity
    sweep, RSI(2)) on ~1y of daily bars and records each hit with its entry/stop/target/R:R, a
    quality score, a Fair-Value-Gap confluence tag and the reasons it fired. Sub-``min_price``
    (penny) names are skipped unless the page's checkbox includes them.
    """
    universe = get_screen_universe()[:sample_size]
    rows: List[Dict] = []
    for sym in universe:
        hist = fetch_symbol_history(sym, days=400)
        if hist.empty or len(hist) < 60 or "Volume" not in hist.columns:
            continue
        closes = _to_series(hist, "Close")
        if min_price and float(closes[-1]) < min_price:
            continue
        opens = _to_series(hist, "Open") if "Open" in hist.columns else closes
        highs = _to_series(hist, "High") if "High" in hist.columns else closes
        lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
        vols = _to_series(hist, "Volume")
        try:
            hits = setups_mod.detect_all(opens, highs, lows, closes, vols)
        except Exception as exc:
            errlog.record("scan_setups", exc, note=sym)
            continue
        if not hits:
            continue
        fvg = setups_mod.fvg_confluence(highs, lows, closes)
        price = float(closes[-1])
        for hit in hits:
            tags = list(hit.tags) + ([fvg] if fvg else [])
            rows.append({
                "symbol": sym, "setup": hit.name, "score": float(hit.score),
                "price": price, "entry": hit.entry, "stop": hit.stop,
                "target": hit.target, "rr": float(hit.risk_reward),
                "tags": ", ".join(tags), "why": " · ".join(hit.reasons),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def run_setup_backtest(setup_name: str, symbols: List[str]) -> Dict:
    """Backtest one setup across ``symbols`` on causal historical bars, honouring the setup's own
    stop/target. Pools trades and returns honest edge stats (expectancy in R, profit factor,
    win rate, Sharpe, max drawdown). Used by the Backtest Lab.
    """
    engine = get_backtest_engine(get_config())
    setup = setups_mod.SETUP_BY_NAME.get(setup_name)
    trades = []  # TradeRecord list pooled across symbols
    n_symbols = 0
    for sym in symbols:
        hist = fetch_symbol_history(sym, days=800)
        if hist.empty or len(hist) < setup.min_bars + 12 or "Volume" not in hist.columns:
            continue
        closes = np.array(_to_series(hist, "Close"), dtype=float)
        opens = np.array(_to_series(hist, "Open") if "Open" in hist.columns else closes, dtype=float)
        highs = np.array(_to_series(hist, "High") if "High" in hist.columns else closes, dtype=float)
        lows = np.array(_to_series(hist, "Low") if "Low" in hist.columns else closes, dtype=float)
        vols = np.array(_to_series(hist, "Volume"), dtype=float)
        try:
            hits = setup.signal_bars(opens, highs, lows, closes, vols)
        except Exception:
            continue
        if not hits:
            continue
        sigs = [{"bar": hh.bar, "entry": hh.entry, "stop": hh.stop,
                 "target": hh.target, "symbol": sym} for hh in hits]
        res = engine.run_signals(closes, highs, lows, sigs)
        trades.extend(res.trades)
        n_symbols += 1

    if not trades:
        return {"trades": [], "n": 0, "n_symbols": n_symbols}

    pnls = np.array([t.pnl_pct for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    rs = [t.pnl_pct / ((t.entry_price - t.stop_price) / t.entry_price)
          for t in trades if (t.entry_price - t.stop_price) > 0]
    eq = np.cumprod(1 + pnls)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))
    dd = (peak - np.concatenate([[1.0], eq])) / peak
    gross_win = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    return {
        "trades": trades,
        "n": len(trades),
        "n_symbols": n_symbols,
        "win_rate": float((pnls > 0).mean()),
        "expectancy_r": float(np.mean(rs)) if rs else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "sharpe": float((pnls.mean() / pnls.std(ddof=1)) * np.sqrt(252)) if len(pnls) > 1 and pnls.std(ddof=1) > 0 else 0.0,
        "max_dd": float(dd.max()),
        "total_return": float(eq[-1] - 1.0),
        "equity": eq.tolist(),
    }


# ETF categories sourced from ETFScreener (single source of truth).
ETF_CATEGORIES = {
    "Broad Market": ETFScreener.BROAD_MARKET_ETFS,
    "Sector": ETFScreener.SECTOR_ETFS,
    "Dividend & Factor": ETFScreener.DIVIDEND_FACTOR_ETFS,
    "Thematic": ETFScreener.THEMATIC_ETFS,
    "International": ETFScreener.INTERNATIONAL_ETFS,
    "Commodities": ETFScreener.COMMODITY_ETFS,
    "Bonds": ETFScreener.BOND_ETFS,
    "Volatility": ETFScreener.VOLATILITY_ETFS,
    "Crypto": ETFScreener.CRYPTO_ETFS,
    "Leveraged / Inverse": ETFScreener.LEVERAGED_ETFS,
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_etf_table(_config, categories: tuple) -> pd.DataFrame:
    """Signals + daily change for the selected ETF categories (cached per selection)."""
    gen = get_signal_generator(_config)
    rows = []
    for category in categories:
        mapping = ETF_CATEGORIES.get(category, {})
        for sym, name in mapping.items():
            hist = fetch_symbol_history(sym, days=160)
            if hist.empty:
                # Keep the ETF visible even when data is briefly unavailable.
                rows.append({"Category": category, "Symbol": sym, "Name": name, "Price": None,
                             "Change %": None, "Trend": "—", "Score": None, "RSI": None,
                             "Recommendation": "—"})
                continue
            closes = _to_series(hist, "Close")
            volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0] * len(closes)
            highs = _to_series(hist, "High") if "High" in hist.columns else closes
            lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
            # min_score=0 → every ETF gets a directional signal + recommendation (no 0.40 gate).
            sig = gen.build_signal(sym, closes, volumes, highs=highs, lows=lows, min_score=0.0)
            price = closes[-1]
            prev = closes[-2] if len(closes) > 1 else price
            change = (price - prev) / prev * 100 if prev else 0.0
            rows.append({
                "Category": category,
                "Symbol": sym,
                "Name": name,
                "Price": price,
                "Change %": change,
                "Trend": (sig.signal_type if sig else "—"),
                "Score": (sig.score if sig else None),
                "RSI": (sig.metadata.get("rsi") if sig else None),
                "Recommendation": recommend_label(sig.score, sig.signal_type) if sig else "—",
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=900)
def fetch_heatmap_data(symbols: List[str]) -> pd.DataFrame:
    """Sector, market cap and daily % change for each symbol — the FinViz market map inputs."""
    rows = []
    for s in symbols:
        info = fetch_symbol_info(s)
        mcap = info.get("market_cap")
        change = info.get("change_pct")
        if not mcap or change is None:
            continue
        rows.append({
            "symbol": s,
            "sector": info.get("sector") or "Unknown",
            "market_cap": float(mcap),
            "change_pct": float(change),
        })
    return pd.DataFrame(rows)


# ── AI helpers (forecast + news digest) ─────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def forecast_symbol(symbol: str, horizon: int = 5) -> Optional[Dict]:
    """Probabilistic price forecast (Chronos or heuristic) for one symbol."""
    hist = fetch_symbol_history(symbol, days=200)
    if hist.empty or len(hist) < 30:
        return None
    return get_forecaster().forecast(_to_series(hist, "Close"), horizon=horizon)


@st.cache_data(ttl=900, show_spinner=False)
def news_digest(symbol: str) -> str:
    """Plain-English digest of a ticker's recent headlines (Yahoo + Google News)."""
    titles = [it["title"] for it in get_ticker_news(symbol, 6)]
    return get_summarizer().summarize(titles)


@st.cache_data(ttl=600, show_spinner=False)
def get_ticker_news(symbol: str, max_items: int = 8) -> List[Dict]:
    """Aggregated ticker news (Yahoo Finance + Google News), cached 10 min."""
    from swingtradeapp.nlp import fetch_news_items
    return fetch_news_items(symbol, max_items)


@st.cache_data(ttl=600, show_spinner=False)
def get_market_news(max_items: int = 24) -> List[Dict]:
    """Broad market news across free outlets (Google News), cached 10 min."""
    from swingtradeapp.nlp import fetch_market_news
    return fetch_market_news(max_items)


# Macro proxies whose news reflects global/local events that move stocks.
MARKET_EVENT_TICKERS = {
    "US Equities": ["SPY", "QQQ", "DIA", "IWM"],
    "Rates": ["^TNX", "TLT"],
    "Commodities": ["CL=F", "GC=F", "USO", "GLD"],
    "Volatility": ["^VIX"],
    "Global": ["EEM", "EFA", "FXI", "EWJ"],
    "Currencies": ["UUP"],
}


@st.cache_data(ttl=900, show_spinner=False)
def scan_market_events(_config, max_per_ticker: int = 4) -> pd.DataFrame:
    """Scan news across macro proxies for events that may move stocks.

    Uses the (optionally upgraded) sentiment model + event classifier; both degrade to
    heuristics. Headlines are deduped so recycled coverage doesn't dominate.
    """
    from swingtradeapp.nlp import NewsEventClassifier, _fetch_headlines
    analyzer = get_sentiment_analyzer(_config)
    classifier = get_event_classifier() if _ai_on("ai_events") else None
    rows = []
    for area, tickers in MARKET_EVENT_TICKERS.items():
        for t in tickers:
            for h in _fetch_headlines(t, max_per_ticker):
                r = analyzer.analyze_text(h)
                event = classifier.classify(h) if classifier else NewsEventClassifier._classify_keywords(h)
                rows.append({
                    "Area": area, "Source": t, "Event": event,
                    "Sentiment": str(r.get("label", "neutral")).lower(),
                    "Score": float(r.get("score", 0.5)), "Headline": h,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["Headline"]).reset_index(drop=True)
    return df


def create_forecast_chart(symbol: str, fc: Dict, history_days: int = 60) -> go.Figure:
    """Recent price history with the forecast p10–p90 cone and p50 path appended."""
    hist = fetch_symbol_history(symbol, days=history_days)
    closes = _to_series(hist, "Close") if not hist.empty else [fc["last_price"]]
    n_hist = len(closes)
    h = fc["horizon"]
    x_hist = list(range(n_hist))
    x_fc = list(range(n_hist - 1, n_hist - 1 + h + 1))  # connect last actual to forecast
    p10 = [closes[-1]] + fc["p10"]
    p50 = [closes[-1]] + fc["p50"]
    p90 = [closes[-1]] + fc["p90"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=closes, name="Price", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=x_fc, y=p90, name="p90", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x_fc, y=p10, name="p10–p90", fill="tonexty",
                             fillcolor="rgba(0,200,80,0.15)", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x_fc, y=p50, name="p50 forecast",
                             line=dict(color="#00a050", dash="dash")))
    fig.update_layout(title=f"{symbol} — {h}-bar forecast ({fc['source']})",
                      height=380, xaxis_title="bar", yaxis_title="Price ($)",
                      hovermode="x unified", margin=dict(t=40, l=10, r=10, b=10))
    return fig


def _forecast_color(v: str) -> str:
    if v == "Confirms":
        return "color:#00c851;font-weight:bold"
    if v == "Caution":
        return "color:#ff4444"
    return "color:#888888"


def _sentiment_emoji(label: str) -> str:
    low = str(label).lower()
    return "🟢" if "pos" in low else "🔴" if "neg" in low else "⚪"


def _render_legend() -> None:
    """Collapsible reference explaining every label used across the dashboard."""
    with st.expander("ℹ️ What the labels mean"):
        st.markdown(
            "| Label | Where | Meaning |\n"
            "|---|---|---|\n"
            "| **Buy** | Recommendation | Bullish signal, confidence ≥ 0.65 |\n"
            "| **Watch** | Recommendation | Bullish, 0.45–0.65 — monitor for entry |\n"
            "| **Weak** | Recommendation | Bullish but < 0.45 — low conviction |\n"
            "| **Avoid (bearish)** | Recommendation | Short/bearish signal — not a long |\n"
            "| **Confirms** | Forecast | Median forecast ends above entry & downside stays above stop |\n"
            "| **Caution** | Forecast | Forecast downside path breaches the stop |\n"
            "| **Neutral** | Forecast | Forecast inconclusive |\n"
            "| **long / short** | Trend | Signal direction |\n"
            "| 🟢 positive / 🔴 negative / ⚪ neutral | Sentiment | News tone from the sentiment model |\n"
            "| earnings · M&A · upgrade · downgrade · legal · guidance · product | Event | News event type |\n"
            "| **Score** | — | Signal confidence 0–1 (higher = stronger) |\n"
            "| **R:R** | — | Reward-to-risk ratio (target distance ÷ stop distance) |\n"
            "| **Vol surge** | — | Volume vs 20-day average (e.g. 2.00x) |\n"
            "| **RSI** | — | Momentum 0–100 (<30 oversold, >70 overbought) |\n"
            "| **Novelty** | News | % of headlines that are genuinely new (not recycled) |\n"
        )


def _mood_cell(v) -> str:
    """Background color for a 0–100 gauge score (matplotlib-free red→green)."""
    try:
        s = float(v)
    except (TypeError, ValueError):
        return ""
    if s >= 60:
        return "background-color:#1b5e20;color:white"
    if s > 45:
        return "background-color:#f9a825;color:black"
    return "background-color:#b71c1c;color:white"


def _mood_label(score: float) -> str:
    if score >= 75:
        return "Extreme Greed"
    if score >= 60:
        return "Greed / Bullish"
    if score > 45:
        return "Neutral"
    if score > 25:
        return "Fear / Bearish"
    return "Extreme Fear"


@st.cache_data(ttl=900, show_spinner=False)
def compute_market_mood(_config) -> Dict:
    """Composite market mood (0–100) from news tone + known quant gauges.

    Blends: live news sentiment across macro proxies, VIX (inverted), market breadth
    (% of SPY above its 200-day MA), and SPY trend vs its 50/200-day moving averages.
    """
    macro = get_macro_context()
    comps: Dict[str, float] = {}

    # 1) News sentiment across macro proxies.
    ev = scan_market_events(_config)
    if not ev.empty:
        pos = int(ev["Sentiment"].str.contains("pos").sum())
        neg = int(ev["Sentiment"].str.contains("neg").sum())
        tot = pos + neg
        comps["News sentiment"] = 50 + 50 * ((pos - neg) / tot) if tot else 50.0
    else:
        comps["News sentiment"] = 50.0

    # 2) VIX inverted: ≤12 → calm/greed (100), ≥32 → stress/fear (0).
    vix = macro.fetch_vix()
    comps["Volatility (VIX)"] = max(0.0, min(100.0, (32 - vix) / (32 - 12) * 100)) if vix else 50.0

    # 3) Market breadth (% of SPY above 200-day MA).
    breadth = macro.fetch_market_breadth()
    comps["Market breadth"] = breadth["bullish_breadth_pct"] * 100 if breadth else 50.0

    # 4) SPY trend vs 50/200-day moving averages.
    spy = fetch_symbol_history("SPY", days=220)
    if not spy.empty and len(spy) >= 200:
        closes = _to_series(spy, "Close")
        price, sma50, sma200 = closes[-1], float(np.mean(closes[-50:])), float(np.mean(closes[-200:]))
        comps["Trend (SPY)"] = 50 + (25 if price > sma50 else -25) + (25 if price > sma200 else -25)
    else:
        comps["Trend (SPY)"] = 50.0

    score = round(sum(comps.values()) / len(comps), 1)
    return {"score": score, "label": _mood_label(score), "components": comps,
            "vix": vix, "as_of": datetime.now().strftime("%Y-%m-%d %H:%M")}


def create_mood_gauge(mood: Dict) -> go.Figure:
    """Fear/Greed-style gauge for the composite market mood."""
    score = mood["score"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Market Mood — {mood['label']}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#333"},
            "steps": [
                {"range": [0, 25], "color": "#ff4444"},
                {"range": [25, 45], "color": "#ff9d4d"},
                {"range": [45, 55], "color": "#ffe066"},
                {"range": [55, 75], "color": "#9ccc65"},
                {"range": [75, 100], "color": "#00c851"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": score},
        },
    ))
    fig.update_layout(height=300, margin=dict(t=50, l=30, r=30, b=10))
    return fig


def render_forecast_panel(symbol: str, config, entry: float = None, stop: float = None,
                          key_prefix: str = "fc") -> None:
    """Chronos/heuristic forecast panel — only when the forecast toggle is on."""
    if not _ai_on("ai_forecast"):
        return
    fc = forecast_symbol(symbol, horizon=5)
    if not fc:
        st.caption("Forecast unavailable for this symbol.")
        return
    st.markdown("#### 🔮 Price forecast")
    st.plotly_chart(create_forecast_chart(symbol, fc), use_container_width=True,
                    key=f"{key_prefix}_fcst_{symbol}")
    er = expected_return_pct(fc)
    shim = SimpleNamespace(signal_type="long",
                           entry_price=entry if entry else fc["last_price"],
                           stop_price=stop if stop else fc["last_price"] * 0.95)
    conf = forecast_confirms(shim, fc)
    fcols = st.columns(3)
    fcols[0].metric("Forecast p50", f"${fc['p50'][-1]:.2f}")
    fcols[1].metric("Exp. return", f"{er:+.2f}%" if er is not None else "—")
    fcols[2].markdown(f"**Forecast check:** "
                      f"<span style='{_forecast_color(conf)}'>{conf}</span>", unsafe_allow_html=True)
    st.caption(f"Forecast source: {fc['source']}")


def render_ticker_news(symbol: str, config, max_items: int = 6) -> None:
    """Readable, always-on news for a single ticker: overall tone + linked headlines.

    Works without any optional models (FinBERT/neutral sentiment + keyword events). When
    the AI toggles are on it adds the model summary, model event tags and novelty.
    """
    from swingtradeapp.nlp import NewsEventClassifier
    items = get_ticker_news(symbol, max_items)
    if not items:
        st.caption("No recent news for this ticker.")
        return

    analyzer = get_sentiment_analyzer(config)
    ec = get_event_classifier() if _ai_on("ai_events") else None
    scored, pos, neg = [], 0, 0
    for it in items:
        r = analyzer.analyze_text(it["title"])
        label = str(r.get("label", "neutral")).lower()
        event = ec.classify(it["title"]) if ec else NewsEventClassifier._classify_keywords(it["title"])
        pos += "pos" in label
        neg += "neg" in label
        scored.append({**it, "label": label, "score": float(r.get("score", 0.5)), "event": event})

    tone = ("🟢 Bullish" if pos > neg else "🔴 Bearish" if neg > pos else "⚪ Mixed / Neutral")
    st.markdown(f"**News tone: {tone}** — {pos} positive · {neg} negative of {len(scored)} recent")

    if _ai_on("ai_summary"):
        digest = news_digest(symbol)
        if digest:
            st.info("📝 " + digest)

    for s in scored:
        emoji = _sentiment_emoji(s["label"])
        ev = f" · `{s['event']}`" if s.get("event") and s["event"] != "other" else ""
        meta = " · ".join(x for x in [s.get("publisher", ""), s.get("time", "")] if x)
        title_md = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
        st.markdown(
            f"{emoji} {title_md}  \n"
            f"<span style='color:#888;font-size:0.85em'>score {s['score']:.2f}{ev}"
            f"{' · ' + meta if meta else ''}</span>",
            unsafe_allow_html=True,
        )


# ── Analyst briefs (template-generated theses; see swingtradeapp/analyst.py) ─────

@st.cache_data(ttl=900, show_spinner=False)
def assemble_dossier(symbol: str, _config, use_forecast: bool = False,
                     use_ml: bool = False):
    """Gather everything the app knows about one ticker into an analyst Dossier (cached 15 min).

    Reuses the same engines the screens use, single-symbol only — cheap when the price store
    is warm. ``use_forecast``/``use_ml`` mirror the Settings AI toggles (plain args so they
    participate in the cache key).
    """
    hist = fetch_symbol_history(symbol, days=400)
    if hist.empty or len(hist) < 26:
        return None
    closes = _to_series(hist, "Close")
    volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0] * len(closes)
    highs = _to_series(hist, "High") if "High" in hist.columns else closes
    lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
    opens = _to_series(hist, "Open") if "Open" in hist.columns else closes

    sig = get_signal_generator(_config).build_signal(
        symbol, closes[-120:], volumes[-120:], highs=highs[-120:], lows=lows[-120:], min_score=0.0)
    info = fetch_symbol_info(symbol)
    try:
        fund = get_fundamentals().get_fundamentals(symbol)
    except Exception:
        fund = {}
    try:
        sentiment = aggregate_news_sentiment(symbol, get_sentiment_analyzer(_config))
    except Exception:
        sentiment = {"positive_pct": None, "count": 0, "headlines": [], "events": []}

    # Named setups + whale footprint straight from the pure detectors (no scan-cache dependency).
    try:
        hits = setups_mod.detect_all(opens, highs, lows, closes, volumes)
    except Exception:
        hits = []
    try:
        whale_row = WhaleDetector(WhaleConfig()).analyze(symbol, opens, highs, lows, closes, volumes)
    except Exception:
        whale_row = None

    regime = None
    try:
        r = get_regime_read()
        regime = {"verdict": r.verdict, "score": r.score, "drivers": list(r.drivers)}
    except Exception:
        pass

    forecast = None
    if use_forecast:
        try:
            forecast = get_forecaster().forecast(closes, horizon=1)
        except Exception:
            forecast = None

    ml_prob = None
    if use_ml and ML_MODEL_PATH.exists():
        model = get_ml_signal_model()
        if model is not None:
            ml_prob = model.predict_proba(ml_signal.extract_features(closes, highs, lows, volumes))

    md = sig.metadata if sig else {}
    votes = {
        "tech": cf.vote_tech(recommend_label(sig.score, sig.signal_type), sig.score) if sig else None,
        "news": cf.vote_news((sentiment.get("positive_pct") or 0) * 100
                             if sentiment.get("count") else None),
        "whale": (cf.vote_whale(whale_row.get("signal"), whale_row.get("whale_score"))
                  if whale_row else None),
    }
    conf_res = cf.score_ticker(votes)

    heads = sentiment.get("headlines") or []
    top_head = (heads[0].get("headline") if heads and isinstance(heads[0], dict)
                else (heads[0] if heads else None))

    return analyst_mod.Dossier(
        symbol=symbol,
        price=info.get("price") or float(closes[-1]),
        change_pct=info.get("change_pct"),
        name=info.get("name") or symbol,
        sector=info.get("sector") or "",
        signal_type=sig.signal_type if sig else None,
        score=sig.score if sig else None,
        rsi=md.get("rsi"), adx=md.get("adx"), macd_hist=md.get("macd_hist"),
        vol_surge=md.get("vol_surge"), atr=md.get("atr"),
        sma20=md.get("sma_20"), sma50=md.get("sma_50"), vwap=md.get("vwap"),
        entry=sig.entry_price if sig else None,
        stop=sig.stop_price if sig else None,
        target=sig.target_price if sig else None,
        reasons=list(md.get("reasons", [])),
        setups=[{"name": h.name, "score": float(h.score), "reasons": list(h.reasons)}
                for h in hits],
        confluence=conf_res,
        regime=regime,
        whale=whale_row,
        forecast=forecast,
        news={"positive_pct": sentiment.get("positive_pct"), "count": sentiment.get("count", 0),
              "top_headline": top_head, "events": list(sentiment.get("events", []))},
        fundamentals={k: fund.get(k) for k in ("pe_ratio", "profit_margin", "roe",
                                               "debt_to_equity", "market_cap")} if fund else None,
        ml_prob=ml_prob,
    )


def render_analyst_brief(symbol: str, config, show_header: bool = True) -> None:
    """The plain-English analyst-brief block: template-generated from the app's own signals,
    optionally rephrased by a local Ollama LLM (Settings → Analyst briefs)."""
    if not st.session_state.get("analyst_briefs", True):
        return
    d = assemble_dossier(symbol, config, use_forecast=_ai_on("ai_forecast"),
                         use_ml=_ai_on("ai_ml_signal"))
    if d is None:
        return
    brief = analyst_mod.build_brief(d)
    md_text = analyst_mod.render_markdown(brief)
    polished = False
    if st.session_state.get("ollama_polish"):
        client = OllamaClient(host=st.session_state.get("ollama_host", ""),
                              model=st.session_state.get("ollama_model", ""))
        with st.spinner("Polishing with local LLM…"):
            out = client.polish(md_text, json.dumps(asdict(d), default=str))
        if out:
            md_text, polished = out, True
    with st.container(border=True):
        if show_header:
            st.markdown("#### 🧠 Analyst brief")
        st.markdown(md_text)
        st.caption(("✨ Polished by local LLM (Ollama) · " if polished else "")
                   + "Based on: " + (", ".join(brief.sources) or "price history only"))


def render_ticker_analysis(symbol: str, config, account_size: float,
                           key_prefix: str = "search") -> None:
    """Full on-demand analysis for ANY ticker: signal, chart, levels, news, forecast."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return
    hist = fetch_symbol_history(symbol, days=120)
    if hist.empty or len(hist) < 26:
        st.warning(f"No usable price history for '{symbol}'. Check the ticker symbol.")
        return
    closes = _to_series(hist, "Close")
    volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0] * len(closes)
    highs = _to_series(hist, "High") if "High" in hist.columns else closes
    lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
    sig = get_signal_generator(config).build_signal(symbol, closes, volumes,
                                                    highs=highs, lows=lows, min_score=0.0)
    info = fetch_symbol_info(symbol)
    price = info.get("price") or (sig.entry_price if sig else closes[-1])
    st.subheader(f"{symbol} — ${price:,.2f}")
    if sig is None:
        st.info("Not enough signal to compute levels; showing news below.")
        render_ticker_news(symbol, config)
        return

    reco = recommend_label(sig.score, sig.signal_type)
    st.markdown(f"**Recommendation:** <span style='{_reco_color(reco)}'>{reco}</span> · "
                f"score {sig.score:.2f} · trend {sig.signal_type}", unsafe_allow_html=True)
    st.plotly_chart(
        create_price_chart(symbol, signal_row=pd.Series(
            {"entry": sig.entry_price, "stop": sig.stop_price, "target": sig.target_price})),
        use_container_width=True, key=f"{key_prefix}_price_{symbol}",
    )
    rr = ((sig.target_price - sig.entry_price) / (sig.entry_price - sig.stop_price)
          if sig.entry_price > sig.stop_price else 0.0)
    c = st.columns(4)
    c[0].metric("Entry", f"${sig.entry_price:.2f}")
    c[1].metric("Stop", f"${sig.stop_price:.2f}",
                delta=f"{(sig.stop_price - sig.entry_price) / sig.entry_price * 100:.2f}%")
    c[2].metric("Target", f"${sig.target_price:.2f}",
                delta=f"+{(sig.target_price - sig.entry_price) / sig.entry_price * 100:.2f}%")
    c[3].metric("R:R", f"{rr:.2f}x")
    note = _limit_note(sig.entry_price, info.get("price") or closes[-1])
    if note:
        st.caption(note)
    m = st.columns(3)
    m[0].metric("RSI", f"{sig.metadata.get('rsi', 0):.2f}")
    m[1].metric("MACD hist", f"{sig.metadata.get('macd_hist', 0):.4f}")
    m[2].metric("Vol surge", f"{sig.metadata.get('vol_surge', 0):.2f}x")
    if sig.metadata.get("reasons"):
        st.write("**Signal reasons:** " + "; ".join(sig.metadata["reasons"]))

    render_analyst_brief(symbol, config)

    st.markdown(f"#### 📰 News for {symbol}")
    render_ticker_news(symbol, config)
    render_forecast_panel(symbol, config, entry=sig.entry_price, stop=sig.stop_price,
                          key_prefix=key_prefix)


def render_market_news(config, top: int = 18) -> None:
    """Overall market news from across free outlets (Google News) with linked articles."""
    from swingtradeapp.nlp import NewsEventClassifier
    items = get_market_news(24)
    if not items:
        st.caption("No market news available right now.")
        return
    analyzer = get_sentiment_analyzer(config)
    ec = get_event_classifier() if _ai_on("ai_events") else None
    pos = neg = 0
    for it in items:
        r = analyzer.analyze_text(it["title"])
        lab = str(r.get("label", "neutral")).lower()
        it["label"], it["score"] = lab, float(r.get("score", 0.5))
        it["event"] = ec.classify(it["title"]) if ec else NewsEventClassifier._classify_keywords(it["title"])
        pos += "pos" in lab
        neg += "neg" in lab
    tone = ("🟢 Bullish" if pos > neg else "🔴 Bearish" if neg > pos else "⚪ Mixed / Neutral")
    st.markdown(f"**Overall market tone: {tone}** — {pos} positive · {neg} negative "
                f"of {len(items)} headlines (Yahoo + Google News)")

    for it in items[:top]:
        emoji = _sentiment_emoji(it["label"])
        ev = f" · `{it['event']}`" if it.get("event") and it["event"] != "other" else ""
        meta = " · ".join(x for x in [it.get("publisher", ""), it.get("time", "")] if x)
        title_md = f"[{it['title']}]({it['url']})" if it.get("url") else it["title"]
        st.markdown(
            f"{emoji} {title_md}  \n"
            f"<span style='color:#888;font-size:0.85em'>score {it['score']:.2f}{ev}"
            f"{' · ' + meta if meta else ''}</span>",
            unsafe_allow_html=True,
        )


# ── Backtest helper ────────────────────────────────────────────────────────────

def run_symbol_backtest(symbol: str, config, days: int = 120) -> Optional[Dict]:
    """Run quick backtest for one symbol. Returns metrics dict or None."""
    try:
        engine = get_backtest_engine(config)
        hist = fetch_symbol_history(symbol, days=days)
        if hist.empty or len(hist) < 30:
            return None
        closes = _to_series(hist, "Close")
        highs = _to_series(hist, "High") if "High" in hist.columns else closes
        lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
        result = engine.run(
            np.array(closes, dtype=float),
            np.array(highs, dtype=float),
            np.array(lows, dtype=float),
        )
        return {
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown_pct,
            "total_return": result.total_return_pct,
            "trades": result.total_trades,
        }
    except Exception as exc:
        errlog.record("symbol_backtest", exc, note=symbol)
        return None


def _to_series(df: pd.DataFrame, col: str) -> List[float]:
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.tolist()


@st.cache_data(ttl=3600, show_spinner=False)
def calibrate_kelly_priors(_config, sample_size: int = 30) -> Optional[Dict]:
    """Aggregate *out-of-sample* edge across a representative universe sample.

    Runs the walk-forward backtester on each sampled symbol and pools the OOS trades, so
    Kelly priors reflect held-out performance rather than the window being traded. The
    leading underscore on ``_config`` tells Streamlit not to hash the config object.
    """
    engine = get_backtest_engine(_config)
    universe = get_screen_universe()[:sample_size]
    wins: List[float] = []
    losses: List[float] = []
    for sym in universe:
        hist = fetch_symbol_history(sym, days=300)
        if hist.empty or len(hist) < 60:
            continue
        closes = np.array(_to_series(hist, "Close"), dtype=float)
        highs = np.array(_to_series(hist, "High") if "High" in hist.columns else closes, dtype=float)
        lows = np.array(_to_series(hist, "Low") if "Low" in hist.columns else closes, dtype=float)
        wf = engine.run_walk_forward(closes, highs, lows)
        for t in wf.trades:
            (wins if t.won else losses).append(abs(t.pnl_pct))
    total = len(wins) + len(losses)
    if total < 20 or not wins or not losses:
        return None
    return {
        "win_rate": len(wins) / total,
        "avg_win": float(np.mean(wins)),
        "avg_loss": float(np.mean(losses)),
        "trades": total,
    }


ML_MODEL_PATH = Path(".data") / "ml_signal_model.joblib"


@st.cache_resource(show_spinner="Training ML signal model (one-time)…")
def get_ml_signal_model(sample_size: int = 40):
    """Walk-forward-trained, calibrated P(up) model (the Screener's optional ML score).

    Loaded from ``.data/ml_signal_model.joblib`` when present; otherwise trained once from a
    universe sample, cached to disk (cross-session) and to the resource cache (in-session).
    Returns ``None`` if scikit-learn or sufficient data is unavailable, so the app falls back to
    the rule-based score. Trained out-of-sample (see ``MLSignalModel.train``).
    """
    cached = MLSignalModel.load(ML_MODEL_PATH)
    if cached is not None and cached.trained:
        return cached
    try:
        histories = []
        for sym in get_screen_universe()[:sample_size]:
            hist = fetch_symbol_history(sym, days=400)
            if hist.empty or len(hist) < 80 or "Volume" not in hist.columns:
                continue
            closes = _to_series(hist, "Close")
            histories.append({
                "symbol": sym, "closes": closes,
                "highs": _to_series(hist, "High") if "High" in hist.columns else closes,
                "lows": _to_series(hist, "Low") if "Low" in hist.columns else closes,
                "volumes": _to_series(hist, "Volume"),
            })
        X, y, times = ml_signal.build_dataset(histories)
        model = MLSignalModel()
        if not model.train(X, y, times) or not model.trained:
            return None
        Path(".data").mkdir(exist_ok=True)
        model.save(ML_MODEL_PATH)
        return model
    except Exception:
        return None


# ── Signal table ───────────────────────────────────────────────────────────────

def _fetch_symbol_extras(symbol: str, fundamentals) -> tuple:
    """(raw_info, fundamentals, headlines) for one symbol — plain network calls only.

    Safe to run from worker threads: no Streamlit elements, no ``st.cache_*``, no ApiBudget.
    Each piece degrades independently so one failed feed doesn't cost the others.
    """
    from swingtradeapp.nlp import _fetch_headlines
    try:
        info = _yf_info(symbol)
    except Exception:
        info = {}
    try:
        fund = fundamentals.get_fundamentals(symbol)
    except Exception:
        fund = {}
    try:
        heads = _fetch_headlines(symbol, 5)
    except Exception:
        heads = []
    return info, fund, heads


def build_signal_table(
    symbols: List[str],
    config,
    threshold: float,
    allocation_scale: float,
    account_size: float,
    filters: Dict,
    include_backtest: bool = False,
    ml_model=None,
) -> pd.DataFrame:
    """Two-pass scan. Pass 1 scores every symbol on (prefetched) price data and applies the
    score gate — CPU-only when the price store is warm. Pass 2 fetches info/fundamentals/news
    concurrently for the handful of survivors, then sentiment scoring (FinBERT) and position
    sizing run back on the main thread."""
    trend_generator = get_signal_generator(config)
    position_sizer = get_sizer(config)
    analyzer = get_sentiment_analyzer(config)
    fundamentals = get_fundamentals()

    # Pass 1 — price-only signal generation + score gate.
    survivors = []
    for symbol in symbols:
        try:
            history = fetch_symbol_history(symbol, days=120)
            if history.empty or len(history) < 26:
                continue

            closes = _to_series(history, "Close")
            volumes = _to_series(history, "Volume") if "Volume" in history.columns else [0] * len(closes)
            highs = _to_series(history, "High") if "High" in history.columns else closes
            lows = _to_series(history, "Low") if "Low" in history.columns else closes

            signal = trend_generator.build_signal(symbol, closes, volumes, highs=highs, lows=lows)
            if signal is None or signal.score < threshold:
                continue
            survivors.append((symbol, signal, closes, highs, lows, volumes))
        except Exception as exc:
            errlog.record("screener_pass1", exc, note=symbol)
            continue

    # Pass 2 — concurrent info/fundamentals/news for survivors only (workers never touch st.*).
    extras: Dict[str, tuple] = {}
    if survivors:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_symbol_extras, sym, fundamentals): sym
                       for sym, *_ in survivors}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    extras[sym] = fut.result()
                except Exception as exc:
                    errlog.record("screener_extras", exc, note=sym)
                    extras[sym] = ({}, {}, [])

    rows = []
    for symbol, signal, closes, highs, lows, volumes in survivors:
        try:
            raw_info, fund, headlines = extras.get(symbol, ({}, {}, []))
            info = _summarize_info(symbol, raw_info)

            # Fundamental filters
            if filters.get("min_market_cap"):
                if not fund.get("market_cap") or fund["market_cap"] < filters["min_market_cap"]:
                    continue
            if filters.get("max_pe") and fund.get("pe_ratio") and fund["pe_ratio"] > filters["max_pe"]:
                continue
            if filters.get("min_dividend_yield"):
                if (fund.get("dividend_yield") or 0) < filters["min_dividend_yield"]:
                    continue
            if filters.get("sectors"):
                if info.get("sector", "Unknown") not in filters["sectors"]:
                    continue

            # Per-symbol backtest is for DISPLAY metrics only (cost-adjusted). Kelly
            # priors are calibrated once, out-of-sample, in calibrate_kelly_priors() —
            # not from this same window (which would be in-sample overfitting).
            bt = run_symbol_backtest(symbol, config, days=120) if include_backtest else None

            sentiment = aggregate_news_sentiment(symbol, analyzer, headlines=headlines)

            # Optional ML probability of an up move over the swing horizon (calibrated, OOS-trained).
            ml_prob = None
            if ml_model is not None:
                ml_prob = ml_model.predict_proba(
                    ml_signal.extract_features(closes, highs, lows, volumes))

            position_size = position_sizer.size_position(
                signal, account_size=account_size, win_prob=ml_prob)
            allocation = float(position_size.fraction) * 100 * allocation_scale

            # Risk-reward ratio
            risk = signal.entry_price - signal.stop_price
            reward = signal.target_price - signal.entry_price
            rr = round(reward / risk, 2) if risk > 0 else 0.0

            row = {
                "symbol": symbol,
                "price": info.get("price") or signal.entry_price,
                "change_pct": info.get("change_pct"),
                "recommendation": recommend_label(signal.score, signal.signal_type),
                "score": signal.score,
                "ml_prob": ml_prob,
                "rsi": signal.metadata.get("rsi"),
                "macd_hist": signal.metadata.get("macd_hist"),
                "atr": signal.metadata.get("atr"),
                "vol_surge": signal.metadata.get("vol_surge"),
                "entry": signal.entry_price,
                "stop": signal.stop_price,
                "target": signal.target_price,
                "risk_reward": rr,
                "allocation_pct": allocation,
                "allocation_usd": round(position_size.dollars * allocation_scale, 0),
                "sentiment_pct": sentiment["positive_pct"] * 100,
                "sector": info.get("sector", "Unknown"),
                "market_cap": fund.get("market_cap"),
                "pe_ratio": fund.get("pe_ratio"),
                "dividend_yield": fund.get("dividend_yield"),
                "reasons": "; ".join(signal.metadata.get("reasons", [])),
                # Backtest columns
                "bt_win_rate": bt["win_rate"] if bt else None,
                "bt_profit_factor": bt["profit_factor"] if bt else None,
                "bt_sharpe": bt["sharpe"] if bt else None,
                "bt_trades": bt["trades"] if bt else None,
            }
            rows.append(row)
        except Exception as exc:
            errlog.record("screener_pass2", exc, note=symbol)
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["score", "allocation_pct", "allocation_usd", "price", "change_pct",
                    "rsi", "macd_hist", "vol_surge", "sentiment_pct", "risk_reward"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Signal Stack: cross-screen conviction board ─────────────────────────────────

# Canonical GICS sector → SPDR sector ETF (covers yfinance's sector-name variants).
SECTOR_TO_ETF = {
    "Technology": "XLK", "Information Technology": "XLK",
    "Health Care": "XLV", "Healthcare": "XLV",
    "Financials": "XLF", "Financial Services": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY", "Consumer Cyclical": "XLY",
    "Consumer Staples": "XLP", "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Materials": "XLB", "Basic Materials": "XLB",
    "Communication Services": "XLC",
}


def yt_social_map() -> Dict[str, tuple]:
    """Per-ticker (bull_pct, mentions) from the persisted YouTube picks store — no network."""
    tally: Dict[str, list] = {}
    for p in load_yt_store().get("picks", []):
        sym = p.get("ticker")
        if not sym:
            continue
        d = tally.setdefault(sym, [0, 0, 0])  # [buy, sell, total]
        d[2] += 1
        if p.get("direction") == "buy":
            d[0] += 1
        elif p.get("direction") == "sell":
            d[1] += 1
    out: Dict[str, tuple] = {}
    for sym, (buy, sell, total) in tally.items():
        decided = buy + sell
        out[sym] = ((buy / decided) if decided else None, total)
    return out


def _vote_arrow(v) -> str:
    if v is None:
        return ""
    return "▲" if v.dir > 0 else ("▼" if v.dir < 0 else "·")


@st.cache_data(ttl=900, show_spinner=False)
def build_signal_stack(_config, n: int = 80) -> pd.DataFrame:
    """Join the cached scan outputs by symbol and score each name by signal confluence.

    The base technical scan (every active name, no score gate) is the only real cost; whale,
    forecast, options, social and sector reads are folded in from cached/persisted sources.
    """
    base = build_signal_table(get_screen_universe()[:n], _config, threshold=0.0,
                              allocation_scale=1.0, account_size=100000.0,
                              filters={}, include_backtest=False)
    if base.empty:
        return pd.DataFrame()

    whale = scan_whale_activity(sample_size=n)
    whale_map = {r["symbol"]: r for _, r in whale.iterrows()} if not whale.empty else {}
    fc = predict_tomorrow(sample_size=n)
    fc_map = {r["Symbol"]: r for _, r in fc.iterrows()} if not fc.empty else {}
    opt = scan_options_flow(sample_size=min(n, 25))
    opt_map = {r["Symbol"]: r for _, r in opt.iterrows()} if not opt.empty else {}
    social = yt_social_map()
    sect_etf = fetch_etf_table(_config, ("Sector",))
    etf_change = ({r["Symbol"]: r["Change %"] for _, r in sect_etf.iterrows()}
                  if not sect_etf.empty else {})

    rows: List[Dict] = []
    for _, b in base.iterrows():
        sym = b["symbol"]
        w, f, o = whale_map.get(sym), fc_map.get(sym), opt_map.get(sym)
        soc_bull, soc_mentions = social.get(sym, (None, 0))
        sector_name = b.get("sector")
        sect_chg = etf_change.get(SECTOR_TO_ETF.get(sector_name)) if sector_name else None

        votes = {
            "tech": cf.vote_tech(b.get("recommendation"), b.get("score")),
            "whale": cf.vote_whale(w["signal"], w["whale_score"]) if w is not None else None,
            "forecast": cf.vote_forecast(f["Direction"], f["Pred Return %"]) if f is not None else None,
            "options": cf.vote_options(o["Sentiment"], o["# Unusual"]) if o is not None else None,
            "news": cf.vote_news(b.get("sentiment_pct")),
            "social": cf.vote_social(soc_bull, soc_mentions),
            "sector": cf.vote_sector(sect_chg),
        }
        res = cf.score_ticker(votes)
        sign = 1 if res["direction"] == "long" else (-1 if res["direction"] == "short" else 0)
        why = [k.title() for k in cf.SIGNAL_ORDER
               if votes.get(k) and sign and votes[k].dir == sign]

        row = {
            "Symbol": sym, "Price": b.get("price"),
            "Direction": res["direction"], "Conviction": res["conviction"],
            "Confluence": res["confluence"], "Coverage": res["coverage"],
            "_agree": res["agree_n"], "_net": res["net"],
            "Why": ", ".join(why), "Sector name": sector_name,
        }
        for k in cf.SIGNAL_ORDER:
            row[k.title()] = _vote_arrow(votes[k])
            row[f"_d_{k}"] = votes[k].detail if votes[k] else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Conviction", "_agree"], ascending=False).reset_index(drop=True)
    return df


# ── Morning Insights helpers (personal, decisive, any-time-of-day) ──────────────

def _market_phase() -> tuple:
    """(phase, greeting, data_session) — canonical source is ``swingtradeapp.clock``."""
    return clock.market_phase()


@st.cache_data(ttl=300, show_spinner=False)
def score_symbols(_config, symbols: tuple) -> pd.DataFrame:
    """Per-symbol technical read for a small, explicit list — the personal-briefing engine.

    One row per scorable symbol: last/prev close, last-session move %, recommendation, score, RSI,
    and entry/stop/target levels (+ the signal's reasons). Bounded by ``len(symbols)``, cached 5 min.
    Indexed by Symbol so callers can ``.loc[sym]`` cheaply.
    """
    gen = get_signal_generator(_config)
    rows: List[Dict] = []
    for sym in symbols:
        hist = fetch_symbol_history(sym, days=120)
        if hist.empty or len(hist) < 26:
            continue
        closes = _to_series(hist, "Close")
        volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0] * len(closes)
        highs = _to_series(hist, "High") if "High" in hist.columns else closes
        lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
        sig = gen.build_signal(sym, closes, volumes, highs=highs, lows=lows)
        if sig is None:
            continue
        last = float(closes[-1])
        prev = float(closes[-2]) if len(closes) > 1 else last
        risk = sig.entry_price - sig.stop_price
        reward = sig.target_price - sig.entry_price
        rows.append({
            "Symbol": sym, "Last": last, "Move %": (last - prev) / prev * 100.0 if prev else 0.0,
            "Reco": _recommendation(sig), "Score": sig.score, "RSI": sig.metadata.get("rsi"),
            "Entry": sig.entry_price, "Stop": sig.stop_price, "Target": sig.target_price,
            "R:R": round(reward / risk, 2) if risk > 0 else 0.0,
            "Why": "; ".join(sig.metadata.get("reasons", []) or [])[:80],
        })
    df = pd.DataFrame(rows)
    return df.set_index("Symbol", drop=False) if not df.empty else df


@st.cache_data(ttl=600, show_spinner=False)
def names_headlines(symbols: tuple) -> Dict[str, tuple]:
    """{symbol: (latest_headline, earnings_imminent)} via the fast Yahoo-only feed (bulk-safe)."""
    from swingtradeapp.nlp import NewsEventClassifier, _fetch_headlines
    out: Dict[str, tuple] = {}
    for sym in symbols:
        try:
            heads = _fetch_headlines(sym, 4)
        except Exception:
            heads = []
        earnings = any(NewsEventClassifier._classify_keywords(h) == "earnings" for h in heads)
        out[sym] = (heads[0] if heads else "", earnings)
    return out


def _eval_alerts(alerts: list, price: float, rsi) -> str:
    """Which of a symbol's alerts are currently triggered → short label ('' if none)."""
    fired = []
    for a in alerts or []:
        if not a.get("active", True):
            continue
        t, comp, val = a.get("type"), a.get("comparison"), a.get("value")
        cur = price if t == "price" else (rsi if t == "rsi" else None)
        if cur is None or val is None:
            continue
        if (comp == "above" and cur >= val) or (comp == "below" and cur <= val):
            fired.append(f"{t} {comp} {val:g}")
    return " · ".join(fired)


@st.cache_data(ttl=900, show_spinner=False)
def _spy_trend() -> str:
    """SPY daily trend vs its 20/50-day SMAs → 'up' | 'down' | 'mixed' | 'unknown'."""
    hist = fetch_symbol_history("SPY", days=90)
    if hist.empty or len(hist) < 50:
        return "unknown"
    c = pd.Series(_to_series(hist, "Close"))
    last, s20, s50 = c.iloc[-1], c.rolling(20).mean().iloc[-1], c.rolling(50).mean().iloc[-1]
    if last > s20 > s50:
        return "up"
    if last < s20 < s50:
        return "down"
    return "mixed"


# ── Alpha Engine data (curated universe + cached panel/benchmark fetchers) ───────

# Curated, liquid, large-cap cross-section. Deliberately fixed (not "today's most actives") so the
# backtest is reproducible; liquid mega/large caps also minimise (never eliminate) survivorship bias.
ALPHA_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "ORCL", "CRM", "ADBE", "AMD",
    "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU", "ADI", "LRCX", "KLAC",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX",
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN", "BMY",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "SPGI", "BLK", "C", "SCHW",
    "CAT", "GE", "BA", "HON", "UPS", "RTX", "UNP", "DE", "LMT",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "LIN", "SHW", "FCX", "NEE", "DUK", "SO",
]


_PANEL_SKIPPED_CHUNKS = {"n": 0}  # last _download_field run's silently-skipped chunk count


def _download_field(symbols: list, years: int, field: str, chunk: int = 25) -> pd.DataFrame:
    """Download one OHLCV ``field`` as a wide (dates × symbols) frame, in chunks.

    Large single-call batches get rate-limited by Yahoo and return half-empty — chunking keeps each
    request small and reliable, then we concat. A failed chunk is retried once, then skipped (not
    fatal); the skip count is surfaced on the Alpha Engine page via ``_PANEL_SKIPPED_CHUNKS``.
    """
    import yfinance as yf
    frames = []
    skipped = 0
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(part, period=f"{years}y", auto_adjust=True, progress=False)
            except Exception:
                raw = None
            if raw is not None and not raw.empty:
                break
            time.sleep(1.0)
        if raw is None or raw.empty:
            skipped += 1
            errlog.record("panel_download", note=f"chunk of {len(part)} symbols skipped ({field})")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            if field in raw.columns.get_level_values(0):
                frames.append(raw[field])
        elif field in raw.columns:                       # single-ticker chunk → flat columns
            f = raw[[field]].copy()
            f.columns = part[:1]
            frames.append(f)
    _PANEL_SKIPPED_CHUNKS["n"] = skipped
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def _min_panel_cols(n_requested: int) -> int:
    """How many symbols a panel must have to be considered healthy (vs a throttled partial)."""
    return n_requested if n_requested < 10 else max(10, int(n_requested * 0.5))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_panel(symbols: tuple, years: int) -> pd.DataFrame:
    """Adjusted-close price panel (dates × symbols) for the Alpha Engine.

    Lake-backed (≤18h), chunked download. Refuses to **accept or cache a degenerate** (too-few-column)
    panel, so a throttled feed can't poison the lake and make every later run fail.
    """
    min_cols = _min_panel_cols(len(symbols))
    key = datalake.panel_key("prices", symbols, years)
    cached = datalake.load_panel(key, max_age_hours=18)
    if cached is not None and cached.shape[1] >= min_cols:
        return cached
    close = _download_field(list(symbols), years, "Close")
    if close.empty:
        return pd.DataFrame()
    close = close.dropna(how="all").dropna(axis=1, thresh=int(len(close) * 0.7)).ffill()
    if close.shape[1] >= min_cols:                        # only persist a healthy panel
        datalake.save_panel(key, close)
    return close


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close_series(symbol: str, years: int) -> pd.Series:
    """Single-symbol adjusted-close series (for SPY benchmark and ^VIX regime), cached 1h."""
    import yfinance as yf
    try:
        raw = yf.download(symbol, period=f"{years}y", auto_adjust=True, progress=False)
    except Exception:
        return pd.Series(dtype=float)
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    c = raw["Close"]
    return c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c


def _vix_regime(vix: pd.Series, calm: float = 18.0, stress: float = 35.0, floor: float = 0.4) -> pd.Series:
    """Map VIX → gross-exposure multiplier in [floor, 1]: full risk when calm, scaled down into stress."""
    r = 1.0 - (vix - calm) / (stress - calm) * (1.0 - floor)
    return r.clip(lower=floor, upper=1.0)


# GICS-ish sector map for the curated universe (used for sector-neutral ranking).
ALPHA_SECTORS = {
    **{s: "Technology" for s in ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD",
                                 "INTC", "CSCO", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU",
                                 "ADI", "LRCX", "KLAC"]},
    **{s: "Communication" for s in ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"]},
    **{s: "Cons. Disc." for s in ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX"]},
    **{s: "Cons. Staples" for s in ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL"]},
    **{s: "Health Care" for s in ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
                                  "AMGN", "BMY"]},
    **{s: "Financials" for s in ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "SPGI", "BLK",
                                 "C", "SCHW"]},
    **{s: "Industrials" for s in ["CAT", "GE", "BA", "HON", "UPS", "RTX", "UNP", "DE", "LMT"]},
    **{s: "Energy" for s in ["XOM", "CVX", "COP", "SLB", "EOG"]},
    **{s: "Materials" for s in ["LIN", "SHW", "FCX"]},
    **{s: "Utilities" for s in ["NEE", "DUK", "SO"]},
}

# Geopolitical / macro stress scenarios. Shocks are expressed as returns of factor proxies:
# mkt=SPY, oil=CL=F, rates=TLT (long bond price; rates-up = TLT down), usd=UUP.
ALPHA_SCENARIOS = {
    "Geopolitical risk-off": {"mkt": -0.08, "oil": 0.15, "rates": 0.03, "usd": 0.03},
    "Oil shock (war/OPEC)": {"oil": 0.30, "mkt": -0.05, "usd": 0.02},
    "Hawkish rate shock": {"rates": -0.08, "mkt": -0.05},
    "Credit / growth scare": {"mkt": -0.12, "rates": 0.05, "oil": -0.10},
    "Risk-on rally": {"mkt": 0.05, "oil": 0.02, "rates": -0.01},
}
ALPHA_FACTOR_PROXIES = {"mkt": "SPY", "oil": "CL=F", "rates": "TLT", "usd": "UUP"}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_volume_panel(symbols: tuple, years: int) -> pd.DataFrame:
    """Share-volume panel (dates × symbols) for ADV-based impact costs. Lake-backed (≤18h), chunked."""
    min_cols = _min_panel_cols(len(symbols))
    key = datalake.panel_key("volume", symbols, years)
    cached = datalake.load_panel(key, max_age_hours=18)
    if cached is not None and cached.shape[1] >= min_cols:
        return cached
    v = _download_field(list(symbols), years, "Volume")
    if v.empty:
        return pd.DataFrame()
    if v.shape[1] >= min_cols:
        datalake.save_panel(key, v)
    return v


# Plain-English phrasing for each factor (positive z-score = attractive to own); themes group
# near-synonyms so a reason never says "low volatility and low volatility".
ALPHA_FACTOR_PHRASES = {
    "momentum": "strong 12-month momentum",
    "residual_momentum": "strong stock-specific momentum",
    "trend": "trending above its 200-day average",
    "high_proximity": "pressing its 52-week high",
    "low_volatility": "low, steady volatility",
    "idiosyncratic_vol": "low stock-specific volatility",
    "reversal": "bouncing after a recent pullback",
}
ALPHA_FACTOR_THEME = {
    "momentum": "momentum", "residual_momentum": "momentum",
    "trend": "trend", "high_proximity": "high",
    "low_volatility": "lowvol", "idiosyncratic_vol": "lowvol",
    "reversal": "reversal",
}


def alpha_factor_reasons(panels: dict, symbols) -> dict:
    """For each symbol, a plain 'why' from its top 1–2 standout factors (one per theme) on the latest date."""
    facs = [k for k in panels if k != "composite"]
    if not facs:
        return {}
    last = panels["composite"].index[-1]
    out = {}
    for sym in symbols:
        scored = []
        for f in facs:
            try:
                z = panels[f].loc[last, sym]
            except Exception:
                z = float("nan")
            if z == z and f in ALPHA_FACTOR_PHRASES:
                scored.append((float(z), f))
        scored.sort(reverse=True)
        picks, seen = [], set()
        for z, f in scored:
            if z <= 0.25:
                break
            theme = ALPHA_FACTOR_THEME.get(f, f)
            if theme in seen:
                continue
            seen.add(theme)
            picks.append(ALPHA_FACTOR_PHRASES[f])
            if len(picks) == 2:
                break
        reason = " and ".join(picks) if picks else "ranks well across the blended model"
        out[sym] = reason[0].upper() + reason[1:]
    return out


def render_alpha_simple_plan(res, account, m, bm, dsr, pbo, reasons=None) -> None:
    """Translate the engine's quant output into a plain 'buy this much' plan for a normal user."""
    def g(d, k):
        v = d.get(k) if d else None
        return v if v is not None and v == v else float("nan")

    book = res.get("latest_book")
    book = book[book["Side"] == "LONG"].copy() if book is not None and not book.empty else None
    if book is None or book.empty:
        st.info("🟡 **No clear buys right now** — the model isn't finding strong setups. Holding cash "
                "is a perfectly fine answer some months.")
        return

    cagr, spy, mdd = g(m, "CAGR"), g(bm, "CAGR"), g(m, "MaxDD")
    conf = ("high" if g(dsr, "DSR") >= 0.95 and g(pbo, "PBO") <= 0.35
            else "low" if (g(dsr, "DSR") < 0.80 or g(pbo, "PBO") > 0.60) else "medium")
    st.success(f"📋 **This month's plan — {len(book)} stocks to buy** for a ${account:,.0f} account")
    bullets = []
    if cagr == cagr and spy == spy:
        tail = f", with dips as deep as ~**{abs(mdd):.0%}** along the way." if mdd == mdd else "."
        bullets.append(f"Historically this approach made about **{cagr:.0%}/yr** vs **{spy:.0%}/yr** "
                       f"for the S&P 500{tail}")
    bullets.append(f"Confidence it's a real edge (not luck): **{conf}**.")
    bullets.append("It's a **monthly** system — check back in ~a month and update to the new list.")
    st.markdown("\n".join(f"- {b}" for b in bullets))

    wsum = float(book["Weight"].sum()) or 1.0
    rows, invested = [], 0.0
    for _, r in book.iterrows():
        w = float(r["Weight"]) / wsum
        price = float(r["Price"]) if r["Price"] == r["Price"] else 0.0
        dollars = w * account
        sh = int(dollars // price) if price > 0 else 0
        invested += sh * price
        rows.append({"Ticker": r["Symbol"], "Buy ≈": dollars, "Shares": sh,
                     "Price": price, "Cost": sh * price, "Weight": w,
                     "Why": (reasons or {}).get(r["Symbol"], "")})
    plan = pd.DataFrame(rows)
    buyable = plan[plan["Shares"] > 0].copy()
    skipped = plan[plan["Shares"] == 0]
    show = buyable if not buyable.empty else plan
    st.dataframe(
        show[["Ticker", "Shares", "Price", "Cost", "Why"]].style.format(
            {"Price": "${:.2f}", "Cost": "${:,.0f}"}),
        use_container_width=True, hide_index=True, height=min(460, 80 + 34 * len(show)))
    note = ""
    if not skipped.empty:
        names = ", ".join(skipped["Ticker"].head(4))
        note = (f" · {len(skipped)} name(s) too pricey for this budget ({names}) — a broker with "
                "**fractional shares** would let you include them")
    st.caption(f"Buy about **${invested:,.0f}** across **{len(buyable)}** stocks · ~${account-invested:,.0f} "
               f"left as cash (whole-share rounding){note}. **Educational, not financial advice** — "
               "start small and never invest money you can't afford to lose.")
    st.download_button("📥 Download my plan (CSV)", show.to_csv(index=False),
                       file_name=f"alpha_plan_{datetime.now():%Y%m%d}.csv", mime="text/csv")


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _source_badge(fig: go.Figure, source: str, note: Optional[str] = None) -> None:
    """Stamp a visible data-source badge on a chart so paid (Polygon) vs free (Yahoo) is obvious
    at a glance — green 📡 for Polygon, grey 🆓 for Yahoo."""
    is_polygon = "Polygon" in (source or "")
    txt = (f"📡 Data: {source}" if is_polygon else f"🆓 Data: {source}")
    if note:
        txt += f"  ·  {note}"
    fig.add_annotation(xref="paper", yref="paper", x=0.005, y=0.99, xanchor="left", yanchor="top",
                       showarrow=False, text=txt, font=dict(size=12, color="#ffffff"),
                       bgcolor=("#00a843" if is_polygon else "#5a5a5a"), borderpad=4, opacity=0.93)


def create_price_chart(symbol: str, signal_row: Optional[pd.Series] = None, days: int = 90) -> go.Figure:
    hist, _src, _note = fetch_history_ondemand(symbol, days=days)
    if hist.empty:
        return go.Figure()
    closes = _to_series(hist, "Close")
    dates = hist.index
    sma20 = pd.Series(closes).rolling(20).mean()
    sma50 = pd.Series(closes).rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closes, mode="lines", name="Close",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=sma20, mode="lines", name="SMA 20",
                             line=dict(color="orange", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=dates, y=sma50, mode="lines", name="SMA 50",
                             line=dict(color="red", width=1, dash="dash")))

    if signal_row is not None:
        last_date = dates[-1]
        for level, color, label in [
            ("entry", "green", "Entry"),
            ("stop", "red", "Stop"),
            ("target", "blue", "Target"),
        ]:
            val = signal_row.get(level)
            if val:
                fig.add_hline(y=val, line_color=color, line_dash="dot",
                              annotation_text=f"{label} ${val:.2f}", annotation_position="right")

    fig.update_layout(
        title=f"{symbol} — Price & Signals · {_src}",
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified", height=450,
    )
    _source_badge(fig, _src, _note)
    return fig


def create_setup_chart(symbol: str, row, days: int = 180) -> go.Figure:
    """Price chart for a Setup Scanner hit, overlaid with the structure that drove it:
    unmitigated Fair-Value-Gap zones, any double-bottom level/neckline, and the entry/stop/target.
    """
    hist, _src, _note = fetch_history_ondemand(symbol, days=days)
    if hist.empty:
        return go.Figure()
    closes = _to_series(hist, "Close")
    highs = _to_series(hist, "High") if "High" in hist.columns else closes
    lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
    dates = hist.index
    sma20 = pd.Series(closes).rolling(20).mean()
    sma50 = pd.Series(closes).rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closes, mode="lines", name="Close",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=dates, y=sma20, mode="lines", name="SMA 20",
                             line=dict(color="orange", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=dates, y=sma50, mode="lines", name="SMA 50",
                             line=dict(color="red", width=1, dash="dash")))

    # Unmitigated Fair-Value-Gap zones (institutional defence zones), most recent few.
    try:
        zones = [z for z in patterns_mod.detect_fair_value_gaps(highs, lows, closes)
                 if not z.mitigated and not z.inverted][-8:]
        for z in zones:
            color = "rgba(0,200,81,0.10)" if z.kind == "bullish" else "rgba(255,68,68,0.10)"
            fig.add_hrect(y0=z.bottom, y1=z.top, fillcolor=color, line_width=0, layer="below")
    except Exception:
        pass

    # Double-bottom support + neckline if present.
    try:
        dp = patterns_mod.detect_double_bottom(highs, lows)
        if dp is not None:
            fig.add_hline(y=dp.level, line_color="#9467bd", line_dash="dot",
                          annotation_text=f"DB support ${dp.level:.2f}", annotation_position="left")
            fig.add_hline(y=dp.neckline, line_color="#9467bd", line_dash="dash",
                          annotation_text=f"Neckline ${dp.neckline:.2f}", annotation_position="left")
    except Exception:
        pass

    # Entry / stop / target from the hit.
    for level, color, label in [("entry", "green", "Entry"), ("stop", "red", "Stop"),
                                ("target", "blue", "Target")]:
        try:
            val = float(row.get(level))
        except (TypeError, ValueError):
            val = None
        if val:
            fig.add_hline(y=val, line_color=color, line_dash="dot",
                          annotation_text=f"{label} ${val:.2f}", annotation_position="right")

    setup_name = row.get("setup", "") if hasattr(row, "get") else ""
    fig.update_layout(title=f"{symbol} — {setup_name} · {_src}".replace(" —  · ", " · "),
                      xaxis_title="Date", yaxis_title="Price ($)",
                      hovermode="x unified", height=460)
    _source_badge(fig, _src, _note)
    return fig


def create_market_heatmap(df: pd.DataFrame) -> go.Figure:
    """FinViz-style market map: tiles sized by market cap, colored by daily % change."""
    if df.empty:
        return go.Figure().add_annotation(text="No data")
    df = df.copy()
    df["sector"] = df["sector"].fillna("Unknown")
    # Embed the 2-decimal % directly in the leaf label so every tile shows it reliably
    # (treemaps blank/aggregate custom_data on parent tiles, which dropped the numbers).
    df["tile"] = df.apply(lambda r: f"{r['symbol']}<br>{r['change_pct']:+.2f}%", axis=1)
    # Symmetric color scale so 0% is the neutral midpoint (green up / red down).
    cap = max(2.0, float(np.nanpercentile(df["change_pct"].abs(), 90)))
    try:
        fig = px.treemap(
            df,
            path=[px.Constant("Market"), "sector", "tile"],
            values="market_cap",
            color="change_pct",
            color_continuous_scale="RdYlGn",
            range_color=[-cap, cap],
        )
        fig.update_traces(
            textposition="middle center",
            texttemplate="%{label}",
            hovertemplate="<b>%{label}</b><extra></extra>",
        )
        fig.update_layout(height=650, margin=dict(t=30, l=10, r=10, b=10),
                          coloraxis_colorbar_title="% chg",
                          uniformtext=dict(minsize=8, mode="hide"))
        return fig
    except Exception as e:
        return go.Figure().add_annotation(text=f"Error: {e}")


def create_sector_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty or "sector" not in df.columns:
        return go.Figure()
    stats = df.groupby("sector").agg(
        avg_change=("change_pct", "mean"), count=("symbol", "count")
    ).reset_index().sort_values("avg_change", ascending=False)
    fig = px.bar(stats, x="sector", y="avg_change", hover_data=["count"],
                 color="avg_change", color_continuous_scale="RdYlGn",
                 title="Avg Daily % Change by Sector", labels={"avg_change": "Avg % chg"})
    fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    return fig


# ── P&L Tracker helpers ────────────────────────────────────────────────────────

TRADE_JOURNAL_PATH = Path(".data/trade_journal.json")


def load_journal() -> List[Dict]:
    return jsonstore.read_json(TRADE_JOURNAL_PATH, default=[])


def save_journal(trades: List[Dict]) -> None:
    jsonstore.atomic_write_json(TRADE_JOURNAL_PATH, trades)


def add_trade(trades: List[Dict], symbol: str, side: str, entry: float,
              stop: float, target: float, qty: int, score: float) -> None:
    trades.append({
        "id": len(trades) + 1,
        "symbol": symbol,
        "side": side,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "qty": qty,
        "score": score,
        "entry_date": datetime.now().isoformat(),
        "exit_price": None,
        "exit_date": None,
        "status": "open",
        "pnl": None,
        "pnl_pct": None,
    })


def close_trade(trades: List[Dict], trade_id: int, exit_price: float) -> None:
    for t in trades:
        if t["id"] == trade_id and t["status"] == "open":
            t["exit_price"] = exit_price
            t["exit_date"] = datetime.now().isoformat()
            t["status"] = "closed"
            t["pnl"] = round((exit_price - t["entry_price"]) * t["qty"], 2)
            t["pnl_pct"] = round((exit_price - t["entry_price"]) / t["entry_price"] * 100, 2)
            break


# ── Main dashboard ─────────────────────────────────────────────────────────────



# Export everything (including _underscore helpers) for the app/screens star-import.
__all__ = [_n for _n in list(globals()) if not _n.startswith('__')]
