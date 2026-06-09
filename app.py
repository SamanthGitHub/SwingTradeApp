"""FinViz-like dashboard: screener, heat maps, watchlists, alerts, backtest metrics, P&L tracker."""

import json
import math
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
    aggregate_news_sentiment,
)
from swingtradeapp.providers import ProviderFactory
from swingtradeapp.retry import with_retry
from swingtradeapp.risk import BayesianKellySizer
from swingtradeapp.signals import TrendSignalGenerator
from swingtradeapp.tickers import get_raw_screen, get_screening_universe, get_tradable_universe
from swingtradeapp import ui
from swingtradeapp import youtube as yt
from swingtradeapp.universe import PreMarketScanner, UniverseFilter
from swingtradeapp.watchlist import WatchlistManager
from swingtradeapp.whale import WhaleConfig, WhaleDetector

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


def _ai_on(key: str) -> bool:
    """Whether an AI feature toggle is enabled on the Settings page."""
    return bool(st.session_state.get(key, False))


# ── Data helpers ───────────────────────────────────────────────────────────────

@with_retry()
def _yf_download(symbol: str, start, end) -> pd.DataFrame:
    return yf.download(symbol, start=start, end=end, progress=False, threads=False)


@with_retry(retries=2)
def _yf_info(symbol: str) -> Dict:
    t = yf.Ticker(symbol)
    return t.info if hasattr(t, "info") else {}


@st.cache_data(ttl=3600)
def fetch_symbol_history(symbol: str, days: int = 90) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        data = _yf_download(symbol, start, end)
        if data.empty:
            return pd.DataFrame()
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
        return data[cols].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_symbol_info(symbol: str) -> Dict:
    try:
        info = _yf_info(symbol)
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        prev = info.get("previousClose")
        change = (price - prev) / prev * 100.0 if price and prev else info.get("regularMarketChangePercent")
        return {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "price": price,
            "change_pct": change,
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return {"sector": "Unknown", "industry": "Unknown", "price": None,
                "change_pct": None, "market_cap": None}


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


def load_yt_store() -> Dict:
    if YT_PICKS_PATH.exists():
        try:
            with open(YT_PICKS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"picks": []}


def save_yt_store(store: Dict) -> None:
    YT_PICKS_PATH.parent.mkdir(exist_ok=True)
    with open(YT_PICKS_PATH, "w") as f:
        json.dump(store, f, indent=2)


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
    return {
        "iv_rank": oa.fetch_iv_rank(symbol),
        "current_iv": current_iv,
        "pc_ratio": oa.fetch_put_call_ratio_symbol(symbol),
        "unusual": oa.detect_unusual_volume(symbol),
        "earnings_date": earn,
        "iv_crush": oa.estimate_iv_crush(symbol, current_iv) if current_iv else None,
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


def _hl_pct(v) -> str:
    """Green/red text color for a signed numeric percentage (dataframe styler)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "color:#00c851" if f > 0 else ("color:#ff4444" if f < 0 else "")


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
                        min_dollar_vol_m: float = 50.0) -> pd.DataFrame:
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


# ETF categories sourced from ETFScreener (single source of truth).
ETF_CATEGORIES = {
    "Sector": ETFScreener.SECTOR_ETFS,
    "Broad Market": ETFScreener.BROAD_MARKET_ETFS,
    "Volatility": ETFScreener.VOLATILITY_ETFS,
    "Commodities": ETFScreener.COMMODITY_ETFS,
    "Bonds": ETFScreener.BOND_ETFS,
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_etf_table(_config) -> pd.DataFrame:
    """Signals + daily change for every tracked ETF, tagged by category."""
    gen = get_signal_generator(_config)
    rows = []
    for category, mapping in ETF_CATEGORIES.items():
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
    m = st.columns(3)
    m[0].metric("RSI", f"{sig.metadata.get('rsi', 0):.2f}")
    m[1].metric("MACD hist", f"{sig.metadata.get('macd_hist', 0):.4f}")
    m[2].metric("Vol surge", f"{sig.metadata.get('vol_surge', 0):.2f}x")
    if sig.metadata.get("reasons"):
        st.write("**Signal reasons:** " + "; ".join(sig.metadata["reasons"]))

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
    except Exception:
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


# ── Signal table ───────────────────────────────────────────────────────────────

def build_signal_table(
    symbols: List[str],
    config,
    threshold: float,
    allocation_scale: float,
    account_size: float,
    filters: Dict,
    include_backtest: bool = False,
) -> pd.DataFrame:
    trend_generator = get_signal_generator(config)
    position_sizer = get_sizer(config)
    analyzer = get_sentiment_analyzer(config)
    fundamentals = get_fundamentals()

    rows = []
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

            info = fetch_symbol_info(symbol)
            fund = fundamentals.get_fundamentals(symbol)

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

            sentiment = aggregate_news_sentiment(symbol, analyzer)
            position_size = position_sizer.size_position(signal, account_size=account_size)
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
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["score", "allocation_pct", "allocation_usd", "price", "change_pct",
                    "rsi", "macd_hist", "vol_surge", "sentiment_pct", "risk_reward"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Chart helpers ──────────────────────────────────────────────────────────────

def create_price_chart(symbol: str, signal_row: Optional[pd.Series] = None, days: int = 90) -> go.Figure:
    hist = fetch_symbol_history(symbol, days=days)
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
        title=f"{symbol} — Price & Signals",
        xaxis_title="Date", yaxis_title="Price ($)",
        hovermode="x unified", height=450,
    )
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
    if TRADE_JOURNAL_PATH.exists():
        try:
            with open(TRADE_JOURNAL_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_journal(trades: List[Dict]) -> None:
    TRADE_JOURNAL_PATH.parent.mkdir(exist_ok=True)
    with open(TRADE_JOURNAL_PATH, "w") as f:
        json.dump(trades, f, indent=2)


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

def run_dashboard() -> None:
    st.set_page_config(
        page_title="SwingTrade Pro Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ui.inject_theme()
    ui.render_header()

    config = get_config()
    # Apply live cost-model overrides set on the Settings page (survive cache clears).
    if "slippage_bps" in st.session_state:
        config.slippage_bps = st.session_state["slippage_bps"]
        config.commission_bps = st.session_state["commission_bps"]
    watchlist_mgr = get_watchlist_manager()

    page = ui.render_nav()

    with st.sidebar:
        st.divider()
        account_size = float(st.number_input("Account size ($)", value=100_000, step=10_000,
                                             min_value=1000))

    # ═══════════════════════ SCREENER ════════════════════════════════════════
    if page == "Screener":
        st.header("Advanced Stock Screener")

        # ── Look up any ticker (independent of the scan) ─────────────────────
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            search_ticker = st.text_input("🔍 Look up any ticker", placeholder="e.g. AAPL, NVDA, TSLA",
                                          key="screener_search").strip().upper()
        with sc2:
            st.write("")
            do_search = st.button("Analyze", use_container_width=True)
        if search_ticker and (do_search or st.session_state.get("last_search") == search_ticker):
            st.session_state["last_search"] = search_ticker
            with st.spinner(f"Analyzing {search_ticker}…"):
                render_ticker_analysis(search_ticker, config, account_size)
            _render_legend()
            st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sample_size = st.number_input("Universe size", 10, 500, 100, 10)
        with c2:
            min_score = st.slider("Min signal score", 0.3, 1.0, 0.45, 0.05)
        with c3:
            allocation_scale = st.slider("Allocation scale", 0.25, 4.0, 1.0, 0.25)
        with c4:
            include_bt = st.checkbox("Include backtest metrics", value=False,
                                     help="Slower — runs per-symbol backtest to show win rate/Sharpe")

        with st.expander("Advanced Filters"):
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                mc_input = st.number_input("Min market cap ($B)", 0.0, 5000.0, 0.0, 1.0)
                min_market_cap = mc_input * 1e9 if mc_input > 0 else None
            with fc2:
                max_pe = st.number_input("Max P/E ratio", 0.0, 500.0, 0.0, 5.0)
            with fc3:
                min_div = st.number_input("Min dividend yield (%)", 0.0, 20.0, 0.0, 0.5) / 100
            with fc4:
                sectors = st.multiselect("Sectors", [
                    "Technology", "Healthcare", "Financials", "Energy",
                    "Consumer Cyclical", "Industrials", "Materials", "Utilities",
                ])
            filters = {
                "min_market_cap": min_market_cap,
                "max_pe": max_pe if max_pe > 0 else None,
                "min_dividend_yield": min_div if min_div > 0 else None,
                "sectors": sectors if sectors else None,
            }

        if st.button("Refresh Scan", use_container_width=False):
            st.cache_data.clear()

        tickers = get_screen_universe()
        scan_universe = tickers[:sample_size]

        # Calibrate Kelly priors once from out-of-sample walk-forward edge (cached 1h).
        with st.spinner("Calibrating position sizer (out-of-sample)…"):
            priors = calibrate_kelly_priors(config)
        if priors:
            get_sizer(config).update_from_backtest(
                win_rate=priors["win_rate"],
                avg_win_pct=priors["avg_win"],
                avg_loss_pct=priors["avg_loss"],
                n_trades=priors["trades"],
            )
            st.caption(f"Sizer calibrated on {priors['trades']} out-of-sample trades · "
                       f"win rate {priors['win_rate']:.2%} · "
                       f"avg win {priors['avg_win']:.2%} / avg loss {priors['avg_loss']:.2%}")
        else:
            st.caption("Sizer using default priors (insufficient out-of-sample trades to calibrate).")

        with st.spinner(f"Scanning {len(scan_universe)} symbols…"):
            df = build_signal_table(scan_universe, config, min_score, allocation_scale,
                                    account_size, filters, include_backtest=include_bt)

        if df.empty:
            st.warning("No signals passed filters. Lower the min score or expand the universe.")
        else:
            # Summary bar
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Signals", len(df))
            m2.metric("Strong (≥0.7)", int((df["score"] >= 0.7).sum()))
            m3.metric("Avg Score", f"{df['score'].mean():.2f}")
            m4.metric("Avg R:R", f"{df['risk_reward'].mean():.1f}x")
            m5.metric("Avg Sentiment", f"{df['sentiment_pct'].mean():.2f}%")

            # Display table
            disp_cols = ["symbol", "price", "change_pct", "recommendation", "score", "rsi",
                         "macd_hist", "vol_surge", "entry", "stop", "target", "risk_reward",
                         "allocation_pct", "allocation_usd", "sentiment_pct", "sector"]
            if include_bt:
                disp_cols += ["bt_win_rate", "bt_profit_factor", "bt_sharpe"]

            disp = df[disp_cols].sort_values("score", ascending=False).copy()

            fmt = {
                "price": "${:.2f}", "change_pct": "{:.2f}%", "score": "{:.2f}",
                "rsi": "{:.1f}", "macd_hist": "{:.4f}", "vol_surge": "{:.1f}x",
                "entry": "${:.2f}", "stop": "${:.2f}", "target": "${:.2f}",
                "risk_reward": "{:.2f}x", "allocation_pct": "{:.2f}%",
                "allocation_usd": "${:,.0f}", "sentiment_pct": "{:.2f}%",
            }
            if include_bt:
                fmt.update({"bt_win_rate": "{:.2%}", "bt_profit_factor": "{:.2f}", "bt_sharpe": "{:.2f}"})

            st.dataframe(
                disp.style.format(fmt, na_rep="—").map(_reco_color, subset=["recommendation"]),
                use_container_width=True, height=420,
            )

            # Export
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

            _render_legend()

            # Ticker drill-down
            st.markdown("---")
            selected = st.selectbox("Inspect ticker", df["symbol"].tolist())
            if selected:
                row = df[df["symbol"] == selected].iloc[0]
                st.plotly_chart(create_price_chart(selected, signal_row=row),
                                use_container_width=True, key=f"drill_price_{selected}")
                reco = row.get("recommendation", "—")
                st.markdown(f"**Recommendation:** <span style='{_reco_color(reco)}'>{reco}</span> "
                            f"· score {row['score']:.2f}", unsafe_allow_html=True)
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Entry", f"${row['entry']:.2f}")
                dc2.metric("Stop", f"${row['stop']:.2f}", delta=f"{(row['stop']-row['entry'])/row['entry']*100:.2f}%")
                dc3.metric("Target", f"${row['target']:.2f}", delta=f"+{(row['target']-row['entry'])/row['entry']*100:.2f}%")
                st.write("**Signal reasons:**", row.get("reasons", ""))

                st.markdown(f"#### 📰 News for {selected}")
                render_ticker_news(selected, config)
                render_forecast_panel(selected, config, entry=row["entry"], stop=row["stop"],
                                      key_prefix="drill")

                # One-click log to journal
                st.markdown("**Log this trade:**")
                j1, j2, j3 = st.columns(3)
                with j1:
                    log_qty = st.number_input("Qty", min_value=1, value=max(1, int(row["allocation_usd"] / row["entry"])), key="log_qty")
                with j2:
                    if st.button("Add to P&L Journal", use_container_width=True):
                        journal = load_journal()
                        add_trade(journal, selected, "long",
                                  float(row["entry"]), float(row["stop"]),
                                  float(row["target"]), log_qty, float(row["score"]))
                        save_journal(journal)
                        st.success(f"Logged {selected} to journal")
                with j3:
                    st.write(f"Alloc: ${row['allocation_usd']:,.0f}  R:R {row['risk_reward']:.1f}x")

            st.session_state["signals_df"] = df

        # ── Overall market news (ticker-independent) ─────────────────────────
        st.markdown("---")
        st.subheader("📰 Overall market news")
        with st.spinner("Loading market news…"):
            render_market_news(config)

    # ═══════════════════════ PRE-MARKET MOVERS ═══════════════════════════════
    elif page == "Pre-Market Movers":
        st.header("Pre-Market Movers")
        st.caption("Biggest gainers and losers right now (pre-market quotes when the pre-market "
                   "session is open, otherwise the regular session). Live from Yahoo Finance.")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            mv_n = st.slider("Show top", 10, 50, 25, 5)
        with mc2:
            mv_min = st.slider("Min move %", 0.0, 10.0, 1.0, 0.5)
        with mc3:
            st.write("")
            if st.button("↻ Refresh now"):
                fetch_movers.clear()

        with st.spinner("Fetching movers…"):
            mv = fetch_movers(top_n=mv_n, min_change_pct=mv_min)

        if mv.empty:
            st.warning("No movers returned — the market data feed may be unavailable right now.")
        else:
            gainers = mv[mv["change_pct"] > 0]
            losers = mv[mv["change_pct"] < 0]
            s1, s2, s3 = st.columns(3)
            s1.metric("Gainers", len(gainers))
            s2.metric("Losers", len(losers))
            s3.metric("Session", mv["session"].mode().iloc[0] if not mv.empty else "—")

            disp = mv.rename(columns={
                "symbol": "Symbol", "name": "Name", "change_pct": "Move %",
                "session": "Session", "price": "Price", "regular_change_pct": "Reg %",
                "volume": "Volume", "market_cap": "Mkt Cap",
            })[["Symbol", "Name", "Move %", "Session", "Price", "Reg %", "Volume", "Mkt Cap"]]

            def _hl(v):
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                disp.style.format({
                    "Move %": "{:+.2f}%", "Price": "${:.2f}", "Reg %": "{:+.2f}%",
                    "Volume": "{:,.0f}", "Mkt Cap": "${:,.0f}",
                }, na_rep="—").map(_hl, subset=["Move %", "Reg %"]),
                use_container_width=True, height=520,
            )

            gl, gr = st.columns(2)
            with gl:
                st.subheader("Top Gainers")
                if not gainers.empty:
                    gfig = px.bar(gainers.head(12).sort_values("change_pct"),
                                  x="change_pct", y="symbol", orientation="h",
                                  color="change_pct", color_continuous_scale="Greens",
                                  labels={"change_pct": "% move", "symbol": ""})
                    gfig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(gfig, use_container_width=True)
            with gr:
                st.subheader("Top Losers")
                if not losers.empty:
                    lfig = px.bar(losers.head(12).sort_values("change_pct", ascending=False),
                                  x="change_pct", y="symbol", orientation="h",
                                  color="change_pct", color_continuous_scale="Reds_r",
                                  labels={"change_pct": "% move", "symbol": ""})
                    lfig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(lfig, use_container_width=True)

    # ═══════════════════════ LIVE MOVERS (RAW) ═══════════════════════════════
    elif page == "Live Movers":
        st.header("⚡ Live Movers")
        st.caption("Raw Yahoo Finance screener feed — no filters, no signals, no recommendations. "
                   "Just who's moving right now. Pick a screener and look.")

        lc1, lc2, lc3 = st.columns([2, 1, 1])
        with lc1:
            screen_name = st.selectbox("Screener", list(RAW_SCREENS.keys()))
        with lc2:
            lm_count = st.slider("How many", 10, 100, 50, 10)
        with lc3:
            st.write("")
            if st.button("↻ Refresh now"):
                fetch_raw_movers.clear()

        with st.spinner(f"Fetching {screen_name}…"):
            raw = fetch_raw_movers(RAW_SCREENS[screen_name], count=lm_count)

        if raw.empty:
            st.warning("No data returned — the screener feed may be unavailable right now.")
        else:
            st.caption(f"{len(raw)} names · live from Yahoo Finance · cached up to 3 min")

            def _chg(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                raw.style.format({
                    "Price": "${:.2f}", "Change %": "{:+.2f}%", "Pre-mkt %": "{:+.2f}%",
                    "Volume": "{:,.0f}", "Avg Vol (3M)": "{:,.0f}", "Mkt Cap": "${:,.0f}",
                }, na_rep="—").map(_chg, subset=["Change %", "Pre-mkt %"]),
                use_container_width=True, height=560,
            )

    # ═══════════════════════ AFTER-HOURS & IPOs ══════════════════════════════
    elif page == "After-Hours & IPOs":
        st.header("🌙 After-Hours & IPOs")
        st.caption("Post-market movers (extended-hours gappers that often set up the next session) "
                   "and a tracker for notable recent IPOs.")

        st.subheader("After-Hours Movers")
        ah1, ah2, ah3 = st.columns(3)
        with ah1:
            ah_n = st.slider("Show top", 10, 50, 25, 5, key="ah_n")
        with ah2:
            ah_min = st.slider("Min move %", 0.0, 10.0, 1.0, 0.5, key="ah_min")
        with ah3:
            st.write("")
            if st.button("↻ Refresh now", key="ah_refresh"):
                fetch_afterhours.clear()

        with st.spinner("Fetching after-hours movers…"):
            ah = fetch_afterhours(top_n=ah_n, min_change_pct=ah_min)

        if ah.empty:
            st.info("No after-hours movers right now — the post-market session is likely closed "
                    "(it runs ~4–8pm ET). Yahoo only reports post-market quotes during/after that "
                    "window.")
        else:
            ah_g = ah[ah["change_pct"] > 0]
            ah_l = ah[ah["change_pct"] < 0]
            s1, s2, s3 = st.columns(3)
            s1.metric("AH Gainers", len(ah_g))
            s2.metric("AH Losers", len(ah_l))
            s3.metric("Biggest move", f"{ah['change_pct'].abs().max():.2f}%")

            ah_disp = ah.rename(columns={
                "symbol": "Symbol", "name": "Name", "change_pct": "AH %",
                "price": "AH Price", "regular_change_pct": "Reg %",
                "volume": "Volume", "market_cap": "Mkt Cap",
            })[["Symbol", "Name", "AH %", "AH Price", "Reg %", "Volume", "Mkt Cap"]]

            def _hl_ah(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                ah_disp.style.format({
                    "AH %": "{:+.2f}%", "AH Price": "${:.2f}", "Reg %": "{:+.2f}%",
                    "Volume": "{:,.0f}", "Mkt Cap": "${:,.0f}",
                }, na_rep="—").map(_hl_ah, subset=["AH %", "Reg %"]),
                use_container_width=True, height=440,
            )

            agl, agr = st.columns(2)
            with agl:
                st.markdown("**Top AH Gainers**")
                if not ah_g.empty:
                    gfig = px.bar(ah_g.head(12).sort_values("change_pct"),
                                  x="change_pct", y="symbol", orientation="h",
                                  color="change_pct", color_continuous_scale="Greens",
                                  labels={"change_pct": "AH %", "symbol": ""})
                    gfig.update_layout(height=360, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(gfig, use_container_width=True, key="ah_up")
            with agr:
                st.markdown("**Top AH Losers**")
                if not ah_l.empty:
                    lfig = px.bar(ah_l.head(12).sort_values("change_pct", ascending=False),
                                  x="change_pct", y="symbol", orientation="h",
                                  color="change_pct", color_continuous_scale="Reds_r",
                                  labels={"change_pct": "AH %", "symbol": ""})
                    lfig.update_layout(height=360, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(lfig, use_container_width=True, key="ah_dn")

        st.markdown("---")
        st.subheader("Recent IPOs")
        st.caption("Curated list of notable recent IPOs. **IPO price is approximate** — the earliest "
                   "close yfinance returns (~2y of history), not the true offer price.")
        with st.spinner("Loading IPO performance…"):
            ipos = ipo_table()
        if ipos.empty:
            st.warning("IPO performance data is unavailable right now.")
        else:
            def _hl_gain(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                ipos.style.format({
                    "IPO price ≈": "${:.2f}", "Current": "${:.2f}",
                    "Gain %": "{:+.2f}%", "Days since IPO": "{:,.0f}",
                }, na_rep="—").map(_hl_gain, subset=["Gain %"]),
                use_container_width=True, height=460,
            )

        with st.expander("🔎 Look up any ticker's run since its earliest data"):
            ip_sym = st.text_input("Symbol", key="ipo_lookup").strip().upper()
            if ip_sym:
                perf = get_ipo_tracker().fetch_ipo_performance(ip_sym)
                if perf:
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.metric("Earliest close ≈", f"${perf['ipo_price_approx']:.2f}")
                    ic2.metric("Current", f"${perf['current_price']:.2f}")
                    ic3.metric("Change", f"{perf['gain_pct']:+.2f}%")
                    st.caption("⚠️ 'Earliest close' is the oldest price in ~2y of history — a rough "
                               "IPO-price proxy only for stocks that listed within that window.")
                else:
                    st.info("No history available for that symbol.")

    # ═══════════════════════ WHALE MOVEMENTS ═════════════════════════════════
    elif page == "Whale Movements":
        st.header("🐋 Whale Movements")
        st.caption("Large-money footprints inferred from the public tape: names trading on an "
                   "outsized volume surge vs their own 20-day baseline, scaled by the dollars "
                   "changing hands and where the day closed in its range. No dark-pool / Level 2 "
                   "data — this is smart-money *inference* from free Yahoo Finance OHLCV.")

        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            w_n = st.slider("Universe size", 40, 250, 120, 20,
                            help="Most-active names are scanned first")
        with wc2:
            w_rvol = st.slider("Min relative volume", 1.5, 8.0, 2.0, 0.5,
                               help="Today's volume vs its 20-day average")
        with wc3:
            w_dollar = st.slider("Min $ traded ($M)", 10, 500, 50, 10,
                                 help="Minimum dollar volume to count as whale size")
        with wc4:
            st.write("")
            if st.button("↻ Refresh now"):
                scan_whale_activity.clear()

        with st.spinner("Scanning the tape for whale activity…"):
            whales = scan_whale_activity(sample_size=w_n, min_rvol=w_rvol,
                                         min_dollar_vol_m=float(w_dollar))

        if whales.empty:
            st.info("No whale-sized volume events right now. Lower the relative-volume or "
                    "$-traded thresholds, or widen the universe.")
        else:
            buying = whales[whales["signal"].isin(["Heavy Buying", "Accumulation"])]
            selling = whales[whales["signal"].isin(["Heavy Selling", "Distribution"])]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Whale events", len(whales))
            m2.metric("Bullish footprints", len(buying))
            m3.metric("Bearish footprints", len(selling))
            m4.metric("Top $ traded", f"${whales['dollar_vol'].max()/1e6:,.0f}M")

            disp = whales.rename(columns={
                "symbol": "Symbol", "signal": "Signal", "whale_score": "Whale Score",
                "rvol": "Rel Vol", "price": "Price", "change_pct": "Change %",
                "dollar_vol": "$ Traded", "close_strength": "Close Str",
                "accum_days": "Accum d", "distrib_days": "Distrib d",
            })[["Symbol", "Signal", "Whale Score", "Rel Vol", "Price", "Change %",
                "$ Traded", "Close Str", "Accum d", "Distrib d"]]

            def _whale_sig_color(v: str) -> str:
                if v in ("Heavy Buying", "Accumulation"):
                    return "color:#00c851;font-weight:bold" if v == "Heavy Buying" else "color:#00c851"
                if v in ("Heavy Selling", "Distribution"):
                    return "color:#ff4444;font-weight:bold" if v == "Heavy Selling" else "color:#ff4444"
                return "color:#888888"

            def _chg_color(v):
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                disp.style.format({
                    "Whale Score": "{:.1f}", "Rel Vol": "{:.2f}x", "Price": "${:.2f}",
                    "Change %": "{:+.2f}%", "$ Traded": "${:,.0f}", "Close Str": "{:.2f}",
                }, na_rep="—")
                .map(_whale_sig_color, subset=["Signal"])
                .map(_chg_color, subset=["Change %"]),
                use_container_width=True, height=520,
            )

            bub = whales.copy()
            bub["dollar_vol_m"] = bub["dollar_vol"] / 1e6
            bfig = px.scatter(
                bub, x="rvol", y="change_pct", size="dollar_vol_m", color="signal",
                hover_name="symbol", size_max=46,
                labels={"rvol": "Relative volume (×)", "change_pct": "Day change %",
                        "dollar_vol_m": "$ traded (M)", "signal": "Signal"},
                color_discrete_map={
                    "Heavy Buying": "#00c851", "Accumulation": "#7cd992",
                    "Heavy Selling": "#ff4444", "Distribution": "#ff8a80", "Churn": "#9e9e9e",
                },
                title="Whale map — volume surge vs price impact (bubble = $ traded)",
            )
            bfig.add_hline(y=0, line_dash="dot", line_color="#888")
            bfig.update_layout(height=440, margin=dict(t=46, l=10, r=10, b=10))
            st.plotly_chart(bfig, use_container_width=True, key="whale_bubble")

            with st.expander("How to read this"):
                st.markdown(
                    "- **Heavy Buying / Accumulation** — big volume on an up day; whales likely "
                    "building positions. Heavy = closed in the top third of the day's range.\n"
                    "- **Heavy Selling / Distribution** — big volume on a down day; likely "
                    "offloading. Heavy = closed in the bottom third.\n"
                    "- **Churn** — outsized volume but the close reverses the day's range "
                    "(indecision / two-sided fight).\n"
                    "- **Rel Vol** — today's volume ÷ its 20-day average (2.00x = double normal).\n"
                    "- **Close Str** — where the close sat in the day's range (1.00 = on the high, "
                    "0.00 = on the low).\n"
                    "- **Accum d / Distrib d** — high-volume up vs down days over the last 20 "
                    "sessions (the multi-day balance behind the latest bar)."
                )

    # ═══════════════════════ OPTIONS FLOW ════════════════════════════════════
    elif page == "Options Flow":
        st.header("🎯 Options Flow")
        st.caption("Options-derived sentiment from live chains: IV rank, put/call ratio, and "
                   "unusual volume (a contract trading above 10% of its open interest = possible "
                   "smart-money positioning). Plus an earnings IV-crush estimate and a Greeks calc.")

        # ── Single-ticker analysis ──────────────────────────────────────────
        oc1, oc2 = st.columns([3, 1])
        with oc1:
            opt_sym = st.text_input("🔍 Analyze a ticker's options", placeholder="e.g. AAPL, NVDA",
                                    key="opt_search").strip().upper()
        with oc2:
            st.write("")
            do_opt = st.button("Analyze", use_container_width=True, key="opt_btn")

        if opt_sym and (do_opt or st.session_state.get("opt_last") == opt_sym):
            st.session_state["opt_last"] = opt_sym
            with st.spinner(f"Pulling option chains for {opt_sym}…"):
                od = analyze_options(opt_sym)

            gcol, mcol = st.columns([1, 1])
            with gcol:
                if od["iv_rank"] is not None:
                    ivfig = go.Figure(go.Indicator(
                        mode="gauge+number", value=od["iv_rank"],
                        title={"text": "IV Rank"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                               "steps": [
                                   {"range": [0, 30], "color": "#9ccc65"},
                                   {"range": [30, 70], "color": "#ffe066"},
                                   {"range": [70, 100], "color": "#ff4444"}]},
                    ))
                    ivfig.update_layout(height=260, margin=dict(t=50, l=30, r=30, b=10))
                    st.plotly_chart(ivfig, use_container_width=True, key=f"iv_gauge_{opt_sym}")
                    st.caption("Low IV (<30) = cheaper premiums (favor buying); High IV (>70) = "
                               "expensive (favor selling).")
                else:
                    st.info("IV rank unavailable (no options chain for this symbol).")
            with mcol:
                pc = od["pc_ratio"]
                sent = _pc_sentiment(pc)
                sent_color = {"Bullish": "#00c851", "Bearish": "#ff4444"}.get(sent, "#888888")
                st.metric("Put/Call ratio", f"{pc:.2f}" if pc is not None else "—")
                st.markdown(f"Sentiment: <span style='color:{sent_color};font-weight:bold'>{sent}"
                            f"</span>", unsafe_allow_html=True)
                st.caption("<0.70 call-heavy (bullish) · >1.00 put-heavy (bearish)")
                if od["current_iv"] is not None:
                    st.metric("Current IV", f"{od['current_iv'] * 100:.2f}%")
                crush = od["iv_crush"]
                if crush:
                    st.metric("Days to earnings", f"{crush['days_to_earnings']}")
                    st.caption(f"Implied move ≈ ±{crush['estimated_move_pct'] * 100:.2f}% · "
                               f"est. IV crush ≈ {crush['iv_crush_pct']:.0f}% post-earnings")
                elif od["earnings_date"]:
                    st.caption(f"Next earnings: {od['earnings_date']:%Y-%m-%d}")

            unusual = od["unusual"]
            if unusual and (unusual.get("unusual_calls") or unusual.get("unusual_puts")):
                st.markdown("#### Unusual options activity")
                ucol, pcol = st.columns(2)
                with ucol:
                    st.markdown("**Calls** (vol > 10% of OI)")
                    uc = pd.DataFrame(unusual.get("unusual_calls", []))
                    if not uc.empty:
                        st.dataframe(uc[["strike", "volume", "openInterest"]].rename(columns={
                            "strike": "Strike", "volume": "Volume", "openInterest": "Open Int"}),
                            use_container_width=True, hide_index=True)
                    else:
                        st.caption("None")
                with pcol:
                    st.markdown("**Puts** (vol > 10% of OI)")
                    up = pd.DataFrame(unusual.get("unusual_puts", []))
                    if not up.empty:
                        st.dataframe(up[["strike", "volume", "openInterest"]].rename(columns={
                            "strike": "Strike", "volume": "Volume", "openInterest": "Open Int"}),
                            use_container_width=True, hide_index=True)
                    else:
                        st.caption("None")
            else:
                st.caption("No unusual options volume detected on the nearest expiration.")

            with st.expander("🧮 Greeks calculator (Black-Scholes call)"):
                spot = (fetch_symbol_info(opt_sym).get("price")) or 100.0
                seed_iv = od["current_iv"] if od["current_iv"] else 0.30
                seed_iv_pct = min(300.0, max(1.0, round(seed_iv * 100, 1)))
                gk1, gk2, gk3 = st.columns(3)
                with gk1:
                    g_strike = st.number_input("Strike $", value=float(round(spot, 2)), step=1.0,
                                               key="g_strike")
                with gk2:
                    g_dte = st.number_input("Days to expiry", 1, 730, 30, key="g_dte")
                with gk3:
                    g_iv = st.number_input("IV (%)", 1.0, 300.0, float(seed_iv_pct),
                                           step=1.0, key="g_iv") / 100.0
                greeks = get_options_analyzer().estimate_greeks(
                    opt_sym, float(spot), float(g_strike), int(g_dte), float(g_iv))
                if greeks:
                    e1, e2, e3, e4, e5 = st.columns(5)
                    e1.metric("Call price", f"${greeks['call_price']:.2f}")
                    e2.metric("Delta", f"{greeks['delta']:.3f}")
                    e3.metric("Gamma", f"{greeks['gamma']:.4f}")
                    e4.metric("Vega", f"{greeks['vega']:.3f}")
                    e5.metric("Theta", f"{greeks['theta']:.3f}")
                    st.caption(f"Spot ${spot:.2f} · simplified estimate, not a pricing engine.")

        st.markdown("---")

        # ── Unusual-flow scan across the most-actives ───────────────────────
        st.subheader("Unusual flow scan")
        st.caption("Scans the most-active names for options sentiment + unusual volume. **Slow** — "
                   "each symbol pulls a live option chain — so it's a small sample, cached 20 min.")
        sf1, sf2 = st.columns([1, 1])
        with sf1:
            sf_n = st.slider("Symbols to scan", 5, 30, 15, 5)
        with sf2:
            st.write("")
            if st.button("↻ Rescan", key="of_refresh"):
                scan_options_flow.clear()

        with st.spinner("Scanning live option chains…"):
            flow = scan_options_flow(sample_size=sf_n)

        if flow.empty:
            st.info("No options data returned for the scanned names right now.")
        else:
            def _sent_color(v: str) -> str:
                if v == "Bullish":
                    return "color:#00c851;font-weight:bold"
                if v == "Bearish":
                    return "color:#ff4444;font-weight:bold"
                return "color:#888888"

            st.dataframe(
                flow.style.format({"P/C Ratio": "{:.2f}", "# Unusual": "{:,.0f}"}, na_rep="—")
                .map(_sent_color, subset=["Sentiment"]),
                use_container_width=True, height=440,
            )

    # ═══════════════════════ PREDICTIONS (TOMORROW) ══════════════════════════
    elif page == "Predictions":
        st.header("🔮 Predictions for Tomorrow")
        st.caption("Next-session probabilistic price forecast for the most-active names. Uses "
                   "Amazon Chronos when the AI extras are installed, otherwise a Monte-Carlo "
                   "random-walk from recent returns. Shows the median predicted close, the "
                   "p10–p90 range, and the implied next-day return. **Model output, not advice.**")

        pc1, pc2, pc3 = st.columns([1, 1, 1])
        with pc1:
            pr_n = st.slider("Universe size", 20, 150, 60, 10,
                             help="Most-active names are scanned first")
        with pc2:
            pr_dir = st.selectbox("Show", ["All", "Bullish only", "Bearish only"])
        with pc3:
            st.write("")
            if st.button("↻ Refresh now"):
                predict_tomorrow.clear()

        with st.spinner("Forecasting tomorrow's moves…"):
            preds = predict_tomorrow(sample_size=pr_n)

        if preds.empty:
            st.warning("No forecasts available right now — the data feed may be unavailable.")
        else:
            view = preds
            if pr_dir == "Bullish only":
                view = preds[preds["Pred Return %"] > 0]
            elif pr_dir == "Bearish only":
                view = preds[preds["Pred Return %"] < 0]

            ups = int((preds["Pred Return %"] > 0).sum())
            downs = int((preds["Pred Return %"] < 0).sum())
            model = preds["Model"].mode().iloc[0] if not preds.empty else "—"
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Forecasts", len(preds))
            pm2.metric("Predicted up", ups)
            pm3.metric("Predicted down", downs)
            pm4.metric("Model", "Chronos AI" if model == "chronos" else "Heuristic MC")
            if model != "chronos":
                st.caption("💡 Install the AI extras (`pip install -r requirements-ai.txt`) to use "
                           "the Chronos foundation model instead of the heuristic.")

            def _dir_color(v: str) -> str:
                if v == "Up":
                    return "color:#00c851;font-weight:bold"
                if v == "Down":
                    return "color:#ff4444;font-weight:bold"
                return "color:#888888"

            def _ret_color(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return ""
                return "color:#00c851" if v > 0 else ("color:#ff4444" if v < 0 else "")

            st.dataframe(
                view.style.format({
                    "Price": "${:.2f}", "Pred Close": "${:.2f}", "Pred Return %": "{:+.2f}%",
                    "Low (p10)": "${:.2f}", "High (p90)": "${:.2f}", "Uncertainty %": "{:.2f}%",
                }, na_rep="—")
                .map(_dir_color, subset=["Direction"])
                .map(_ret_color, subset=["Pred Return %"]),
                use_container_width=True, height=480,
            )

            tcol, bcol = st.columns(2)
            with tcol:
                st.subheader("Top predicted gainers")
                top_up = preds[preds["Pred Return %"] > 0].head(12)
                if not top_up.empty:
                    ufig = px.bar(top_up.sort_values("Pred Return %"),
                                  x="Pred Return %", y="Symbol", orientation="h",
                                  color="Pred Return %", color_continuous_scale="Greens",
                                  labels={"Pred Return %": "Pred. next-day %", "Symbol": ""})
                    ufig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(ufig, use_container_width=True, key="pred_up")
            with bcol:
                st.subheader("Top predicted losers")
                top_dn = preds[preds["Pred Return %"] < 0].tail(12)
                if not top_dn.empty:
                    dfig = px.bar(top_dn.sort_values("Pred Return %", ascending=False),
                                  x="Pred Return %", y="Symbol", orientation="h",
                                  color="Pred Return %", color_continuous_scale="Reds_r",
                                  labels={"Pred Return %": "Pred. next-day %", "Symbol": ""})
                    dfig.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(dfig, use_container_width=True, key="pred_dn")

            # ── Drill-down: forecast cone for one ticker (next 5 sessions for context) ──
            st.markdown("---")
            st.subheader("Inspect a forecast")
            insp = st.selectbox("Ticker", view["Symbol"].tolist(), key="pred_inspect")
            if insp:
                fc5 = forecast_symbol(insp, horizon=5)
                if fc5:
                    st.plotly_chart(create_forecast_chart(insp, fc5),
                                    use_container_width=True, key=f"pred_cone_{insp}")
                    er = expected_return_pct(fc5)
                    st.caption(f"5-session median path · expected return {er:+.2f}% · "
                               f"source: {fc5['source']}")
                else:
                    st.info("Not enough history to chart a forecast for this ticker.")

            st.caption("⚠️ Forecasts are probabilistic and frequently wrong, especially around "
                       "news/earnings. Use as one input, not a guarantee.")

    # ═══════════════════════ AUTO WATCHLIST ══════════════════════════════════
    elif page == "Auto Watchlist":
        st.header("Auto Watchlist — Pre-Market Bulls")
        st.caption("Auto-built from today's bullish pre-market movers, with recommended "
                   "entry / stop / target and the key momentum indicators.")

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            aw_n = st.slider("Max names", 5, 40, 20, 5)
        with ac2:
            aw_min = st.slider("Min pre-market move %", 1.0, 10.0, 2.0, 0.5)
        with ac3:
            st.write("")
            auto_save = st.checkbox("Save to watchlist", value=True,
                                    help="Overwrites the 'Pre-Market Bulls' watchlist on each run")

        use_fc = _ai_on("ai_forecast")
        with st.spinner("Scanning bullish movers and building signals…"):
            awl = build_auto_watchlist(config, top_n=aw_n, min_change_pct=aw_min,
                                       with_forecast=use_fc)

        if awl.empty:
            st.warning("No bullish pre-market movers found right now. Try lowering the min move %.")
        else:
            buys = int((awl["Recommendation"] == "Buy").sum())
            s1, s2, s3 = st.columns(3)
            s1.metric("Candidates", len(awl))
            s2.metric("Buy-rated", buys)
            s3.metric("Avg pre-mkt move", f"{awl['Pre-mkt %'].mean():+.2f}%")

            awl_fmt = {
                "Pre-mkt %": "{:+.2f}%", "Price": "${:.2f}", "Score": "{:.2f}",
                "RSI": "{:.2f}", "MACD hist": "{:.4f}", "Vol surge": "{:.2f}x",
                "Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "R:R": "{:.2f}x",
            }
            styler = awl.style.format({**awl_fmt, **({"Fcst ret %": "{:+.2f}%"} if use_fc else {})},
                                      na_rep="—").map(_reco_color, subset=["Recommendation"])
            if use_fc and "Forecast" in awl.columns:
                styler = styler.map(_forecast_color, subset=["Forecast"])
            st.dataframe(styler, use_container_width=True, height=460)

            if auto_save:
                wl_name = "Pre-Market Bulls"
                existing = set(watchlist_mgr.get_watchlist(wl_name))
                for sym in awl["Symbol"]:
                    if sym not in existing:
                        watchlist_mgr.add_symbol(wl_name, sym)
                st.success(f"Saved {len(awl)} symbols to watchlist '{wl_name}'.")

            st.download_button("Download CSV", awl.to_csv(index=False),
                               file_name=f"auto_watchlist_{datetime.now():%Y%m%d_%H%M}.csv",
                               mime="text/csv")
            _render_legend()

    # ═══════════════════════ ETF SCREENER ════════════════════════════════════
    elif page == "ETF Screener":
        st.header("ETF Screener — by Category")
        st.caption("Sector, broad-market, volatility, commodity and bond ETFs with daily "
                   "change and trend signal — all on one screen.")

        with st.spinner("Loading ETF signals…"):
            etf_df = fetch_etf_table(config)

        if etf_df.empty:
            st.warning("No ETF data available right now.")
        else:
            # Sector rotation summary from the sector rows.
            sectors = etf_df[etf_df["Category"] == "Sector"].dropna(subset=["Change %"])
            if not sectors.empty:
                best = sectors.loc[sectors["Change %"].idxmax()]
                worst = sectors.loc[sectors["Change %"].idxmin()]
                r1, r2, r3 = st.columns(3)
                r1.metric("Leading sector", best["Name"], f"{best['Change %']:+.2f}%")
                r2.metric("Lagging sector", worst["Name"], f"{worst['Change %']:+.2f}%")
                r3.metric("Avg sector move", f"{sectors['Change %'].mean():+.2f}%")

            etf_fmt = {"Price": "${:.2f}", "Change %": "{:+.2f}%",
                       "Score": "{:.2f}", "RSI": "{:.2f}"}

            def _trend_color(v):
                return ("color:#00c851" if v == "long"
                        else ("color:#ff4444" if v == "short" else ""))

            for category in ETF_CATEGORIES:
                cat_df = etf_df[etf_df["Category"] == category]
                if cat_df.empty:
                    continue
                st.subheader(category)
                disp = cat_df[["Symbol", "Name", "Price", "Change %", "Trend",
                               "Score", "RSI", "Recommendation"]]
                st.dataframe(
                    disp.style.format(etf_fmt, na_rep="—")
                        .map(_trend_color, subset=["Trend"])
                        .map(_reco_color, subset=["Recommendation"]),
                    use_container_width=True,
                )
            _render_legend()

    # ═══════════════════════ MARKET EVENTS ═══════════════════════════════════
    elif page == "Market Events":
        st.header("Market Events — Global & Local")
        st.caption("Macro regime, the scheduled economic calendar, and live news across "
                   "equities, rates, commodities, currencies and global markets — the events "
                   "most likely to move stocks.")

        macro = get_macro_context()

        # ── Market Mood (news tone + VIX + breadth + trend) ──────────────────
        st.subheader("Market Mood")
        with st.spinner("Gauging market mood…"):
            mood = compute_market_mood(config)
        mg1, mg2 = st.columns([1, 1])
        with mg1:
            st.plotly_chart(create_mood_gauge(mood), use_container_width=True)
        with mg2:
            st.markdown(f"**Overall: {mood['label']} ({mood['score']:.1f}/100)**")
            st.caption("Composite of live news tone and known quant gauges. "
                       "0 = extreme fear · 100 = extreme greed.")
            comp_df = pd.DataFrame(
                [{"Gauge": k, "Score": round(v, 1)} for k, v in mood["components"].items()]
            )
            st.dataframe(
                comp_df.style.format({"Score": "{:.1f}"}).map(_mood_cell, subset=["Score"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(f"As of {mood['as_of']}")

        st.divider()

        # Macro regime snapshot.
        vix = macro.fetch_vix()
        breadth = macro.fetch_market_breadth()
        skip, reason = macro.should_skip_entries()
        e1, e2, e3 = st.columns(3)
        if vix is not None:
            regime = ("High stress" if vix > 25 else "Complacent" if vix < 12 else "Normal")
            e1.metric("VIX", f"{vix:.2f}", regime)
        else:
            e1.metric("VIX", "—")
        e2.metric("Bullish breadth", f"{breadth['bullish_breadth_pct']:.0%}" if breadth else "—")
        e3.metric("Entry climate", "Caution" if skip else "OK", reason)

        # Scheduled economic calendar (programmatic — never goes stale).
        st.subheader("Upcoming economic calendar")
        events = macro.get_upcoming_macro_events(days_ahead=45)
        if events:
            cal = pd.DataFrame(
                [{"Date": d.strftime("%a %b %d, %Y"),
                  "In days": (d.date() - datetime.now().date()).days,
                  "Event": name} for d, name in events]
            )
            st.dataframe(cal, use_container_width=True, hide_index=True)
        else:
            st.caption("No scheduled events in the next 45 days.")

        # Live news scan across macro proxies.
        st.subheader("Live market-moving news")
        with st.spinner("Scanning global & local market news…"):
            ev = scan_market_events(config)
        if ev.empty:
            st.warning("No market news available right now.")
        else:
            areas = st.multiselect("Filter areas", list(MARKET_EVENT_TICKERS.keys()),
                                   default=list(MARKET_EVENT_TICKERS.keys()))
            shown = ev[ev["Area"].isin(areas)] if areas else ev

            net = {"positive": 0, "negative": 0, "neutral": 0}
            for s in shown["Sentiment"]:
                key = "positive" if "pos" in s else "negative" if "neg" in s else "neutral"
                net[key] += 1
            n1, n2, n3 = st.columns(3)
            n1.metric("Positive", net["positive"])
            n2.metric("Negative", net["negative"])
            n3.metric("Neutral", net["neutral"])

            def _sent_color(v: str) -> str:
                return ("color:#00c851" if "pos" in v
                        else "color:#ff4444" if "neg" in v else "color:#888888")

            st.dataframe(
                shown[["Area", "Source", "Event", "Sentiment", "Score", "Headline"]]
                    .style.format({"Score": "{:.2f}"})
                    .map(_sent_color, subset=["Sentiment"]),
                use_container_width=True, height=460,
            )
            if not _ai_on("ai_events"):
                st.caption("Tip: enable **News event tagging** in Settings for model-based "
                           "event classification (currently using keyword heuristics).")
        _render_legend()

    # ═══════════════════════ HEAT MAP ════════════════════════════════════════
    elif page == "Heat Map":
        st.header("Market Heat Map")
        st.caption("Tiles sized by market cap, colored by today's % change — green up, red down. Grouped by sector.")
        hm_size = st.slider("Universe size", 30, 300, 120, 10)

        tickers = get_screen_universe()[:hm_size]
        with st.spinner("Loading market data…"):
            hm_df = fetch_heatmap_data(tickers)
        if not hm_df.empty:
            adv = int((hm_df["change_pct"] > 0).sum())
            dec = int((hm_df["change_pct"] < 0).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Advancing", adv)
            m2.metric("Declining", dec)
            m3.metric("Avg % change", f"{hm_df['change_pct'].mean():+.2f}%")
            st.plotly_chart(create_market_heatmap(hm_df), use_container_width=True)
            st.plotly_chart(create_sector_bar(hm_df), use_container_width=True)
        else:
            st.warning("No market data available — try a larger universe.")

    # ═══════════════════════ WATCHLISTS ══════════════════════════════════════
    elif page == "Watchlists":
        st.header("Watchlists")
        wl_names = watchlist_mgr.list_watchlists()
        wc1, wc2 = st.columns([2, 1])
        with wc1:
            sel_wl = st.selectbox("Select watchlist", wl_names)
        with wc2:
            new_wl = st.text_input("New watchlist name")
            if st.button("Create") and new_wl:
                watchlist_mgr.create_watchlist(new_wl)
                st.rerun()

        if sel_wl:
            syms = watchlist_mgr.get_watchlist(sel_wl)
            wa1, wa2 = st.columns([3, 1])
            with wa1:
                new_sym = st.text_input("Add symbol").upper()
            with wa2:
                if st.button("Add") and new_sym:
                    watchlist_mgr.add_symbol(sel_wl, new_sym)
                    st.rerun()

            if syms:
                trend_gen = get_signal_generator(config)
                wl_rows = []
                for sym in syms:
                    hist = fetch_symbol_history(sym, 90)
                    info = fetch_symbol_info(sym)
                    if not hist.empty:
                        closes = _to_series(hist, "Close")
                        volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0]*len(closes)
                        highs = _to_series(hist, "High") if "High" in hist.columns else closes
                        lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
                        sig = trend_gen.build_signal(sym, closes, volumes, highs=highs, lows=lows)
                        if sig:
                            risk = sig.entry_price - sig.stop_price
                            reward = sig.target_price - sig.entry_price
                            wl_rows.append({
                                "Symbol": sym, "Price": info.get("price"),
                                "Recommendation": recommend_label(sig.score, sig.signal_type),
                                "Score": sig.score,
                                "RSI": sig.metadata.get("rsi"), "Entry": sig.entry_price,
                                "Stop": sig.stop_price, "Target": sig.target_price,
                                "R:R": round(reward/risk, 2) if risk > 0 else 0,
                                "Sector": info.get("sector"),
                            })
                if wl_rows:
                    wl_df = pd.DataFrame(wl_rows)
                    st.dataframe(
                        wl_df.style.format({"Price": "${:.2f}", "Score": "{:.2f}",
                                            "RSI": "{:.1f}", "Entry": "${:.2f}",
                                            "Stop": "${:.2f}", "Target": "${:.2f}", "R:R": "{:.2f}x"}
                                           ).map(_reco_color, subset=["Recommendation"]),
                        use_container_width=True,
                    )
                # Remove buttons
                cols = st.columns(min(len(syms), 8))
                for i, sym in enumerate(syms):
                    with cols[i % len(cols)]:
                        if st.button(f"Remove {sym}", key=f"rm_{sym}"):
                            watchlist_mgr.remove_symbol(sel_wl, sym)
                            st.rerun()
            else:
                st.info("Watchlist is empty.")

            if sel_wl != "Default" and st.button(f"Delete watchlist '{sel_wl}'"):
                watchlist_mgr.delete_watchlist(sel_wl)
                st.rerun()

    # ═══════════════════════ COMPARE ═════════════════════════════════════════
    elif page == "Compare":
        st.header("Compare Stocks")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            sym1 = st.text_input("Stock 1", "AAPL").upper()
        with cc2:
            sym2 = st.text_input("Stock 2", "MSFT").upper()
        with cc3:
            sym3 = st.text_input("Stock 3 (optional)", "").upper()

        symbols = [s for s in [sym1, sym2, sym3] if s]
        if symbols:
            for i, sym in enumerate(symbols):
                st.plotly_chart(create_price_chart(sym, days=180), use_container_width=True,
                                key=f"cmp_price_{i}_{sym}")

            fundamentals = get_fundamentals()
            trend_gen = get_signal_generator(config)
            cmp_rows = []
            for sym in symbols:
                info = fetch_symbol_info(sym)
                fund = fundamentals.get_fundamentals(sym)
                hist = fetch_symbol_history(sym, 90)
                if not hist.empty:
                    closes = _to_series(hist, "Close")
                    volumes = _to_series(hist, "Volume") if "Volume" in hist.columns else [0]*len(closes)
                    highs = _to_series(hist, "High") if "High" in hist.columns else closes
                    lows = _to_series(hist, "Low") if "Low" in hist.columns else closes
                    sig = trend_gen.build_signal(sym, closes, volumes, highs=highs, lows=lows)
                    bt = run_symbol_backtest(sym, config)
                    crow = {
                        "Symbol": sym, "Price": info.get("price"),
                        "Recommendation": recommend_label(sig.score, sig.signal_type) if sig else "—",
                        "Score": sig.score if sig else None,
                        "RSI": sig.metadata.get("rsi") if sig else None,
                        "Entry": sig.entry_price if sig else None,
                        "Stop": sig.stop_price if sig else None,
                        "Target": sig.target_price if sig else None,
                        "P/E": fund.get("pe_ratio"), "Div Yield": fund.get("dividend_yield"),
                        "BT Win Rate": bt["win_rate"] if bt else None,
                        "BT PF": bt["profit_factor"] if bt else None,
                        "Sector": info.get("sector"),
                    }
                    if _ai_on("ai_forecast"):
                        fc = get_forecaster().forecast(closes, horizon=5)
                        crow["Fcst ret %"] = expected_return_pct(fc)
                        crow["Forecast"] = forecast_confirms(sig, fc)
                    cmp_rows.append(crow)
            if cmp_rows:
                cmp_styler = pd.DataFrame(cmp_rows).style.format({
                    "Price": "${:.2f}", "Score": "{:.2f}", "RSI": "{:.1f}",
                    "Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}",
                    "P/E": "{:.1f}", "Div Yield": "{:.2%}",
                    "BT Win Rate": "{:.2%}", "BT PF": "{:.2f}",
                    **({"Fcst ret %": "{:+.2f}%"} if _ai_on("ai_forecast") else {}),
                }, na_rep="—").map(_reco_color, subset=["Recommendation"])
                if _ai_on("ai_forecast"):
                    cmp_styler = cmp_styler.map(_forecast_color, subset=["Forecast"])
                st.dataframe(cmp_styler, use_container_width=True)

    # ═══════════════════════ P&L TRACKER ═════════════════════════════════════
    elif page == "P&L Tracker":
        st.header("P&L Tracker — Trade Journal")

        journal = load_journal()

        # ── Add trade manually ─────────────────────────────────────────────
        with st.expander("Add Trade"):
            ta1, ta2, ta3, ta4, ta5, ta6 = st.columns(6)
            with ta1:
                t_sym = st.text_input("Symbol", key="t_sym").upper()
            with ta2:
                t_entry = st.number_input("Entry $", value=0.0, step=0.01, key="t_entry")
            with ta3:
                t_stop = st.number_input("Stop $", value=0.0, step=0.01, key="t_stop")
            with ta4:
                t_target = st.number_input("Target $", value=0.0, step=0.01, key="t_target")
            with ta5:
                t_qty = st.number_input("Qty", min_value=1, value=1, key="t_qty")
            with ta6:
                t_score = st.number_input("Score", 0.0, 1.0, 0.5, 0.01, key="t_score")
            if st.button("Log Trade") and t_sym and t_entry > 0:
                add_trade(journal, t_sym, "long", t_entry, t_stop, t_target, t_qty, t_score)
                save_journal(journal)
                st.success(f"Logged {t_sym}")
                st.rerun()

        # ── Close open trade ───────────────────────────────────────────────
        open_trades = [t for t in journal if t["status"] == "open"]
        if open_trades:
            with st.expander("Close a Trade"):
                open_syms = [f"#{t['id']} {t['symbol']} @ {t['entry_price']}" for t in open_trades]
                sel = st.selectbox("Select trade to close", open_syms, key="close_sel")
                exit_px = st.number_input("Exit price $", value=0.0, step=0.01, key="exit_px")
                if st.button("Close Trade") and exit_px > 0:
                    trade_id = int(sel.split("#")[1].split(" ")[0])
                    close_trade(journal, trade_id, exit_px)
                    save_journal(journal)
                    st.rerun()

        if not journal:
            st.info("No trades logged yet. Use the Screener to log a signal, or add one manually above.")
        else:
            df_j = pd.DataFrame(journal)

            # ── Summary metrics ────────────────────────────────────────────
            closed = df_j[df_j["status"] == "closed"]
            if not closed.empty:
                total_pnl = closed["pnl"].sum()
                wins = closed[closed["pnl"] > 0]
                losses = closed[closed["pnl"] <= 0]
                win_rate = len(wins) / len(closed)
                avg_win = wins["pnl"].mean() if not wins.empty else 0
                avg_loss = losses["pnl"].mean() if not losses.empty else 0
                profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())
                                 if not losses.empty and losses["pnl"].sum() != 0 else float("inf"))

                pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                pm1.metric("Total P&L", f"${total_pnl:,.2f}",
                           delta_color="normal" if total_pnl >= 0 else "inverse")
                pm2.metric("Win Rate", f"{win_rate:.2%}")
                pm3.metric("Avg Win", f"${avg_win:,.2f}")
                pm4.metric("Avg Loss", f"${avg_loss:,.2f}")
                pm5.metric("Profit Factor", f"{profit_factor:.2f}")

                # Equity curve
                closed_sorted = closed.sort_values("exit_date")
                eq = closed_sorted["pnl"].cumsum()
                fig_eq = go.Figure(go.Scatter(
                    x=list(range(len(eq))), y=eq,
                    mode="lines+markers", fill="tonexty",
                    line=dict(color="green" if total_pnl >= 0 else "red", width=2)
                ))
                fig_eq.update_layout(title="Equity Curve (closed trades)", height=300,
                                     xaxis_title="Trade #", yaxis_title="Cumulative P&L ($)")
                st.plotly_chart(fig_eq, use_container_width=True)

            # ── Trade table ────────────────────────────────────────────────
            st.subheader("All Trades")
            show_cols = ["id", "symbol", "side", "entry_price", "stop_price", "target_price",
                         "qty", "score", "entry_date", "exit_price", "exit_date", "status", "pnl", "pnl_pct"]
            disp_j = df_j[show_cols].copy()
            st.dataframe(
                disp_j.style.format({
                    "entry_price": "${:.2f}", "stop_price": "${:.2f}",
                    "target_price": "${:.2f}", "exit_price": "${:.2f}",
                    "score": "{:.2f}", "pnl": "${:,.2f}", "pnl_pct": "{:.2f}%",
                }, na_rep="—"),
                use_container_width=True,
            )
            # Export journal
            st.download_button("Download Journal CSV", df_j.to_csv(index=False),
                               file_name="trade_journal.csv", mime="text/csv")

    # ═══════════════════════ ALERTS ══════════════════════════════════════════
    elif page == "Alerts":
        st.header("Price & Metric Alerts")
        ac1, ac2 = st.columns([2, 1])
        with ac1:
            alert_sym = st.text_input("Symbol").upper()
        with ac2:
            alert_type = st.selectbox("Alert type", ["Price Above", "Price Below", "RSI Above", "RSI Below"])
        alert_val = st.number_input("Value", value=0.0, step=0.1)
        if st.button("Create Alert") and alert_sym and alert_val:
            key_map = {"Price Above": ("price", "above"), "Price Below": ("price", "below"),
                       "RSI Above": ("rsi", "above"), "RSI Below": ("rsi", "below")}
            atype, acomp = key_map[alert_type]
            watchlist_mgr.add_alert(alert_sym, atype, alert_val, acomp)
            st.success(f"Alert created for {alert_sym}")

        st.subheader("Active Alerts")
        all_alerts = watchlist_mgr.get_all_alerts()
        if all_alerts:
            for sym, alerts in all_alerts.items():
                with st.expander(f"{sym} ({len(alerts)} alerts)"):
                    for i, alert in enumerate(alerts):
                        ai1, ai2 = st.columns([4, 1])
                        with ai1:
                            st.write(f"**{alert['type'].upper()}** {alert['comparison']} {alert['value']}")
                        with ai2:
                            if st.button("Delete", key=f"del_{sym}_{i}"):
                                watchlist_mgr.remove_alert(sym, i)
                                st.rerun()
        else:
            st.info("No alerts set.")

    # ═══════════════════════ SETTINGS ════════════════════════════════════════
    elif page == "Information":
        st.header("ℹ️ Information & Guide")
        st.caption("How SwingTrade Pro works, what each page is for, when to run it, "
                   "and the limits of what it can — and can't — tell you.")

        st.warning(
            "**Not financial advice.** This is a research and screening tool, not a proven "
            "profitable strategy. It has **no live track record**, and its backtest metrics are "
            "**survivorship-biased** (the universe is today's listed names). Treat every signal as "
            "a starting point for your own research — never an instruction to trade."
        )

        tab_guide, tab_timing, tab_glossary, tab_data, tab_method = st.tabs(
            ["📄 Page guide", "⏰ When to run", "📖 Glossary", "🗄️ Data & sources", "⚖️ Method & limits"]
        )

        with tab_guide:
            st.markdown(
                "| Page | What it's for |\n"
                "|---|---|\n"
                "| **Screener** | The core workflow: scans the universe for trend setups, sizes "
                "positions (half-Kelly), and drills into any ticker with chart, news & forecast. |\n"
                "| **Pre-Market Movers** | Biggest gainers/losers right now (pre-market quotes when "
                "that session is open, else regular session). |\n"
                "| **Live Movers** | Raw Yahoo predefined-screener feed — no signals, no filters. "
                "Just who's moving. |\n"
                "| **After-Hours & IPOs** | Post-market movers (~4–8pm ET) plus a curated recent-IPO "
                "tracker. |\n"
                "| **Whale Movements** | Infers large-money footprints from daily volume/$-traded/"
                "closing-strength → a 0–100 whale score. |\n"
                "| **Options Flow** | Single-ticker options analysis (IV rank, put/call, unusual "
                "volume, Greeks) + a small flow scan. |\n"
                "| **Predictions** | Next-session forecast (Chronos → heuristic fallback) + a "
                "5-session drill-down cone. |\n"
                "| **Auto Watchlist** | Auto-built watchlist of the strongest current signals. |\n"
                "| **ETF Screener** | Screen ETFs by category (sector, broad market, volatility, "
                "commodities, bonds). |\n"
                "| **Market Events** | Market-wide news + a mood gauge (news tone + VIX + breadth + "
                "SPY trend). |\n"
                "| **Heat Map** | Visual map of moves across the universe. |\n"
                "| **Watchlists / Compare** | Save names to track; compare several side by side. |\n"
                "| **P&L Tracker** | Manual trade journal — log signals and track realized P&L. |\n"
                "| **Alerts** | Threshold alerts on watched names. |\n"
                "| **Settings** | Cost model, optional local AI toggles, cache/data management. |\n"
            )

        with tab_timing:
            st.markdown("#### Best time to run each screen *(all times ET)*")
            st.markdown(
                "| Screen | Best window | Why |\n"
                "|---|---|---|\n"
                "| **Pre-Market Movers** | **8:00–9:15 AM** | Overnight news, earnings, European "
                "session and 8:30 econ data are priced in; real volume, gap mostly formed. |\n"
                "| **Screener** | After 9:45 AM, or evening | Lets the opening gap settle; or scan "
                "after the close to plan tomorrow. |\n"
                "| **Live Movers** | Intraday (9:30 AM–4:00 PM) | It's a live session feed. |\n"
                "| **After-Hours & IPOs** | **4:00–8:00 PM** | Post-market fields are empty outside "
                "this window. |\n"
                "| **Predictions / Auto Watchlist** | After the close | Uses completed daily bars; "
                "stable until the next session. |\n"
            )
            st.info(
                "**On data days, wait until after 8:30 AM.** CPI / jobs / FOMC releases reshuffle "
                "the movers completely. And always check the **Volume** column — a big % move on "
                "tiny volume usually fades at the open."
            )

        with tab_glossary:
            st.markdown("Every label used across the dashboard:")
            _render_legend()

        with tab_data:
            st.markdown(
                "- **yfinance** — quotes, history, fundamentals, news, predefined screeners.\n"
                "- **Nasdaq Trader symbol directory** — the tradable universe (cached 24h).\n"
                "- **Google News RSS** — broad free news (on-demand single-ticker & market views only).\n"
                "- **Alpaca** — execution (paper/live bracket orders); offline if no keys in `.env`.\n"
            )
            st.caption(
                "All network calls are retry-wrapped (Yahoo 401/429 are usually transient). Data is "
                "free, delayed/best-effort, and occasionally wrong — corporate actions, thin-volume "
                "prints and bad ticks happen. Caches refresh on their own TTLs; force a refresh per "
                "page or via Settings → Clear all caches."
            )

        with tab_method:
            st.markdown("""
**Signals** — Trend + RSI · MACD · Bollinger Bands · ATR-based stops · Volume surge, combined
into a 0–1 score with an entry / stop / target and a reward-to-risk ratio.

**Risk** — Half-Kelly position sizing with shrinkage, a portfolio heat limit, and a daily
circuit breaker. Kelly priors are calibrated **once, out-of-sample** on walk-forward trades —
not on the window being traded.

**Backtest** — Vectorized walk-forward simulation, **net of costs** (slippage + commission,
editable in Settings). Reports win rate, profit factor and Sharpe.

**Optional AI** — Local, open-source models (price forecasting, news event tagging,
summarization, novelty), each with a heuristic fallback. Off by default; toggle in Settings.
            """)
            st.error(
                "**Read this before trusting any number.** The backtest is survivorship-biased "
                "(point-in-time data isn't wired up yet), so its win rate / profit factor are "
                "optimistic. The signals are standard, widely-arbitraged indicators with no proven "
                "edge, and there is no forward paper-trading record. The only honest way to know if "
                "the strategy works is to paper-trade it forward for months and compare to SPY "
                "buy-and-hold **after costs**."
            )

        st.markdown("---")
        st.caption("SwingTrade Pro · for research & education only · you alone are responsible for "
                   "your trades.")

    elif page == "YouTube":
        st.header("📺 YouTube — what top traders are saying")
        st.caption("Scans recent uploads from a curated set of finance YouTubers, reads the full "
                   "transcript, and surfaces tickers, calls, pullback/merger chatter — and a "
                   "running track record of who's actually right. Free & key-less.")
        st.warning("**Opinions, not signals.** Finfluencer picks frequently underperform the "
                   "market. Treat mentions as leads to research, and watch the track record below "
                   "before weighting anyone's view.")

        yc1, yc2, yc3 = st.columns([1.2, 2, 1])
        with yc1:
            within_hours = st.slider("Lookback (hours)", 12, 48, 48, 6)
        with yc2:
            picked = st.multiselect("Channels", list(yt.TRADER_CHANNELS.values()),
                                    default=list(yt.TRADER_CHANNELS.values()))
        with yc3:
            st.write("")
            if st.button("↻ Refresh", use_container_width=True):
                fetch_yt_uploads.clear()
                get_yt_transcript.clear()
                get_yt_channel_id.clear()

        handle_by_name = {name: handle for handle, name in yt.TRADER_CHANNELS.items()}
        selected_handles = [(handle_by_name[n], n) for n in picked]

        universe = get_universe_set()
        analyzer = get_sentiment_analyzer(config)
        event_clf = get_event_classifier() if _ai_on("ai_events") else None
        summarizer = get_summarizer() if _ai_on("ai_summary") else None

        # ── Fetch + analyze ──────────────────────────────────────────────────
        analyses: List[yt.VideoAnalysis] = []
        skipped_channels: List[str] = []
        with st.spinner("Resolving channels & fetching recent uploads…"):
            uploads: List[yt.Upload] = []
            for handle, name in selected_handles:
                cid = get_yt_channel_id(handle)
                if not cid:
                    skipped_channels.append(name)
                    continue
                uploads += fetch_yt_uploads(cid, name, float(within_hours))

        uploads.sort(key=lambda u: u.published_ts, reverse=True)
        uploads = uploads[:40]  # cap transcript fetches so the scan stays responsive

        if uploads:
            prog = st.progress(0.0, text="Reading transcripts…")
            for i, up in enumerate(uploads):
                segs = get_yt_transcript(up.video_id)
                analyses.append(yt.analyze_video(
                    up, segs, universe,
                    analyzer=analyzer, event_classifier=event_clf, summarizer=summarizer))
                prog.progress((i + 1) / len(uploads), text=f"Analyzed {i + 1}/{len(uploads)} videos")
            prog.empty()

        if skipped_channels:
            st.caption("Couldn't resolve: " + ", ".join(skipped_channels) +
                       " (handle renamed or rate-limited).")

        if not analyses:
            st.info("No uploads from the selected channels in the last "
                    f"{within_hours}h. Widen the lookback or pick more channels.")
        else:
            # Persist any new extracted picks (idempotent), then grade the whole history.
            store = load_yt_store()
            new_picks = sum(yt.record_picks(store, a, yt_current_price) for a in analyses)
            if new_picks:
                save_yt_store(store)
            graded = yt.grade_picks(store.get("picks", []), yt_current_price, yt_spy_return_since)
            board = yt.creator_leaderboard(graded)

            transcribed = sum(1 for a in analyses if a.has_transcript)
            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Videos", len(analyses))
            mm2.metric("With transcript", f"{transcribed}/{len(analyses)}")
            mm3.metric("Tickers mentioned", len({t for a in analyses for t in a.tickers}))
            mm4.metric("Picks tracked", len(store.get("picks", [])))

            # ── Creator track record ─────────────────────────────────────────
            st.subheader("🏆 Creator track record")
            st.caption("Graded vs actual price move and SPY over the same window. Builds up as "
                       "picks accrue — sparse at first.")
            if board:
                bdf = pd.DataFrame(board)[["channel", "picks", "graded", "win_rate", "avg_alpha_pct"]]
                bdf = bdf.rename(columns={"channel": "Creator", "picks": "Picks", "graded": "Graded",
                                          "win_rate": "Win rate", "avg_alpha_pct": "Avg alpha"})
                st.dataframe(
                    bdf.style.format({"Win rate": "{:.2%}", "Avg alpha": "{:+.2f}%"}, na_rep="—")
                       .map(lambda v: _hl_pct(v), subset=["Avg alpha"]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("No graded picks yet — check back after a few scans across some days.")

            # ── Consensus: what they're talking about ────────────────────────
            st.subheader("🗣️ What they're talking about")
            consensus = yt.ticker_consensus(analyses)
            wl_mgr = get_watchlist_manager()
            for c in consensus[:8]:
                sym = c["ticker"]
                rc1, rc2, rc3, rc4, rc5 = st.columns([1, 1.4, 1.4, 1.4, 1.2])
                rc1.markdown(f"**{sym}**")
                price = yt_current_price(sym)
                rc2.markdown(f"${price:.2f}" if price else "—")
                bull = c["bull_pct"]
                rc3.markdown(f"{bull:.0%} bull" if bull is not None else "—")
                rc4.markdown(f"{c['videos']} vids · {c['channels']} ch · conv {c['conviction']:.2f}")
                if rc5.button("+ Watchlist", key=f"yt_wl_{sym}"):
                    wl_mgr.add_symbol("YouTube", sym)
                    st.toast(f"Added {sym} to YouTube watchlist")

            # ── Pullbacks & mergers ──────────────────────────────────────────
            pc1, pc2 = st.columns(2)
            with pc1:
                st.subheader("📉 Pullbacks & impacting statements")
                hits = [a for a in analyses if a.flags.get("pullback")]
                if not hits:
                    st.caption("Nothing flagged.")
                for a in hits[:6]:
                    st.markdown(f"**{a.upload.channel}** — [{a.upload.title[:70]}]({a.upload.url})")
                    for snip in a.flags["pullback"][:2]:
                        st.caption(snip)
            with pc2:
                st.subheader("🤝 Merger news & rumors")
                hits = [a for a in analyses if a.flags.get("merger") or "M&A" in a.events]
                if not hits:
                    st.caption("Nothing flagged.")
                for a in hits[:6]:
                    st.markdown(f"**{a.upload.channel}** — [{a.upload.title[:70]}]({a.upload.url})")
                    for snip in a.flags.get("merger", [])[:2]:
                        st.caption(snip)

            # ── Extracted calls this scan ────────────────────────────────────
            st.subheader("🎯 Extracted calls (this scan)")
            pick_rows = [{
                "Creator": a.upload.channel, "Ticker": p.ticker,
                "Dir": (p.direction or "—").upper(), "Target": p.price_target,
                "Horizon": p.horizon or "—",
                "At": f"{p.timestamp_sec // 60}:{p.timestamp_sec % 60:02d}",
            } for a in analyses for p in a.picks]
            if pick_rows:
                st.dataframe(pd.DataFrame(pick_rows).style.format({"Target": "${:.2f}"}, na_rep="—"),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("No explicit buy/sell/target calls parsed this scan.")

            # ── Per-video detail ─────────────────────────────────────────────
            st.subheader("📂 Videos")
            _sent_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
            for a in analyses:
                up = a.upload
                head = f"{_sent_emoji.get(a.sentiment, '⚪')} {up.channel} · {up.title[:80]}"
                with st.expander(head):
                    tx = "transcript" if a.has_transcript else "title+description (no transcript)"
                    st.caption(f"{up.published_str} · {a.sentiment} ({a.sentiment_score:.2f}) · {tx}"
                               + (f" · events: {', '.join(a.events)}" if a.events else ""))
                    st.markdown(f"[▶ Open video]({up.url})")
                    if a.digest:
                        st.markdown(f"**Digest:** {a.digest}")
                    if a.mentions:
                        st.markdown("**Mentions** (jump to the moment):")
                        for mn in a.mentions[:10]:
                            ts = f"{mn.timestamp_sec // 60}:{mn.timestamp_sec % 60:02d}"
                            st.markdown(f"- **{mn.ticker}** [{ts}]({mn.deep_link}) — {mn.snippet[:140]}")

        _render_legend()

    elif page == "Settings":
        st.header("Settings")

        st.subheader("Backtest Cost Model")
        st.caption("Backtest metrics are net of these costs, and the position sizer is "
                   "calibrated on out-of-sample walk-forward trades (not the window being traded).")
        bc1, bc2 = st.columns(2)
        with bc1:
            slip = st.number_input("Slippage (bps per fill)", 0.0, 100.0,
                                   float(config.slippage_bps), 1.0)
        with bc2:
            comm = st.number_input("Commission (bps per side)", 0.0, 50.0,
                                   float(config.commission_bps), 0.5)
        if (slip, comm) != (config.slippage_bps, config.commission_bps):
            st.session_state["slippage_bps"] = slip
            st.session_state["commission_bps"] = comm
            st.cache_data.clear()
            st.cache_resource.clear()
            st.info("Cost model updated — caches cleared. Re-run the Screener to recalibrate.")
            st.rerun()

        st.subheader("AI Features (open-source, local)")
        st.caption("Each loads a Hugging Face model on first use (CPU). If a model isn't "
                   "installed, the feature falls back to a heuristic. Install extras with: "
                   "`pip install transformers torch chronos-forecasting sentence-transformers`.")
        ai1, ai2 = st.columns(2)
        with ai1:
            st.session_state["ai_forecast"] = st.checkbox(
                "Price forecasting (Chronos)", value=_ai_on("ai_forecast"),
                help="Probabilistic p10/p50/p90 forecast + signal confirmation")
            st.session_state["ai_events"] = st.checkbox(
                "News event tagging", value=_ai_on("ai_events"),
                help="Zero-shot earnings / M&A / downgrade / lawsuit tags")
        with ai2:
            st.session_state["ai_summary"] = st.checkbox(
                "News summarization digest", value=_ai_on("ai_summary"),
                help="distilbart digest of recent headlines")
            st.session_state["ai_novelty"] = st.checkbox(
                "Semantic news novelty", value=_ai_on("ai_novelty"),
                help="Dedup recycled headlines via sentence-transformers")
        st.caption("Forecast appears on Auto Watchlist + Screener drill-down; news AI on the "
                   "Screener drill-down.")

        st.subheader("Cache Management")
        if st.button("Clear all caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared!")

        st.subheader("About")
        st.markdown("""
**SwingTrade Pro Dashboard**

Signal engine: RSI · MACD · Bollinger Bands · ATR stops · Volume surge
Risk: Half-Kelly sizing · Portfolio heat limit · Daily circuit breaker
Backtest: Walk-forward vectorized simulation · Win rate · Profit factor · Sharpe
Execution: Alpaca bracket orders (entry / stop-loss / take-profit)
        """)

        data_dir = Path(".data")
        if data_dir.exists():
            st.subheader("Data Files")
            for f in data_dir.iterdir():
                if f.is_file():
                    st.write(f"✅ `{f.name}` — {f.stat().st_size:,} bytes")
            if st.button("Reset all data"):
                import shutil
                shutil.rmtree(data_dir)
                st.success("Data reset!")
                st.rerun()


if __name__ == "__main__":
    run_dashboard()
