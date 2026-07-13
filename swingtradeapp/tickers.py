"""US stock universe for screening.

The live universe is fetched dynamically (see ``get_sp500_universe``); the static
``MAJOR_US_STOCKS`` list below is only an offline fallback for when the live fetch
and the on-disk cache are both unavailable.
"""

# Offline fallback only — the live universe is fetched at runtime. Deduped, currently-listed,
# liquid large/mid caps (last audited July 2026); delisted/renamed names were purged (SIVB,
# CRAY, BRCM, FISV, SQ, XLRN, CONE, BOOKING, …) along with ETFs and bogus symbols.
MAJOR_US_STOCKS = [
    # Tech megacap & software
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AVGO', 'ORCL', 'COST',
    'NFLX', 'ADBE', 'CRM', 'INTC', 'AMD', 'MU', 'LRCX', 'MRVL', 'MCHP', 'QCOM',
    'INTU', 'SNPS', 'CDNS', 'NOW', 'BKNG', 'PYPL', 'UBER', 'SHOP', 'PLTR', 'ANET',
    'TSM', 'ASML', 'SMCI', 'ENTG', 'MKSI', 'ANSS', 'OKTA', 'NET', 'GTLB', 'AKAM',
    # Financials & payments
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'USB', 'PNC',
    'AXP', 'V', 'MA', 'CME', 'ICE', 'PGR', 'AIG', 'HIG', 'MKL', 'SOFI',
    'DFS', 'COF', 'SYF', 'PAYC', 'FIS', 'JKHY', 'GPN', 'WEX', 'PAYX', 'BRK-B',
    # Energy & materials
    'XOM', 'CVX', 'COP', 'MPC', 'PSX', 'VLO', 'OXY', 'EOG', 'SLB', 'HAL',
    'FANG', 'APA', 'FCX', 'SCCO', 'NEM', 'ALB', 'SQM', 'MP', 'CAT', 'MOS',
    # Consumer & retail
    'PG', 'KO', 'PEP', 'PM', 'MO', 'EL', 'CLX', 'CHD', 'KMB', 'MDLZ',
    'WMT', 'TGT', 'HD', 'LOW', 'BBY', 'DLTR', 'FIVE', 'RCL', 'CCL', 'MAR',
    'EBAY', 'ETSY', 'BABA', 'PDD', 'SE', 'ABNB', 'EXPE', 'MNST', 'STZ', 'KR',
    # Healthcare & pharma
    'JNJ', 'UNH', 'LLY', 'PFE', 'ABBV', 'MRK', 'BMY', 'AZN', 'AMGN', 'GILD',
    'BIIB', 'VRTX', 'BMRN', 'REGN', 'ALNY', 'ILMN', 'EXAS', 'DXCM', 'CVS', 'HCA',
    'UHS', 'LH', 'DGX', 'ISRG', 'MDT', 'TMO', 'DHR', 'ABT', 'SYK', 'BSX',
    # Industrials, defense & transport
    'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LDOS', 'HII', 'TXT', 'AXON', 'GE',
    'MMM', 'HON', 'ITW', 'PH', 'PCAR', 'NDSN', 'IR', 'WM', 'RSG', 'ADM',
    'UPS', 'FDX', 'XPO', 'KNX', 'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'CPRT',
    # Utilities, telecom & real estate
    'NEE', 'DUK', 'SO', 'EXC', 'AEP', 'XEL', 'WEC', 'ETR', 'PCG', 'FE',
    'VZ', 'T', 'TMUS', 'CMCSA', 'CHTR', 'IBM', 'AMT', 'PLD', 'CCI', 'EQIX',
    'DLR', 'IRM', 'PSA', 'EQR', 'AVB', 'SPG', 'O', 'WY', 'SHW', 'APD',
]

import json
import logging
import re
import time
from pathlib import Path

from .jsonstore import atomic_write_json
from .retry import with_retry

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(".data/universe.json")
_CACHE_TTL = 24 * 3600  # refresh the listed-symbol directory at most once a day
_USER_AGENT = "SwingTradeApp/1.0 (contact: ops@swingtradeapp.local)"

# Official Nasdaq Trader symbol directory (pipe-delimited, updated nightly).
# nasdaqlisted = Nasdaq issues; otherlisted = NYSE / NYSE American / Arca / BATS / IEX.
_NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
_OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

# Common-stock ticker shape: 1–5 letters, optional single-letter share class (BRK-B, BF-B).
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")

# Security-name markers for non-common-stock instruments (warrants, units, rights,
# preferreds, notes, etc.) that the directory lists alongside ordinary shares.
_NON_COMMON_RE = re.compile(
    r"\b(WARRANTS?|UNITS?|RIGHTS?|PREFERRED|DEPOSITARY|DEBENTURES?|NOTES?|"
    r"SUBORDINAT\w*|CONVERTIBLE|BONDS?|ETN)\b",
    re.IGNORECASE,
)


@with_retry()
def _http_get(url: str, timeout: int = 20) -> str:
    import ssl
    import urllib.request

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_directory(text: str, symbol_col: str) -> list:
    """Parse a Nasdaq Trader pipe-delimited file into a list of common-stock symbols.

    Drops the trailing 'File Creation Time' footer, test issues, ETFs, and
    non-common-stock symbols (warrants, units, rights, preferreds).
    """
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    if not lines:
        return []
    header = lines[0].split("|")
    idx = {name: i for i, name in enumerate(header)}
    if symbol_col not in idx:
        return []

    name_idx = idx.get("Security Name", -1)
    out = []
    for line in lines[1:]:
        fields = line.split("|")
        if len(fields) < len(header):
            continue
        if fields[idx.get("Test Issue", -1)] == "Y":
            continue
        if "ETF" in idx and fields[idx["ETF"]] == "Y":
            continue
        # Nasdaq financial status: keep Normal ('N') / blank; skip deficient/delinquent/bankrupt.
        if "Financial Status" in idx and fields[idx["Financial Status"]] not in ("N", ""):
            continue
        # Drop warrants / units / rights / preferreds / notes by their security name.
        if 0 <= name_idx < len(fields) and _NON_COMMON_RE.search(fields[name_idx]):
            continue
        sym = fields[idx[symbol_col]].strip().upper().replace(".", "-")
        if _TICKER_RE.match(sym):
            out.append(sym)
    return out


def _fetch_listed_universe() -> list:
    """Fetch the full set of US exchange-listed common stocks from Nasdaq Trader."""
    nasdaq = _parse_directory(_http_get(_NASDAQ_LISTED_URL), "Symbol")
    other = _parse_directory(_http_get(_OTHER_LISTED_URL), "ACT Symbol")
    symbols = sorted(set(nasdaq) | set(other))
    if len(symbols) < 1000:
        raise ValueError(f"Unexpected listed-symbol count: {len(symbols)}")
    logger.info("Loaded %d listed symbols (%d Nasdaq, %d other)", len(symbols), len(nasdaq), len(other))
    return symbols


def _read_cache() -> list:
    try:
        return json.loads(_CACHE_PATH.read_text()).get("symbols", [])
    except Exception:
        return []


def _write_cache(symbols: list) -> None:
    try:
        atomic_write_json(_CACHE_PATH, {"fetched_at": time.time(), "symbols": symbols}, indent=0)
    except Exception:
        logger.debug("Could not persist universe cache", exc_info=True)


def get_tradable_universe(force_refresh: bool = False) -> list:
    """Return current US exchange-listed common stocks (Nasdaq Trader directory).

    Preference order: fresh on-disk cache → live fetch → stale cache → static fallback.
    """
    if not force_refresh:
        try:
            payload = json.loads(_CACHE_PATH.read_text())
            if time.time() - payload.get("fetched_at", 0) < _CACHE_TTL and payload.get("symbols"):
                return payload["symbols"]
        except Exception:
            pass

    try:
        symbols = _fetch_listed_universe()
        _write_cache(symbols)
        return symbols
    except Exception:
        logger.warning("Live universe fetch failed; falling back to cache/static list", exc_info=True)

    return _read_cache() or MAJOR_US_STOCKS


@with_retry()
def _yf_screen(predefined: str, count: int):
    import yfinance as yf
    return yf.screen(predefined, count=count)


def get_raw_screen(predefined: str, count: int = 50) -> list:
    """Raw quote dicts from one of Yahoo's predefined screeners — no filtering applied.

    Passthrough for the "who's moving" view: returns the screener's quotes exactly as Yahoo
    sends them (symbol, prices, change %, volume, market cap, pre-market fields, …). Empty list
    on any failure so callers can render gracefully.
    """
    try:
        result = _yf_screen(predefined, min(count, 250))
        return result.get("quotes", []) if isinstance(result, dict) else []
    except Exception:
        logger.debug("raw screen %s failed", predefined, exc_info=True)
        return []


def get_active_symbols(count: int = 100) -> list:
    """Most-active US equities right now, via Yahoo's predefined screener (live)."""
    try:
        result = _yf_screen("most_actives", min(count, 250))
        quotes = result.get("quotes", []) if isinstance(result, dict) else []
        return [q.get("symbol") for q in quotes if q.get("symbol")]
    except Exception:
        logger.debug("most_actives screen failed", exc_info=True)
        return []


def get_screening_universe(active_count: int = 100) -> list:
    """Listed universe ordered with today's most-active names first.

    Keeps the screener/heat-map slices on liquid stocks while still exposing the
    full listed universe behind them.
    """
    listed = get_tradable_universe()
    listed_set = set(listed)
    actives = [s for s in get_active_symbols(active_count) if s in listed_set]
    seen = set(actives)
    return actives + [s for s in listed if s not in seen]


# Backwards-compatible alias (universe is now the full listed set, not just the S&P 500).
def get_sp500_universe(force_refresh: bool = False) -> list:
    return get_tradable_universe(force_refresh=force_refresh)


def filter_non_penny_stocks(tickers: list, min_price: float = 5.0,
                            last_price: dict = None) -> list:
    """Drop names whose last close is under ``min_price``.

    ``last_price`` maps symbol → last close, from price data the caller already has in
    hand (e.g. the prefetch store) — no network call is made here. Symbols without a
    known price are kept, so a missing quote never silently shrinks the universe.
    """
    if not last_price:
        return list(tickers)
    kept = []
    for t in tickers:
        p = last_price.get(t)
        if p is None or p >= min_price:
            kept.append(t)
    return kept
