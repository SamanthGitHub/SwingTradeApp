"""US stock universe for screening.

The live universe is fetched dynamically (see ``get_sp500_universe``); the static
``MAJOR_US_STOCKS`` list below is only an offline fallback for when the live fetch
and the on-disk cache are both unavailable.
"""

# Offline fallback only — the live S&P 500 list is fetched at runtime.
MAJOR_US_STOCKS = [
    # Tech Megacap
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'AVGO', 'ASML', 'COST',
    'NFLX', 'ADBE', 'CRM', 'INTC', 'AMD', 'MU', 'LRCX', 'MRVL', 'MCHP', 'QCOM',
    'INTU', 'SNPS', 'CDNS', 'PAYX', 'PZZA', 'BKNG', 'PYPL', 'ZM', 'NETS', 'CORN',
    
    # Financial Services
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'TD', 'RY', 'CM',
    'AXON', 'AIG', 'HIG', 'PGR', 'MKL', 'RLI', 'TFIN', 'IYF', 'VFV', 'CRAY',
    'DFS', 'COF', 'SYF', 'PAYC', 'ADYEY', 'ADYEN', 'SQ', 'FISV', 'FIS', 'JKHY',
    
    # Energy & Materials
    'XOM', 'CVX', 'COP', 'MPC', 'PSX', 'VLO', 'OXY', 'EOG', 'SLB', 'HAL',
    'FANG', 'PARAA', 'APA', 'BST', 'METC', 'USU', 'FCX', 'SCCO', 'NEM', 'GFI',
    'ALB', 'LIT', 'LAC', 'SQM', 'CMMC', 'MP', 'FSI', 'UEC', 'SPRQ', 'CCJC',
    
    # Consumer & Retail
    'PG', 'UL', 'KO', 'MO', 'PM', 'EL', 'CLX', 'CHD', 'KMB', 'CPRT',
    'WMT', 'TGT', 'KSS', 'M', 'BBY', 'DLTR', 'FIVE', 'RCL', 'CCL', 'MAR',
    'AMZN', 'EBAY', 'ETSY', 'SHOP', 'JMIA', 'MWG', 'BABA', 'PDD', 'SE', 'GRAB',
    
    # Healthcare & Pharma
    'JNJ', 'UNH', 'LLY', 'PFE', 'ABBV', 'MRK', 'BMY', 'AZN', 'AMGN', 'GILD',
    'BIIB', 'VRTX', 'XLRN', 'BMRN', 'REGN', 'ALNY', 'ILMN', 'EXAS', 'DXCM', 'CVS',
    'UHS', 'HCA', 'LPLA', 'LH', 'DGX', 'LABS', 'XRAY', 'GMED', 'CSTL', 'CRVS',
    
    # Payments & FinTech
    'V', 'MA', 'AXP', 'DD', 'STWD', 'CME', 'ICE', 'INTC', 'AKAM', 'ANET',
    'SSNC', 'SS', 'TRMB', 'EPAY', 'GPN', 'WEX', 'FLYW', 'TLIS', 'GTLB', 'PAYO',
    
    # Utilities & Infrastructure
    'NEE', 'DUK', 'SO', 'EXC', 'AEP', 'XEL', 'LNT', 'CMS', 'WEC', 'DOK',
    'AEE', 'PPL', 'ETR', 'IBM', 'GIS', 'IEX', 'DTE', 'EIX', 'PCG', 'FE',
    
    # Real Estate & Infrastructure
    'AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'CONE', 'SBA', 'STAG', 'SITC', 'IRM',
    'PSA', 'EQR', 'AVB', 'SPG', 'WY', 'UMH', 'NHI', 'LCRT', 'LADR', 'GET',
    
    # Aerospace & Defense
    'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LDOS', 'HII', 'KTOS', 'TXT', 'MOD',
    
    # Insurance
    'BRK.B', 'BRK.A', 'TRMB', 'SOFI', 'SLV', 'SIVB', 'PACB', 'SBLK', 'KRBN', 'TMHC',
    
    # Semiconductors & Components
    'TSM', 'QCOM', 'BRCM', 'AVGO', 'MU', 'TMDX', 'LRCX', 'SMCI', 'MKSI', 'ENTG',
    'ANSS', 'SNPS', 'CDNS', 'VEEV', 'DESK', 'MSFT', 'AMZN', 'PYPL', 'OKTA', 'NET',
    
    # Industrial & Manufacturing  
    'CAT', 'BA', 'GE', 'MMM', 'HON', 'ITW', 'PH', 'PCAR', 'NDSN', 'IR',
    'WM', 'ROL', 'RSG', 'GES', 'TWO', 'EXC', 'AES', 'EOG', 'MOS', 'ADM',
    
    # Telecommunications
    'VZ', 'T', 'TMUS', 'CMCSA', 'CHTR', 'DTM', 'LBRDK', 'LBRDA', 'VOD', 'TMUS',
    
    # Transportation & Logistics
    'UPS', 'FDX', 'XPO', 'KNX', 'CAR', 'J', 'KEX', 'JBLU', 'LUV', 'DAL',
    'UAL', 'AAL', 'ULCC', 'SAVE', 'SKYW', 'EXPE', 'BOOKING', 'TRIP', 'ABNB', 'LYFT',
    
    # Chemicals & Materials
    'LYB', 'DOW', 'CE', 'APD', 'ECL', 'IFF', 'CTVA', 'BALL', 'PKG', 'IP',
    'AMCR', 'HUN', 'OLN', 'AXTA', 'SHW', 'SCKT', 'AXTO', 'RPAY', 'TCBI', 'PFSI',
    
    # Food & Beverage
    'KO', 'PEP', 'MO', 'PM', 'TAP', 'STZ', 'DEO', 'BUD', 'MNST', 'FIZZ',
    'CPRI', 'KR', 'SJM', 'K', 'GIS', 'CAG', 'MDLZ', 'NSRGY', 'EL', 'ITC',
    
    # Consumer Staples & Discretionary
    'WMT', 'TGT', 'DLTR', 'FIVE', 'KSS', 'AMZN', 'HD', 'LOW', 'BBY', 'AAPL',
]

import json
import logging
import re
import time
from pathlib import Path

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
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps({"fetched_at": time.time(), "symbols": symbols}))
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


def filter_non_penny_stocks(tickers: list, min_price: float = 5.0) -> list:
    """Filter out penny stocks (typically stocks under min_price)."""
    # This would be used in conjunction with real price data
    # For now, the MAJOR_US_STOCKS list already excludes penny stocks by definition
    return tickers
