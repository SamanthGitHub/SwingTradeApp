import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import TradingConfig
from .providers import YFinanceProvider
from .retry import with_retry
from .tickers import get_tradable_universe

logger = logging.getLogger(__name__)


@with_retry()
def _yf_screen(predefined: str, count: int = 100):
    import yfinance as yf
    return yf.screen(predefined, count=count)


@with_retry()
def _yf_history(symbol: str, period: str):
    import yfinance as yf
    return yf.Ticker(symbol).history(period=period)


@dataclass
class AssetCandidate:
    symbol: str
    market_cap: float
    price: float
    avg_volume: float
    spread_pct: float
    float_shares: Optional[int] = None


class UniverseFilter:
    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        # Use YFinanceProvider for fundamentals (yfinance-only mode)
        self.fundamentals_provider = YFinanceProvider()

    def fetch_screened_symbols(self) -> List[str]:
        logger.debug('Fetching listed US common-stock universe (Nasdaq Trader directory)...')
        # Live exchange-listed common stocks; excludes ETFs, test issues, warrants/units.
        return get_tradable_universe()

    def is_liquid_asset(self, symbol: str) -> bool:
        logger.debug('Checking liquidity profile for %s', symbol)
        fundamentals = self.fundamentals_provider.fetch_fundamentals(symbol)
        profile = fundamentals.get('profile', {}) if isinstance(fundamentals, dict) else {}
        market_cap = float(profile.get('mktCap', 0))
        price = float(profile.get('price', 0.0))
        avg_volume = float(profile.get('avgVolume', 0))

        spread_pct = 0.0
        if market_cap < self.config.min_market_cap:
            return False
        if price < self.config.min_price:
            return False
        if avg_volume < self.config.min_avg_volume:
            return False
        if spread_pct > self.config.max_spread_pct:
            return False
        return True

    def get_daily_universe(self) -> List[str]:
        candidates = self.fetch_screened_symbols()
        return [symbol for symbol in candidates if self.is_liquid_asset(symbol)]


class PreMarketScanner:
    def __init__(self, config: TradingConfig) -> None:
        self.config = config

    def is_gap_up_candidate(self, symbol: str) -> bool:
        """
        Check if stock is gapping up at market open.
        Criteria: gap% >= threshold, volume surge, float < threshold
        """
        try:
            hist = _yf_history(symbol, "5d")
            if len(hist) < 2:
                return False

            yesterday_close = hist["Close"].iloc[-2]
            today_open = hist["Open"].iloc[-1]
            gap_pct = (today_open - yesterday_close) / yesterday_close

            # Today's volume
            today_volume = hist["Volume"].iloc[-1]
            avg_volume = hist["Volume"].iloc[:-1].mean()

            # Criteria
            gap_above_threshold = gap_pct >= self.config.gap_pct_threshold
            volume_above_threshold = today_volume >= self.config.premarket_volume_threshold
            volume_surge = today_volume >= avg_volume * 1.5

            return gap_above_threshold and (volume_above_threshold or volume_surge)
        except Exception:
            return False

    def fetch_movers(self, top_n: int = 25, min_change_pct: float = 1.0) -> List[Dict[str, Any]]:
        """Live pre-market / session movers via Yahoo's predefined screeners.

        Prefers pre-market fields when the pre-market session is active, otherwise
        falls back to the regular-session change. Returns gainers and losers merged,
        sorted by the magnitude of the move.
        """
        rows: Dict[str, Dict[str, Any]] = {}
        try:
            for screen in ("day_gainers", "day_losers", "most_actives"):
                try:
                    result = _yf_screen(screen, count=100)
                except Exception:
                    continue
                quotes = result.get("quotes", []) if isinstance(result, dict) else []
                for q in quotes:
                    sym = q.get("symbol")
                    if not sym or sym in rows:
                        continue
                    pre_pct = q.get("preMarketChangePercent")
                    reg_pct = q.get("regularMarketChangePercent")
                    # Use pre-market move when present (pre-market session), else regular.
                    change = pre_pct if pre_pct not in (None, 0) else reg_pct
                    session = "pre-market" if pre_pct not in (None, 0) else "regular"
                    if change is None:
                        continue
                    rows[sym] = {
                        "symbol": sym,
                        "change_pct": float(change),
                        "session": session,
                        "price": q.get("preMarketPrice") or q.get("regularMarketPrice"),
                        "regular_change_pct": reg_pct,
                        "volume": q.get("regularMarketVolume"),
                        "market_cap": q.get("marketCap"),
                        "name": q.get("shortName") or q.get("longName") or sym,
                    }
        except Exception:
            return []

        movers = [r for r in rows.values() if abs(r["change_pct"]) >= min_change_pct]
        return sorted(movers, key=lambda r: abs(r["change_pct"]), reverse=True)[:top_n]

    def fetch_gap_up_movers(self, symbols: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
        """Fetch today's biggest gap-ups from a list of symbols."""
        movers = []
        try:
            for symbol in symbols[:50]:  # Limit to avoid API rate limit
                try:
                    hist = _yf_history(symbol, "5d")
                except Exception:
                    continue
                if len(hist) < 2:
                    continue
                yesterday_close = hist["Close"].iloc[-2]
                today_open = hist["Open"].iloc[-1]
                gap_pct = (today_open - yesterday_close) / yesterday_close

                if gap_pct > 0.02:  # Only gaps > 2%
                    movers.append({
                        "symbol": symbol,
                        "gap_pct": gap_pct * 100,
                        "yesterday_close": yesterday_close,
                        "today_open": today_open,
                    })
        except Exception:
            pass

        return sorted(movers, key=lambda x: x["gap_pct"], reverse=True)[:top_n]
