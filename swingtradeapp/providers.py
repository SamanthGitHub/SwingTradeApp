import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .retry import with_retry

logger = logging.getLogger(__name__)


@with_retry()
def _ticker_info(symbol: str) -> Dict[str, Any]:
    """Fetch yfinance ``.info`` with retry/backoff on transient (rate-limit) errors."""
    import yfinance as yf
    t = yf.Ticker(symbol)
    return t.info if hasattr(t, "info") else {}


class DataProviderBase(ABC):
    @abstractmethod
    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError()


class YFinanceProvider(DataProviderBase):
    """Free market and fundamental data via yfinance (Yahoo Finance)."""

    def _get_ticker(self, symbol: str):
        try:
            import yfinance as yf
            return yf.Ticker(symbol)
        except Exception:
            return None

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        try:
            info = _ticker_info(symbol)
        except Exception:
            info = {}
        return {
            'ticker': symbol,
            'last': info.get('regularMarketPrice') or info.get('previousClose'),
            'ask': info.get('ask'),
            'bid': info.get('bid'),
        }

    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        try:
            info = _ticker_info(symbol)
        except Exception:
            info = {}
        if not info:
            return {'profile': {'mktCap': 0, 'price': 0.0, 'avgVolume': 0}}
        return {
            'profile': {
                'mktCap': info.get('marketCap', 0),
                'price': info.get('regularMarketPrice') or info.get('previousClose', 0.0),
                'avgVolume': info.get('averageVolume', 0),
            }
        }

    @with_retry()
    def _option_chain(self, symbol: str, date: str):
        import yfinance as yf
        return yf.Ticker(symbol).option_chain(date)

    def fetch_options_chain(self, symbol: str, date: str) -> Dict[str, Any]:
        try:
            chain = self._option_chain(symbol, date)
            return {
                'symbol': symbol,
                'date': date,
                'chains': {
                    'calls': chain.calls.to_dict(orient='records'),
                    'puts': chain.puts.to_dict(orient='records'),
                },
            }
        except Exception:
            return {'symbol': symbol, 'date': date, 'chains': []}


class PolygonProvider(DataProviderBase):
    """Polygon.io (rebranded "Massive") — a paid API used strictly on its FREE tier (5 req/min).

    Every call is gated by an :class:`~swingtradeapp.ratelimit.ApiBudget` so we never exceed the
    free limit (and so never get charged). On a missing key, an exhausted budget, or any HTTP
    error, methods return ``None`` / ``{}`` so callers fall back to a free source (yfinance).
    Intended for **on-demand single-symbol** lookups only — never bulk scans.
    """

    BASE = "https://api.polygon.io"

    def __init__(self, api_key: str = "", budget: Any = None) -> None:
        self.api_key = (api_key or "").strip()
        self.budget = budget

    def available(self) -> bool:
        return bool(self.api_key)

    def _reserve(self) -> bool:
        """Check the budget and, if a slot is free, reserve it (record the call up front).

        Reserving before the request guarantees we never exceed the cap even if the request
        then fails or retries — we err toward under-using, never over.
        """
        if not self.available():
            return False
        if self.budget is None:
            return True
        allowed, _ = self.budget.check("polygon")
        if not allowed:
            return False
        self.budget.record("polygon")
        return True

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # Deliberately NOT wrapped in @with_retry: the ApiBudget is our rate control, and retrying
        # on a 429/401 would fire extra HTTP requests we didn't count — breaking the "never exceed"
        # guarantee. On any error we fail fast and the caller falls back to free yfinance data.
        import requests
        p = dict(params or {})
        p["apiKey"] = self.api_key
        resp = requests.get(f"{self.BASE}{path}", params=p, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def fetch_daily_bars(self, symbol: str, days: int = 180):
        """Daily OHLCV as a DataFrame indexed by date (same shape as ``fetch_symbol_history``).

        Returns ``None`` on missing key / exhausted budget / error / empty so callers fall back.
        """
        if not self._reserve():
            return None
        import pandas as pd
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=days)
        path = f"/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
        try:
            data = self._get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})
        except Exception:
            return None
        results = data.get("results") or []
        if not results:
            return None
        rows = [{"Open": r.get("o"), "High": r.get("h"), "Low": r.get("l"),
                 "Close": r.get("c"), "Volume": r.get("v")} for r in results]
        idx = pd.to_datetime([r.get("t") for r in results], unit="ms")
        df = pd.DataFrame(rows, index=idx).dropna()
        df.index.name = "Date"
        return df if not df.empty else None

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        if not self._reserve():
            return {}
        try:
            data = self._get(f"/v2/aggs/ticker/{symbol.upper()}/prev", {"adjusted": "true"})
        except Exception:
            return {}
        res = (data.get("results") or [{}])[0]
        return {"ticker": symbol, "last": res.get("c"), "open": res.get("o"),
                "high": res.get("h"), "low": res.get("l"), "volume": res.get("v")}

    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        if not self._reserve():
            return {"profile": {"mktCap": 0, "price": 0.0, "avgVolume": 0}}
        try:
            data = self._get(f"/v3/reference/tickers/{symbol.upper()}", {})
        except Exception:
            return {"profile": {"mktCap": 0, "price": 0.0, "avgVolume": 0}}
        r = data.get("results") or {}
        return {"profile": {"mktCap": r.get("market_cap", 0), "name": r.get("name"),
                            "exchange": r.get("primary_exchange"), "price": 0.0, "avgVolume": 0}}


class ProviderFactory:
    def __init__(self, config: Any) -> None:
        self.config = config

    def create_polygon_provider(self, budget: Any = None) -> "PolygonProvider":
        """Build a budget-guarded Polygon provider from the config's key.

        Always wired to an ``ApiBudget`` (a fresh one if none is supplied) so a factory-built
        provider is still hard-capped to the free tier.
        """
        if budget is None:
            from .ratelimit import ApiBudget
            budget = ApiBudget()
        return PolygonProvider(api_key=getattr(self.config, "polygon_api_key", ""), budget=budget)

    def create_theta_provider(self) -> YFinanceProvider:
        return YFinanceProvider()

    def create_fmp_provider(self) -> YFinanceProvider:
        return YFinanceProvider()
