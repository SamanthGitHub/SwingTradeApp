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


class ProviderFactory:
    def __init__(self, config: Any) -> None:
        self.config = config

    def create_polygon_provider(self) -> YFinanceProvider:
        return YFinanceProvider()

    def create_theta_provider(self) -> YFinanceProvider:
        return YFinanceProvider()

    def create_fmp_provider(self) -> YFinanceProvider:
        return YFinanceProvider()
