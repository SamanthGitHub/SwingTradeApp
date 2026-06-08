"""Extract fundamental data from yfinance with caching."""

from typing import Any, Dict, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path


class FundamentalsExtractor:
    """Extract and cache fundamental data from yfinance."""
    
    def __init__(self, cache_hours: int = 24):
        self.cache_hours = cache_hours
        self.cache_dir = Path('.data/fundamentals_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for symbol."""
        return self.cache_dir / f"{symbol}.json"
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cache is still valid."""
        cache_path = self._get_cache_path(symbol)
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                timestamp = datetime.fromisoformat(data.get('timestamp', ''))
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                return age_hours < self.cache_hours
        except Exception:
            return False
    
    def _load_from_cache(self, symbol: str) -> Optional[Dict]:
        """Load fundamentals from cache."""
        cache_path = self._get_cache_path(symbol)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, symbol: str, data: Dict) -> None:
        """Save fundamentals to cache."""
        data['timestamp'] = datetime.now().isoformat()
        cache_path = self._get_cache_path(symbol)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a symbol with caching."""
        # Check cache first
        if self._is_cache_valid(symbol):
            cached = self._load_from_cache(symbol)
            if cached:
                cached.pop('timestamp', None)
                return cached
        
        fundamentals = {
            'symbol': symbol,
            'market_cap': None,
            'pe_ratio': None,
            'dividend_yield': None,
            'earnings_date': None,
            'avg_volume': None,
            '52_week_high': None,
            '52_week_low': None,
            'debt_to_equity': None,
            'profit_margin': None,
            'return_on_equity': None,
            'revenue': None,
            'roe': None,
            'debt_equity': None,
            'current_price': None,
            'fifty_day_avg': None,
            'two_hundred_day_avg': None,
        }
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info if hasattr(ticker, 'info') else {}
            
            # Extract available fields
            fundamentals['market_cap'] = info.get('marketCap')
            fundamentals['pe_ratio'] = info.get('trailingPE')
            fundamentals['dividend_yield'] = info.get('dividendYield')
            fundamentals['earnings_date'] = info.get('lastEarningsDate')
            fundamentals['avg_volume'] = info.get('averageVolume')
            fundamentals['52_week_high'] = info.get('fiftyTwoWeekHigh')
            fundamentals['52_week_low'] = info.get('fiftyTwoWeekLow')
            fundamentals['debt_to_equity'] = info.get('debtToEquity')
            fundamentals['profit_margin'] = info.get('profitMargins')
            fundamentals['return_on_equity'] = info.get('returnOnEquity')
            fundamentals['revenue'] = info.get('totalRevenue')
            fundamentals['roe'] = info.get('returnOnEquity')
            fundamentals['debt_equity'] = info.get('debtToEquity')
            fundamentals['current_price'] = info.get('currentPrice')
            fundamentals['fifty_day_avg'] = info.get('fiftyDayAverage')
            fundamentals['two_hundred_day_avg'] = info.get('twoHundredDayAverage')
            
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
        
        # Cache the result
        self._save_to_cache(symbol, fundamentals.copy())
        
        return fundamentals
    
    def get_batch_fundamentals(self, symbols: list) -> pd.DataFrame:
        """Get fundamentals for multiple symbols as DataFrame."""
        data = []
        for symbol in symbols:
            try:
                fund = self.get_fundamentals(symbol)
                data.append(fund)
            except Exception:
                pass
        
        return pd.DataFrame(data) if data else pd.DataFrame()
    
    def format_value(self, value: any, value_type: str = 'number') -> str:
        """Format fundamental value for display."""
        if value is None:
            return 'N/A'
        
        try:
            if value_type == 'percent':
                return f"{float(value) * 100:.1f}%"
            elif value_type == 'market_cap':
                val = float(value)
                if val >= 1e9:
                    return f"${val/1e9:.1f}B"
                elif val >= 1e6:
                    return f"${val/1e6:.1f}M"
                return f"${val:,.0f}"
            elif value_type == 'price':
                return f"${float(value):.2f}"
            else:
                return f"{float(value):.2f}"
        except Exception:
            return str(value)
