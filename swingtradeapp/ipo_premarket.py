"""
IPO tracking and screening for early-stage swing trades.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta

import yfinance as yf


class IPOTracker:
    """Identify and track IPOs for swing trading."""

    # Curated list of notable recent IPOs (2023-2025). MANUALLY MAINTAINED — refresh periodically
    # (like the Fed calendar in macro_filters). Kept within ~2y so fetch_ipo_performance's
    # earliest-close IPO-price proxy stays meaningful.
    RECENT_IPOS = {
        "ARM":  {"ipo_date": datetime(2023, 9, 14), "name": "Arm Holdings"},
        "CART": {"ipo_date": datetime(2023, 9, 19), "name": "Instacart (Maplebear)"},
        "KVYO": {"ipo_date": datetime(2023, 9, 20), "name": "Klaviyo"},
        "BIRK": {"ipo_date": datetime(2023, 10, 11), "name": "Birkenstock"},
        "CAVA": {"ipo_date": datetime(2023, 6, 15), "name": "Cava Group"},
        "RDDT": {"ipo_date": datetime(2024, 3, 21), "name": "Reddit"},
        "ALAB": {"ipo_date": datetime(2024, 3, 20), "name": "Astera Labs"},
        "RBRK": {"ipo_date": datetime(2024, 4, 25), "name": "Rubrik"},
        "TEM":  {"ipo_date": datetime(2024, 6, 14), "name": "Tempus AI"},
        "LINE": {"ipo_date": datetime(2024, 7, 25), "name": "Lineage"},
        "CRWV": {"ipo_date": datetime(2025, 3, 28), "name": "CoreWeave"},
        "CRCL": {"ipo_date": datetime(2025, 6, 5), "name": "Circle Internet Group"},
    }

    def is_recent_ipo(self, symbol: str, days_since_ipo: int = 365) -> bool:
        """Check if stock is a recent IPO (within N days)."""
        if symbol not in self.RECENT_IPOS:
            return False
        ipo_date = self.RECENT_IPOS[symbol]["ipo_date"]
        age_days = (datetime.now() - ipo_date).days
        return 0 < age_days < days_since_ipo

    def get_ipo_age(self, symbol: str) -> Optional[int]:
        """Get days since IPO."""
        if symbol not in self.RECENT_IPOS:
            return None
        ipo_date = self.RECENT_IPOS[symbol]["ipo_date"]
        return (datetime.now() - ipo_date).days

    def fetch_ipo_performance(self, symbol: str) -> Optional[Dict]:
        """Get IPO price vs current (not available via yfinance without prospectus)."""
        try:
            hist = yf.Ticker(symbol).history(period="2y")
            if hist.empty:
                return None
            ipo_price = hist["Close"].iloc[0]  # Approximate with earliest available
            current_price = hist["Close"].iloc[-1]
            gain_pct = (current_price - ipo_price) / ipo_price * 100
            return {
                "ipo_price_approx": float(ipo_price),
                "current_price": float(current_price),
                "gain_pct": gain_pct,
            }
        except Exception:
            return None

    def screen_ipo_candidates(
        self,
        symbol: str,
        days_since_ipo_min: int = 30,
        days_since_ipo_max: int = 180,
        min_price_change_pct: float = 20.0,
    ) -> Optional[Dict]:
        """
        Screen IPO for swing trade setup.
        Best window: 1-6 months old, significant move off IPO, consolidation forming.
        """
        age = self.get_ipo_age(symbol)
        if age is None or age < days_since_ipo_min or age > days_since_ipo_max:
            return None

        perf = self.fetch_ipo_performance(symbol)
        if perf is None:
            return None

        if abs(perf["gain_pct"]) < min_price_change_pct:
            return None  # Too close to IPO price, consolidating

        return {
            "symbol": symbol,
            "days_since_ipo": age,
            "ipo_price": perf["ipo_price_approx"],
            "current_price": perf["current_price"],
            "gain_pct": perf["gain_pct"],
            "trading_phase": "early" if age < 90 else "established",
        }


class PreAfterMarketScanner:
    """Track pre-market and after-hours price action."""

    def fetch_premarket_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch pre-market price data.
        yfinance has limited pre-market data; Alpaca is better.
        """
        try:
            # yfinance doesn't directly support pre-market
            # This is a placeholder; use Alpaca API in production
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                return None

            # Calculate gap from yesterday's close to today's open
            yesterday_close = hist["Close"].iloc[-2] if len(hist) >= 2 else None
            today_open = hist["Open"].iloc[-1]

            if yesterday_close is None:
                return None

            gap_pct = (today_open - yesterday_close) / yesterday_close * 100
            return {
                "symbol": symbol,
                "yesterday_close": float(yesterday_close),
                "today_open": float(today_open),
                "gap_pct": gap_pct,
                "gap_direction": "up" if gap_pct > 0 else "down",
            }
        except Exception:
            return None

    def fetch_afterhours_data(self, symbol: str) -> Optional[Dict]:
        """
        After-hours price and volume (also limited via yfinance).
        """
        try:
            # Similar limitation; Alpaca recommended
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                return None

            close = hist["Close"].iloc[-1]
            volume = hist["Volume"].iloc[-1]

            return {
                "symbol": symbol,
                "close": float(close),
                "volume": float(volume),
                "note": "yfinance limited; use Alpaca for true after-hours",
            }
        except Exception:
            return None

    def find_premarket_movers(
        self,
        symbols: List[str],
        gap_threshold_pct: float = 3.0,
    ) -> List[Dict]:
        """
        Find symbols with significant pre-market gaps.
        """
        movers = []
        for symbol in symbols:
            pm_data = self.fetch_premarket_data(symbol)
            if pm_data and abs(pm_data["gap_pct"]) > gap_threshold_pct:
                movers.append(pm_data)

        return sorted(movers, key=lambda x: abs(x["gap_pct"]), reverse=True)

    def find_afterhours_movers(
        self,
        symbols: List[str],
        volume_multiple: float = 2.0,
    ) -> List[Dict]:
        """
        Find after-hours volume spikes (possible next-day setup).
        """
        movers = []
        avg_volume = None

        for symbol in symbols:
            ah_data = self.fetch_afterhours_data(symbol)
            if ah_data:
                # Would need to compare to average volume
                # Placeholder implementation
                movers.append(ah_data)

        return movers
