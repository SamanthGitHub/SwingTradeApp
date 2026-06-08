"""
Macro filters for context-aware trading.
Checks VIX, Fed calendar, economic events before allowing entries.
"""

import calendar
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import yfinance as yf

# Scheduled 2026 FOMC decision dates (second day of each meeting).
_FOMC_2026 = [
    datetime(2026, 1, 28), datetime(2026, 3, 18), datetime(2026, 4, 29),
    datetime(2026, 6, 17), datetime(2026, 7, 29), datetime(2026, 9, 16),
    datetime(2026, 10, 28), datetime(2026, 12, 9),
]


class MacroContext:
    """Fetches and tracks macro conditions that affect trading edge."""

    def __init__(self):
        self.vix_history: Dict[datetime, float] = {}
        self.fed_dates: List[Tuple[datetime, str]] = self._build_fed_calendar()

    # ── VIX Monitoring ─────────────────────────────────────────────────────────

    def fetch_vix(self) -> Optional[float]:
        """Get current VIX level. >25 = high volatility/stress."""
        try:
            vix_data = yf.Ticker("^VIX").history(period="5d")
            if vix_data.empty:
                return None
            return float(vix_data["Close"].iloc[-1])
        except Exception:
            return None

    def fetch_vix_ma(self, days: int = 20) -> Optional[float]:
        """Get VIX simple moving average (trend)."""
        try:
            vix_data = yf.Ticker("^VIX").history(period="90d")
            if len(vix_data) < days:
                return None
            return float(vix_data["Close"].rolling(days).mean().iloc[-1])
        except Exception:
            return None

    def is_high_volatility_regime(self) -> bool:
        """Returns True if VIX > 25 (market stress, avoid new entries)."""
        vix = self.fetch_vix()
        return vix is not None and vix > 25.0

    def is_complacent_regime(self) -> bool:
        """Returns True if VIX < 12 (very low vol, low signal reliability)."""
        vix = self.fetch_vix()
        return vix is not None and vix < 12.0

    # ── Fed Calendar ───────────────────────────────────────────────────────────

    def _build_fed_calendar(self) -> List[Tuple[datetime, str]]:
        """Hardcoded 2024-2025 Fed calendar (update annually)."""
        return [
            (datetime(2024, 1, 31), "FOMC Decision"),
            (datetime(2024, 3, 20), "FOMC Decision"),
            (datetime(2024, 5, 1), "FOMC Decision"),
            (datetime(2024, 6, 18), "FOMC Decision"),
            (datetime(2024, 7, 31), "FOMC Decision"),
            (datetime(2024, 9, 18), "FOMC Decision"),
            (datetime(2024, 11, 7), "FOMC Decision"),
            (datetime(2024, 12, 18), "FOMC Decision"),
            (datetime(2025, 1, 29), "FOMC Decision"),
            (datetime(2025, 3, 19), "FOMC Decision"),
            (datetime(2025, 5, 7), "FOMC Decision"),
            (datetime(2025, 6, 18), "FOMC Decision"),
            # CPI (second Tuesday of month)
            (datetime(2024, 2, 13), "CPI Release"),
            (datetime(2024, 3, 12), "CPI Release"),
            (datetime(2024, 4, 10), "CPI Release"),
            (datetime(2024, 5, 15), "CPI Release"),
            (datetime(2024, 6, 12), "CPI Release"),
            (datetime(2024, 7, 11), "CPI Release"),
            (datetime(2024, 8, 14), "CPI Release"),
            (datetime(2024, 9, 11), "CPI Release"),
            (datetime(2024, 10, 10), "CPI Release"),
            (datetime(2024, 11, 13), "CPI Release"),
            (datetime(2024, 12, 11), "CPI Release"),
            (datetime(2025, 1, 14), "CPI Release"),
            # Jobs (first Friday of month)
            (datetime(2024, 2, 2), "Jobs Report"),
            (datetime(2024, 3, 8), "Jobs Report"),
            (datetime(2024, 4, 5), "Jobs Report"),
            (datetime(2024, 5, 3), "Jobs Report"),
            (datetime(2024, 6, 7), "Jobs Report"),
            (datetime(2024, 7, 5), "Jobs Report"),
            (datetime(2024, 8, 2), "Jobs Report"),
            (datetime(2024, 9, 6), "Jobs Report"),
            (datetime(2024, 10, 4), "Jobs Report"),
            (datetime(2024, 11, 1), "Jobs Report"),
            (datetime(2024, 12, 6), "Jobs Report"),
            (datetime(2025, 1, 10), "Jobs Report"),
        ] + [(d, "FOMC Decision") for d in _FOMC_2026]

    # ── Upcoming events (programmatic; recurring releases generated on the fly) ──

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
        """The n-th given weekday of a month (e.g. 1st Friday, 2nd Wednesday)."""
        count, day = 0, 1
        last = calendar.monthrange(year, month)[1]
        while day <= last:
            if datetime(year, month, day).weekday() == weekday:
                count += 1
                if count == n:
                    return datetime(year, month, day)
            day += 1
        return datetime(year, month, last)

    def get_upcoming_macro_events(self, days_ahead: int = 45) -> List[Tuple[datetime, str]]:
        """Upcoming scheduled US macro events in the window (FOMC + recurring Jobs/CPI).

        Jobs (first Friday) and CPI (~second Wednesday) dates are generated for the next
        few months so the calendar never goes stale.
        """
        now = datetime.now()
        end = now + timedelta(days=days_ahead)
        events: List[Tuple[datetime, str]] = list(self.fed_dates)
        for i in range(4):
            mm = (now.month - 1 + i) % 12 + 1
            yy = now.year + (now.month - 1 + i) // 12
            events.append((self._nth_weekday(yy, mm, calendar.FRIDAY, 1), "Jobs Report (est.)"))
            events.append((self._nth_weekday(yy, mm, calendar.WEDNESDAY, 2), "CPI Release (est.)"))
        upcoming = {(d, n) for d, n in events if now.date() <= d.date() <= end.date()}
        return sorted(upcoming)

    def is_near_fed_event(self, hours_before: int = 24, hours_after: int = 4) -> Tuple[bool, Optional[str]]:
        """Check if trading near major Fed event. Returns (is_near_event, event_name)."""
        now = datetime.now()
        window_start = now - timedelta(hours=hours_after)
        window_end = now + timedelta(hours=hours_before)

        for event_date, event_name in self.fed_dates:
            if window_start <= event_date <= window_end:
                return True, event_name

        return False, None

    # ── Economic Context ───────────────────────────────────────────────────────

    def fetch_market_breadth(self) -> Optional[Dict[str, float]]:
        """
        Fetch broad market advance/decline info.
        Returns % of SPX stocks above their 200-day MA.
        """
        try:
            # Proxy: check if SPX near 200MA
            spy = yf.Ticker("SPY").history(period="300d")
            sma_200 = spy["Close"].rolling(200).mean()
            breadth_pct = (spy["Close"].iloc[-20:] > sma_200.iloc[-20:]).sum() / 20
            return {"bullish_breadth_pct": float(breadth_pct)}
        except Exception:
            return None

    def fetch_put_call_ratio(self) -> Optional[float]:
        """
        Market-wide put/call ratio via VIX put/call data.
        > 1.0 = bearish, < 0.5 = bullish
        Returns None if unavailable.
        """
        # Note: Requires premium data source. This is a placeholder.
        # In production, use market options data via Polygon, IB, or Alpaca.
        return None

    # ── Macro-Aware Position Sizing ────────────────────────────────────────────

    def get_macro_risk_adjustment(self) -> float:
        """
        Returns multiplier for position size based on macro conditions.
        Example: 0.5x in high VIX, 1.0x in normal, 0.8x in low VIX.
        """
        vix = self.fetch_vix()
        if vix is None:
            return 1.0

        if vix > 30:
            return 0.3  # Extreme stress: reduce position size 70%
        elif vix > 25:
            return 0.5  # High vol: reduce 50%
        elif vix > 20:
            return 0.8  # Elevated: reduce 20%
        elif vix < 12:
            return 0.8  # Complacent: reduce 20% (low reliability)
        else:
            return 1.0  # Normal: full size

    def should_skip_entries(self) -> Tuple[bool, str]:
        """
        Returns (should_skip, reason) for when to avoid new entries.
        """
        # High volatility
        vix = self.fetch_vix()
        if vix and vix > 30:
            return True, f"VIX={vix:.0f} (extreme stress)"

        # Fed event risk
        is_near, event = self.is_near_fed_event(hours_before=24, hours_after=2)
        if is_near:
            return True, f"Near {event}"

        # Very low breadth
        breadth = self.fetch_market_breadth()
        if breadth and breadth["bullish_breadth_pct"] < 0.4:
            return True, f"Low breadth ({breadth['bullish_breadth_pct']:.0%} bullish)"

        return False, "OK"


# Example usage in main.py:
# macro = MacroContext()
# skip, reason = macro.should_skip_entries()
# if skip:
#     print(f"Skip entries: {reason}")
#     continue
#
# adjustment = macro.get_macro_risk_adjustment()
# position_size = position_sizer.size_position(signal, account_size) * adjustment
