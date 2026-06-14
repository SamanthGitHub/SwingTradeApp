"""Daily market-regime / bias gate + weekday seasonality.

The research's directive #1: *never deploy a setup in an analytical vacuum.* This module turns the
broad-market picture (SPY vs its 200-day SMA + slope, market breadth, VIX) into a single
**Trade / Caution / Stand-aside** verdict that the Setup Scanner uses to gate long ideas. It also
computes per-weekday seasonality (up-day rate, average return, overnight-gap-fill rate) — the
docx's Tue/Wed/Thu/Fri edges — on free daily data.

Pure / Streamlit-free: the caller (``app.py``) fetches & caches SPY/VIX/breadth and passes them in.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class RegimeRead:
    verdict: str               # "Trade" | "Caution" | "Stand-aside"
    score: int                 # 0–100 risk-on score
    spy_above_200: bool
    spy_trend_up: bool
    breadth_pct: Optional[float]   # fraction of recent SPY closes above the 200-SMA (0–1)
    vix: Optional[float]
    drivers: List[str] = field(default_factory=list)

    @property
    def allows_long(self) -> bool:
        """Whether new long setups are sanctioned (Stand-aside blocks them)."""
        return self.verdict != "Stand-aside"


def assess_regime(spy_closes: Sequence[float], vix: Optional[float] = None,
                  breadth_pct: Optional[float] = None) -> RegimeRead:
    """Combine SPY structure, breadth and VIX into a 0–100 risk-on score and a verdict.

    ``spy_closes`` should be ~250+ daily closes; ``breadth_pct`` the fraction of recent closes
    above the 200-SMA (e.g. from ``MacroContext.fetch_market_breadth``); ``vix`` the current level.
    """
    c = np.asarray(spy_closes, dtype=float)
    drivers: List[str] = []
    score = 50.0

    spy_above_200 = False
    spy_trend_up = False
    if len(c) >= 200:
        sma200 = float(np.mean(c[-200:]))
        spy_above_200 = bool(c[-1] > sma200)
        score += 20 if spy_above_200 else -20
        drivers.append(f"SPY {'above' if spy_above_200 else 'below'} 200-SMA")
        if len(c) >= 220:
            sma200_prev = float(np.mean(c[-220:-20]))
            spy_trend_up = sma200 > sma200_prev
            score += 10 if spy_trend_up else -10
            drivers.append(f"200-SMA {'rising' if spy_trend_up else 'falling'}")

    if breadth_pct is not None:
        score += (breadth_pct - 0.5) * 40.0
        drivers.append(f"breadth {breadth_pct*100:.0f}% above 200-SMA")

    if vix is not None:
        if vix < 15:
            score += 10
        elif vix < 20:
            score += 5
        elif vix < 25:
            score -= 5
        elif vix < 30:
            score -= 15
        else:
            score -= 25
        drivers.append(f"VIX {vix:.1f}")

    score = int(max(0, min(100, round(score))))
    verdict = "Trade" if score >= 60 else ("Caution" if score >= 40 else "Stand-aside")
    return RegimeRead(verdict, score, spy_above_200, spy_trend_up, breadth_pct, vix, drivers)


def weekday_seasonality(dates: Sequence, opens: Sequence[float], highs: Sequence[float],
                        lows: Sequence[float], closes: Sequence[float]) -> pd.DataFrame:
    """Per-weekday stats on daily bars: up-day rate, average return, and overnight gap-fill rate.

    A day's overnight gap is ``open[t]`` vs ``close[t-1]``; it "fills" when the day's range trades
    back through the prior close. Returns a DataFrame indexed Mon→Fri.
    """
    o = np.asarray(opens, dtype=float)
    h = np.asarray(highs, dtype=float)
    lo = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    idx = pd.to_datetime(pd.Index(dates))
    n = len(c)
    rows = {wd: {"days": 0, "up": 0, "ret": [], "gaps": 0, "filled": 0} for wd in range(5)}

    for t in range(1, n):
        wd = int(idx[t].weekday())
        if wd > 4:
            continue
        prev_close = c[t - 1]
        if prev_close <= 0:
            continue
        r = rows[wd]
        r["days"] += 1
        r["ret"].append((c[t] - prev_close) / prev_close)
        if c[t] > prev_close:
            r["up"] += 1
        gap_up = o[t] > prev_close
        gap_dn = o[t] < prev_close
        if gap_up or gap_dn:
            r["gaps"] += 1
            if (gap_up and lo[t] <= prev_close) or (gap_dn and h[t] >= prev_close):
                r["filled"] += 1

    out = []
    for wd in range(5):
        r = rows[wd]
        d = r["days"]
        out.append({
            "Weekday": _WEEKDAYS[wd],
            "Days": d,
            "Up day %": (r["up"] / d * 100.0) if d else float("nan"),
            "Avg return %": (float(np.mean(r["ret"])) * 100.0) if r["ret"] else float("nan"),
            "Gap-fill %": (r["filled"] / r["gaps"] * 100.0) if r["gaps"] else float("nan"),
        })
    return pd.DataFrame(out).set_index("Weekday")
