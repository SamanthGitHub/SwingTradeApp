"""Market clock: one canonical US-Eastern time source + session-phase helper.

Every timestamp, fetch window and "is the market open?" decision in the app goes through
here, so a machine in another timezone (or a naive ``datetime.now()``) can never shift a
date window off by a day or mislabel the session.
"""

from __future__ import annotations

from datetime import datetime
from typing import Tuple


def now_et() -> datetime:
    """Current time in US/Eastern (market time) — falls back to local naive time only if
    zoneinfo/tzdata is unavailable on the host."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def market_phase() -> Tuple[str, str, str]:
    """``(phase, greeting, data_session)`` for the current US-Eastern moment.

    phase/data_session ∈ {"pre-market", "regular", "after-hours", "closed"} — drives page
    titles and which movers feed has live data. (US-holiday closures are not modelled; a
    holiday reads as a quiet "regular" session.)
    """
    now = now_et()
    if now.weekday() >= 5:
        return ("closed", "Weekend plan", "closed")
    hm = now.hour * 60 + now.minute
    if 240 <= hm < 570:    # 04:00–09:30
        return ("pre-market", "Good morning", "pre-market")
    if 570 <= hm < 960:    # 09:30–16:00
        return ("regular", "Markets are open", "regular")
    if 960 <= hm < 1200:   # 16:00–20:00
        return ("after-hours", "After the close", "after-hours")
    return ("closed", "Tonight's plan", "closed")
