"""Shared in-process price-history store backing the bulk-prefetch fast path.

One longest-span daily-bar frame is kept per symbol, so a single 400-day prefetch serves
every screen's window (Setup Scanner 400d, Predictions 200d, Rally Radar 160d, Screener
120d, Whale 60d) via date-cutoff tail slices. Pure module — no Streamlit, no network —
so it stays thread-safe and unit-testable. Entries expire after ``TTL_SECONDS`` and are
only ever *upgraded* to a longer span, never silently shortened.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

TTL_SECONDS = 3600

# symbol → (fetched_at_monotonic, span_days, frame)
_store: Dict[str, Tuple[float, int, pd.DataFrame]] = {}
_lock = threading.Lock()


def _fresh(entry: Tuple[float, int, pd.DataFrame]) -> bool:
    return (time.monotonic() - entry[0]) < TTL_SECONDS


def get(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """A ``days``-window tail slice for ``symbol``, or ``None`` on miss/stale/short-span.

    Returns a copy so callers can mutate freely (matches the isolation ``st.cache_data``'s
    pickle round-trip used to provide).
    """
    with _lock:
        entry = _store.get(symbol)
        if entry is None or not _fresh(entry) or entry[1] < days:
            return None
        df = entry[2]
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
    try:
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            cutoff = cutoff.tz_localize(idx.tz)
        return df[idx >= cutoff].copy()
    except Exception:
        return df.copy()


def put_many(frames: Dict[str, pd.DataFrame], span_days: int) -> None:
    """Store per-symbol frames fetched over a ``span_days`` window.

    An existing fresh entry with a longer span is kept — a shorter fetch must never
    downgrade the data a wider screen already paid for.
    """
    now = time.monotonic()
    with _lock:
        for symbol, df in frames.items():
            if df is None or df.empty:
                continue
            old = _store.get(symbol)
            if old is not None and _fresh(old) and old[1] > span_days:
                continue
            _store[symbol] = (now, span_days, df)


def missing(symbols: List[str], days: int) -> List[str]:
    """The subset of ``symbols`` with no fresh entry covering ``days`` (order preserved)."""
    with _lock:
        return [s for s in symbols
                if (e := _store.get(s)) is None or not _fresh(e) or e[1] < days]


def clear() -> None:
    with _lock:
        _store.clear()


def stats() -> Dict[str, int]:
    """Store health for status displays: total entries and how many are still fresh."""
    with _lock:
        fresh = sum(1 for e in _store.values() if _fresh(e))
        return {"symbols": len(_store), "fresh": fresh}
