"""Session error ledger — makes silent data failures visible.

The app deliberately degrades instead of crashing (a failed fetch returns an empty frame and
the UI shows "unavailable"), but historically those failures were swallowed by bare
``except Exception: pass`` blocks, so a persistent provider outage looked identical to "no
data right now". This module is the middle ground: failures still never crash a page, but
every one is recorded here and surfaced as a "⚠ N data issues" note in the data-status strip
plus a **Data health** table on the Settings page.

Pure / Streamlit-free; process-wide (one ledger per Streamlit process, shared by all tabs).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Dict, List, Optional

_MAX_ENTRIES = 200

_lock = threading.Lock()
_entries: deque = deque(maxlen=_MAX_ENTRIES)


def record(source: str, exc: Optional[BaseException] = None, note: str = "") -> None:
    """Log one failure. ``source`` is where it happened (e.g. ``"fetch_symbol_history"``),
    ``exc`` the caught exception (optional), ``note`` any extra context (e.g. the symbol)."""
    entry = {
        "ts": time.time(),
        "source": source,
        "error": f"{type(exc).__name__}: {exc}" if exc is not None else "",
        "note": note,
    }
    with _lock:
        _entries.append(entry)


@contextmanager
def soft(source: str, note: str = ""):
    """``with errlog.soft("scan_setups", note=sym): ...`` — the drop-in replacement for
    ``try: ... except Exception: pass`` that records what was swallowed."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 — this is the explicit swallow-with-receipt
        record(source, exc, note)


def entries() -> List[Dict]:
    """Most-recent-first snapshot of the ledger."""
    with _lock:
        return list(reversed(_entries))


def count() -> int:
    with _lock:
        return len(_entries)


def summary() -> Dict[str, int]:
    """``{source: n}`` counts, for a compact roll-up."""
    out: Dict[str, int] = {}
    with _lock:
        for e in _entries:
            out[e["source"]] = out.get(e["source"], 0) + 1
    return out


def clear() -> None:
    with _lock:
        _entries.clear()
