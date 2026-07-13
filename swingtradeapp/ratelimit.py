"""Per-provider free-tier usage ledger + hard-stop guard.

This is the single source of truth that the app **never exceeds a free API tier** (so a paid
provider used on its free plan never starts charging). Each supported provider declares its free
limits (per-minute / per-day / per-month); every paid call is checked BEFORE it's made and recorded
after. When a window is full, ``check()`` returns ``False`` and the caller must fall back to a free
source (yfinance) instead of calling the paid API.

Pure / Streamlit-free (like ``whale.py`` / ``ratelimit`` is infra) so it can be unit-tested in
isolation; ``app.py`` owns the cached singleton. The ledger is a small JSON file under ``.data/``
(gitignored, same convention as ``portfolio_state.json``) so it survives reruns, restarts and
multiple browser tabs sharing one Streamlit process.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .jsonstore import atomic_write_json, read_json


@dataclass(frozen=True)
class ProviderLimit:
    key: str
    name: str
    env_var: str
    per_minute: Optional[int] = None
    per_day: Optional[int] = None
    per_month: Optional[int] = None
    docs_url: str = ""
    recommended_use: str = ""


# Registry of supported free-tier providers. To add the next one (Finnhub / Alpha Vantage / FMP /
# Tiingo …) add an entry here AND a DataProviderBase subclass in providers.py — the Settings hub,
# the usage meters and the guard all pick it up automatically from this dict.
PROVIDERS: Dict[str, ProviderLimit] = {
    "polygon": ProviderLimit(
        key="polygon",
        name="Polygon · Massive",
        env_var="POLYGON_API_KEY",
        per_minute=5,            # free "Basic" plan: 5 requests / minute (verified June 2026)
        per_day=None,
        per_month=None,
        docs_url="https://massive.com/pricing",
        recommended_use="single-symbol drill-downs (5 calls/min — not bulk scans)",
    ),
}

_MINUTE = 60
_DAY = 86_400
_MONTH = 30 * _DAY
_RETAIN = _MONTH + _DAY  # prune timestamps older than the largest window we track


class ApiBudget:
    """File-backed sliding-window call counter. Thread-safe within one process."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else Path(".data") / "api_usage.json"
        self._lock = threading.Lock()

    # ── internal io ──────────────────────────────────────────────────────────
    def _load(self) -> Dict[str, List[float]]:
        data = read_json(self.path, default={})
        try:
            return {k: [float(t) for t in v] for k, v in data.items()}
        except (AttributeError, TypeError, ValueError):
            return {}

    def _save(self, data: Dict[str, List[float]]) -> None:
        try:
            atomic_write_json(self.path, data, indent=0)
        except Exception:
            pass

    @staticmethod
    def _prune(ts: List[float], now: float) -> List[float]:
        cutoff = now - _RETAIN
        return [t for t in ts if t >= cutoff]

    @staticmethod
    def _count_within(ts: List[float], now: float, window: int) -> int:
        cutoff = now - window
        return sum(1 for t in ts if t >= cutoff)

    @staticmethod
    def _reset_in(ts: List[float], now: float, window: int) -> int:
        """Seconds until the oldest in-window call ages out (i.e. one slot frees up)."""
        in_window = [t for t in ts if t >= now - window]
        if not in_window:
            return 0
        return max(0, int(window - (now - min(in_window))))

    # ── public api ───────────────────────────────────────────────────────────
    def check(self, provider: str, n: int = 1) -> Tuple[bool, str]:
        """Would ``n`` more calls right now breach any declared window for ``provider``?

        Returns ``(allowed, reason)``. Unknown providers are unmetered (always allowed).
        """
        spec = PROVIDERS.get(provider)
        if spec is None:
            return True, ""
        now = time.time()
        with self._lock:
            ts = self._prune(self._load().get(provider, []), now)
        for window, limit, label in ((_MINUTE, spec.per_minute, "minute"),
                                     (_DAY, spec.per_day, "day"),
                                     (_MONTH, spec.per_month, "month")):
            if limit is None:
                continue
            if self._count_within(ts, now, window) + n > limit:
                return False, f"{spec.name} {label} limit ({limit}/{label})"
        return True, ""

    def try_acquire(self, provider: str, n: int = 1) -> Tuple[bool, str]:
        """Atomically check-and-record ``n`` calls: the load → prune → check → record → save
        sequence runs under one lock, so two threads can never both pass a ``check()`` and
        together breach a window (the TOCTOU gap between separate ``check``/``record`` calls).
        This is what providers must call immediately before a metered request.

        Returns ``(acquired, reason)``; on ``False`` nothing was recorded.
        """
        spec = PROVIDERS.get(provider)
        if spec is None:
            return True, ""
        now = time.time()
        with self._lock:
            data = self._load()
            ts = self._prune(data.get(provider, []), now)
            for window, limit, label in ((_MINUTE, spec.per_minute, "minute"),
                                         (_DAY, spec.per_day, "day"),
                                         (_MONTH, spec.per_month, "month")):
                if limit is None:
                    continue
                if self._count_within(ts, now, window) + n > limit:
                    return False, f"{spec.name} {label} limit ({limit}/{label})"
            ts.extend([now] * max(1, n))
            data[provider] = ts
            self._save(data)
        return True, ""

    def record(self, provider: str, n: int = 1) -> None:
        """Reserve/record ``n`` calls against ``provider`` (call this when you make the request)."""
        if provider not in PROVIDERS:
            return
        now = time.time()
        with self._lock:
            data = self._load()
            ts = self._prune(data.get(provider, []), now)
            ts.extend([now] * max(1, n))
            data[provider] = ts
            self._save(data)

    def status(self, provider: str) -> Dict[str, object]:
        """Current usage for the meters: used/remaining per window + seconds to reset."""
        spec = PROVIDERS.get(provider)
        if spec is None:
            return {"name": provider}
        now = time.time()
        with self._lock:
            ts = self._prune(self._load().get(provider, []), now)
        out: Dict[str, object] = {"name": spec.name}
        if spec.per_minute is not None:
            used = self._count_within(ts, now, _MINUTE)
            out.update(per_minute=spec.per_minute, used_minute=used,
                       remaining_minute=max(0, spec.per_minute - used),
                       reset_in_s=self._reset_in(ts, now, _MINUTE))
        if spec.per_day is not None:
            out.update(per_day=spec.per_day, used_day=self._count_within(ts, now, _DAY))
        if spec.per_month is not None:
            out.update(per_month=spec.per_month, used_month=self._count_within(ts, now, _MONTH))
        return out
