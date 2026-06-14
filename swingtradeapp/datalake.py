"""Local Parquet data lake for the Alpha Engine — free, dependency-light persistence.

Two jobs:
1. **Persistent panel cache** — the heavy multi-symbol price/volume downloads survive app restarts
   (Streamlit's ``@st.cache_data`` is in-memory only), so a backtest reloads in milliseconds instead
   of re-hitting Yahoo every cold start.
2. **Reproducible snapshots** — pin a dated copy of a panel so a backtest can be re-run on *exactly*
   the data it used, the first step toward point-in-time discipline.

Storage is Parquet via pandas/pyarrow (already a pandas 3.x dependency — no new install, $0). DuckDB,
if present, is offered as an optional SQL query layer but is never required. Everything lives under
``.data/lake/`` (``.data/`` is gitignored).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

LAKE = Path(".data/lake")


def _dir(sub: str = "") -> Path:
    p = LAKE / sub if sub else LAKE
    p.mkdir(parents=True, exist_ok=True)
    return p


def panel_key(prefix: str, symbols, years: int) -> str:
    """Stable cache key for a (universe, horizon) panel — order-independent via a hash of symbols."""
    h = hashlib.md5(",".join(sorted(symbols)).encode()).hexdigest()[:10]
    return f"{prefix}_{years}y_{len(list(symbols))}_{h}"


# ── Persistent panel cache ───────────────────────────────────────────────────────

def save_panel(name: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        df.to_parquet(_dir() / f"{name}.parquet")
    except Exception:
        pass


def load_panel(name: str, max_age_hours: Optional[float] = None) -> Optional[pd.DataFrame]:
    """Read a cached panel if present (and fresh, when ``max_age_hours`` is given); else ``None``."""
    f = _dir() / f"{name}.parquet"
    if not f.exists():
        return None
    if max_age_hours is not None and (time.time() - f.stat().st_mtime) / 3600.0 > max_age_hours:
        return None
    try:
        return pd.read_parquet(f)
    except Exception:
        return None


def panel_age_hours(name: str) -> Optional[float]:
    f = _dir() / f"{name}.parquet"
    return (time.time() - f.stat().st_mtime) / 3600.0 if f.exists() else None


# ── Reproducible snapshots ───────────────────────────────────────────────────────

def save_snapshot(df: pd.DataFrame, tag: str) -> str:
    """Pin a dated copy of a panel for exact later reproduction. Returns the snapshot name."""
    name = f"{tag}__{time.strftime('%Y%m%d_%H%M')}"
    try:
        df.to_parquet(_dir("snapshots") / f"{name}.parquet")
    except Exception:
        return ""
    return name


def list_snapshots(prefix: str = "") -> List[str]:
    return sorted((p.stem for p in _dir("snapshots").glob(f"{prefix}*.parquet")), reverse=True)


def load_snapshot(name: str) -> Optional[pd.DataFrame]:
    f = _dir("snapshots") / f"{name}.parquet"
    try:
        return pd.read_parquet(f) if f.exists() else None
    except Exception:
        return None


# ── Optional DuckDB query layer (never required) ─────────────────────────────────

def query(sql: str):
    """Run SQL over the lake's Parquet files via DuckDB, if installed; else ``None``."""
    try:
        import duckdb
        return duckdb.query(sql).to_df()
    except Exception:
        return None
