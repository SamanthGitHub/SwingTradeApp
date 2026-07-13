"""The per-render context handed from app.py's router to every screens/* page."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Ctx:
    config: Any            # TradingConfig (with live Settings overrides applied)
    account_size: float    # sidebar "Account size ($)" input
    watchlist_mgr: Any     # shared WatchlistManager singleton
