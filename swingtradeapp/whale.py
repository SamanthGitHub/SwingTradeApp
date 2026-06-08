"""Whale-movement detection: surface large-money ("smart money") footprints from daily OHLCV.

We only have free Yahoo Finance data — no real Level 2 / time-and-sales / dark-pool prints —
so "whale activity" is *inferred* from the public tape: an outsized volume surge relative to a
symbol's own 20-day baseline, scaled by the dollars actually changing hands and by where the
day closed in its range (closing strength = who won the day). The result is a 0–100 whale score
and a plain-English signal (Heavy Buying / Heavy Selling / Accumulation / Distribution / Churn).

Pure / dependency-free of Streamlit so it can be unit-tested in isolation; ``app.py`` owns the
caching + universe loop (same pattern as the ETF screener and auto-watchlist).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class WhaleConfig:
    """Thresholds for flagging a daily bar as whale activity."""

    lookback: int = 20          # trailing days for the volume / dollar-volume baseline
    min_rvol: float = 2.0       # today's volume must be ≥ this × its 20-day average
    min_dollar_vol: float = 50_000_000.0  # and ≥ this much capital must change hands
    # Score component caps (the value at which a component maxes out its weight).
    rvol_cap: float = 5.0       # 5× relative volume → full volume weight
    dollar_cap: float = 1.0e9   # $1B traded → full dollar weight
    impact_cap: float = 8.0     # 8% daily move → full price-impact weight
    # Component weights (sum to 100).
    w_rvol: float = 50.0
    w_dollar: float = 30.0
    w_impact: float = 20.0


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _classify(change_pct: float, close_strength: float, accumulation_days: int,
              distribution_days: int) -> str:
    """Map the day's character + recent accumulation balance to a plain-English signal.

    ``close_strength`` is where the close sits in the day's range (0 = on the low, 1 = on the
    high). Big volume + up day + strong close ⇒ whales buying; big volume + down day + weak
    close ⇒ whales selling. A move that reverses its range intraday is "Churn" (indecision).
    """
    strong_close = close_strength >= 0.66
    weak_close = close_strength <= 0.34
    if change_pct > 0 and strong_close:
        return "Heavy Buying"
    if change_pct < 0 and weak_close:
        return "Heavy Selling"
    if change_pct > 0:
        return "Accumulation"
    if change_pct < 0:
        return "Distribution"
    # Flat close: lean on the multi-day balance.
    if accumulation_days > distribution_days:
        return "Accumulation"
    if distribution_days > accumulation_days:
        return "Distribution"
    return "Churn"


class WhaleDetector:
    """Infer large-money footprints from a single symbol's recent daily OHLCV."""

    def __init__(self, config: Optional[WhaleConfig] = None) -> None:
        self.config = config or WhaleConfig()

    def analyze(self, symbol: str, opens: Sequence[float], highs: Sequence[float],
                lows: Sequence[float], closes: Sequence[float],
                volumes: Sequence[float]) -> Optional[Dict]:
        """Return a whale-activity record for ``symbol``, or ``None`` if it isn't a whale bar.

        Looks at the most recent bar against the trailing ``lookback`` baseline. Returns
        ``None`` when there's too little history, the data is degenerate, or the bar fails the
        relative-volume / dollar-volume gates.
        """
        cfg = self.config
        closes = np.asarray(closes, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        n = len(closes)
        if n < cfg.lookback + 2 or len(volumes) != n:
            return None

        last_vol = float(volumes[-1])
        # Trailing baseline EXCLUDES today so a spike doesn't inflate its own average.
        base_vol = volumes[-(cfg.lookback + 1):-1]
        avg_vol = float(np.mean(base_vol)) if base_vol.size else 0.0
        if avg_vol <= 0 or last_vol <= 0:
            return None

        price = float(closes[-1])
        prev = float(closes[-2])
        if price <= 0 or prev <= 0:
            return None

        rvol = last_vol / avg_vol
        dollar_vol = price * last_vol
        if rvol < cfg.min_rvol or dollar_vol < cfg.min_dollar_vol:
            return None

        change_pct = (price - prev) / prev * 100.0

        # Closing strength within the day's range (who controlled the tape into the close).
        hi, lo = float(highs[-1]), float(lows[-1])
        rng = hi - lo
        close_strength = (price - lo) / rng if rng > 0 else 0.5

        # Multi-day accumulation/distribution balance over the lookback window: count up-volume
        # vs down-volume days where volume ran above the trailing average (institutional days).
        acc_days = dist_days = 0
        avg_all = float(np.mean(volumes[-cfg.lookback:]))
        for i in range(n - cfg.lookback, n):
            if i <= 0:
                continue
            if volumes[i] <= avg_all:
                continue
            if closes[i] > closes[i - 1]:
                acc_days += 1
            elif closes[i] < closes[i - 1]:
                dist_days += 1

        # 0–100 whale score: volume surge dominates, scaled by dollars + price impact.
        score = (
            _clip01(rvol / cfg.rvol_cap) * cfg.w_rvol
            + _clip01(dollar_vol / cfg.dollar_cap) * cfg.w_dollar
            + _clip01(abs(change_pct) / cfg.impact_cap) * cfg.w_impact
        )

        signal = _classify(change_pct, close_strength, acc_days, dist_days)
        # Direction sign: bullish footprints positive, bearish negative (for sorting / color).
        bullish = signal in ("Heavy Buying", "Accumulation")

        return {
            "symbol": symbol,
            "signal": signal,
            "bullish": bullish,
            "whale_score": round(score, 1),
            "rvol": round(rvol, 2),
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "dollar_vol": dollar_vol,
            "volume": last_vol,
            "avg_volume": avg_vol,
            "close_strength": round(close_strength, 2),
            "accum_days": acc_days,
            "distrib_days": dist_days,
        }

    def scan(self, records: Sequence[Dict]) -> List[Dict]:
        """Analyze many symbols and return whale bars sorted by score (strongest first).

        ``records`` is a sequence of dicts each with ``symbol`` and OHLCV sequences
        (``opens``/``highs``/``lows``/``closes``/``volumes``). Symbols that aren't whale bars
        are dropped.
        """
        out: List[Dict] = []
        for r in records:
            res = self.analyze(
                r["symbol"], r.get("opens", []), r.get("highs", []), r.get("lows", []),
                r["closes"], r["volumes"],
            )
            if res is not None:
                out.append(res)
        out.sort(key=lambda d: d["whale_score"], reverse=True)
        return out
