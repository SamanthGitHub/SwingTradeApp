"""Rally Radar: detect stocks whose momentum is *igniting* — about to start a move — rather than
trends that have already run.

The main Screener rewards established trends (price above rising MAs, strong RSI, positive MACD).
This module looks one step earlier, for the **coiled-spring / inflection** setups that precede a
rally:

  • Volatility squeeze   — Bollinger-band width compressed to a low percentile of its own recent
                           history (a quiet base that tends to be followed by an expansion move).
  • Volume accumulation  — the last few sessions' volume rising vs the 20-day baseline (quiet
                           accumulation before price reacts).
  • MACD inflection      — the histogram turning up / about to cross (momentum shifting from down
                           to up), rewarded *before* it's fully confirmed.
  • RSI ignition zone    — RSI rising up through ~45–65 (turning up, NOT already overbought).
  • 20-day MA reclaim    — price reclaiming a flattening/rising 20-day average (trend inflection).
  • Base-breakout edge   — close pressing the top of its recent range (about to break resistance).

Each component contributes to a 0–100 **Rally Readiness** score and a list of plain-English
reasons, plus a `stage` label (Coiling → Igniting → Breaking out). Names that are already
overbought, selling off hard, or in a clean downtrend with no inflection are dropped — those
belong on the Screener, not here.

Pure / Streamlit-free (like ``whale.py``) so it can be unit-tested in isolation; ``app.py`` owns
the caching + universe loop (``scan_rally_radar``).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .num import clip01 as _clip01, ema as _ema
from .signals import compute_rsi


@dataclass
class RallyConfig:
    """Lookbacks + component weights for the Rally Readiness score (weights sum to 100)."""

    bb_period: int = 20
    bb_hist: int = 100          # window for the band-width percentile (squeeze) calc
    base_window: int = 20       # recent range used for the base-breakout proximity
    min_bars: int = 60          # need at least this much history to judge a setup

    # Component weights (sum to 100).
    w_squeeze: float = 25.0
    w_volume: float = 20.0
    w_macd: float = 20.0
    w_rsi: float = 15.0
    w_reclaim: float = 10.0
    w_breakout: float = 10.0

    # Exclusion gates — these aren't "about to start", so drop them.
    rsi_overbought: float = 78.0    # already extended
    max_down_day: float = -4.0      # selling off hard today


def _bb_width_series(closes: np.ndarray, period: int) -> np.ndarray:
    """Rolling Bollinger-band width ((upper-lower)/mid = 4·σ/mean) as a series."""
    n = len(closes)
    if n < period:
        return np.array([])
    widths = []
    for i in range(period, n + 1):
        win = closes[i - period:i]
        m = float(win.mean())
        if m > 0:
            widths.append(4.0 * float(win.std()) / m)
    return np.asarray(widths, dtype=float)


class RallyDetector:
    """Score a single symbol's daily OHLCV for an *early* / pre-breakout bullish setup."""

    def __init__(self, config: Optional[RallyConfig] = None) -> None:
        self.config = config or RallyConfig()

    def analyze(self, symbol: str, closes: Sequence[float], volumes: Sequence[float],
                highs: Optional[Sequence[float]] = None,
                lows: Optional[Sequence[float]] = None) -> Optional[Dict]:
        """Return a Rally Readiness record for ``symbol``, or ``None`` if it isn't an early setup.

        Drops names that are already overbought, gapping down hard today, or in a clean downtrend
        with no upward inflection — none of those are "about to start a rally".
        """
        cfg = self.config
        c = np.asarray(closes, dtype=float)
        v = np.asarray(volumes, dtype=float)
        n = len(c)
        if n < cfg.min_bars or len(v) != n:
            return None
        if np.any(~np.isfinite(c)) or c[-1] <= 0:
            return None

        h = np.asarray(highs, dtype=float) if highs is not None else c
        lo = np.asarray(lows, dtype=float) if lows is not None else c

        price = float(c[-1])
        prev = float(c[-2])
        change_pct = (price - prev) / prev * 100.0 if prev > 0 else 0.0

        sma20 = float(np.mean(c[-20:]))
        sma20_prev = float(np.mean(c[-25:-5])) if n >= 25 else sma20
        sma50 = float(np.mean(c[-50:])) if n >= 50 else sma20

        rsi_now = compute_rsi(c)
        rsi_prev = compute_rsi(c[:-3]) if n > 17 else rsi_now
        rsi_rising = rsi_now > rsi_prev

        # MACD (12/26/9) histogram series for the inflection read.
        ema12, ema26 = _ema(c, 12), _ema(c, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        hist = macd_line - signal_line
        h0 = float(hist[-1])
        h1 = float(hist[-2]) if n >= 2 else h0
        h2 = float(hist[-3]) if n >= 3 else h1
        macd_above = macd_line[-1] > signal_line[-1]
        hist_rising = h0 > h1

        # ── Exclusion gates (belongs on the Screener / not an early setup) ──────────
        if rsi_now >= cfg.rsi_overbought:
            return None
        if rsi_now < 35:  # deeply oversold = still falling, not yet igniting
            return None
        if change_pct <= cfg.max_down_day:
            return None
        # Clean downtrend / falling knife: clearly below a bearish (20<50) MA stack with weak,
        # non-rising momentum. A flat base (MAs ~equal, RSI ~50) is NOT excluded by this.
        downtrend = price < sma50 * 0.97 and sma20 < sma50 and rsi_now < 48
        if downtrend and not hist_rising:
            return None

        reasons: List[str] = []

        # A. Volatility squeeze — band width compressed to a low percentile of its history.
        widths = _bb_width_series(c[-(cfg.bb_hist + cfg.bb_period):], cfg.bb_period)
        bb_pctile = 1.0
        score_sq = 0.0
        if widths.size >= 10:
            cur_w = widths[-1]
            bb_pctile = float(np.mean(widths <= cur_w))  # 0 = tightest in the window
            score_sq = _clip01((0.5 - bb_pctile) / 0.5) * cfg.w_squeeze
            if bb_pctile <= 0.25:
                reasons.append(f"Volatility squeeze (band width in {bb_pctile*100:.0f}th pctile)")

        # B. Volume accumulation — last 5 sessions vs the 20-day baseline.
        base_vol = float(np.mean(v[-20:]))
        rvol5 = float(np.mean(v[-5:])) / base_vol if base_vol > 0 else 0.0
        rvol_today = float(v[-1]) / base_vol if base_vol > 0 else 0.0
        vol_metric = max(rvol5, rvol_today * 0.8)
        score_vol = _clip01((vol_metric - 0.9) / 0.6) * cfg.w_volume
        if rvol5 >= 1.15 or rvol_today >= 1.5:
            reasons.append(f"Volume building ({rvol5:.2f}× 5-day vs 20-day)")

        # C. MACD inflection — reward the turn *before* it's fully confirmed.
        score_macd = 0.0
        if macd_above and h0 > 0 and hist_rising:
            score_macd = cfg.w_macd
            reasons.append("MACD crossed up & accelerating")
        elif macd_above and h0 > 0:
            score_macd = cfg.w_macd * 0.75
            reasons.append("MACD bullish")
        elif hist_rising and h0 > h2 and h0 < 0:
            score_macd = cfg.w_macd * 0.5
            reasons.append("MACD histogram rising toward a cross")
        elif hist_rising:
            score_macd = cfg.w_macd * 0.3

        # D. RSI ignition zone — rising up through ~45–65 (not overbought).
        score_rsi = 0.0
        if rsi_rising and 50 <= rsi_now <= 65:
            score_rsi = cfg.w_rsi
            reasons.append(f"RSI rising through ignition zone ({rsi_now:.0f})")
        elif rsi_rising and 45 <= rsi_now < 50:
            score_rsi = cfg.w_rsi * 0.7
            reasons.append(f"RSI turning up toward 50 ({rsi_now:.0f})")
        elif rsi_rising and 65 < rsi_now <= 72:
            score_rsi = cfg.w_rsi * 0.5
        elif rsi_rising:
            score_rsi = cfg.w_rsi * 0.25

        # E. 20-day MA reclaim — price reclaiming a flattening / rising average.
        score_reclaim = 0.0
        recently_below = bool(np.any(c[-10:-1] < np.mean(c[-20:])))
        if price > sma20 and recently_below:
            score_reclaim = cfg.w_reclaim
            reasons.append("Reclaimed the 20-day average")
        elif price > sma20 and sma20 >= sma20_prev:
            score_reclaim = cfg.w_reclaim * 0.7
        elif price > sma20:
            score_reclaim = cfg.w_reclaim * 0.4

        # F. Base-breakout edge — close pressing the top of its recent range.
        win_hi = float(np.max(h[-cfg.base_window:]))
        win_lo = float(np.min(lo[-cfg.base_window:]))
        rng = win_hi - win_lo
        pos_in_range = (price - win_lo) / rng if rng > 0 else 0.5
        score_break = _clip01((pos_in_range - 0.7) / 0.3) * cfg.w_breakout
        dist_to_high_pct = (win_hi - price) / price * 100.0 if price > 0 else 0.0
        if pos_in_range >= 0.85:
            reasons.append(f"Pressing the {cfg.base_window}-day high "
                           f"({dist_to_high_pct:.1f}% away)")

        score = score_sq + score_vol + score_macd + score_rsi + score_reclaim + score_break

        # Stage: where in the ignition sequence the setup is (earliest → latest).
        if score_break >= 7 and score_vol >= 10:
            stage = "Breaking out"
        elif score_macd >= 10 and (score_rsi >= 7 or score_vol >= 8):
            stage = "Igniting"
        elif score_sq >= 15:
            stage = "Coiling"
        else:
            stage = "Building"

        return {
            "symbol": symbol,
            "rally_score": round(score, 1),
            "stage": stage,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi_now, 1),
            "macd_hist": round(h0, 4),
            "rvol5": round(rvol5, 2),
            "bb_pctile": round(bb_pctile * 100, 0),
            "dist_to_high_pct": round(dist_to_high_pct, 2),
            "reasons": "; ".join(reasons) if reasons else "Early-stage setup forming",
        }

    def scan(self, records: Sequence[Dict], min_score: float = 0.0) -> List[Dict]:
        """Analyze many symbols; return early-rally candidates sorted by score (strongest first).

        ``records`` is a sequence of dicts each with ``symbol`` + ``closes``/``volumes`` (and
        optionally ``highs``/``lows``). Non-setups and anything below ``min_score`` are dropped.
        """
        out: List[Dict] = []
        for r in records:
            res = self.analyze(r["symbol"], r["closes"], r["volumes"],
                               r.get("highs"), r.get("lows"))
            if res is not None and res["rally_score"] >= min_score:
                out.append(res)
        out.sort(key=lambda d: d["rally_score"], reverse=True)
        return out
