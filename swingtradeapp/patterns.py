"""Reusable chart-pattern detectors: candlesticks, double bottom/top, Fair Value Gaps,
and liquidity sweeps (Turtle Soup).

Pure / Streamlit-free (like ``whale.py`` and ``momentum_radar.py``) so they can be unit-tested in
isolation and composed by ``setups.py``. These are the building blocks the research docs lean on:
the *engulfing / hammer* entry trigger, the *double bottom/top* reversal structure, the *FVG*
imbalance (traded as a trend-continuation defence zone, not a magnet), and the *liquidity sweep*
("close back inside" the swept extreme).

All functions operate on plain sequences of OHLC(V) and look only **backward** from the evaluated
bar, so they're causal — safe to call bar-by-bar in a backtest without lookahead.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

EPS = 1e-9


def _arr(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _norm_index(i: int, n: int) -> int:
    """Normalise a possibly-negative index into [0, n)."""
    return i + n if i < 0 else i


# ── Candlestick patterns (evaluate a single bar) ─────────────────────────────────
# Each takes OHLC sequences and a bar index ``i`` (default the last bar) and returns a bool.

def is_bullish_engulfing(opens, highs, lows, closes, i: int = -1) -> bool:
    """Current candle is up and its real body fully engulfs the prior down candle's body."""
    o, c = _arr(opens), _arr(closes)
    i = _norm_index(i, len(c))
    if i < 1:
        return False
    prev_down = c[i - 1] < o[i - 1]
    curr_up = c[i] > o[i]
    engulfs = c[i] >= o[i - 1] and o[i] <= c[i - 1]
    return bool(prev_down and curr_up and engulfs)


def is_bearish_engulfing(opens, highs, lows, closes, i: int = -1) -> bool:
    """Current candle is down and its real body fully engulfs the prior up candle's body."""
    o, c = _arr(opens), _arr(closes)
    i = _norm_index(i, len(c))
    if i < 1:
        return False
    prev_up = c[i - 1] > o[i - 1]
    curr_down = c[i] < o[i]
    engulfs = o[i] >= c[i - 1] and c[i] <= o[i - 1]
    return bool(prev_up and curr_down and engulfs)


def is_hammer(opens, highs, lows, closes, i: int = -1) -> bool:
    """Bullish hammer: small body near the top, long lower wick (≥2× body), short upper wick."""
    o, h, lo, c = _arr(opens), _arr(highs), _arr(lows), _arr(closes)
    i = _norm_index(i, len(c))
    rng = h[i] - lo[i]
    if rng <= EPS:
        return False
    body = abs(c[i] - o[i])
    lower_wick = min(o[i], c[i]) - lo[i]
    upper_wick = h[i] - max(o[i], c[i])
    return bool(lower_wick >= 2 * body and upper_wick <= body and body <= 0.4 * rng)


def pin_bar_direction(opens, highs, lows, closes, i: int = -1) -> Optional[str]:
    """A rejection pin bar: a dominant wick (≥⅔ of range) on one side, small body.

    Returns ``"bullish"`` (long lower wick), ``"bearish"`` (long upper wick), or ``None``.
    """
    o, h, lo, c = _arr(opens), _arr(highs), _arr(lows), _arr(closes)
    i = _norm_index(i, len(c))
    rng = h[i] - lo[i]
    if rng <= EPS:
        return None
    body = abs(c[i] - o[i])
    lower_wick = min(o[i], c[i]) - lo[i]
    upper_wick = h[i] - max(o[i], c[i])
    if body > 0.35 * rng:
        return None
    if lower_wick >= 0.66 * rng:
        return "bullish"
    if upper_wick >= 0.66 * rng:
        return "bearish"
    return None


def is_doji(opens, highs, lows, closes, i: int = -1) -> bool:
    """Indecision: real body ≤ 10% of the bar's range."""
    o, h, lo, c = _arr(opens), _arr(highs), _arr(lows), _arr(closes)
    i = _norm_index(i, len(c))
    rng = h[i] - lo[i]
    if rng <= EPS:
        return False
    return bool(abs(c[i] - o[i]) <= 0.10 * rng)


def bullish_trigger(opens, highs, lows, closes, i: int = -1) -> Optional[str]:
    """Any bullish entry trigger candle (engulfing / hammer / bullish pin). Returns the name."""
    if is_bullish_engulfing(opens, highs, lows, closes, i):
        return "bullish engulfing"
    if is_hammer(opens, highs, lows, closes, i):
        return "hammer"
    if pin_bar_direction(opens, highs, lows, closes, i) == "bullish":
        return "bullish pin bar"
    return None


# ── Swing pivots ─────────────────────────────────────────────────────────────────
# Complements MarketStructure.find_swing_levels (which returns only the *most recent* pivot) by
# returning *all* fractal pivots in the window — needed to find two matching lows/highs.

def swing_low_indices(lows, left: int = 2, right: int = 2) -> List[int]:
    lo = _arr(lows)
    out: List[int] = []
    for i in range(left, len(lo) - right):
        win = lo[i - left:i + right + 1]
        if lo[i] == win.min() and np.argmin(win) == left:
            out.append(i)
    return out


def swing_high_indices(highs, left: int = 2, right: int = 2) -> List[int]:
    h = _arr(highs)
    out: List[int] = []
    for i in range(left, len(h) - right):
        win = h[i - left:i + right + 1]
        if h[i] == win.max() and np.argmax(win) == left:
            out.append(i)
    return out


# ── Double bottom / top ──────────────────────────────────────────────────────────

@dataclass
class DoublePattern:
    kind: str          # "double_bottom" | "double_top"
    level: float       # the twice-defended support (bottom) / resistance (top)
    neckline: float    # the intervening peak (bottom) / trough (top) = breakout trigger
    first_bar: int
    second_bar: int


def detect_double_bottom(highs, lows, lookback: int = 60, tol: float = 0.04) -> Optional[DoublePattern]:
    """Two swing lows within ``tol`` of each other, separated by an intervening peak (neckline).

    Returns the most recent such pair in the window, or ``None``.
    """
    h, lo = _arr(highs), _arr(lows)
    n = len(lo)
    if n < 12:
        return None
    start = max(0, n - lookback)
    lows_idx = [i for i in swing_low_indices(lo) if i >= start]
    for b in range(len(lows_idx) - 1, 0, -1):
        i2 = lows_idx[b]
        for a in range(b - 1, -1, -1):
            i1 = lows_idx[a]
            base = max(lo[i1], EPS)
            if abs(lo[i2] - lo[i1]) / base <= tol and i2 - i1 >= 3:
                neckline = float(np.max(h[i1:i2 + 1]))
                return DoublePattern("double_bottom", float(min(lo[i1], lo[i2])),
                                     neckline, i1, i2)
    return None


def detect_double_top(highs, lows, lookback: int = 60, tol: float = 0.04) -> Optional[DoublePattern]:
    """Two swing highs within ``tol`` of each other, separated by an intervening trough."""
    h, lo = _arr(highs), _arr(lows)
    n = len(h)
    if n < 12:
        return None
    start = max(0, n - lookback)
    highs_idx = [i for i in swing_high_indices(h) if i >= start]
    for b in range(len(highs_idx) - 1, 0, -1):
        i2 = highs_idx[b]
        for a in range(b - 1, -1, -1):
            i1 = highs_idx[a]
            base = max(h[i1], EPS)
            if abs(h[i2] - h[i1]) / base <= tol and i2 - i1 >= 3:
                neckline = float(np.min(lo[i1:i2 + 1]))
                return DoublePattern("double_top", float(max(h[i1], h[i2])),
                                     neckline, i1, i2)
    return None


# ── Fair Value Gaps (FVG) ──────────────────────────────────────────────────────────

@dataclass
class FVGZone:
    kind: str       # "bullish" | "bearish"
    top: float
    bottom: float
    bar: int        # index of the middle (displacement) candle
    mitigated: bool # price has since closed back through the gap
    inverted: bool  # a once-respected gap that was closed through → flips bias


def detect_fair_value_gaps(highs, lows, closes, max_lookback: int = 80) -> List[FVGZone]:
    """Three-candle imbalances. A bullish FVG = low[i+1] > high[i-1] (a price void left by a
    burst of buying); bearish = high[i+1] < low[i-1]. Each zone is flagged ``mitigated`` if a
    later candle closed back inside it, and ``inverted`` (an Inverse FVG) if a later candle
    closed fully through it — flipping it into the opposite-bias defence zone.
    """
    h, lo, c = _arr(highs), _arr(lows), _arr(closes)
    n = len(c)
    zones: List[FVGZone] = []
    start = max(1, n - max_lookback)
    for i in range(start, n - 1):
        # Bullish gap between candle i-1 high and candle i+1 low (displacement candle = i).
        if lo[i + 1] > h[i - 1]:
            top, bottom = float(lo[i + 1]), float(h[i - 1])
            after = c[i + 2:]
            mitigated = bool(np.any((after <= top) & (after >= bottom)))
            inverted = bool(np.any(after < bottom))
            zones.append(FVGZone("bullish", top, bottom, i, mitigated, inverted))
        # Bearish gap.
        elif h[i + 1] < lo[i - 1]:
            top, bottom = float(lo[i - 1]), float(h[i + 1])
            after = c[i + 2:]
            mitigated = bool(np.any((after <= top) & (after >= bottom)))
            inverted = bool(np.any(after > top))
            zones.append(FVGZone("bearish", top, bottom, i, mitigated, inverted))
    return zones


def nearest_unmitigated_fvg(highs, lows, closes, kind: str = "bullish",
                            max_lookback: int = 80) -> Optional[FVGZone]:
    """The most recent still-unmitigated, non-inverted FVG of ``kind`` below/above price."""
    zones = [z for z in detect_fair_value_gaps(highs, lows, closes, max_lookback)
             if z.kind == kind and not z.mitigated and not z.inverted]
    return zones[-1] if zones else None


# ── Liquidity sweep / Turtle Soup ──────────────────────────────────────────────────

@dataclass
class LiquiditySweep:
    direction: str     # "long" (swept a low, closed back up) | "short" (swept a high, closed back down)
    swept_level: float
    bar: int


def detect_liquidity_sweep(opens, highs, lows, closes, i: int = -1,
                           lookback: int = 20, tol: float = 0.0015) -> Optional[LiquiditySweep]:
    """Turtle Soup: at bar ``i`` price sweeps just beyond a prior swing extreme (triggering resting
    stops) then **closes back inside** the range — a failed breakout that reverses.

    Long: wick pierces below a prior swing low but the candle closes back above it.
    Short: wick pierces above a prior swing high but the candle closes back below it.
    """
    o, h, lo, c = _arr(opens), _arr(highs), _arr(lows), _arr(closes)
    n = len(c)
    i = _norm_index(i, n)
    if i < 5:
        return None
    win_start = max(0, i - lookback)
    prior_low = float(np.min(lo[win_start:i]))
    prior_high = float(np.max(h[win_start:i]))

    # Bullish sweep: dipped below prior low, closed back above it.
    if lo[i] < prior_low and c[i] > prior_low and c[i] > o[i]:
        if (prior_low - lo[i]) / max(prior_low, EPS) <= 0.05:  # a sweep, not a collapse
            return LiquiditySweep("long", prior_low, i)
    # Bearish sweep: poked above prior high, closed back below it.
    if h[i] > prior_high and c[i] < prior_high and c[i] < o[i]:
        if (h[i] - prior_high) / max(prior_high, EPS) <= 0.05:
            return LiquiditySweep("short", prior_high, i)
    return None
