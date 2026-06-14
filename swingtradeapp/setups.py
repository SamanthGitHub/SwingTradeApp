"""Named, research-backed swing setups — each turns a confluence of indicators + chart patterns
into a concrete, level-bearing trade idea (entry / stop / target) with plain-English reasons.

Setups implemented (all long-biased, daily timeframe, free data only):

  • **VCP / Minervini SEPA** — price-based 8-point trend template (price stacked above rising
    50/150/200 SMAs, near the 52-wk high, well off the low) + a volatility contraction + a
    breakout on a ≥40% volume surge. Fundamentals (earnings growth / RS / inst. ownership) are
    layered on as *tags* by the scanner when available — they aren't point-in-time, so the
    backtest core is price/volume only.
  • **RSI(2) Connors** — Larry Connors' micro-reversion: in an uptrend (price > 200-SMA), price
    below its 5-EMA and a 2-period RSI under 10. (Exit is condition-based — RSI(2)>70 / close
    back above the 5-EMA — approximated by a tight bracket for backtesting.)
  • **20-EMA pullback** — trend continuation: uptrend, price tags the 20-EMA, RSI 35–50,
    contracting pullback volume, and a bullish trigger candle.
  • **Double-bottom breakout** — two defended lows + a break of the neckline.
  • **Liquidity sweep (Turtle Soup)** — a swept prior low that closes back inside the range.

Each ``Setup`` exposes ``detect`` (latest bar, for the scanner) and ``signal_bars`` (every
historical trigger bar, for the Backtest Lab). The condition is evaluated **causally** — only
data up to the evaluated bar is used — so the same code drives both without lookahead.

Pure / Streamlit-free; ``app.py`` owns caching + the universe loop.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .signals import _ema, compute_atr, compute_rsi, compute_volume_surge
from .momentum_radar import _bb_width_series
from .patterns import (
    bullish_trigger,
    detect_double_bottom,
    detect_liquidity_sweep,
    nearest_unmitigated_fvg,
)


@dataclass
class SetupHit:
    name: str
    direction: str            # "long" (all current setups are long-biased)
    entry: float
    stop: float
    target: float
    score: float              # 0–1 quality, for ranking
    bar: int                  # index where it fired
    tags: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def risk_reward(self) -> float:
        risk = self.entry - self.stop
        return (self.target - self.entry) / risk if risk > 0 else 0.0


def _sma(c: np.ndarray, window: int) -> float:
    return float(np.mean(c[-window:])) if len(c) >= window else float(np.mean(c))


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


class Setup:
    """Base class. Subclasses implement ``_fires_at`` on a causal slice ending at bar ``i``."""

    name: str = "setup"
    min_bars: int = 60

    def _fires_at(self, o, h, l, c, v, i: int, ctx: Optional[Dict] = None) -> Optional[SetupHit]:
        raise NotImplementedError

    def detect(self, opens, highs, lows, closes, volumes,
               ctx: Optional[Dict] = None) -> Optional[SetupHit]:
        """Evaluate the latest bar (used by the Setup Scanner). ``ctx`` may carry fundamentals."""
        o, h, l, c, v = (np.asarray(x, dtype=float) for x in (opens, highs, lows, closes, volumes))
        if len(c) < self.min_bars:
            return None
        return self._fires_at(o, h, l, c, v, len(c) - 1, ctx)

    def signal_bars(self, opens, highs, lows, closes, volumes, min_gap: int = 3) -> List[SetupHit]:
        """Every historical bar where the setup fired (used by the Backtest Lab), de-clustered
        so two triggers can't sit within ``min_gap`` bars of each other."""
        o, h, l, c, v = (np.asarray(x, dtype=float) for x in (opens, highs, lows, closes, volumes))
        n = len(c)
        hits: List[SetupHit] = []
        last = -10 ** 9
        for i in range(self.min_bars, n):
            hit = self._fires_at(o, h, l, c, v, i, None)
            if hit is not None and i - last >= min_gap:
                hits.append(hit)
                last = i
        return hits

    # shared helper: long-trade levels with a positive, valid bracket
    @staticmethod
    def _levels(price: float, stop: float, rr: float) -> Optional[tuple]:
        risk = price - stop
        if risk <= 0:
            return None
        return price, stop, price + rr * risk


class VCPSetup(Setup):
    name = "VCP breakout"
    min_bars = 150

    def _fires_at(self, O, H, L, C, V, i, ctx=None):
        o, h, l, c, v = O[:i + 1], H[:i + 1], L[:i + 1], C[:i + 1], V[:i + 1]
        n = len(c)
        price = float(c[-1])

        sma50, sma150 = _sma(c, 50), _sma(c, 150)
        sma200 = _sma(c, 200) if n >= 200 else sma150
        stacked = price > sma50 > sma150 >= sma200 - 1e-9
        # 200-SMA rising vs ~1 month ago (only when enough history; else don't penalise).
        rising = True
        if n >= 221:
            rising = float(np.mean(c[-200:])) > float(np.mean(c[-221:-21]))

        win = c[-252:] if n >= 252 else c
        hi52, lo52 = float(np.max(win)), float(np.min(win))
        near_high = price >= 0.75 * hi52
        above_low = price >= 1.30 * lo52
        if not (stacked and rising and near_high and above_low):
            return None

        widths = _bb_width_series(c, 20)
        if len(widths) < 30:
            return None
        pctile = float((widths < widths[-1]).mean() * 100.0)   # current band-width percentile
        contracting = pctile <= 40.0

        pivot = float(np.max(h[-16:-1]))         # base high, excluding today
        vol_surge = compute_volume_surge(v)
        breakout = price > pivot and vol_surge >= 1.4
        if not (contracting and breakout):
            return None

        base_low = float(np.min(l[-16:-1]))
        stop = max(base_low, price * 0.92)       # Minervini's ≤8% hard stop, or the base low
        lv = self._levels(price, stop, rr=3.0)
        if lv is None:
            return None
        entry, stop, target = lv

        score = 0.70 + 0.10 * (1 - pctile / 40.0) + 0.05 * min(vol_surge / 2.0, 1.0)
        tags = ["VCP", "trend template"]
        reasons = [f"price stacked >50/150/200-SMA, {((price/hi52-1)*100):+.1f}% vs 52-wk high",
                   f"band-width squeeze ({pctile:.0f}th pctile), broke ${pivot:.2f} on {vol_surge:.1f}× vol"]

        if ctx:  # optional fundamental overlay (live scan only — not point-in-time)
            rs = ctx.get("rs_rating")
            if rs is not None and rs >= 70:
                score += 0.05
                reasons.append(f"RS rating {rs:.0f}")
            eg = ctx.get("earnings_growth")
            if eg is not None and eg >= 0.20:
                tags.append("earnings accel")
        return SetupHit(self.name, "long", round(entry, 2), round(stop, 2), round(target, 2),
                        _clip01(score), i, tags, reasons)


class RSI2Setup(Setup):
    name = "RSI(2) reversion"
    min_bars = 60

    def _fires_at(self, O, H, L, C, V, i, ctx=None):
        c = C[:i + 1]
        n = len(c)
        price = float(c[-1])
        sma200 = _sma(c, 200) if n >= 200 else _sma(c, n)
        ema5 = float(_ema(c, 5)[-1])
        rsi2 = compute_rsi(c, period=2)
        if not (price > sma200 and price < ema5 and rsi2 < 10.0):
            return None
        lv = self._levels(price, price * 0.95, rr=0.6)   # negatively-skewed RR (~1:1.5), small win
        if lv is None:
            return None
        entry, stop, target = lv
        score = 0.55 + (10.0 - rsi2) / 40.0
        return SetupHit(self.name, "long", round(entry, 2), round(stop, 2), round(target, 2),
                        _clip01(score), i, ["RSI(2)", "mean reversion"],
                        [f"RSI(2)={rsi2:.0f} <10 oversold", "price below 5-EMA dip",
                         "above 200-SMA uptrend"])


class EMA20PullbackSetup(Setup):
    name = "20-EMA pullback"
    min_bars = 60

    def _fires_at(self, O, H, L, C, V, i, ctx=None):
        o, h, l, c, v = O[:i + 1], H[:i + 1], L[:i + 1], C[:i + 1], V[:i + 1]
        n = len(c)
        price = float(c[-1])
        sma50 = _sma(c, 50)
        sma200 = _sma(c, 200) if n >= 200 else sma50
        ema20 = float(_ema(c, 20)[-1])
        uptrend = price > sma50 and sma50 > sma200
        touched = float(l[-1]) <= ema20 * 1.005          # tagged the 20-EMA
        rsi = compute_rsi(c, 14)
        in_zone = 35.0 <= rsi <= 50.0
        vol_declining = float(np.mean(v[-3:])) < float(np.mean(v[-20:]))
        trig = bullish_trigger(o, h, l, c)
        if not (uptrend and touched and in_zone and vol_declining and trig):
            return None
        atr = compute_atr(h, l, c)
        if atr <= 0:
            return None
        swing_low = float(np.min(l[-10:]))
        stop = min(swing_low, price - 1.5 * atr)
        lv = self._levels(price, stop, rr=2.0)
        if lv is None:
            return None
        entry, stop, target = lv
        score = 0.60 + 0.10 * (1 - abs(rsi - 42.5) / 7.5)
        return SetupHit(self.name, "long", round(entry, 2), round(stop, 2), round(target, 2),
                        _clip01(score), i, ["20-EMA pullback", trig],
                        [f"uptrend pullback to 20-EMA, RSI={rsi:.0f}", f"{trig} trigger on light volume"])


class DoubleBottomSetup(Setup):
    name = "Double bottom"
    min_bars = 40

    def _fires_at(self, O, H, L, C, V, i, ctx=None):
        h, l, c = H[:i + 1], L[:i + 1], C[:i + 1]
        dp = detect_double_bottom(h, l, lookback=60)
        if dp is None:
            return None
        price = float(c[-1])
        recent = (len(c) - 1) - dp.second_bar <= 25
        # Broke (and not yet over-extended past) the neckline.
        if not (recent and dp.neckline < price <= dp.neckline * 1.08):
            return None
        lv = self._levels(price, dp.level * 0.985, rr=2.0)
        if lv is None:
            return None
        entry, stop, target = lv
        return SetupHit(self.name, "long", round(entry, 2), round(stop, 2), round(target, 2),
                        0.60, i, ["double bottom"],
                        [f"two lows ~${dp.level:.2f} defended, broke neckline ${dp.neckline:.2f}"])


class LiquiditySweepSetup(Setup):
    name = "Liquidity sweep"
    min_bars = 30

    def _fires_at(self, O, H, L, C, V, i, ctx=None):
        o, h, l, c = O[:i + 1], H[:i + 1], L[:i + 1], C[:i + 1]
        sw = detect_liquidity_sweep(o, h, l, c, lookback=20)
        if sw is None or sw.direction != "long":
            return None
        price = float(c[-1])
        lv = self._levels(price, float(l[-1]) * 0.997, rr=2.0)
        if lv is None:
            return None
        entry, stop, target = lv
        return SetupHit(self.name, "long", round(entry, 2), round(stop, 2), round(target, 2),
                        0.55, i, ["liquidity sweep", "turtle soup"],
                        [f"swept prior low ${sw.swept_level:.2f} & closed back inside"])


# Registry — order = display/priority order in the scanner.
ALL_SETUPS: List[Setup] = [
    VCPSetup(),
    EMA20PullbackSetup(),
    DoubleBottomSetup(),
    LiquiditySweepSetup(),
    RSI2Setup(),
]

SETUP_BY_NAME: Dict[str, Setup] = {s.name: s for s in ALL_SETUPS}


def detect_all(opens, highs, lows, closes, volumes, ctx: Optional[Dict] = None) -> List[SetupHit]:
    """Run every setup against the latest bar; return all that fired (best score first)."""
    hits = [s.detect(opens, highs, lows, closes, volumes, ctx) for s in ALL_SETUPS]
    out = [hit for hit in hits if hit is not None]
    out.sort(key=lambda hh: hh.score, reverse=True)
    return out


def fvg_confluence(highs, lows, closes) -> Optional[str]:
    """A short tag if price sits just above an unmitigated bullish FVG (trend-continuation support)."""
    z = nearest_unmitigated_fvg(highs, lows, closes, kind="bullish")
    if z is None:
        return None
    price = float(np.asarray(closes, dtype=float)[-1])
    if z.bottom <= price <= z.top * 1.02:
        return "bullish FVG support"
    return None
