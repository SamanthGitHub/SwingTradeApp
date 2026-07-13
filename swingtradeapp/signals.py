import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import TradingConfig
from .num import ema as _ema  # canonical EMA (was a local copy; setups.py imports it from here)

logger = logging.getLogger(__name__)

# ── Pullback / limit-entry tuning (ATR multiples) ────────────────────────────────
# Entry is a *limit* placed into a pullback (longs) / bounce (shorts) rather than "buy at market
# now". The stop is anchored to structure (recent swing) and the target to the expected move from
# the current price — so a better fill improves the reward:risk instead of it being a fixed 2:1.
ENTRY_PULLBACK_ATR = 0.5   # default distance of the limit from price when no MA support is nearby
MIN_PULLBACK_ATR = 0.25    # the limit must sit at least this far into the pullback (a real wait)
MAX_PULLBACK_ATR = 1.5     # ...but no deeper than this (don't wait for an unlikely fill)
STOP_BUFFER_ATR = 1.5      # stop sits at least this far beyond entry (a sane swing stop, not noise-tight)
TARGET_ATR = 3.0           # target distance measured from the *current* price (the move thesis)
SWING_LOOKBACK = 10        # bars used for the structure (swing low/high) stop


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


@dataclass
class Signal:
    ticker: str
    score: float
    signal_type: str  # "long" or "short"
    metadata: Dict[str, Any]
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0


class VolumeAnomalyDetector:
    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        try:
            from sklearn.ensemble import IsolationForest
            self.model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
            self.is_trained = False
        except Exception:
            self.model = None
            self.is_trained = False

    def fit(self, volume_series: List[float]) -> None:
        if self.model is None or len(volume_series) < 2:
            return
        data = np.array(volume_series).reshape(-1, 1)
        self.model.fit(data)
        self.is_trained = True

    def is_volume_anomaly(self, volume_series: List[float]) -> bool:
        if self.model is None or not self.is_trained or len(volume_series) == 0:
            return False
        try:
            data = np.array(volume_series).reshape(-1, 1)
            scores = self.model.decision_function(data)
            return bool(scores[-1] < -0.3)
        except Exception:
            return False


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's recursive moving average: ``s[i] = s[i-1] + (v[i] - s[i-1]) / period``.

    Seeded with the simple mean of the first ``period`` values; positions before the seed
    hold the seed so the output keeps the input's length.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    if n <= period:
        out[:] = float(np.mean(values))
        return out
    seed = float(np.mean(values[:period]))
    out[:period] = seed
    for i in range(period, n):
        out[i] = out[i - 1] + (values[i] - out[i - 1]) / period
    return out


def compute_rsi_series(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """True Wilder RSI over the whole series, O(n). The warm-up (first ``period`` bars) pads
    to a neutral 50.0 so the output aligns 1:1 with ``closes``."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if n < period + 1:
        return np.full(n, 50.0)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = wilder_smooth(gains, period)
    avg_loss = wilder_smooth(losses, period)
    rsi = np.where(avg_loss > 0,
                   100.0 - 100.0 / (1.0 + np.divide(avg_gain, avg_loss,
                                                    out=np.zeros_like(avg_gain),
                                                    where=avg_loss > 0)),
                   np.where(avg_gain > 0, 100.0, 50.0))
    out = np.empty(n, dtype=float)
    out[1:] = rsi
    out[:period] = 50.0
    return out


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Wilder RSI — returns value for last bar."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return 50.0
    return float(compute_rsi_series(closes, period)[-1])


def compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average Directional Index — trend strength 0-100 (>25 = strong trend).

    Full Wilder construction: RMA-smoothed TR and ±DM → DI± → DX series → ADX = RMA of DX.
    """
    if len(closes) < period * 2 + 1:
        return 20.0

    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)

    # True Range
    tr = np.maximum(highs[1:] - lows[1:],
          np.maximum(np.abs(highs[1:] - closes[:-1]),
                     np.abs(lows[1:] - closes[:-1])))

    # Directional movements
    up = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                  np.maximum(highs[1:] - highs[:-1], 0), 0)
    dn = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                  np.maximum(lows[:-1] - lows[1:], 0), 0)

    atr = wilder_smooth(tr, period)
    safe_atr = np.where(atr > 0, atr, 1.0)
    plus_di = np.where(atr > 0, 100.0 * wilder_smooth(up, period) / safe_atr, 0.0)
    minus_di = np.where(atr > 0, 100.0 * wilder_smooth(dn, period) / safe_atr, 0.0)
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / np.where(di_sum > 0, di_sum, 1.0), 0.0)

    # DX is meaningful once the DI smoothing has seeded (index period-1 in the diff arrays).
    adx = wilder_smooth(dx[period - 1:], period)
    return float(min(adx[-1], 100.0))


def compute_stoch_rsi(rsi_series: np.ndarray, period: int = 14,
                      smooth_k: int = 3, smooth_d: int = 3) -> Tuple[float, float]:
    """Stochastic RSI — momentum of RSI. Returns ``(%K, %D)``.

    %K is the ``smooth_k``-SMA of the raw stochastic of RSI over ``period`` bars; %D (the
    signal line) is the ``smooth_d``-SMA of %K — the standard construction.
    """
    rsi_series = np.asarray(rsi_series, dtype=float)
    raw_needed = smooth_k + smooth_d - 1                  # raw stoch values for both smoothings
    if len(rsi_series) < period + raw_needed:
        return 50.0, 50.0

    raws: List[float] = []
    for offset in range(raw_needed - 1, -1, -1):          # oldest → newest
        end = len(rsi_series) - offset
        window = rsi_series[end - period:end]
        lo, hi = float(np.min(window)), float(np.max(window))
        raws.append(50.0 if hi == lo else 100.0 * (float(window[-1]) - lo) / (hi - lo))
    k_vals = [float(np.mean(raws[i:i + smooth_k])) for i in range(len(raws) - smooth_k + 1)]
    d = float(np.mean(k_vals[-smooth_d:]))
    return float(k_vals[-1]), d


def compute_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 volumes: np.ndarray) -> float:
    """Volume-Weighted Average Price — cumulative from start of data."""
    if len(closes) < 2:
        return float(closes[-1])
    tp = (highs + lows + closes) / 3.0
    total_volume = np.sum(volumes)
    if total_volume <= 0:  # e.g. index symbols like ^VIX carry no volume
        return float(np.mean(tp))
    return float(np.sum(tp * volumes) / total_volume)


def compute_macd(closes: np.ndarray) -> Tuple[float, float, float]:
    """MACD (12, 26, 9). Returns (macd_line, signal_line, histogram)."""
    if len(closes) < 26:
        return 0.0, 0.0, 0.0
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line
    return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])


def compute_bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    """Bollinger Bands. Returns (upper, mid, lower) for last bar."""
    if len(closes) < period:
        mid = float(closes[-1])
        return mid, mid, mid
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    return mid + num_std * std, mid, mid - num_std * std


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range."""
    if len(closes) < period + 1:
        return float(np.mean(highs[-period:] - lows[-period:])) if len(highs) >= 1 else 0.0
    tr = np.maximum(highs[1:] - lows[1:],
          np.maximum(np.abs(highs[1:] - closes[:-1]),
                     np.abs(lows[1:] - closes[:-1])))
    return float(np.mean(tr[-period:]))


def compute_volume_surge(volumes: np.ndarray, lookback: int = 20) -> float:
    """Ratio of current volume to rolling average (last `lookback` bars)."""
    if len(volumes) < lookback + 1:
        return 1.0
    avg_vol = float(np.mean(volumes[-(lookback + 1):-1]))
    if avg_vol == 0:
        return 1.0
    return float(volumes[-1]) / avg_vol


class TrendSignalGenerator:
    def __init__(self, config: TradingConfig) -> None:
        self.config = config

    def build_signal(
        self,
        symbol: str,
        closes: List[float],
        volumes: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        min_score: float = 0.40,
    ) -> Optional[Signal]:
        """
        Multi-factor signal engine supporting LONG and SHORT.
        Factors: RSI, MACD crossover, Bollinger Band position, volume surge, SMA trend, ADX, VWAP.
        Returns entry/stop/target prices in the Signal so the execution bridge can act immediately.
        """
        if len(closes) < 26:
            return None

        c = np.array(closes, dtype=float)
        v = np.array(volumes, dtype=float)
        h = np.array(highs, dtype=float) if highs else c
        lo = np.array(lows, dtype=float) if lows else c

        current_price = float(c[-1])
        sma_20 = float(np.mean(c[-20:]))
        sma_50 = float(np.mean(c[-50:])) if len(c) >= 50 else sma_20

        rsi_series = compute_rsi_series(c)
        rsi = float(rsi_series[-1])
        stoch_rsi, stoch_sig = compute_stoch_rsi(rsi_series)
        macd_val, macd_sig, macd_hist = compute_macd(c)
        bb_upper, bb_mid, bb_lower = compute_bollinger(c)
        atr = compute_atr(h, lo, c)
        vol_surge = compute_volume_surge(v)
        adx = compute_adx(h, lo, c)
        vwap = compute_vwap(h, lo, c, v)

        # ─────────────────────────────────── LONG SIGNAL ───────────────────────────────────
        long_score = 0.0
        long_reasons: List[str] = []

        # RSI oversold (weight 0.20)
        if rsi < 35:
            long_score += 0.20
            long_reasons.append(f"RSI={rsi:.0f} oversold")
        elif rsi < 50:
            long_score += 0.10
            long_reasons.append(f"RSI={rsi:.0f} neutral-low")

        # MACD bullish (weight 0.25)
        if macd_hist > 0 and macd_val > macd_sig:
            long_score += 0.15
            long_reasons.append(f"MACD bullish hist={macd_hist:+.4f}")

        # Bollinger Band mean reversion (weight 0.20)
        if current_price <= bb_lower * 1.02:
            long_score += 0.20
            long_reasons.append(f"BB lower bounce")
        elif current_price < bb_mid * 0.98:
            long_score += 0.10
            long_reasons.append(f"BB below mid")

        # Uptrend alignment (weight 0.15)
        if current_price > sma_20 > sma_50:
            long_score += 0.15
            long_reasons.append("Uptrend SMA align")
        elif current_price > sma_20:
            long_score += 0.08
            long_reasons.append("Price>SMA20")

        # Volume confirmation (weight 0.10)
        if vol_surge >= 1.8:
            long_score += 0.10
            long_reasons.append(f"Vol surge {vol_surge:.1f}x")

        # Trend strength filter (ADX)
        if adx > 25:
            long_score += 0.05
            long_reasons.append(f"Strong trend ADX={adx:.0f}")
        elif adx < 20:
            long_score *= 0.8  # Penalize choppy markets
            long_reasons.append(f"Weak trend ADX={adx:.0f}")

        # Price above VWAP
        if current_price > vwap * 1.01:
            long_score += 0.05
            long_reasons.append("Price > VWAP")

        long_score = min(long_score, 1.0)

        # ─────────────────────────────────── SHORT SIGNAL ───────────────────────────────────
        short_score = 0.0
        short_reasons: List[str] = []

        # RSI overbought (weight 0.20)
        if rsi > 65:
            short_score += 0.20
            short_reasons.append(f"RSI={rsi:.0f} overbought")
        elif rsi > 50:
            short_score += 0.10
            short_reasons.append(f"RSI={rsi:.0f} neutral-high")

        # MACD bearish (weight 0.25)
        if macd_hist < 0 and macd_val < macd_sig:
            short_score += 0.15
            short_reasons.append(f"MACD bearish hist={macd_hist:+.4f}")

        # Bollinger Band rejection (weight 0.20)
        if current_price >= bb_upper * 0.98:
            short_score += 0.20
            short_reasons.append(f"BB upper rejection")
        elif current_price > bb_mid * 1.02:
            short_score += 0.10
            short_reasons.append(f"BB above mid")

        # Downtrend alignment (weight 0.15)
        if current_price < sma_20 < sma_50:
            short_score += 0.15
            short_reasons.append("Downtrend SMA align")
        elif current_price < sma_20:
            short_score += 0.08
            short_reasons.append("Price<SMA20")

        # Volume confirmation (weight 0.10)
        if vol_surge >= 1.8:
            short_score += 0.10
            short_reasons.append(f"Vol surge {vol_surge:.1f}x")

        # Trend strength filter (ADX)
        if adx > 25:
            short_score += 0.05
            short_reasons.append(f"Strong trend ADX={adx:.0f}")
        elif adx < 20:
            short_score *= 0.8
            short_reasons.append(f"Weak trend ADX={adx:.0f}")

        # Price below VWAP
        if current_price < vwap * 0.99:
            short_score += 0.05
            short_reasons.append("Price < VWAP")

        short_score = min(short_score, 1.0)

        # ─────────────────────────────────── RETURN BEST SIGNAL ───────────────────────────────
        # Return whichever has higher confidence
        signal_type = "long" if long_score >= short_score else "short"
        final_score = max(long_score, short_score)
        reasons = long_reasons if signal_type == "long" else short_reasons

        if final_score < min_score:
            return None

        # ── Entry / Stop / Target ────────────────────────────────────────────
        # Entry is a *limit* into a pullback (long) / bounce (short), not market-now. Stop is
        # structure-anchored (recent swing) and target is measured from the current price, so a
        # better fill yields a better R:R instead of a fixed 2:1. Falls back to a market entry when
        # ATR is unusable so degenerate data still yields a signal.
        if atr > 0:
            entry_type = "pullback-limit"
            if signal_type == "long":
                support = sma_20 if sma_20 < current_price else current_price - ENTRY_PULLBACK_ATR * atr
                entry_price = _clamp(support, current_price - MAX_PULLBACK_ATR * atr,
                                     current_price - MIN_PULLBACK_ATR * atr)
                recent_low = float(np.min(lo[-SWING_LOOKBACK:]))
                stop_price = min(entry_price - STOP_BUFFER_ATR * atr, recent_low - 0.1 * atr)
                target_price = current_price + TARGET_ATR * atr
            else:  # short — enter on a bounce toward resistance above price
                resistance = sma_20 if sma_20 > current_price else current_price + ENTRY_PULLBACK_ATR * atr
                entry_price = _clamp(resistance, current_price + MIN_PULLBACK_ATR * atr,
                                     current_price + MAX_PULLBACK_ATR * atr)
                recent_high = float(np.max(h[-SWING_LOOKBACK:]))
                stop_price = max(entry_price + STOP_BUFFER_ATR * atr, recent_high + 0.1 * atr)
                target_price = current_price - TARGET_ATR * atr
        else:  # ATR unavailable → market entry with config-based stop/target
            entry_type = "market"
            entry_price = current_price
            if signal_type == "long":
                stop_price = current_price * (1 - self.config.stop_loss_pct)
                target_price = current_price * (1 + self.config.take_profit_pct)
            else:
                stop_price = current_price * (1 + self.config.stop_loss_pct)
                target_price = current_price * (1 - self.config.take_profit_pct)

        return Signal(
            ticker=symbol,
            score=final_score,
            signal_type=signal_type,
            entry_price=round(entry_price, 2),
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            metadata={
                "rsi": float(rsi),
                "stoch_rsi": float(stoch_rsi),
                "stoch_rsi_signal": float(stoch_sig),
                "adx": float(adx),
                "macd": float(macd_val),
                "macd_signal": float(macd_sig),
                "macd_hist": float(macd_hist),
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "atr": float(atr),
                "vol_surge": float(vol_surge),
                "vwap": float(vwap),
                "sma_20": sma_20,
                "sma_50": sma_50,
                "price": current_price,
                "entry_type": entry_type,
                "long_score": long_score,
                "short_score": short_score,
                "reasons": reasons,
            },
        )

    # Keep for backward compat with dashboard.py
    def compute_rsi(self, closes: List[float], period: int = 14) -> float:
        return compute_rsi(np.array(closes, dtype=float), period)
