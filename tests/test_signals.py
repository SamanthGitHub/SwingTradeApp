"""Indicator correctness — known-answer checks against independent reference
implementations (this is what locks in the Phase-2 Wilder-smoothing fixes)."""

import numpy as np
import pytest

from swingtradeapp.signals import (
    TrendSignalGenerator,
    compute_adx,
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_rsi_series,
    compute_stoch_rsi,
    compute_volume_surge,
    wilder_smooth,
)
from tests.conftest import make_ohlcv


# ── Reference implementations (plain loops, written independently of signals.py) ──

def ref_wilder(values, period):
    values = np.asarray(values, dtype=float)
    out = np.empty(len(values))
    out[:period] = np.nan
    s = values[:period].mean()
    out[period - 1] = s
    for i in range(period, len(values)):
        s = s + (values[i] - s) / period
        out[i] = s
    return out


def ref_rsi_last(closes, period=14):
    closes = np.asarray(closes, dtype=float)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def ref_adx_last(highs, lows, closes, period=14):
    highs, lows, closes = (np.asarray(a, dtype=float) for a in (highs, lows, closes))
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    up_m = highs[1:] - highs[:-1]
    dn_m = lows[:-1] - lows[1:]
    plus_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    minus_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    atr = ref_wilder(tr, period)
    p_di = 100.0 * ref_wilder(plus_dm, period) / np.where(atr > 0, atr, np.nan)
    m_di = 100.0 * ref_wilder(minus_dm, period) / np.where(atr > 0, atr, np.nan)
    dx = 100.0 * np.abs(p_di - m_di) / (p_di + m_di)
    dx = dx[period - 1:]                       # first defined DX
    return ref_wilder(dx, period)[-1]


# ── wilder_smooth ────────────────────────────────────────────────────────────────

def test_wilder_smooth_matches_reference():
    rng = np.random.default_rng(3)
    vals = rng.uniform(1, 5, 100)
    got = wilder_smooth(vals, 14)
    want = ref_wilder(vals, 14)
    assert got[-1] == pytest.approx(want[-1], rel=1e-9)


def test_wilder_smooth_constant_series_is_flat():
    out = wilder_smooth(np.full(60, 3.0), 14)
    assert out[-1] == pytest.approx(3.0)


# ── RSI ──────────────────────────────────────────────────────────────────────────

def test_rsi_series_matches_reference_wilder_rsi():
    closes = make_ohlcv(120)["Close"].to_numpy()
    got = compute_rsi_series(closes)[-1]
    assert got == pytest.approx(ref_rsi_last(closes), abs=1e-6)


def test_rsi_extremes():
    rising = np.linspace(100, 200, 60)
    falling = np.linspace(200, 100, 60)
    assert compute_rsi(rising) == pytest.approx(100.0)
    assert compute_rsi(falling) < 5.0


def test_rsi_series_bounds_and_length(ohlcv):
    closes = ohlcv["Close"].to_numpy()
    series = compute_rsi_series(closes)
    assert len(series) == len(closes)
    assert np.all((series >= 0) & (series <= 100))


def test_rsi_short_history_neutral():
    assert compute_rsi(np.array([1.0, 2.0, 3.0])) == 50.0


# ── ADX ──────────────────────────────────────────────────────────────────────────

def test_adx_matches_reference_wilder_adx():
    df = make_ohlcv(200, trend=0.002, vol=0.012, seed=5)
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    got = compute_adx(h, l, c)
    want = ref_adx_last(h, l, c)
    assert got == pytest.approx(want, rel=0.02), "ADX must be Wilder-smoothed, not single-bar DX"


def test_adx_trend_beats_chop():
    trend = make_ohlcv(200, trend=0.006, vol=0.004, seed=1)
    chop = make_ohlcv(200, trend=0.0, vol=0.02, seed=2)
    adx_trend = compute_adx(trend["High"].to_numpy(), trend["Low"].to_numpy(), trend["Close"].to_numpy())
    adx_chop = compute_adx(chop["High"].to_numpy(), chop["Low"].to_numpy(), chop["Close"].to_numpy())
    assert adx_trend > adx_chop


def test_adx_short_history_fallback():
    assert compute_adx(np.ones(10), np.ones(10), np.ones(10)) == 20.0


# ── StochRSI ─────────────────────────────────────────────────────────────────────

def test_stoch_rsi_bounds_and_real_signal_line(ohlcv):
    rsi = compute_rsi_series(ohlcv["Close"].to_numpy())
    k, d = compute_stoch_rsi(rsi)
    assert 0.0 <= k <= 100.0 and 0.0 <= d <= 100.0
    # regression for the old placeholder: %D must respond to the data, not sit at 50.0
    rsi_hot = np.linspace(40, 90, 60)
    k_hot, d_hot = compute_stoch_rsi(rsi_hot)
    assert k_hot > 95.0 and d_hot > 95.0


def test_stoch_rsi_short_history_neutral():
    assert compute_stoch_rsi(np.array([50.0, 55.0])) == (50.0, 50.0)


# ── MACD / Bollinger / ATR / volume ─────────────────────────────────────────────

def test_macd_constant_series_is_zero():
    line, sig, hist = compute_macd(np.full(60, 100.0))
    assert line == pytest.approx(0.0, abs=1e-9)
    assert hist == pytest.approx(0.0, abs=1e-9)


def test_bollinger_ordering(ohlcv):
    closes = ohlcv["Close"].to_numpy()
    upper, mid, lower = compute_bollinger(closes)
    assert lower < mid < upper
    assert mid == pytest.approx(closes[-20:].mean(), rel=1e-9)


def test_atr_positive(arrays):
    _, highs, lows, closes, _ = arrays
    assert compute_atr(highs, lows, closes) > 0


def test_volume_surge_detects_spike():
    vols = np.full(40, 1e6)
    vols[-1] = 5e6
    assert compute_volume_surge(vols) == pytest.approx(5.0, rel=0.3)


# ── TrendSignalGenerator end-to-end ─────────────────────────────────────────────

def test_build_signal_long_on_oversold_dip():
    """The generator is a mean-reversion entry engine: its bread-and-butter long is a
    sharp, high-volume flush to the lower Bollinger band inside a long uptrend."""
    from tests.conftest import offline_config
    gen = TrendSignalGenerator(offline_config())
    df = make_ohlcv(240, trend=0.0015, vol=0.009, seed=3)
    last = df["Close"].iloc[-1]
    dip = last * np.array([0.985, 0.965, 0.94, 0.925])       # 4-bar ~7.5% flush
    closes = np.concatenate([df["Close"].to_numpy(), dip])
    highs = np.concatenate([df["High"].to_numpy(), dip * 1.012])
    lows = np.concatenate([df["Low"].to_numpy(), dip * 0.99])
    vols = np.concatenate([df["Volume"].to_numpy(),
                           df["Volume"].mean() * np.array([1.5, 2.0, 2.6, 3.0])])
    sig = gen.build_signal("TEST", closes.tolist(), vols.tolist(),
                           highs=highs.tolist(), lows=lows.tolist())
    assert sig is not None
    assert sig.signal_type == "long"
    assert 0.4 <= sig.score <= 1.0
    assert sig.stop_price < sig.entry_price < sig.target_price
    assert sig.metadata.get("reasons")


def test_build_signal_parabolic_run_reads_overextended(ohlcv_trending):
    """A relentless exponential run is *not* a fresh long entry to this engine — it reads
    overbought (short-side mean reversion) or nothing. Locks in the design intent."""
    from tests.conftest import offline_config
    gen = TrendSignalGenerator(offline_config())
    df = ohlcv_trending
    sig = gen.build_signal("TEST", df["Close"].tolist(), df["Volume"].tolist(),
                           highs=df["High"].tolist(), lows=df["Low"].tolist())
    assert sig is None or sig.signal_type == "short"


def test_build_signal_survives_short_history():
    from tests.conftest import offline_config
    gen = TrendSignalGenerator(offline_config())
    closes = list(np.linspace(10, 12, 30))
    sig = gen.build_signal("TINY", closes, [1e6] * 30)
    assert sig is None or 0.0 <= sig.score <= 1.0
