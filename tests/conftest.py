"""Shared fixtures — all tests run fully offline on deterministic synthetic OHLCV."""

import numpy as np
import pandas as pd
import pytest

from swingtradeapp.config import TradingConfig


def offline_config() -> TradingConfig:
    """A no-keys, offline TradingConfig for engines that require one."""
    return TradingConfig("", "", "", offline_mode=True)


def make_ohlcv(n: int = 250, trend: float = 0.0005, vol: float = 0.015,
               seed: int = 7, start_price: float = 100.0) -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV: a seeded geometric random walk with
    intraday ranges and log-normal volume. Business-day DatetimeIndex ending today."""
    rng = np.random.default_rng(seed)
    rets = trend + vol * rng.standard_normal(n)
    closes = start_price * np.exp(np.cumsum(rets))
    spread = np.abs(vol * closes * rng.standard_normal(n)) + closes * 0.002
    highs = closes + spread * rng.uniform(0.3, 1.0, n)
    lows = closes - spread * rng.uniform(0.3, 1.0, n)
    opens = lows + (highs - lows) * rng.uniform(0.2, 0.8, n)
    volumes = np.exp(rng.normal(14, 0.4, n)).round()
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                         "Close": closes, "Volume": volumes}, index=idx)


@pytest.fixture
def ohlcv():
    """Default 250-bar synthetic frame (mild uptrend)."""
    return make_ohlcv()


@pytest.fixture
def ohlcv_trending():
    """Strongly trending frame — should read as a clean uptrend everywhere."""
    return make_ohlcv(trend=0.004, vol=0.008, seed=11)


@pytest.fixture
def ohlcv_choppy():
    """Flat, noisy frame — no trend for the detectors to find."""
    return make_ohlcv(trend=0.0, vol=0.02, seed=23)


@pytest.fixture
def arrays(ohlcv):
    """The (opens, highs, lows, closes, volumes) numpy tuple most engines take."""
    return (ohlcv["Open"].to_numpy(), ohlcv["High"].to_numpy(), ohlcv["Low"].to_numpy(),
            ohlcv["Close"].to_numpy(), ohlcv["Volume"].to_numpy())
