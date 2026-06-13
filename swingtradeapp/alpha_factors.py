"""Cross-sectional factor library for the Alpha Engine.

Everything here is **pure** (no Streamlit, no network) and operates on a *price panel*: a wide
``pd.DataFrame`` indexed by date (ascending), one column per symbol, values = adjusted close.
That shape makes every factor a vectorised, lookahead-free transform — a factor value at row ``t``
only ever reads prices at ``t`` or earlier (via ``.shift``/``.rolling``), so a backtest that ranks
on these at date ``t`` and trades at ``t+1`` cannot peek into the future.

Design notes
------------
* **Why fixed, a-priori factors (not fit weights):** the classic equity anomalies — cross-sectional
  *momentum* (12-1), short-term *reversal*, *trend*, the *low-volatility* anomaly, and *52-week-high
  proximity* — have decades of out-of-sample literature. Using them with fixed signs/weights avoids
  the single biggest way retail backtests lie to themselves: fitting the combination in-sample.
* **Survivorship bias (honest limitation):** free daily data only knows *currently listed* names, so
  delisted losers are missing. We cannot fully remove that with $0 data. We *can* remove **lookahead**
  bias (above) and minimise delisting impact by running on a liquid large-cap universe.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── Individual factors (higher score = more attractive to go LONG) ───────────────

def momentum(px: pd.DataFrame, lookback: int = TRADING_DAYS, skip: int = 21) -> pd.DataFrame:
    """Classic 12-1 momentum: total return from ``t-lookback`` to ``t-skip``.

    Skipping the most recent ~month sidesteps the short-term reversal effect that contaminates raw
    12-month momentum. Higher = stronger trailing winner.
    """
    return px.shift(skip) / px.shift(lookback) - 1.0


def reversal(px: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Short-term reversal (contrarian): the *negative* of the last ``window`` days' return.

    Recent big losers tend to bounce and recent big winners to fade over ~1 month, so we flip the
    sign — a recent loser gets a high score.
    """
    return -(px / px.shift(window) - 1.0)


def trend(px: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Distance of price above its long moving average (price/SMA - 1). Higher = healthier uptrend."""
    return px / px.rolling(window).mean() - 1.0


def low_volatility(px: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """The low-volatility anomaly: *negative* annualised realised vol of daily returns.

    Lower-vol names have historically earned better risk-adjusted returns, so we negate vol → a calm
    stock scores high.
    """
    rets = px.pct_change()
    vol = rets.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return -vol


def high_proximity(px: pd.DataFrame, window: int = TRADING_DAYS) -> pd.DataFrame:
    """Proximity to the trailing 52-week high (price/rolling-max - 1, in [-1, 0]).

    Names pressing their highs tend to keep working (the '52-week-high' effect). 0 = at the high.
    """
    return px / px.rolling(window).max() - 1.0


def realized_vol(px: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualised realised vol — used by the engine for inverse-vol position sizing (not a score)."""
    return px.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)


# Default factor set and weights. Signs are already baked into the factors (all "higher = long").
DEFAULT_FACTORS = {
    "momentum": (momentum, 0.35),
    "trend": (trend, 0.25),
    "high_proximity": (high_proximity, 0.15),
    "low_volatility": (low_volatility, 0.15),
    "reversal": (reversal, 0.10),
}


# ── Cross-sectional standardisation & combination ────────────────────────────────

def zscore_cross_section(factor: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """Standardise each *row* (date) across symbols → mean 0, std 1, winsorised to ±``winsor``.

    Cross-sectional (per-date) z-scoring is what makes heterogeneous factors comparable and lets us
    rank names *relative to their peers on that day*, which is how factor alpha is actually harvested.
    """
    mu = factor.mean(axis=1)
    sigma = factor.std(axis=1).replace(0.0, np.nan)
    z = factor.sub(mu, axis=0).div(sigma, axis=0)
    return z.clip(-winsor, winsor)


def composite_score(
    px: pd.DataFrame,
    factors: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute z-scored factor panels + the weighted composite from a price panel.

    Returns a dict: each factor name → its cross-sectional z-score panel, plus ``"composite"`` =
    the weight-blended score used for ranking. All panels share ``px``'s index/columns.
    """
    factors = factors or DEFAULT_FACTORS
    out: Dict[str, pd.DataFrame] = {}
    composite = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    weight_sum = 0.0
    for name, (fn, weight) in factors.items():
        z = zscore_cross_section(fn(px))
        out[name] = z
        composite = composite.add(z * weight, fill_value=0.0)
        weight_sum += weight
    if weight_sum:
        composite = composite / weight_sum
    out["composite"] = composite
    return out


def sector_neutralize(score: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """Demean each date's scores *within sector* so the book ranks names against sector peers.

    A raw composite can quietly become a sector bet (e.g. "long whatever Tech is doing"). Subtracting
    the per-sector mean each day strips that out, leaving **stock-selection** alpha — the same idea as an
    institutional risk model neutralising sector exposure, done cheaply with a known sector map. Names
    with no mapped sector (or singleton sectors) are left as-is.
    """
    sm = pd.Series(sector_map)
    out = score.copy()
    for sec in sm.unique():
        cols = [c for c in sm[sm == sec].index if c in score.columns]
        if len(cols) >= 2:
            out[cols] = score[cols].sub(score[cols].mean(axis=1), axis=0)
    return out
