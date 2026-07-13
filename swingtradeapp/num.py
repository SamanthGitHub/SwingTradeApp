"""Tiny shared numeric helpers — the single home for utilities that used to be
copy-pasted across confluence / momentum_radar / whale / setups / signals."""

from __future__ import annotations

import numpy as np


def clip01(x) -> float:
    """Clamp to [0, 1]; non-numeric input clamps to 0.0 instead of raising."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average over an array (same length as input)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out
