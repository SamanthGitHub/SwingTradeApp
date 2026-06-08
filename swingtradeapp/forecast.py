"""Probabilistic price forecasting.

Primary path: Amazon Chronos-Bolt (a zero-shot time-series foundation model) producing
quantile forecast paths. Fallback path: a Monte-Carlo simulation from historical
log-returns — so the feature always yields uncertainty bands even when the heavy model
(or torch) is not installed. Mirrors the lazy-load + graceful-fallback pattern used by
the FinBERT analyzer in nlp.py.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_QUANTILES = [0.1, 0.5, 0.9]


class PriceForecaster:
    """Forecast forward price quantiles (p10 / p50 / p90) for a close-price series."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._pipeline = None
        self._tried_load = False

    # ── Model loading ───────────────────────────────────────────────────────────

    def _ensure_pipeline(self) -> None:
        """Lazy-load Chronos-Bolt once; leave ``_pipeline`` None if unavailable."""
        if self._tried_load:
            return
        self._tried_load = True
        try:
            import torch  # noqa: F401  (chronos needs torch)
            from chronos import ChronosBoltPipeline

            self._pipeline = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small")
            logger.info("Loaded Chronos-Bolt forecasting model")
        except Exception:
            self._pipeline = None
            logger.info("Chronos unavailable — price forecasts will use heuristic fallback")

    # ── Public API ──────────────────────────────────────────────────────────────

    def forecast(self, closes: List[float], horizon: int = 5) -> Optional[Dict[str, Any]]:
        """Return forward quantile paths for the next ``horizon`` bars.

        Result: ``{p10, p50, p90: List[float], last_price, horizon, source}`` where
        ``source`` is ``"chronos"`` or ``"heuristic"``. Returns None on bad input.
        """
        series = np.asarray([c for c in closes if c is not None and np.isfinite(c)], dtype=float)
        if series.size < 30:
            return None

        self._ensure_pipeline()
        if self._pipeline is not None:
            try:
                return self._forecast_chronos(series, horizon)
            except Exception:
                logger.debug("Chronos forecast failed; using heuristic", exc_info=True)
        return self._forecast_heuristic(series, horizon)

    # ── Implementations ─────────────────────────────────────────────────────────

    def _forecast_chronos(self, series: np.ndarray, horizon: int) -> Dict[str, Any]:
        import torch

        context = torch.tensor(series, dtype=torch.float32)
        # ChronosBolt returns quantiles directly: shape [batch, horizon, n_quantiles].
        q, _mean = self._pipeline.predict_quantiles(
            context=context, prediction_length=horizon, quantile_levels=_QUANTILES,
        )
        arr = q[0].cpu().numpy()  # [horizon, 3]
        return {
            "p10": arr[:, 0].tolist(),
            "p50": arr[:, 1].tolist(),
            "p90": arr[:, 2].tolist(),
            "last_price": float(series[-1]),
            "horizon": horizon,
            "source": "chronos",
        }

    def _forecast_heuristic(self, series: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Monte-Carlo geometric random walk from historical log-returns."""
        log_ret = np.diff(np.log(series))
        mu = float(np.mean(log_ret))
        sigma = float(np.std(log_ret)) or 1e-4
        last = float(series[-1])

        rng = np.random.default_rng(42)
        n_paths = 1000
        shocks = rng.normal(mu, sigma, size=(n_paths, horizon))
        paths = last * np.exp(np.cumsum(shocks, axis=1))  # [n_paths, horizon]
        p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
        return {
            "p10": p10.tolist(),
            "p50": p50.tolist(),
            "p90": p90.tolist(),
            "last_price": last,
            "horizon": horizon,
            "source": "heuristic",
        }


def forecast_confirms(signal: Any, fc: Optional[Dict[str, Any]]) -> str:
    """Cross-check a forecast against a signal's entry/stop.

    "Confirms"  → median path ends above entry and downside path stays above stop.
    "Caution"   → downside path breaches the stop.
    "Neutral"   → anything else (or no forecast / non-long signal).
    """
    if fc is None or signal is None or getattr(signal, "signal_type", "long") != "long":
        return "Neutral"
    entry = float(getattr(signal, "entry_price", 0.0) or 0.0)
    stop = float(getattr(signal, "stop_price", 0.0) or 0.0)
    if entry <= 0:
        return "Neutral"
    p50_end = fc["p50"][-1]
    p10_min = min(fc["p10"])
    if p10_min <= stop:
        return "Caution"
    if p50_end > entry and p10_min > stop:
        return "Confirms"
    return "Neutral"


def expected_return_pct(fc: Optional[Dict[str, Any]]) -> Optional[float]:
    """Median forecast return over the horizon, in percent."""
    if not fc or not fc.get("last_price"):
        return None
    return (fc["p50"][-1] - fc["last_price"]) / fc["last_price"] * 100.0
