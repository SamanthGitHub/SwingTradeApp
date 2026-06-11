"""ML signal model — a calibrated probability that a name rises over the swing horizon.

The Screener ranks with a rule-based score; this adds the cornerstone *algorithmic-trading* ML
technique on top: a supervised classifier over the technical features the app already computes,
producing a **calibrated P(up)** that's surfaced in the table and fed into Kelly sizing.

Design (matches the app's conventions):

* **No lookahead.** Every indicator here is computed on ``series[:idx+1]`` only — the existing
  ``signals.compute_*`` helpers each return the value for the *last* bar of the array they're given,
  so slicing to ``idx`` is sufficient. Labels look forward, but only during training.
* **Walk-forward / OOS.** ``MLSignalModel.train`` splits each symbol's samples by time fraction
  (older → train, newest → validation) with a purge band between them, so reported AUC / accuracy /
  Brier are out-of-sample (same philosophy as ``calibrate_kelly_priors``).
* **$0 / local & graceful.** Uses scikit-learn (already a core dependency) + joblib. Every sklearn
  import is inside a method; if anything is missing the model simply stays untrained and
  ``predict_proba`` returns ``None`` so the app falls back to the heuristic path.

Pure / Streamlit-free (mirrors ``whale.py`` / ``momentum_radar.py``); ``app.py`` owns the caching,
the universe loop and the on-disk model cache.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from .signals import (
    compute_adx,
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_volume_surge,
)

logger = logging.getLogger(__name__)

# Minimum bars of history required before a feature row can be formed (SMA50, ADX, 55-day range).
MIN_BARS = 60
# Default swing horizon (sessions) the label looks forward over.
DEFAULT_HORIZON = 5

# Fixed, ordered feature names — kept stable so a saved model and live prediction agree.
FEATURE_NAMES: List[str] = [
    "rsi",
    "macd_hist_pct",
    "macd_minus_signal_pct",
    "bb_pctb",
    "bb_width",
    "atr_pct",
    "vol_surge",
    "adx",
    "stoch_rsi",
    "px_vs_sma20",
    "px_vs_sma50",
    "sma20_vs_sma50",
    "mom_5",
    "mom_10",
    "mom_20",
    "dist_to_high_20",
    "dist_to_low_20",
    "dist_to_high_55",
]


def _rsi_series(closes: np.ndarray, length: int = 14, period: int = 14) -> np.ndarray:
    """The last ``length`` RSI values (for a cheap Stochastic-RSI), each computed causally."""
    n = len(closes)
    out = []
    for k in range(length):
        end = n - length + 1 + k
        out.append(compute_rsi(closes[:end]) if end >= period + 1 else 50.0)
    return np.asarray(out, dtype=float)


def extract_features(closes: Sequence[float], highs: Sequence[float], lows: Sequence[float],
                     volumes: Sequence[float], idx: Optional[int] = None) -> Optional[np.ndarray]:
    """Feature vector at bar ``idx`` (default = last bar) using only data up to and including ``idx``.

    Returns ``None`` when there isn't enough causal history (``< MIN_BARS`` bars up to ``idx``).
    The output order matches :data:`FEATURE_NAMES`.
    """
    c = np.asarray(closes, dtype=float)
    h = np.asarray(highs, dtype=float)
    lo = np.asarray(lows, dtype=float)
    v = np.asarray(volumes, dtype=float)
    n = len(c)
    if idx is None:
        idx = n - 1
    if idx < MIN_BARS - 1 or idx >= n:
        return None
    if len(h) != n or len(lo) != n or len(v) != n:
        return None

    # Causal slices — nothing beyond idx is ever read.
    cc = c[:idx + 1]
    hh = h[:idx + 1]
    ll = lo[:idx + 1]
    vv = v[:idx + 1]
    price = float(cc[-1])
    if not np.isfinite(price) or price <= 0:
        return None

    sma20 = float(np.mean(cc[-20:]))
    sma50 = float(np.mean(cc[-50:]))

    macd_line, macd_sig, macd_hist = compute_macd(cc)
    bb_up, bb_mid, bb_low = compute_bollinger(cc)
    bb_rng = bb_up - bb_low
    atr = compute_atr(hh, ll, cc)
    stoch_rsi, _ = _stoch_from_series(_rsi_series(cc))

    hi20 = float(np.max(hh[-20:]))
    lo20 = float(np.min(ll[-20:]))
    hi55 = float(np.max(hh[-55:]))

    feats = [
        compute_rsi(cc),                                            # rsi
        macd_hist / price,                                          # macd_hist_pct
        (macd_line - macd_sig) / price,                             # macd_minus_signal_pct
        (price - bb_low) / bb_rng if bb_rng > 0 else 0.5,           # bb_pctb
        bb_rng / bb_mid if bb_mid > 0 else 0.0,                     # bb_width
        atr / price,                                                # atr_pct
        compute_volume_surge(vv),                                   # vol_surge
        compute_adx(hh, ll, cc),                                    # adx
        stoch_rsi,                                                  # stoch_rsi
        price / sma20 - 1.0,                                        # px_vs_sma20
        price / sma50 - 1.0,                                        # px_vs_sma50
        sma20 / sma50 - 1.0 if sma50 > 0 else 0.0,                  # sma20_vs_sma50
        price / float(cc[-6]) - 1.0,                                # mom_5
        price / float(cc[-11]) - 1.0,                               # mom_10
        price / float(cc[-21]) - 1.0,                               # mom_20
        (hi20 - price) / price,                                     # dist_to_high_20
        (price - lo20) / price,                                     # dist_to_low_20
        (hi55 - price) / price,                                     # dist_to_high_55
    ]
    arr = np.asarray(feats, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _stoch_from_series(rsi_series: np.ndarray, period: int = 14):
    if len(rsi_series) < 1:
        return 50.0, 50.0
    window = rsi_series[-period:]
    rmin, rmax = float(np.min(window)), float(np.max(window))
    rng = rmax - rmin
    stoch = 100.0 * (rsi_series[-1] - rmin) / rng if rng > 0 else 50.0
    return float(stoch), 50.0


def forward_label(closes: Sequence[float], idx: int, horizon: int = DEFAULT_HORIZON,
                  band: float = 0.0) -> Optional[int]:
    """1 if the close ``horizon`` bars ahead is up by more than ``band`` (fraction), else 0.

    Returns ``None`` when there's no bar ``horizon`` ahead. **Training only** — never used live.
    """
    c = np.asarray(closes, dtype=float)
    j = idx + horizon
    if idx < 0 or j >= len(c) or c[idx] <= 0:
        return None
    ret = c[j] / c[idx] - 1.0
    return int(ret > band)


def build_dataset(histories: Sequence[Dict], horizon: int = DEFAULT_HORIZON, step: int = 3):
    """Assemble (X, y, times) across symbols.

    ``histories``: dicts with ``closes`` (required) and ``highs``/``lows``/``volumes`` (optional).
    ``times`` is each sample's within-symbol time fraction (``idx / n``) — used for the walk-forward
    split in :meth:`MLSignalModel.train` so older bars train and the newest bars validate.
    ``step`` strides the sliding window to reduce overlap/autocorrelation between samples.
    """
    X: List[np.ndarray] = []
    y: List[int] = []
    times: List[float] = []
    for hst in histories:
        c = np.asarray(hst.get("closes", []), dtype=float)
        n = len(c)
        if n < MIN_BARS + horizon + 1:
            continue
        h = np.asarray(hst.get("highs", c), dtype=float)
        lo = np.asarray(hst.get("lows", c), dtype=float)
        v = np.asarray(hst.get("volumes", np.zeros(n)), dtype=float)
        if len(h) != n or len(lo) != n or len(v) != n:
            h, lo, v = c, c, np.zeros(n)
        for idx in range(MIN_BARS - 1, n - horizon, step):
            feats = extract_features(c, h, lo, v, idx)
            if feats is None:
                continue
            lbl = forward_label(c, idx, horizon)
            if lbl is None:
                continue
            X.append(feats)
            y.append(lbl)
            times.append(idx / n)
    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,)), np.empty((0,))
    return np.vstack(X), np.asarray(y, dtype=int), np.asarray(times, dtype=float)


class MLSignalModel:
    """Calibrated gradient-boosting classifier over :data:`FEATURE_NAMES`. Graceful when untrained."""

    # Walk-forward split fractions: train on the oldest 65%, validate on the newest 30%; the
    # 0.65–0.70 gap is the purge band so no training label leaks into the validation window.
    TRAIN_FRAC = 0.65
    VAL_FRAC = 0.70

    def __init__(self) -> None:
        self.clf = None
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.horizon: int = DEFAULT_HORIZON
        self.metrics: Dict[str, float] = {}

    @property
    def trained(self) -> bool:
        return self.clf is not None

    def train(self, X: np.ndarray, y: np.ndarray, times: np.ndarray,
              horizon: int = DEFAULT_HORIZON) -> Dict[str, float]:
        """Fit the calibrated classifier walk-forward; return OOS metrics (or {} on failure)."""
        self.horizon = horizon
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
        except Exception:
            logger.info("scikit-learn unavailable — ML signal model disabled")
            return {}

        if len(y) < 300 or len(np.unique(y)) < 2:
            logger.info("Not enough data to train ML signal model (n=%d)", len(y))
            return {}

        train_mask = times <= self.TRAIN_FRAC
        val_mask = times >= self.VAL_FRAC
        if train_mask.sum() < 200 or val_mask.sum() < 50 \
                or len(np.unique(y[train_mask])) < 2 or len(np.unique(y[val_mask])) < 2:
            return {}

        base = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.06, max_iter=200, l2_regularization=1.0,
            random_state=42)
        method = "isotonic" if train_mask.sum() >= 1000 else "sigmoid"
        try:
            clf = CalibratedClassifierCV(base, method=method, cv=3)
            clf.fit(X[train_mask], y[train_mask])
        except Exception:
            logger.exception("ML signal model training failed")
            return {}

        p_val = clf.predict_proba(X[val_mask])[:, 1]
        self.clf = clf
        self.metrics = {
            "auc": float(roc_auc_score(y[val_mask], p_val)),
            "accuracy": float(accuracy_score(y[val_mask], (p_val > 0.5).astype(int))),
            "brier": float(brier_score_loss(y[val_mask], p_val)),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "base_rate": float(np.mean(y[val_mask])),
        }
        logger.info("ML signal model trained: %s", self.metrics)
        return self.metrics

    def predict_proba(self, features: Optional[np.ndarray]) -> Optional[float]:
        """Calibrated P(up) for one feature vector, or ``None`` if untrained / no features."""
        if self.clf is None or features is None:
            return None
        try:
            x = np.asarray(features, dtype=float).reshape(1, -1)
            if x.shape[1] != len(self.feature_names):
                return None
            return float(self.clf.predict_proba(x)[0, 1])
        except Exception:
            return None

    # ── Persistence (joblib ships with scikit-learn) ───────────────────────────────
    def save(self, path) -> bool:
        if self.clf is None:
            return False
        try:
            import joblib
            joblib.dump({"clf": self.clf, "feature_names": self.feature_names,
                         "horizon": self.horizon, "metrics": self.metrics}, path)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, path) -> Optional["MLSignalModel"]:
        try:
            import joblib
            blob = joblib.load(path)
            m = cls()
            m.clf = blob["clf"]
            m.feature_names = blob.get("feature_names", list(FEATURE_NAMES))
            m.horizon = blob.get("horizon", DEFAULT_HORIZON)
            m.metrics = blob.get("metrics", {})
            # A model trained on a different feature set is unusable — treat as absent.
            if m.feature_names != list(FEATURE_NAMES):
                return None
            return m
        except Exception:
            return None
