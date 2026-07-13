"""ml_signal — feature shape/determinism/finiteness and the feature-version retrain gate."""

import numpy as np

from swingtradeapp import ml_signal
from swingtradeapp.ml_signal import FEATURE_NAMES, FEATURE_VERSION, extract_features
from tests.conftest import make_ohlcv


def _arrays(n=250, seed=7):
    df = make_ohlcv(n, seed=seed)
    return (df["Close"].to_numpy(), df["High"].to_numpy(),
            df["Low"].to_numpy(), df["Volume"].to_numpy())


def test_feature_vector_shape_and_finite():
    closes, highs, lows, volumes = _arrays()
    feats = extract_features(closes, highs, lows, volumes)
    assert feats is not None
    assert len(feats) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(feats))


def test_features_deterministic():
    closes, highs, lows, volumes = _arrays()
    f1 = extract_features(closes, highs, lows, volumes)
    f2 = extract_features(closes, highs, lows, volumes)
    assert np.allclose(f1, f2)


def test_features_use_only_given_bars():
    """Causality: features from a truncated window must not change when later bars are
    appended and the same truncation is applied — no hidden global state."""
    closes, highs, lows, volumes = _arrays(300)
    cut = 250
    f_before = extract_features(closes[:cut], highs[:cut], lows[:cut], volumes[:cut])
    # mutate the "future" then re-truncate: identical input window → identical features
    closes2 = closes.copy(); closes2[cut:] = 1.0
    f_after = extract_features(closes2[:cut], highs[:cut], lows[:cut], volumes[:cut])
    assert np.allclose(f_before, f_after)


def test_short_history_returns_none_or_finite():
    closes, highs, lows, volumes = _arrays(30)
    feats = extract_features(closes, highs, lows, volumes)
    assert feats is None or np.all(np.isfinite(feats))


def test_feature_version_gate_forces_retrain(tmp_path):
    """A model blob saved under an older FEATURE_VERSION must not load (→ retrain)."""
    import joblib
    path = tmp_path / "model.joblib"
    joblib.dump({"model": None, "meta": {}, "feature_version": FEATURE_VERSION - 1}, path)
    assert ml_signal.MLSignalModel.load(path) is None


def test_forward_label_is_future_looking():
    closes = np.linspace(100, 200, 100)  # strictly rising → every forward label is "up"
    lab = ml_signal.forward_label(closes, idx=50)
    assert lab == 1
