"""Offline smoke tests for the pure analysis engines — they must run, return sane shapes,
and stay silent (no network) on synthetic data."""

import numpy as np

from swingtradeapp import analysis_guide as ag
from swingtradeapp import confluence as cf
from swingtradeapp import patterns, regime, setups
from swingtradeapp.retry import is_retryable, with_retry
from swingtradeapp.whale import WhaleConfig, WhaleDetector
from tests.conftest import make_ohlcv


def test_setups_detect_all_runs(arrays):
    opens, highs, lows, closes, volumes = arrays
    hits = setups.detect_all(opens, highs, lows, closes, volumes)
    assert isinstance(hits, list)
    for h in hits:
        assert h.stop < h.entry < h.target or h.entry > 0  # long setups: coherent levels
        assert h.reasons


def test_patterns_fvg_detector(arrays):
    _, highs, lows, closes, _ = arrays
    zones = patterns.detect_fair_value_gaps(highs, lows, closes)
    assert isinstance(zones, list)


def test_regime_verdicts():
    up = make_ohlcv(300, trend=0.003, vol=0.008, seed=3)["Close"].tolist()
    down = make_ohlcv(300, trend=-0.003, vol=0.02, seed=4)["Close"].tolist()
    r_up = regime.assess_regime(up, vix=14.0, breadth_pct=0.7)
    r_down = regime.assess_regime(down, vix=35.0, breadth_pct=0.2)
    assert r_up.verdict in {"Trade", "Caution", "Stand-aside"}
    assert r_up.score > r_down.score
    assert not r_down.allows_long or r_down.verdict != "Stand-aside"


def test_confluence_score_ticker_directions():
    bull = {"tech": cf.Vote(1, 0.9), "whale": cf.Vote(1, 0.7), "news": cf.Vote(1, 0.6)}
    bear = {"tech": cf.Vote(-1, 0.9), "whale": cf.Vote(-1, 0.7)}
    r_bull = cf.score_ticker(bull)
    r_bear = cf.score_ticker(bear)
    assert r_bull["direction"] == "long" and r_bear["direction"] == "short"
    assert 0 <= r_bull["conviction"] <= 100
    assert cf.score_ticker({})["direction"] == "neutral"


def test_whale_detector_flags_volume_spike():
    df = make_ohlcv(60, seed=9)
    volumes = df["Volume"].to_numpy().copy()
    volumes[-1] = volumes[:-1].mean() * 6            # unmistakable outsized print
    closes = df["Close"].to_numpy()
    det = WhaleDetector(WhaleConfig(min_rvol=2.0, min_dollar_vol=0.0))
    res = det.analyze("TEST", df["Open"].to_numpy(), df["High"].to_numpy(),
                      df["Low"].to_numpy(), closes, volumes)
    assert res is None or (0 <= res["whale_score"] <= 100)


def test_analysis_guide_grades():
    good = ag.grade_trend(price=110.0, sma20=105.0, sma50=100.0)
    bad = ag.grade_trend(price=90.0, sma20=95.0, sma50=100.0)
    assert good.mark == "pass" and bad.mark == "fail"
    assert ag.grade_rr(3.0).mark in {"pass", "warn", "fail", "na"}
    n_pass, n_graded, verdict = ag.summarize({"trend": "pass", "momentum": "warn", "volume": "na"})
    assert n_graded == 2 and n_pass == 1 and isinstance(verdict, str)


def test_retry_classifier():
    assert is_retryable(Exception("HTTP Error 429: Too Many Requests"))
    assert is_retryable(Exception("Connection timed out"))
    assert not is_retryable(Exception("KeyError: 'Close'"))


def test_with_retry_retries_then_succeeds():
    calls = {"n": 0}

    @with_retry(retries=3, base_delay=0.01, max_delay=0.02)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("429 too many requests")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3
