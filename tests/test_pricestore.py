"""pricestore — the shared prefetch store's slice/TTL/span-upgrade semantics."""

import pandas as pd
import pytest

from swingtradeapp import pricestore
from tests.conftest import make_ohlcv


@pytest.fixture(autouse=True)
def clean_store():
    pricestore.clear()
    yield
    pricestore.clear()


def test_get_miss_returns_none():
    assert pricestore.get("NOPE", 60) is None


def test_put_then_get_slices_window():
    df = make_ohlcv(300)  # ~300 business days ≈ 420 calendar days
    pricestore.put_many({"AAA": df}, span_days=400)
    got = pricestore.get("AAA", 60)
    assert got is not None and not got.empty
    assert len(got) < len(df), "a 60-day request must be a tail slice, not the whole span"
    assert got.index[-1] == df.index[-1]
    span = (got.index[-1] - got.index[0]).days
    assert span <= 60


def test_short_span_cannot_serve_wider_request():
    df = make_ohlcv(40)
    pricestore.put_many({"BBB": df}, span_days=60)
    assert pricestore.get("BBB", 200) is None, "a 60d entry must not silently serve a 200d request"


def test_span_upgrade_and_no_downgrade():
    small = make_ohlcv(40, seed=1)
    big = make_ohlcv(300, seed=2)
    pricestore.put_many({"CCC": small}, span_days=60)
    pricestore.put_many({"CCC": big}, span_days=400)      # upgrade
    upgraded = pricestore.get("CCC", 400)
    assert upgraded is not None and len(upgraded) > len(small)
    pricestore.put_many({"CCC": small}, span_days=60)     # attempted downgrade
    kept = pricestore.get("CCC", 400)
    assert kept is not None and len(kept) > len(small), \
        "a shorter fetch must never shrink the stored span"


def test_get_returns_a_copy():
    df = make_ohlcv(100)
    pricestore.put_many({"DDD": df}, span_days=120)
    got = pricestore.get("DDD", 120)
    got.iloc[0, 0] = -999.0
    again = pricestore.get("DDD", 120)
    assert again.iloc[0, 0] != -999.0


def test_ttl_expiry(monkeypatch):
    df = make_ohlcv(100)
    pricestore.put_many({"EEE": df}, span_days=120)
    real = pricestore.time.monotonic()
    monkeypatch.setattr(pricestore.time, "monotonic",
                        lambda: real + pricestore.TTL_SECONDS + 1)
    assert pricestore.get("EEE", 120) is None


def test_missing_lists_only_unserved():
    pricestore.put_many({"FFF": make_ohlcv(300)}, span_days=400)
    assert pricestore.missing(["FFF", "GGG"], 120) == ["GGG"]


def test_empty_frames_ignored():
    pricestore.put_many({"HHH": pd.DataFrame()}, span_days=120)
    assert pricestore.get("HHH", 120) is None
