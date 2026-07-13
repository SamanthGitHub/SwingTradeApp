"""ApiBudget — the "never exceed the free tier" guarantee, including under threads."""

from concurrent.futures import ThreadPoolExecutor

from swingtradeapp import ratelimit
from swingtradeapp.ratelimit import ApiBudget


class FakeTime:
    """Stand-in for the ``time`` module inside ratelimit — lets tests move the clock."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def time(self):
        return self.t


def budget(tmp_path):
    return ApiBudget(tmp_path / "usage.json")


def test_minute_cap_enforced(tmp_path):
    b = budget(tmp_path)
    results = [b.try_acquire("polygon")[0] for _ in range(8)]
    assert results.count(True) == 5
    assert results[:5] == [True] * 5


def test_window_ages_out(tmp_path, monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(ratelimit, "time", fake)
    b = budget(tmp_path)
    for _ in range(5):
        assert b.try_acquire("polygon")[0]
    assert not b.try_acquire("polygon")[0]
    fake.t += 61
    assert b.try_acquire("polygon")[0], "a slot must free up after the minute window passes"


def test_try_acquire_thread_hammer_never_breaches(tmp_path):
    b = budget(tmp_path)
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: b.try_acquire("polygon")[0], range(64)))
    assert results.count(True) == 5, "concurrent acquires must never exceed the 5/min cap"


def test_check_does_not_consume(tmp_path):
    b = budget(tmp_path)
    for _ in range(20):
        assert b.check("polygon")[0]
    assert b.status("polygon")["used_minute"] == 0


def test_unknown_provider_unmetered(tmp_path):
    b = budget(tmp_path)
    assert b.try_acquire("nonexistent") == (True, "")


def test_corrupt_ledger_recovers(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text('{"polygon": [12345')  # truncated mid-write, pre-atomic era
    b = ApiBudget(path)
    assert b.try_acquire("polygon")[0]
    assert b.status("polygon")["used_minute"] == 1


def test_status_fields(tmp_path):
    b = budget(tmp_path)
    b.try_acquire("polygon")
    s = b.status("polygon")
    assert s["per_minute"] == 5
    assert s["used_minute"] == 1
    assert s["remaining_minute"] == 4
