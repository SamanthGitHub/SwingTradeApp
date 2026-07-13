"""WatchlistManager — CRUD round-trips, persistence across instances, corruption recovery."""

from swingtradeapp.watchlist import WatchlistManager


def test_default_watchlist_exists(tmp_path):
    m = WatchlistManager(str(tmp_path))
    assert "Default" in m.list_watchlists()


def test_crud_round_trip(tmp_path):
    m = WatchlistManager(str(tmp_path))
    m.create_watchlist("Tech")
    m.add_symbol("Tech", "AAPL")
    m.add_symbol("Tech", "NVDA")
    m.remove_symbol("Tech", "AAPL")
    assert m.get_watchlist("Tech") == ["NVDA"]
    m.delete_watchlist("Tech")
    assert "Tech" not in m.list_watchlists()


def test_persists_across_instances(tmp_path):
    m1 = WatchlistManager(str(tmp_path))
    m1.add_symbol("Swing", "TSLA")
    m1.add_alert("TSLA", "price", 250.0, "above")
    m2 = WatchlistManager(str(tmp_path))
    assert m2.get_watchlist("Swing") == ["TSLA"]
    assert m2.get_alerts("TSLA")[0]["value"] == 250.0


def test_duplicate_symbol_not_added_twice(tmp_path):
    m = WatchlistManager(str(tmp_path))
    m.add_symbol("Default", "AMD")
    m.add_symbol("Default", "AMD")
    assert m.get_watchlist("Default") == ["AMD"]


def test_default_cannot_be_deleted(tmp_path):
    m = WatchlistManager(str(tmp_path))
    m.delete_watchlist("Default")
    assert "Default" in m.list_watchlists()


def test_corrupt_files_recover_clean(tmp_path):
    (tmp_path / "watchlists.json").write_text('{"Default": ["AAP')
    (tmp_path / "alerts.json").write_text("not json at all")
    m = WatchlistManager(str(tmp_path))
    assert m.watchlists == {"Default": []}
    assert m.alerts == {}


def test_alert_remove_by_index(tmp_path):
    m = WatchlistManager(str(tmp_path))
    m.add_alert("SPY", "price", 500.0)
    m.add_alert("SPY", "rsi", 70.0)
    m.remove_alert("SPY", 0)
    assert len(m.get_alerts("SPY")) == 1
    assert m.get_alerts("SPY")[0]["type"] == "rsi"
