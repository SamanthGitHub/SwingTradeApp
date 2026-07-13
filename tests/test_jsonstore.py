"""jsonstore — atomic write/read behavior and corrupt-file recovery."""

import os

from swingtradeapp.jsonstore import atomic_write_json, atomic_write_text, read_json


def test_round_trip(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1, "b": [1, 2]})
    assert read_json(p) == {"a": 1, "b": [1, 2]}


def test_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "x.json"
    atomic_write_json(p, [1])
    assert read_json(p) == [1]


def test_no_tmp_droppings(tmp_path):
    p = tmp_path / "x.json"
    for i in range(5):
        atomic_write_json(p, {"i": i})
    assert [f for f in os.listdir(tmp_path) if f != "x.json"] == []


def test_read_corrupt_returns_default(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"trunc')
    assert read_json(p, default={"ok": True}) == {"ok": True}


def test_read_missing_returns_default(tmp_path):
    assert read_json(tmp_path / "ghost.json", default=[]) == []


def test_overwrite_replaces_whole_content(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"old": "long content " * 100})
    atomic_write_json(p, {"new": 1})
    assert read_json(p) == {"new": 1}


def test_atomic_write_text(tmp_path):
    p = tmp_path / "e.env"
    atomic_write_text(p, "KEY=value\n")
    assert p.read_text() == "KEY=value\n"


def test_non_serializable_falls_back_to_str(tmp_path):
    p = tmp_path / "odd.json"
    atomic_write_json(p, {"when": object()})  # default=str stringifies
    assert "when" in read_json(p)
