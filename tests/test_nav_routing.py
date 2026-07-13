"""The nav-label ↔ page-routing contract: every label in ui.NAV_ITEMS must be routed
(either a ``page == "…"`` branch in app.py or a key in screens.PAGES), and vice versa —
this turns the app's string-matched routing fragility into a caught test failure."""

import re
from pathlib import Path

from swingtradeapp import ui

APP = Path(__file__).resolve().parent.parent / "app.py"


def routed_labels() -> set:
    src = APP.read_text()
    labels = set(re.findall(r'page == "([^"]+)"', src))
    try:
        from screens import PAGES  # post-split registry (optional while migration is underway)
        labels |= set(PAGES.keys())
    except ImportError:
        pass
    return labels


def test_every_nav_item_has_a_route():
    nav = {label for label, _icon, _blurb in ui.NAV_ITEMS}
    missing = nav - routed_labels()
    assert not missing, f"nav items with no page branch: {missing}"


def test_no_orphan_routes():
    nav = {label for label, _icon, _blurb in ui.NAV_ITEMS}
    orphans = routed_labels() - nav
    assert not orphans, f"page branches unreachable from the nav: {orphans}"


def test_nav_items_unique():
    labels = [label for label, _icon, _blurb in ui.NAV_ITEMS]
    assert len(labels) == len(set(labels))
