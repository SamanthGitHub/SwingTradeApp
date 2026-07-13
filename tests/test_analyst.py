"""analyst — briefs must be non-empty, deterministic within a day, and grounded
(every number in the text exists in the dossier)."""

import re

from swingtradeapp import analyst
from swingtradeapp.analyst import Brief, Dossier, build_brief, market_brief, render_markdown


def full_dossier() -> Dossier:
    return Dossier(
        symbol="TEST", price=105.20, change_pct=1.8, name="Test Corp", sector="Technology",
        signal_type="long", score=0.71, rsi=61.5, adx=28.0, macd_hist=0.42, vol_surge=2.4,
        atr=2.1, sma20=101.0, sma50=97.5, vwap=100.2,
        entry=104.0, stop=99.8, target=112.4,
        reasons=["RSI in momentum zone", "MACD histogram positive"],
        setups=[{"name": "VCP", "score": 78.0, "reasons": ["volatility contraction"]}],
        confluence={"direction": "long", "conviction": 64, "confluence": "5/7", "coverage": "6/7"},
        regime={"verdict": "Trade", "score": 72, "drivers": ["SPY above 200-SMA"]},
        whale={"signal": "Accumulation", "whale_score": 66.0, "rvol": 2.8},
        forecast={"p10": 102.0, "p50": 106.5, "p90": 111.0, "last_price": 105.2, "source": "heuristic"},
        news={"positive_pct": 0.7, "count": 10, "top_headline": "Test Corp beats estimates", "events": []},
        fundamentals={"pe_ratio": 21.0, "profit_margin": 0.18, "roe": 0.25,
                      "debt_to_equity": 40.0, "market_cap": 5e10},
        earnings_days=12, ml_prob=0.63,
    )


def test_full_dossier_brief_has_all_sections():
    brief = build_brief(full_dossier())
    assert isinstance(brief, Brief)
    assert brief.headline
    titles = [t for t, _ in brief.paragraphs]
    assert len(titles) >= 4
    assert 0 <= brief.confidence <= 100
    assert brief.sources


def test_brief_is_deterministic_same_day():
    b1 = build_brief(full_dossier())
    b2 = build_brief(full_dossier())
    assert b1.headline == b2.headline
    assert b1.paragraphs == b2.paragraphs


def test_sparse_dossier_never_blank_or_none_text():
    brief = build_brief(Dossier(symbol="THIN", price=12.0))
    md = render_markdown(brief)
    assert md.strip()
    assert "None" not in md and "nan" not in md.lower().replace("finance", "")


def test_brief_numbers_are_grounded():
    """Every $-figure in the trade-plan text must come from the dossier (no hallucinated levels)."""
    d = full_dossier()
    md = render_markdown(build_brief(d))
    dossier_numbers = {f"{v:,.2f}" for v in
                       [d.price, d.entry, d.stop, d.target, d.sma20, d.sma50, d.vwap, d.atr,
                        d.forecast["p10"], d.forecast["p50"], d.forecast["p90"]] if v is not None}
    for m in re.finditer(r"\$([\d,]+\.\d{2})", md):
        assert m.group(1) in dossier_numbers, f"${m.group(1)} not found in dossier"


def test_conflicting_signals_are_surfaced():
    d = full_dossier()
    d.whale = {"signal": "Distribution", "whale_score": 70.0, "rvol": 3.0}  # against the long read
    md = render_markdown(build_brief(d)).lower()
    assert "distribution" in md


def test_market_brief_renders():
    mb = market_brief(0.62, "Long",
                      {"verdict": "Trade", "score": 70, "drivers": ["breadth improving"]},
                      vix=15.0, breadth_pct=0.63, next_event=("FOMC Decision", 3))
    md = render_markdown(mb)
    assert md.strip()
    assert mb.headline
