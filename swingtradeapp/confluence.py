"""Signal Stack — cross-screen conviction scoring (pure logic, no Streamlit).

Each screen in the app reads a stock through one lens. This module turns those reads into
directional **votes** and combines them into a single conviction score, so the user can see where
*independent* signals agree. ``app.py`` gathers the raw scan outputs (mostly from cached functions)
and feeds primitive values to the ``vote_*`` builders; the Streamlit page owns rendering.

A vote is ``{dir: +1 bullish / 0 neutral / -1 bearish, strength: 0–1, detail}``. Absent signals
(``None``) don't count toward coverage. The composite is a weighted average of dir·strength; the
**confluence** is how many present signals agree with the net direction.

Note: the signals aren't fully independent (tech & forecast both lean on price; news & social are
both sentiment), so treat confluence as suggestive, not statistical proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .num import clip01 as _clip01

# Display/iteration order and relative trust of each signal.
SIGNAL_ORDER = ["tech", "whale", "forecast", "options", "news", "social", "sector"]
N_SIGNALS = len(SIGNAL_ORDER)  # coverage denominator (total signals the app can provide)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "tech": 1.0,      # the core technical read
    "whale": 0.8,     # smart-money volume footprint
    "forecast": 0.7,  # model's next-session direction
    "options": 0.7,   # derivatives positioning
    "news": 0.5,      # headline sentiment
    "social": 0.4,    # YouTube finfluencer buzz
    "sector": 0.4,    # macro/sector tailwind
}


@dataclass
class Vote:
    dir: int            # +1 bullish, 0 neutral, -1 bearish
    strength: float     # 0..1 confidence/magnitude
    detail: str = ""


# ── Per-signal vote builders (pure; primitive inputs) ────────────────────────────

def vote_tech(recommendation: Optional[str], score: Optional[float]) -> Optional[Vote]:
    """From the Screener recommendation label + score. 'Avoid (bearish)' → short."""
    if not recommendation or recommendation == "—" or score is None:
        return None
    s = _clip01(score)
    if recommendation.startswith("Avoid"):
        return Vote(-1, max(s, 0.4), f"{recommendation} · {s:.2f}")
    if recommendation in ("Buy", "Watch", "Weak"):
        return Vote(1, s, f"{recommendation} · {s:.2f}")
    return Vote(0, s, f"{recommendation} · {s:.2f}")


def vote_news(sentiment_pct: Optional[float]) -> Optional[Vote]:
    """From the % of positive headlines. Near-zero usually means *no* news → absent."""
    if sentiment_pct is None:
        return None
    p = float(sentiment_pct)
    if p <= 5:
        return None  # effectively no news, not a bearish read
    if p >= 55:
        return Vote(1, _clip01((p - 50) / 50), f"{p:.0f}% positive")
    if p <= 45:
        return Vote(-1, _clip01((50 - p) / 50), f"{p:.0f}% positive")
    return Vote(0, 0.2, f"{p:.0f}% positive")


_WHALE_BULL = {"Heavy Buying", "Accumulation"}
_WHALE_BEAR = {"Heavy Selling", "Distribution"}


def vote_whale(signal: Optional[str], whale_score: Optional[float]) -> Optional[Vote]:
    if not signal or whale_score is None:
        return None
    s = _clip01(float(whale_score) / 100.0)
    if signal in _WHALE_BULL:
        return Vote(1, s, f"{signal} ({whale_score:.0f})")
    if signal in _WHALE_BEAR:
        return Vote(-1, s, f"{signal} ({whale_score:.0f})")
    return Vote(0, s, f"{signal} ({whale_score:.0f})")  # Churn


def vote_options(sentiment: Optional[str], n_unusual: Optional[int]) -> Optional[Vote]:
    if not sentiment or sentiment == "—":
        return None
    low = str(sentiment).lower()
    s = _clip01(0.4 + float(n_unusual or 0) / 10.0)
    if "bull" in low:
        return Vote(1, s, f"{sentiment}, {n_unusual or 0} unusual")
    if "bear" in low:
        return Vote(-1, s, f"{sentiment}, {n_unusual or 0} unusual")
    return Vote(0, 0.3, f"{sentiment}")


def vote_forecast(direction: Optional[str], ret_pct: Optional[float]) -> Optional[Vote]:
    if not direction or direction in ("—", "Flat"):
        return None
    r = abs(float(ret_pct or 0.0))
    s = _clip01(r / 3.0)  # a ~3% predicted move = full strength
    if direction == "Up":
        return Vote(1, s, f"+{float(ret_pct or 0):.2f}% next session")
    if direction == "Down":
        return Vote(-1, s, f"{float(ret_pct or 0):.2f}% next session")
    return None


def vote_social(bull_pct: Optional[float], mentions: Optional[int]) -> Optional[Vote]:
    if not mentions:
        return None
    if bull_pct is None:
        return Vote(0, 0.2, f"{mentions} mention(s), mixed")
    b = float(bull_pct)
    s = _clip01(abs(b - 0.5) * 2)
    if b >= 0.6:
        return Vote(1, s, f"{b:.0%} bull · {mentions} mention(s)")
    if b <= 0.4:
        return Vote(-1, s, f"{b:.0%} bull · {mentions} mention(s)")
    return Vote(0, 0.2, f"{b:.0%} bull · {mentions} mention(s)")


def vote_sector(change_pct: Optional[float]) -> Optional[Vote]:
    if change_pct is None:
        return None
    c = float(change_pct)
    s = _clip01(abs(c) / 2.0)  # a 2% sector-ETF move = full strength
    if c >= 0.2:
        return Vote(1, s, f"sector {c:+.2f}%")
    if c <= -0.2:
        return Vote(-1, s, f"sector {c:+.2f}%")
    return Vote(0, 0.2, f"sector {c:+.2f}%")


# ── Aggregation ──────────────────────────────────────────────────────────────────

def score_ticker(votes: Dict[str, Optional[Vote]],
                 weights: Optional[Dict[str, float]] = None) -> Dict:
    """Combine per-signal votes into a coverage-weighted conviction and confluence count.

    ``net = Σ w·dir·strength / Σ w(present)`` ∈ [-1, 1] sets the **direction** and raw strength.
    Conviction also scales with **breadth** so broad agreement beats one strong signal:
    ``conviction = round(100·|net|·√(present_n / N_SIGNALS))``. ``confluence`` is how many present
    signals agree with the net sign over the number present (e.g. ``"5/7"``); ``coverage`` is how
    many of the app's signals had a read (e.g. ``"3/7"``).
    """
    weights = weights or DEFAULT_WEIGHTS
    present = {k: v for k, v in votes.items() if v is not None}
    if not present:
        return {"conviction": 0, "direction": "neutral", "net": 0.0, "present_n": 0,
                "agree_n": 0, "confluence": "0/0", "coverage": f"0/{N_SIGNALS}"}

    wsum = sum(weights.get(k, 0.5) for k in present)
    net = (sum(weights.get(k, 0.5) * v.dir * v.strength for k, v in present.items()) / wsum
           if wsum else 0.0)
    sign = 1 if net > 0 else (-1 if net < 0 else 0)
    direction = "long" if net > 0.02 else ("short" if net < -0.02 else "neutral")
    agree_n = sum(1 for v in present.values() if v.dir == sign and v.dir != 0) if sign else 0
    present_n = len(present)
    coverage_factor = (present_n / N_SIGNALS) ** 0.5  # sub-linear breadth discount
    return {
        "conviction": int(round(100 * abs(net) * coverage_factor)),
        "direction": direction,
        "net": round(net, 4),
        "present_n": present_n,
        "agree_n": agree_n,
        "confluence": f"{agree_n}/{present_n}",
        "coverage": f"{present_n}/{N_SIGNALS}",
    }
