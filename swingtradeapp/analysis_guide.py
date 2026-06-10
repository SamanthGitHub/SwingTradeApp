"""How to Analyze — the teaching curriculum + per-step grading (pure logic, no Streamlit).

Two halves:
  * ``STEPS`` — the framework guide as data (what / why / how / which screen) for each of the seven
    analysis steps. The page renders these as the "method".
  * ``grade_*`` functions — turn a ticker's live numbers into a ✅/⚠️/❌ read with a one-line note,
    using documented rules of thumb. ``summarize`` rolls the marks into an overall verdict.

Thresholds are deliberately simple, educational defaults — not financial advice. ``app.py`` fetches
the live values (reusing the existing signal/fundamentals/sentiment engines) and feeds primitives
here; the Streamlit page owns rendering.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, Optional, Tuple

# mark → emoji used by the page
MARKS = {"pass": "✅", "warn": "⚠️", "fail": "❌", "na": "➖"}

Grade = namedtuple("Grade", ["mark", "note"])

# ── The method (framework guide) ─────────────────────────────────────────────────
STEPS = [
    {
        "key": "trend", "title": "1 · Trend & structure",
        "what": "Which way is the stock actually moving? Compare price to its 20- and 50-day moving "
                "averages and look for higher highs / higher lows.",
        "why": "Trading with the trend stacks the odds in your favor — most swing gains come from "
               "continuation, not calling tops and bottoms.",
        "how": "Bullish when price > SMA20 > SMA50 (averages stacked and rising). Below both = "
               "downtrend. Tangled = no trend; wait.",
        "screen": "Screener",
    },
    {
        "key": "momentum", "title": "2 · Momentum",
        "what": "How strong and stretched is the move? RSI (0–100) and the MACD histogram measure it.",
        "why": "Momentum confirms a trend has fuel — but extremes warn that a pullback is near.",
        "how": "RSI 40–65 with a positive MACD histogram is healthy. RSI > 70 is overbought "
               "(pullback risk); < 30 is oversold.",
        "screen": "Screener",
    },
    {
        "key": "volume", "title": "3 · Volume & smart money",
        "what": "Is real participation behind the move? Compare today's volume to its 20-day average "
                "and check for institutional 'whale' footprints.",
        "why": "Price moves on heavy volume are far more trustworthy than the same move on thin "
               "volume, which tends to fade.",
        "how": "A volume surge ≥ 1.5× the average plus an accumulation footprint confirms; "
               "distribution on heavy volume is a red flag.",
        "screen": "Whale Movements",
    },
    {
        "key": "rr", "title": "4 · Levels & risk:reward",
        "what": "Define your entry, a stop (where you're wrong), and a target — then the reward-to-"
                "risk ratio.",
        "why": "Even a 50% win rate is profitable if winners are 2×+ your losers. Risk is the one "
               "thing you control.",
        "how": "Stops are ATR-based (volatility-aware). Want R:R ≥ 2:1. Between 1–2 is thin; below 1, "
               "skip it.",
        "screen": "Screener",
    },
    {
        "key": "fundamentals", "title": "5 · Fundamentals",
        "what": "Is the underlying business sound? Valuation (P/E), profitability (margin, ROE) and "
                "leverage (debt/equity).",
        "why": "Even for a swing trade, weak fundamentals add downside risk on any bad headline; "
               "quality companies recover faster.",
        "how": "Reasonable P/E (< ~30), positive margins, ROE > 10%, manageable debt. Rich valuation "
               "or losses = handle with care.",
        "screen": "Screener",
    },
    {
        "key": "sentiment", "title": "6 · Sentiment & catalysts",
        "what": "What's the news tone, and is a catalyst (earnings) imminent?",
        "why": "Sentiment moves price short term; an earnings date inside your horizon is a binary "
               "event that can blow through your stop.",
        "how": "Positive headline balance helps; mixed/negative is caution. Earnings within ~7 days "
               "= event risk — size down or wait.",
        "screen": "Market Events",
    },
    {
        "key": "risk", "title": "7 · Position sizing & risk management",
        "what": "How much to buy. Size from your edge (half-Kelly), cap risk per trade, and respect "
                "portfolio heat.",
        "why": "Position sizing — not stock picking — is what keeps you in the game. The best setup "
               "still fails sometimes.",
        "how": "Risk ≤ 1–2% of the account per trade, size via half-Kelly, honor the stop, and don't "
               "let total open risk pile up.",
        "screen": "Settings",
    },
]


# ── Per-step grading (pure; primitive inputs) ────────────────────────────────────

def grade_trend(price: Optional[float], sma20: Optional[float], sma50: Optional[float]) -> Grade:
    if price is None or sma20 is None or sma50 is None:
        return Grade("na", "Not enough history for moving averages.")
    if price > sma20 > sma50:
        return Grade("pass", "Uptrend — price above both rising averages.")
    if price < sma20 < sma50:
        return Grade("fail", "Downtrend — price below both averages.")
    return Grade("warn", "Mixed — averages not aligned; no clean trend yet.")


def grade_momentum(rsi: Optional[float], macd_hist: Optional[float]) -> Grade:
    if rsi is None:
        return Grade("na", "No momentum reading.")
    if rsi > 70:
        return Grade("warn", f"Overbought (RSI {rsi:.0f}) — pullback risk.")
    if rsi < 30:
        return Grade("warn", f"Oversold (RSI {rsi:.0f}) — bounce possible, trend weak.")
    if macd_hist is not None and macd_hist < 0:
        return Grade("fail", f"MACD histogram negative — momentum fading (RSI {rsi:.0f}).")
    if 40 <= rsi <= 68 and (macd_hist or 0) >= 0:
        return Grade("pass", f"Healthy momentum (RSI {rsi:.0f}), MACD positive.")
    return Grade("warn", f"Neutral momentum (RSI {rsi:.0f}).")


def grade_volume(vol_surge: Optional[float], whale_signal: Optional[str]) -> Grade:
    bull = whale_signal in ("Heavy Buying", "Accumulation")
    bear = whale_signal in ("Heavy Selling", "Distribution")
    if vol_surge is None:
        return Grade("na", "No volume data.")
    if bear:
        return Grade("fail", f"Distribution on {vol_surge:.1f}× volume — sellers in control.")
    if vol_surge >= 1.5 and bull:
        return Grade("pass", f"{vol_surge:.1f}× volume with an accumulation footprint.")
    if vol_surge >= 1.5:
        return Grade("pass", f"{vol_surge:.1f}× average volume — participation confirms the move.")
    return Grade("warn", f"Light volume ({vol_surge:.1f}×) — weak conviction.")


def grade_rr(rr: Optional[float]) -> Grade:
    if rr is None:
        return Grade("na", "No levels computed.")
    if rr >= 2:
        return Grade("pass", f"R:R {rr:.2f}× — reward ≥ 2× risk.")
    if rr >= 1:
        return Grade("warn", f"R:R {rr:.2f}× — acceptable but thin.")
    return Grade("fail", f"R:R {rr:.2f}× — risk outweighs reward; skip.")


def _norm_de(debt_to_equity: Optional[float]) -> Optional[float]:
    """yfinance reports debtToEquity as a percentage (e.g. 150 = 1.5×). Normalize to a ratio."""
    if debt_to_equity is None:
        return None
    return debt_to_equity / 100.0 if debt_to_equity > 5 else debt_to_equity


def grade_fundamentals(pe: Optional[float], profit_margin: Optional[float],
                       roe: Optional[float], debt_to_equity: Optional[float]) -> Grade:
    notes, score, n = [], 0, 0
    if pe is not None:
        n += 1
        if pe < 30:
            score += 1; notes.append(f"P/E {pe:.0f} (reasonable)")
        elif pe <= 50:
            notes.append(f"P/E {pe:.0f} (elevated)")
        else:
            notes.append(f"P/E {pe:.0f} (rich)")
    if profit_margin is not None:
        n += 1
        if profit_margin > 0:
            score += 1; notes.append(f"margin {profit_margin * 100:.0f}%")
        else:
            notes.append("unprofitable")
    if roe is not None:
        n += 1
        if roe > 0.10:
            score += 1; notes.append(f"ROE {roe * 100:.0f}%")
        else:
            notes.append(f"ROE {roe * 100:.0f}%")
    de = _norm_de(debt_to_equity)
    if de is not None:
        n += 1
        if de < 1.0:
            score += 1; notes.append(f"D/E {de:.1f}")
        else:
            notes.append(f"D/E {de:.1f} (levered)")
    if n == 0:
        return Grade("na", "No fundamental data available.")
    frac = score / n
    note = "; ".join(notes)
    if frac >= 0.7:
        return Grade("pass", note)
    if frac >= 0.4:
        return Grade("warn", note)
    return Grade("fail", note)


def grade_sentiment(positive_pct: Optional[float], days_to_earnings: Optional[int]) -> Grade:
    if positive_pct is None:
        return Grade("na", "No recent headlines.")
    earnings_soon = days_to_earnings is not None and 0 <= days_to_earnings <= 7
    if positive_pct >= 55:
        base = Grade("pass", f"{positive_pct:.0f}% positive headlines.")
    elif positive_pct <= 45:
        base = Grade("fail", f"{positive_pct:.0f}% positive — negative tone.")
    else:
        base = Grade("warn", f"{positive_pct:.0f}% positive — mixed tone.")
    if earnings_soon:
        mark = "warn" if base.mark == "pass" else base.mark
        return Grade(mark, base.note + f" · earnings in {days_to_earnings}d — event risk.")
    return base


def grade_risk(kelly_fraction: Optional[float]) -> Grade:
    if kelly_fraction is None:
        return Grade("na", "No size computed.")
    if kelly_fraction > 0:
        return Grade("pass", f"Half-Kelly sizes this at {kelly_fraction * 100:.1f}% of the account "
                             f"— keep risk per trade ≤ 1–2%.")
    return Grade("warn", "No measurable edge — Kelly suggests sitting out or sizing tiny.")


# ── Verdict ──────────────────────────────────────────────────────────────────────

_TECH_KEYS = ("trend", "momentum", "volume", "rr")


def summarize(marks: Dict[str, str]) -> Tuple[int, int, str]:
    """(n_pass, n_graded, verdict) from a {step_key: mark} dict. 'na' steps don't count."""
    graded = {k: m for k, m in marks.items() if m != "na"}
    n_graded = len(graded)
    n_pass = sum(1 for m in graded.values() if m == "pass")

    tech = [graded[k] for k in _TECH_KEYS if k in graded]
    tech_pass = sum(1 for m in tech if m == "pass")
    parts = []
    if tech and tech_pass >= max(1, len(tech) - 1):
        parts.append("strong technical setup")
    elif tech and any(m == "fail" for m in tech):
        parts.append("weak technicals")
    else:
        parts.append("mixed technicals")
    if graded.get("fundamentals") in ("warn", "fail"):
        parts.append("watch the fundamentals")
    if graded.get("sentiment") == "fail":
        parts.append("negative sentiment")
    if graded.get("rr") == "fail":
        parts.append("poor risk:reward")
    verdict = ", ".join(parts)
    return n_pass, n_graded, verdict
