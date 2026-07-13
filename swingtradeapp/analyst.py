"""Analyst briefs — plain-English trade theses composed from the app's own signals.

Pure template-NLG (no network, no models, no Streamlit): every sentence is generated from
structured values the app already computed — the trend signal, named setups, confluence
votes, regime read, whale footprint, forecast, news sentiment and fundamentals — so a
brief can never claim a number that isn't in its :class:`Dossier`. This is deliberate:
a deterministic composer can't hallucinate, and disagreements between signals become
explicit "watch out" sentences instead of being averaged away.

Wording is seeded per ``(symbol, day)`` so a brief is stable across reruns within a day
but phrasing varies across names. ``app.py`` assembles the dossier (it owns fetching and
caching); an optional local-LLM pass (see ``llm_local``) may *rephrase* the finished
brief but never adds facts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from . import analysis_guide as ag

DISCLAIMER = "Generated locally from the app's own signals — not financial advice."


@dataclass
class Dossier:
    """Everything known about one ticker, gathered from the app's engines. All optional
    except the symbol — the brief says less when it knows less, never guesses."""
    symbol: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    name: str = ""
    sector: str = ""
    # Trend signal (signals.TrendSignalGenerator output)
    signal_type: Optional[str] = None          # "long" | "short"
    score: Optional[float] = None              # 0–1
    rsi: Optional[float] = None
    adx: Optional[float] = None
    macd_hist: Optional[float] = None
    vol_surge: Optional[float] = None
    atr: Optional[float] = None
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    vwap: Optional[float] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    # Named setups (setups.detect_all hits): [{"name", "score", "reasons": [...]}]
    setups: List[Dict] = field(default_factory=list)
    # Confluence (confluence.score_ticker result): {"direction", "conviction", "confluence", "coverage"}
    confluence: Optional[Dict] = None
    # Regime (regime.RegimeRead fields): {"verdict", "score", "drivers": [...]}
    regime: Optional[Dict] = None
    # Whale row: {"signal", "whale_score", "rvol"}
    whale: Optional[Dict] = None
    # Forecast (forecast.PriceForecaster dict): {"p10","p50","p90","last_price","source"} (horizon 1)
    forecast: Optional[Dict] = None
    # News: {"positive_pct" (0–1), "count", "top_headline", "events": [...]}
    news: Optional[Dict] = None
    # Fundamentals: {"pe_ratio","profit_margin","roe","debt_to_equity","market_cap"}
    fundamentals: Optional[Dict] = None
    earnings_days: Optional[int] = None        # days to next earnings, if known
    ml_prob: Optional[float] = None            # calibrated P(up), 0–1


@dataclass
class Brief:
    symbol: str
    headline: str
    paragraphs: List[Tuple[str, str]]          # (section title, text)
    confidence: int                            # 0–100
    sources: List[str]                         # which engines contributed


def _fmt(x: Optional[float], nd: int = 2) -> str:
    return f"{x:,.{nd}f}" if x is not None else "n/a"


def _rr(d: Dossier) -> Optional[float]:
    if not (d.entry and d.stop and d.target):
        return None
    risk = abs(d.entry - d.stop)
    reward = abs(d.target - d.entry)
    return round(reward / risk, 2) if risk > 0 else None


def _pick(rng: random.Random, options: List[str]) -> str:
    return options[rng.randrange(len(options))]


# ── Section builders ─────────────────────────────────────────────────────────────

def _trend_paragraph(d: Dossier, rng: random.Random) -> Optional[str]:
    if d.price is None:
        return None
    bits: List[str] = []
    tg = ag.grade_trend(d.price, d.sma20, d.sma50)
    if tg.mark == "pass":
        bits.append(_pick(rng, [
            f"{d.symbol} is in a clean uptrend — price (${_fmt(d.price)}) sits above both its "
            f"20-day (${_fmt(d.sma20)}) and 50-day (${_fmt(d.sma50)}) averages",
            f"The trend is working for the bulls: at ${_fmt(d.price)}, {d.symbol} trades above a "
            f"rising 20-day average (${_fmt(d.sma20)}) which in turn sits above the 50-day (${_fmt(d.sma50)})",
        ]))
    elif tg.mark == "fail":
        bits.append(_pick(rng, [
            f"{d.symbol} is in a downtrend — price (${_fmt(d.price)}) is below both the 20-day "
            f"(${_fmt(d.sma20)}) and 50-day (${_fmt(d.sma50)}) averages",
            f"Structure favors the bears: at ${_fmt(d.price)}, {d.symbol} sits under both key moving averages",
        ]))
    elif d.sma20 is not None:
        bits.append(f"The moving averages are tangled (price ${_fmt(d.price)}, 20-day ${_fmt(d.sma20)}, "
                    f"50-day ${_fmt(d.sma50)}) — no clean trend yet")

    if d.rsi is not None:
        if d.rsi >= 70:
            bits.append(f"momentum is stretched (RSI {d.rsi:.0f} — overbought territory, pullback risk)")
        elif d.rsi <= 30:
            bits.append(f"momentum is washed out (RSI {d.rsi:.0f} — oversold; bounces happen here, "
                        "but so do further legs down)")
        elif 40 <= d.rsi <= 65:
            bits.append(f"momentum is healthy (RSI {d.rsi:.0f})")
        else:
            bits.append(f"momentum is neutral (RSI {d.rsi:.0f})")
    if d.macd_hist is not None:
        bits.append("the MACD histogram is " + ("positive — buyers still pressing" if d.macd_hist > 0
                                                else "negative — momentum fading"))
    if d.adx is not None:
        if d.adx >= 25:
            bits.append(f"and trend strength is real (ADX {d.adx:.0f})")
        elif d.adx < 20:
            bits.append(f"but trend strength is weak (ADX {d.adx:.0f} — choppy tape, signals less reliable)")
    return ". ".join(s[0].upper() + s[1:] for s in bits) + "." if bits else None


def _volume_paragraph(d: Dossier, rng: random.Random) -> Optional[str]:
    bits: List[str] = []
    if d.vol_surge is not None:
        if d.vol_surge >= 1.8:
            bits.append(_pick(rng, [
                f"Participation is loud — volume is running {d.vol_surge:.1f}× its 20-day average",
                f"Volume confirms: {d.vol_surge:.1f}× the recent average is real participation",
            ]))
        elif d.vol_surge >= 1.3:
            bits.append(f"Volume is modestly elevated at {d.vol_surge:.1f}× average")
        else:
            bits.append(f"Volume is unremarkable ({d.vol_surge:.1f}× average) — conviction behind the "
                        "move is limited")
    w = d.whale or {}
    wsig, wscore = w.get("signal"), w.get("whale_score")
    if wsig in ("Heavy Buying", "Accumulation"):
        bits.append(f"the tape shows an institutional **{wsig.lower()}** footprint"
                    + (f" (whale score {wscore:.0f}/100)" if wscore is not None else ""))
    elif wsig in ("Heavy Selling", "Distribution"):
        bits.append(f"caution: the tape shows **{wsig.lower()}** — large sellers at work"
                    + (f" (whale score {wscore:.0f}/100)" if wscore is not None else ""))
    return ". ".join(s[0].upper() + s[1:] for s in bits) + "." if bits else None


def _plan_paragraph(d: Dossier) -> Optional[str]:
    if not (d.entry and d.stop and d.target and d.signal_type):
        return None
    rr = _rr(d)
    side = "long" if d.signal_type == "long" else "short"
    verb = "pullback toward" if side == "long" else "bounce toward"
    txt = (f"The plan is a {side}: work a limit on a {verb} **${_fmt(d.entry)}**, "
           f"stop at **${_fmt(d.stop)}** (the trade is wrong below it)" if side == "long" else
           f"The plan is a {side}: work a limit on a {verb} **${_fmt(d.entry)}**, "
           f"stop at **${_fmt(d.stop)}** (the trade is wrong above it)")
    txt += f", target **${_fmt(d.target)}**"
    if rr is not None:
        if rr >= 2:
            txt += f" — a {rr:.1f}:1 reward-to-risk, comfortably worth taking"
        elif rr >= 1:
            txt += f" — a thin {rr:.1f}:1 reward-to-risk; a better fill matters"
        else:
            txt += f" — only {rr:.1f}:1 reward-to-risk, which argues for skipping it"
    if d.atr:
        txt += f". Expect roughly ${_fmt(d.atr)} of daily range (ATR) while managing it"
    if d.setups:
        names = ", ".join(s.get("name", "?") for s in d.setups[:3])
        txt += f". Named setup{'s' if len(d.setups) > 1 else ''} in play: {names}"
    return txt + "."


def _catalyst_paragraph(d: Dossier) -> Optional[str]:
    n = d.news or {}
    bits: List[str] = []
    pos, cnt = n.get("positive_pct"), n.get("count", 0)
    if cnt and pos is not None:
        pct = pos * 100 if pos <= 1 else pos
        tone = "positive" if pct >= 55 else ("negative" if pct <= 45 else "mixed")
        bits.append(f"Headline tone is {tone} ({pct:.0f}% positive across {cnt} recent stories)")
    top = n.get("top_headline")
    if top:
        bits.append(f'most recent: "{top}"')
    events = [e for e in (n.get("events") or []) if e and e != "other"]
    if events:
        bits.append("flagged news events: " + ", ".join(sorted(set(events))[:4]))
    if d.earnings_days is not None and 0 <= d.earnings_days <= 14:
        bits.append(f"**earnings in {d.earnings_days} day{'s' if d.earnings_days != 1 else ''}** — a "
                    "binary event that can gap through any stop; size down or wait")
    f = d.forecast or {}
    if f.get("p50") is not None and f.get("last_price"):
        last, p50 = float(f["last_price"]), float(f["p50"] if not isinstance(f["p50"], (list, tuple))
                                                  else f["p50"][0])
        ret = (p50 - last) / last * 100.0
        bits.append(f"the {f.get('source', 'forecast')} model's next-session median is "
                    f"${_fmt(p50)} ({ret:+.1f}%)")
    return ". ".join(s[0].upper() + s[1:] for s in bits) + "." if bits else None


def _risk_paragraph(d: Dossier) -> Optional[str]:
    """Conflicts and invalidations — the paragraph a cheerleader would leave out."""
    bits: List[str] = []
    long_side = d.signal_type == "long"
    wsig = (d.whale or {}).get("signal")
    if long_side and wsig in ("Heavy Selling", "Distribution"):
        bits.append("the bullish technical read conflicts with a bearish smart-money footprint "
                    f"({wsig.lower()}) — one of them is wrong")
    if not long_side and d.signal_type and wsig in ("Heavy Buying", "Accumulation"):
        bits.append("the bearish read conflicts with institutional accumulation on the tape")
    if long_side and d.rsi is not None and d.rsi >= 70:
        bits.append("chasing an overbought name means a routine pullback can stop you out of a "
                    "correct thesis")
    reg = d.regime or {}
    if reg.get("verdict") == "Stand-aside":
        bits.append("the market regime gate reads **Stand-aside** — broad conditions argue against "
                    "new entries regardless of this chart")
    elif reg.get("verdict") == "Caution":
        bits.append("the regime read is **Caution** — take smaller size than usual")
    conf = d.confluence or {}
    if conf.get("direction") and d.signal_type and conf["direction"] not in (d.signal_type, "mixed"):
        bits.append(f"the cross-screen confluence points {conf['direction']} against this trade")
    rr = _rr(d)
    if rr is not None and rr < 1.5:
        bits.append(f"reward-to-risk is thin at {rr:.1f}:1")
    if d.stop is not None:
        bits.append(f"the thesis is invalidated at ${_fmt(d.stop)} — honor the stop")
    return ". ".join(s[0].upper() + s[1:] for s in bits) + "." if bits else None


def _bottom_line(d: Dossier, confidence: int) -> str:
    label = ("high-conviction" if confidence >= 70 else
             "reasonable" if confidence >= 50 else
             "low-conviction" if confidence >= 30 else "weak")
    side = d.signal_type or "no-trade"
    conf = d.confluence or {}
    agree = f" with {conf['confluence']} independent signals agreeing" if conf.get("confluence") else ""
    if side == "no-trade":
        return f"No actionable trade here right now — keep {d.symbol} on the watch list."
    txt = f"A {label} {side} idea{agree} (confidence {confidence}/100)."
    if d.ml_prob is not None:
        txt += f" The calibrated ML model puts P(up over the swing horizon) at {d.ml_prob * 100:.0f}%."
    return txt + " " + DISCLAIMER


def _confidence(d: Dossier) -> int:
    conf = 50.0
    if d.score is not None:
        conf = d.score * 100.0 * 0.8                      # technical score is the base
    c = d.confluence or {}
    if c.get("conviction"):
        conf = 0.7 * conf + 0.3 * float(c["conviction"])  # cross-screen agreement folds in
    reg = (d.regime or {}).get("verdict")
    if reg == "Stand-aside":
        conf -= 25
    elif reg == "Caution":
        conf -= 10
    wsig = (d.whale or {}).get("signal")
    if d.signal_type == "long" and wsig in ("Heavy Selling", "Distribution"):
        conf -= 15
    if d.signal_type == "long" and d.rsi is not None and d.rsi >= 70:
        conf -= 10
    if d.earnings_days is not None and 0 <= d.earnings_days <= 7:
        conf -= 10
    rr = _rr(d)
    if rr is not None and rr >= 2:
        conf += 5
    return int(max(5, min(95, round(conf))))


def build_brief(d: Dossier) -> Brief:
    """Compose the multi-paragraph analyst brief for one ticker."""
    rng = random.Random(f"{d.symbol}:{date.today().isoformat()}")
    confidence = _confidence(d)

    sections: List[Tuple[str, Optional[str]]] = [
        ("Trend & momentum", _trend_paragraph(d, rng)),
        ("Smart money & volume", _volume_paragraph(d, rng)),
        ("Trade plan", _plan_paragraph(d)),
        ("Catalysts & news", _catalyst_paragraph(d)),
        ("Risks & invalidation", _risk_paragraph(d)),
        ("Bottom line", _bottom_line(d, confidence)),
    ]
    paragraphs = [(t, p) for t, p in sections if p]

    sources = []
    if d.score is not None:
        sources.append("trend signal")
    if d.setups:
        sources.append("setup scanner")
    if d.confluence:
        sources.append("signal stack")
    if d.whale:
        sources.append("whale detector")
    if d.forecast:
        sources.append("price forecast")
    if d.news and d.news.get("count"):
        sources.append("news sentiment")
    if d.fundamentals:
        sources.append("fundamentals")
    if d.regime:
        sources.append("market regime")
    if d.ml_prob is not None:
        sources.append("ML signal model")

    side = (d.signal_type or "watch").upper()
    tone = "🟢" if (d.signal_type == "long" and confidence >= 50) else \
           ("🔴" if d.signal_type == "short" and confidence >= 50 else "🟡")
    headline = f"{tone} {d.symbol} — {side} · confidence {confidence}/100"
    return Brief(symbol=d.symbol, headline=headline, paragraphs=paragraphs,
                 confidence=confidence, sources=sources)


def market_brief(mood_score: Optional[float], bias: Optional[str], regime: Optional[Dict],
                 vix: Optional[float], breadth_pct: Optional[float],
                 next_event: Optional[Tuple[str, int]] = None) -> Brief:
    """The morning tape-read: one brief for the whole market instead of one ticker.

    ``next_event`` is ``(name, days_away)`` for the nearest scheduled macro event.
    """
    rng = random.Random(f"MARKET:{date.today().isoformat()}")
    bits: List[str] = []
    if mood_score is not None:
        read = ("risk-on" if mood_score >= 58 else "risk-off" if mood_score <= 42 else "undecided")
        bits.append(_pick(rng, [
            f"The tape reads {read} this morning (market mood {mood_score:.0f}/100)",
            f"Market mood sits at {mood_score:.0f}/100 — a {read} tape",
        ]))
    reg = regime or {}
    if reg.get("verdict"):
        drivers = "; ".join(reg.get("drivers", [])[:3])
        bits.append(f"the regime gate says **{reg['verdict']}**" + (f" ({drivers})" if drivers else ""))
    if vix is not None:
        vtxt = "stressed" if vix > 25 else ("complacent" if vix < 13 else "normal")
        bits.append(f"VIX at {vix:.1f} is {vtxt}")
    if breadth_pct is not None:
        b = breadth_pct * 100 if breadth_pct <= 1 else breadth_pct
        bits.append(f"breadth has {b:.0f}% of recent closes above trend")
    tape = ". ".join(s[0].upper() + s[1:] for s in bits) + "." if bits else "No market data available."

    plan_bits: List[str] = []
    if bias == "Long":
        plan_bits.append("Conditions favor working long setups at normal size")
    elif bias == "Defensive":
        plan_bits.append("Defensive day — smaller size, wider berth, fewer new entries; capital "
                         "preservation beats a forced trade")
    else:
        plan_bits.append("Mixed conditions — trade only A-grade setups and keep size modest")
    if next_event:
        name, days = next_event
        when = "today" if days == 0 else (f"in {days} day{'s' if days != 1 else ''}")
        plan_bits.append(f"nearest scheduled macro event: **{name}** {when} — expect volatility around it")
    plan = ". ".join(s[0].upper() + s[1:] for s in plan_bits) + "."

    conf = int(mood_score) if mood_score is not None else 50
    return Brief(symbol="MARKET",
                 headline=f"Morning read — bias: {bias or 'Neutral'}",
                 paragraphs=[("The tape", tape), ("The plan", plan)],
                 confidence=conf,
                 sources=[s for s, present in [("market mood", mood_score is not None),
                                               ("regime", bool(reg)), ("VIX", vix is not None),
                                               ("breadth", breadth_pct is not None)] if present])


def render_markdown(brief: Brief) -> str:
    lines = [f"**{brief.headline}**", ""]
    for title, text in brief.paragraphs:
        lines.append(f"**{title}.** {text}")
        lines.append("")
    return "\n".join(lines).rstrip()
