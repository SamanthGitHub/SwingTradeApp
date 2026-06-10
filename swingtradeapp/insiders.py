"""Insider activity — real SEC Form 4 transactions (pure logic, no Streamlit).

Corporate insiders (officers, directors, 10%+ owners) must file Form 4 when they trade their own
stock; yfinance exposes that feed for **free** (``Ticker.insider_transactions`` /
``Ticker.insider_purchases``). Quirk: the feed's ``Transaction`` column is empty, so buy/sell is
parsed from the human-readable ``Text`` ("Sale at price …" / "Purchase at price …").

``app.py`` owns the (cached) yfinance fetch; this module just classifies, tidies and summarizes —
so it stays testable (mirrors ``whale.py`` / ``confluence.py``). Cluster buying (several *different*
insiders buying in a short window) is a well-known bullish tell and is surfaced here.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

_BUY_WORDS = ("purchase", "bought", "acquire", "buy")
_SELL_WORDS = ("sale", "sold", "sell", "disposition", "disposed")

TIDY_COLS = ["Date", "Insider", "Position", "Action", "Shares", "Value", "Note"]


def classify(text) -> str:
    """Buy / Sell / Other from the Form-4 text. Gifts, awards, option exercises → Other."""
    t = str(text or "").lower()
    if any(w in t for w in _BUY_WORDS):
        return "Buy"
    if any(w in t for w in _SELL_WORDS):
        return "Sell"
    return "Other"


def tidy_transactions(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize yfinance ``insider_transactions`` → Date/Insider/Position/Action/Shares/Value/Note,
    newest first. Returns an empty (typed) frame when there's nothing."""
    if raw is None or getattr(raw, "empty", True):
        return pd.DataFrame(columns=TIDY_COLS)
    df = raw
    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df.get("Start Date"), errors="coerce")
    out["Insider"] = df.get("Insider").astype("string") if "Insider" in df else pd.NA
    out["Position"] = df.get("Position").astype("string") if "Position" in df else pd.NA
    out["Action"] = df.get("Text").map(classify) if "Text" in df else "Other"
    out["Shares"] = pd.to_numeric(df.get("Shares"), errors="coerce")
    out["Value"] = pd.to_numeric(df.get("Value"), errors="coerce")
    out["Note"] = df.get("Text").astype("string") if "Text" in df else pd.NA
    return out.sort_values("Date", ascending=False, na_position="last").reset_index(drop=True)


def summarize(tidy: pd.DataFrame, days: int = 180) -> Dict:
    """Net insider sentiment over the last ``days``: buy/sell counts, $ values, a -100..100 score
    (+100 = all buying, -100 = all selling), a label, and the count of *distinct* recent buyers
    (a cluster-buy ≥ 2 is bullish)."""
    base = {"buys": 0, "sells": 0, "buy_value": 0.0, "sell_value": 0.0, "net_value": 0.0,
            "score": 0, "label": "No data", "cluster_buyers": 0, "window_days": days}
    if tidy is None or tidy.empty:
        return base
    if tidy["Date"].notna().any():
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        recent = tidy[tidy["Date"] >= cutoff]
    else:
        recent = tidy
    buys = recent[recent["Action"] == "Buy"]
    sells = recent[recent["Action"] == "Sell"]
    bv = float(buys["Value"].fillna(0).sum())
    sv = float(sells["Value"].fillna(0).sum())
    total = bv + sv
    score = int(round((bv - sv) / total * 100)) if total > 0 else 0
    if score >= 40:
        label = "Heavy buying"
    elif score > 0:
        label = "Net buying"
    elif score <= -40:
        label = "Heavy selling"
    elif score < 0:
        label = "Net selling"
    else:
        label = "Neutral"
    return {"buys": int(len(buys)), "sells": int(len(sells)),
            "buy_value": bv, "sell_value": sv, "net_value": bv - sv,
            "score": score, "label": label,
            "cluster_buyers": int(buys["Insider"].nunique()), "window_days": days}
