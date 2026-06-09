"""YouTube finfluencer scanner — what top traders are saying in the last 24–48h.

Pure logic, no Streamlit (mirrors ``whale.py``). Everything here is **free and key-less**:

* recent uploads via YouTube's per-channel **RSS feed** (no API key, no quota),
* full transcripts via the optional ``youtube-transcript-api`` package, with a graceful
  fallback to the video title + description when captions are missing/blocked,
* ticker / pick / sentiment analysis via heuristics + the app's existing **local** NLP
  stack (FinBERT-style sentiment, zero-shot events, distilbart digest) — **no paid LLM**.

The companion Streamlit page in ``app.py`` owns caching, persistence IO and rendering.

Note on transcripts: YouTube *auto*-captions are typically lowercase and unpunctuated, so the
``NAME_TO_TICKER`` company-name map carries most of the ticker detection for spoken content;
the cashtag / uppercase paths mainly help on manual captions, titles and descriptions.
"""

from __future__ import annotations

import logging
import re
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Curated starter set of finance/trading YouTubers, stored as @handles (human-editable).
# Handles are resolved to channel IDs lazily and cached, so a renamed/wrong handle is simply
# skipped rather than breaking the page. MANUALLY MAINTAINED (like ``IPOTracker.RECENT_IPOS``).
TRADER_CHANNELS: Dict[str, str] = {
    "MeetKevin": "Meet Kevin",
    "GrahamStephan": "Graham Stephan",
    "JosephCarlsonShow": "Joseph Carlson",
    "ClearValueTax": "ClearValue Tax",
    "ZipTrader": "ZipTrader",
    "TomNash": "Tom Nash",
    "AndreiJikh": "Andrei Jikh",
    "MinorityMindset": "Minority Mindset",
    "StockMoe": "Stock Moe",
    "FinancialEducation": "Financial Education",
}

_UA = {"User-Agent": "Mozilla/5.0 (SwingTradeApp YouTube scanner)"}


# ── HTTP plumbing (mirrors nlp._fetch_google_news) ───────────────────────────────

def _ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_get(url: str, timeout: int = 15) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


# ── Channel handle → channel_id resolution ───────────────────────────────────────

_CHANNEL_ID_RE = re.compile(r'"(?:channelId|externalId)":"(UC[0-9A-Za-z_-]{22})"')


def resolve_channel_id(handle: str) -> Optional[str]:
    """Resolve an @handle to its ``UC…`` channel ID by scraping the channel page.

    Returns ``None`` on any failure (network, renamed handle, parse miss). Callers should
    cache the result — a channel's ID never changes.
    """
    handle = handle.lstrip("@").strip()
    if not handle:
        return None
    html = _http_get(f"https://www.youtube.com/@{urllib.parse.quote(handle)}/videos")
    if not html:
        return None
    m = _CHANNEL_ID_RE.search(html)
    return m.group(1) if m else None


# ── Recent uploads via per-channel RSS ───────────────────────────────────────────

_YT_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


@dataclass
class Upload:
    video_id: str
    title: str
    url: str
    published_ts: float
    published_str: str
    description: str
    channel: str
    channel_id: str


def fetch_channel_uploads(channel_id: str, channel_name: str,
                          within_hours: float = 48.0) -> List[Upload]:
    """Recent uploads for one channel from its free RSS feed, filtered to ``within_hours``."""
    xml = _http_get(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}")
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
    except Exception:
        return []
    cutoff = datetime.now(timezone.utc).timestamp() - within_hours * 3600
    out: List[Upload] = []
    for entry in root.findall("a:entry", _YT_NS):
        vid = entry.findtext("yt:videoId", default="", namespaces=_YT_NS) or ""
        title = (entry.findtext("a:title", default="", namespaces=_YT_NS) or "").strip()
        if not vid or not title:
            continue
        ts, pub_str = 0.0, ""
        pub = entry.findtext("a:published", default="", namespaces=_YT_NS)
        if pub:
            try:
                dt = datetime.fromisoformat(pub)
                ts = dt.timestamp()
                pub_str = dt.astimezone().strftime("%b %d, %H:%M")
            except Exception:
                pass
        if ts and ts < cutoff:
            continue
        desc = ""
        group = entry.find("media:group", _YT_NS)
        if group is not None:
            desc = (group.findtext("media:description", default="", namespaces=_YT_NS) or "").strip()
        out.append(Upload(
            video_id=vid, title=title,
            url=f"https://www.youtube.com/watch?v={vid}",
            published_ts=ts, published_str=pub_str, description=desc,
            channel=channel_name, channel_id=channel_id,
        ))
    return out


# ── Transcripts (optional dep, graceful fallback) ────────────────────────────────

@dataclass
class Segment:
    text: str
    start: float


def fetch_transcript_segments(video_id: str) -> Optional[List[Segment]]:
    """Full transcript as timestamped segments via ``youtube-transcript-api``.

    Returns ``None`` on any failure — dependency not installed, no captions, or throttled —
    so the caller falls back to the title + description. Supports both the legacy classmethod
    API (≤0.6) and the instance API (≥1.0).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        return None
    raw = None
    try:  # ≥1.0 instance API
        fetched = YouTubeTranscriptApi().fetch(video_id)
        raw = [{"text": getattr(s, "text", ""), "start": getattr(s, "start", 0.0)} for s in fetched]
    except Exception:
        try:  # ≤0.6 classmethod API
            raw = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception:
            return None
    if not raw:
        return None
    try:
        return [Segment(text=str(r.get("text", "")), start=float(r.get("start", 0.0))) for r in raw]
    except Exception:
        return None


def segments_text(segments: List[Segment]) -> str:
    return " ".join(s.text for s in segments if s.text)


def deep_link(video_id: str, sec: float) -> str:
    """A watch URL that jumps to ``sec`` seconds — links a mention to the exact moment."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(sec)}s"


# ── Ticker detection ─────────────────────────────────────────────────────────────

# Spoken transcripts say company names, not symbols — this map carries most detection.
NAME_TO_TICKER: Dict[str, str] = {
    "nvidia": "NVDA", "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "advanced micro devices": "AMD",
    "palantir": "PLTR", "coinbase": "COIN", "micron": "MU", "broadcom": "AVGO",
    "intel": "INTC", "boeing": "BA", "disney": "DIS", "walmart": "WMT",
    "starbucks": "SBUX", "super micro": "SMCI", "supermicro": "SMCI",
    "rivian": "RIVN", "lucid": "LCID", "robinhood": "HOOD", "sofi": "SOFI",
    "paypal": "PYPL", "shopify": "SHOP", "salesforce": "CRM", "oracle": "ORCL",
    "qualcomm": "QCOM", "snowflake": "SNOW", "crowdstrike": "CRWD",
    "eli lilly": "LLY", "costco": "COST", "ford motor": "F", "general motors": "GM",
}

# All-caps words that are also valid tickers — ignored unless cash-tagged.
_TICKER_STOPWORDS: Set[str] = {
    "A", "I", "IT", "ON", "OR", "BY", "BE", "GO", "SO", "AT", "AM", "PM", "TV", "US",
    "ALL", "ARE", "FOR", "AND", "THE", "YOU", "CEO", "CFO", "IPO", "ETF", "USA", "ER",
    "USD", "GDP", "CPI", "SEC", "FED", "EPS", "ATH", "DD", "AI", "EV", "OK", "NO", "PE",
    "YES", "BUY", "NEW", "NOW", "ONE", "TWO", "WSB", "YOLO", "FOMO", "IMO", "EOD", "NYSE",
}

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_UPPER_RE = re.compile(r"\b([A-Z]{2,5})\b")


def extract_tickers(text: str, universe: Set[str]) -> Counter:
    """Detect tickers via cashtags, validated uppercase tokens, and company-name matches."""
    found: Counter = Counter()
    if not text:
        return found
    for m in _CASHTAG_RE.finditer(text):
        found[m.group(1).upper()] += 1
    for m in _UPPER_RE.finditer(text):
        sym = m.group(1)
        if sym not in _TICKER_STOPWORDS and sym in universe:
            found[sym] += 1
    low = text.lower()
    for name, sym in NAME_TO_TICKER.items():
        c = low.count(name)
        if c:
            found[sym] += c
    return found


@dataclass
class Mention:
    ticker: str
    snippet: str
    timestamp_sec: int
    deep_link: str


def find_mentions(video_id: str, segments: List[Segment], universe: Set[str],
                  max_per_ticker: int = 3) -> List[Mention]:
    """Timestamped mentions: walk segments and link each ticker hit to its moment."""
    mentions: List[Mention] = []
    counts: Counter = Counter()
    for seg in segments:
        for sym in extract_tickers(seg.text, universe):
            if counts[sym] >= max_per_ticker:
                continue
            counts[sym] += 1
            mentions.append(Mention(sym, seg.text.strip(), int(seg.start),
                                    deep_link(video_id, seg.start)))
    return mentions


# ── Pick extraction (direction + target + horizon) — heuristic, no paid LLM ──────

_DIRECTION = {
    "buy": ["buy", "buying", "bullish", "long ", "load up", "back up the truck",
            "accumulate", "adding", "scooping"],
    "sell": ["sell", "selling", "bearish", "short", "shorting", "dump", "trim",
             "avoid", "puts on", "take profit"],
    "hold": ["hold", "holding", "hodl", "wait on"],
}
_PT_CONTEXT_RE = re.compile(
    r"(?:price target|target of|sees|see it at|hits?|reach(?:es)?|"
    r"go(?:es)? to|going to|up to)\s+\$?\s?(\d{1,5}(?:\.\d{1,2})?)", re.I)
_HORIZON_RE = re.compile(
    r"(by (?:the )?end of (?:the )?(?:year|month|quarter)|year[- ]end|"
    r"next (?:week|month|quarter|year)|\d+\s*(?:day|week|month|year)s?|"
    r"long[- ]term|short[- ]term)", re.I)


@dataclass
class Pick:
    ticker: str
    direction: Optional[str]
    price_target: Optional[float]
    horizon: Optional[str]
    snippet: str
    timestamp_sec: int
    deep_link: str


def extract_picks(video_id: str, segments: List[Segment], universe: Set[str]) -> List[Pick]:
    """Best-effort calls: only segments naming exactly one ticker, with a direction or target.

    Heuristic and intentionally conservative — noise is acceptable because the track-record
    grading exposes wrong calls over time. No paid model involved.
    """
    picks: List[Pick] = []
    seen: Set[tuple] = set()
    for seg in segments:
        tickers = extract_tickers(seg.text, universe)
        if len(tickers) != 1:
            continue
        sym = next(iter(tickers))
        low = seg.text.lower()
        direction = next((d for d, words in _DIRECTION.items()
                          if any(w in low for w in words)), None)
        target = None
        mt = _PT_CONTEXT_RE.search(seg.text)
        if mt:
            try:
                target = float(mt.group(1))
            except ValueError:
                target = None
        horizon = None
        mh = _HORIZON_RE.search(seg.text)
        if mh:
            horizon = mh.group(1).strip()
        if direction is None and target is None:
            continue
        key = (sym, direction)
        if key in seen:
            continue
        seen.add(key)
        picks.append(Pick(sym, direction, target, horizon, seg.text.strip(),
                          int(seg.start), deep_link(video_id, seg.start)))
    return picks


# ── Pullback / merger flagging ───────────────────────────────────────────────────

PULLBACK_KEYWORDS = [
    "pullback", "pull back", "correction", "crash", "sell-off", "selloff", "overbought",
    "bubble", "plunge", "tank", "top is in", "bearish", "downturn", "recession",
    "capitulation", "overvalued", "rug pull",
]
MERGER_KEYWORDS = [
    "merger", "merge with", "acquire", "acquisition", "buyout", "take over", "takeover",
    "rumor", "rumour", "in talks", "deal to buy", "bid for", "acquiring",
]


def _snippets_for(text: str, keywords: List[str], width: int = 90, max_hits: int = 3) -> List[str]:
    low = text.lower()
    hits: List[str] = []
    for kw in keywords:
        idx = low.find(kw)
        if idx >= 0:
            start = max(0, idx - width)
            end = min(len(text), idx + len(kw) + width)
            snip = "…" + text[start:end].strip() + "…"
            if snip not in hits:  # overlapping keywords can yield the same window
                hits.append(snip)
            if len(hits) >= max_hits:
                break
    return hits


def scan_flags(text: str) -> Dict[str, List[str]]:
    """Context snippets around pullback/correction language and merger/rumor language."""
    return {
        "pullback": _snippets_for(text, PULLBACK_KEYWORDS),
        "merger": _snippets_for(text, MERGER_KEYWORDS),
    }


# ── Conviction + sentiment + digest ──────────────────────────────────────────────

_CONVICTION_WORDS = [
    "huge", "massive", "strongly", "favorite", "conviction", "all in", "back up the truck",
    "load up", "biggest position", "screaming buy", "no brainer", "table pounding",
    "aggressively", "pounding the table",
]
_BULL_WORDS = ["buy", "bullish", "long", "upside", "rally", "breakout", "moon", "undervalued",
               "strong", "accumulate"]
_BEAR_WORDS = ["sell", "bearish", "short", "downside", "crash", "pullback", "overvalued",
               "weak", "dump", "avoid"]


def score_conviction(text: str, mention_count: int) -> float:
    """0–1 blend of repetition (how often named) and intensity language."""
    low = text.lower()
    intensity = sum(low.count(w) for w in _CONVICTION_WORDS)
    raw = 0.5 * min(mention_count / 5.0, 1.0) + 0.5 * min(intensity / 4.0, 1.0)
    return round(raw, 3)


def _sentiment(text: str, analyzer) -> tuple:
    """(label, score) — local model across sampled chunks, keyword fallback otherwise."""
    if analyzer is not None:
        try:
            chunks = [text[i:i + 512] for i in range(0, min(len(text), 512 * 4), 512)] or [text[:512]]
            pos = neg = 0
            score_acc = 0.0
            for c in chunks:
                r = analyzer.analyze_text(c)
                lbl = str(r.get("label", "neutral")).lower()
                score_acc += float(r.get("score", 0.5))
                if "pos" in lbl:
                    pos += 1
                elif "neg" in lbl:
                    neg += 1
            if chunks and (pos or neg):
                return ("bullish" if pos >= neg else "bearish"), round(score_acc / len(chunks), 3)
        except Exception:
            pass
    low = text.lower()
    b = sum(low.count(w) for w in _BULL_WORDS)
    s = sum(low.count(w) for w in _BEAR_WORDS)
    if b == s:
        return "neutral", 0.5
    conf = min(0.5 + abs(b - s) / max(b + s, 1) * 0.5, 0.99)
    return ("bullish" if b > s else "bearish"), round(conf, 3)


def _digest(text: str, upload: "Upload", summarizer) -> str:
    if summarizer is not None and text:
        try:
            chunks = [text[i:i + 1000] for i in range(0, min(len(text), 3000), 1000)]
            parts = [summarizer.summarize([c]) for c in chunks if c.strip()]
            d = " ".join(p for p in parts if p).strip()
            if d:
                return d[:600]
        except Exception:
            pass
    base = upload.description or text
    sents = re.split(r"(?<=[.!?])\s+", base.strip())
    return " ".join(sents[:2])[:400]


@dataclass
class VideoAnalysis:
    upload: Upload
    has_transcript: bool
    text: str
    sentiment: str
    sentiment_score: float
    tickers: Counter
    mentions: List[Mention]
    picks: List[Pick]
    events: List[str]
    flags: Dict[str, List[str]]
    digest: str
    conviction: Dict[str, float] = field(default_factory=dict)


def analyze_video(upload: Upload, segments: Optional[List[Segment]], universe: Set[str], *,
                  analyzer=None, event_classifier=None, summarizer=None) -> VideoAnalysis:
    """Full per-video analysis. Falls back to title+description when there's no transcript."""
    has_tx = bool(segments)
    if segments:
        text = segments_text(segments)
        segs = segments
    else:
        text = f"{upload.title}. {upload.description}".strip()
        segs = [Segment(text=text, start=0.0)]

    mentions = find_mentions(upload.video_id, segs, universe)
    picks = extract_picks(upload.video_id, segs, universe)
    tickers = extract_tickers(text, universe)
    sentiment, sscore = _sentiment(text, analyzer)

    events: List[str] = []
    if event_classifier is not None:
        try:
            ev = event_classifier.classify((upload.title + ". " + text[:400]).strip())
            if ev and ev != "other":
                events.append(ev)
        except Exception:
            pass

    conviction = {sym: score_conviction(text, cnt) for sym, cnt in tickers.items()}
    return VideoAnalysis(
        upload=upload, has_transcript=has_tx, text=text,
        sentiment=sentiment, sentiment_score=sscore, tickers=tickers,
        mentions=mentions, picks=picks, events=events,
        flags=scan_flags(text), digest=_digest(text, upload, summarizer),
        conviction=conviction,
    )


# ── Cross-video consensus ────────────────────────────────────────────────────────

def ticker_consensus(analyses: List[VideoAnalysis]) -> List[Dict]:
    """Aggregate per ticker: mention count, #videos/#channels, bull%, peak conviction."""
    agg: Dict[str, Dict] = {}
    for a in analyses:
        for sym, cnt in a.tickers.items():
            d = agg.setdefault(sym, {"ticker": sym, "mentions": 0, "videos": 0,
                                     "bull": 0, "bear": 0, "channels": set(), "conviction": 0.0})
            d["mentions"] += cnt
            d["videos"] += 1
            d["channels"].add(a.upload.channel)
            d["conviction"] = max(d["conviction"], a.conviction.get(sym, 0.0))
            if a.sentiment == "bullish":
                d["bull"] += 1
            elif a.sentiment == "bearish":
                d["bear"] += 1
    rows: List[Dict] = []
    for d in agg.values():
        total = d["bull"] + d["bear"]
        rows.append({
            "ticker": d["ticker"], "mentions": d["mentions"], "videos": d["videos"],
            "channels": len(d["channels"]),
            "bull_pct": (d["bull"] / total) if total else None,
            "conviction": round(d["conviction"], 3),
        })
    rows.sort(key=lambda r: (r["videos"], r["mentions"]), reverse=True)
    return rows


# ── Track record / accountability (pure; app.py owns file IO) ─────────────────────

def make_pick_id(video_id: str, ticker: str, direction: Optional[str]) -> str:
    return f"{video_id}:{ticker}:{direction or 'na'}"


def record_picks(store: Dict, analysis: VideoAnalysis,
                 price_at_mention_fn: Callable[[str], Optional[float]],
                 today: Optional[str] = None) -> int:
    """Append any *new* picks from ``analysis`` to ``store['picks']`` with entry price.

    Idempotent per (video, ticker, direction) so re-scanning a video doesn't double-log.
    Returns the number of newly recorded picks.
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    picks = store.setdefault("picks", [])
    existing = {p["id"] for p in picks}
    added = 0
    for pk in analysis.picks:
        pid = make_pick_id(analysis.upload.video_id, pk.ticker, pk.direction)
        if pid in existing:
            continue
        picks.append({
            "id": pid, "channel": analysis.upload.channel,
            "channel_id": analysis.upload.channel_id, "video_id": analysis.upload.video_id,
            "video_title": analysis.upload.title, "video_url": analysis.upload.url,
            "ticker": pk.ticker, "direction": pk.direction,
            "price_target": pk.price_target, "horizon": pk.horizon,
            "snippet": pk.snippet, "timestamp_sec": pk.timestamp_sec, "deep_link": pk.deep_link,
            "mention_date": today, "price_at_mention": price_at_mention_fn(pk.ticker),
        })
        existing.add(pid)
        added += 1
    return added


def grade_picks(picks: List[Dict],
                current_price_fn: Callable[[str], Optional[float]],
                bench_return_fn: Callable[[str], Optional[float]]) -> List[Dict]:
    """Annotate each stored pick with return, directional return, and alpha vs benchmark.

    ``bench_return_fn(mention_date)`` returns the benchmark (SPY) fractional return since that
    date. A short/sell call profits from a decline, so its directional return is negated.
    """
    graded: List[Dict] = []
    for p in picks:
        rec = dict(p)
        entry = p.get("price_at_mention")
        cur = current_price_fn(p["ticker"]) if entry else None
        if entry and cur and entry > 0:
            raw = (cur - entry) / entry
            directional = -raw if p.get("direction") == "sell" else raw
            bench = bench_return_fn(p.get("mention_date", ""))
            rec["return_pct"] = round(raw * 100, 2)
            rec["directional_pct"] = round(directional * 100, 2)
            rec["alpha_pct"] = round((directional - bench) * 100, 2) if bench is not None else None
            rec["win"] = bool(directional > 0)
        else:
            rec["return_pct"] = rec["directional_pct"] = rec["alpha_pct"] = None
            rec["win"] = None
        graded.append(rec)
    return graded


def creator_leaderboard(graded: List[Dict]) -> List[Dict]:
    """Per-creator scorecard: #picks, win rate, average alpha — ranked by alpha."""
    by: Dict[str, Dict] = {}
    for p in graded:
        c = by.setdefault(p["channel"], {"channel": p["channel"], "picks": 0, "wins": 0,
                                         "graded": 0, "alpha_sum": 0.0, "alpha_n": 0})
        c["picks"] += 1
        if p.get("win") is not None:
            c["graded"] += 1
            c["wins"] += int(bool(p["win"]))
        if p.get("alpha_pct") is not None:
            c["alpha_sum"] += p["alpha_pct"]
            c["alpha_n"] += 1
    rows: List[Dict] = []
    for c in by.values():
        rows.append({
            "channel": c["channel"], "picks": c["picks"], "graded": c["graded"],
            "win_rate": (c["wins"] / c["graded"]) if c["graded"] else None,
            "avg_alpha_pct": (c["alpha_sum"] / c["alpha_n"]) if c["alpha_n"] else None,
        })
    rows.sort(key=lambda r: (r["avg_alpha_pct"] if r["avg_alpha_pct"] is not None else -1e9),
              reverse=True)
    return rows
