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
        if not ts or ts < cutoff:
            continue  # unknown or older than the lookback window
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

    langs = ["en", "en-US", "en-GB"]

    def _attempt() -> Optional[List[dict]]:
        try:  # ≥1.0 instance API, English preferred
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=langs)
        except TypeError:  # signature without languages kwarg
            fetched = YouTubeTranscriptApi().fetch(video_id)
        except Exception:  # ≤0.6 classmethod API
            return YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        return [{"text": getattr(s, "text", ""), "start": getattr(s, "start", 0.0)} for s in fetched]

    raw = None
    for attempt in range(2):  # one retry — transcript endpoint throttles intermittently
        try:
            raw = _attempt()
            if raw:
                break
        except Exception:
            pass
        if attempt == 0:
            import time
            time.sleep(0.4)
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
    # Indices / broad ETFs (finfluencers talk macro constantly).
    "s&p 500": "SPY", "s and p 500": "SPY", "nasdaq 100": "QQQ",
    "russell 2000": "IWM", "dow jones": "DIA",
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
# Word-boundary patterns so "artificial intelligence" ≠ INTC and "afford" ≠ F.
# Longest names first so "advanced micro devices" wins before any shorter overlap.
_NAME_PATTERNS = [
    (re.compile(r"\b" + re.escape(name) + r"\b"), sym)
    for name, sym in sorted(NAME_TO_TICKER.items(), key=lambda kv: -len(kv[0]))
]


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
    for pat, sym in _NAME_PATTERNS:
        c = len(pat.findall(low))
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
# Price targets: "price target 220" / "target of 220" (no $ needed) OR a verb + an explicit
# "$220" (the $ avoids capturing "to 5 years" / "go to 5 percent" as a $5 target).
_PT_RE = re.compile(
    r"(?:price target|target of|target at|target\s+is)\s+\$?\s?(\d{1,5}(?:\.\d{1,2})?)"
    r"|(?:sees|hits?|reach(?:es)?|going to|go(?:es)? to|up to|to)\s+\$\s?(\d{1,5}(?:\.\d{1,2})?)",
    re.I)
_PT_BAD_SUFFIX = re.compile(r"^\s*(?:%|percent|year|month|week|day|x\b)", re.I)
_HORIZON_RE = re.compile(
    r"(by (?:the )?end of (?:the )?(?:year|month|quarter)|year[- ]end|"
    r"next (?:week|month|quarter|year)|\d+\s*(?:day|week|month|year)s?|"
    r"long[- ]term|short[- ]term)", re.I)


def _find_target(text: str) -> Optional[float]:
    """First plausible price target, skipping numbers that are really %/durations."""
    for m in _PT_RE.finditer(text):
        num = m.group(1) or m.group(2)
        if not num or _PT_BAD_SUFFIX.match(text[m.end():]):
            continue
        try:
            return float(num)
        except ValueError:
            continue
    return None


def _call_context(segments: List["Segment"], i: int, universe: Set[str],
                  sym: str, max_chars: int = 160) -> str:
    """Text of segment ``i`` plus a short lookahead — so a call whose direction/target lands a
    caption or two later isn't lost. The lookahead stops as soon as a *different* ticker
    appears, to avoid attributing the next ticker's target to this one."""
    parts = [segments[i].text]
    size = len(segments[i].text)
    j = i + 1
    while j < len(segments) and size < max_chars:
        nxt = segments[j].text
        other = extract_tickers(nxt, universe)
        if other and set(other) != {sym}:
            break
        parts.append(nxt)
        size += len(nxt)
        j += 1
    return " ".join(parts)


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
    for i, seg in enumerate(segments):
        tickers = extract_tickers(seg.text, universe)
        if len(tickers) != 1:  # anchor on a segment naming exactly one ticker
            continue
        sym = next(iter(tickers))
        ctx = _call_context(segments, i, universe, sym)
        low = ctx.lower()
        direction = next((d for d, words in _DIRECTION.items()
                          if any(w in low for w in words)), None)
        target = _find_target(ctx)
        horizon = None
        mh = _HORIZON_RE.search(ctx)
        if mh:
            horizon = mh.group(1).strip()
        if direction is None and target is None:
            continue
        key = (sym, direction)
        if key in seen:
            continue
        seen.add(key)
        picks.append(Pick(sym, direction, target, horizon, ctx.strip(),
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
    ticker_sentiment: Dict[str, str] = field(default_factory=dict)


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

    # Per-ticker sentiment from the segments that actually name each ticker (falls back to the
    # whole-video read when a ticker only surfaced via a cross-segment name match).
    snips_by: Dict[str, List[str]] = {}
    for m in mentions:
        snips_by.setdefault(m.ticker, []).append(m.snippet)
    ticker_sentiment = {
        sym: _sentiment(" ".join(snips_by.get(sym, []))[:1500] or text, analyzer)[0]
        for sym in tickers
    }

    conviction = {sym: score_conviction(text, cnt) for sym, cnt in tickers.items()}
    return VideoAnalysis(
        upload=upload, has_transcript=has_tx, text=text,
        sentiment=sentiment, sentiment_score=sscore, tickers=tickers,
        mentions=mentions, picks=picks, events=events,
        flags=scan_flags(text), digest=_digest(text, upload, summarizer),
        conviction=conviction, ticker_sentiment=ticker_sentiment,
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
            tsent = a.ticker_sentiment.get(sym, a.sentiment)  # per-ticker, not whole-video
            if tsent == "bullish":
                d["bull"] += 1
            elif tsent == "bearish":
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
                 price_on_fn: Callable[[str, str], Optional[float]],
                 today: Optional[str] = None) -> int:
    """Append any *new* picks from ``analysis`` to ``store['picks']``.

    The call is dated to the video's **publish date** and the entry price is the close on that
    date (``price_on_fn(ticker, date)``), not the scan-time price — otherwise a creator's track
    record would be graded from whenever you happened to run the scan. Idempotent per
    (video, ticker, direction). Returns the number of newly recorded picks.
    """
    up = analysis.upload
    call_date = None
    if up.published_ts:
        try:
            call_date = datetime.fromtimestamp(up.published_ts).strftime("%Y-%m-%d")
        except Exception:
            call_date = None
    call_date = call_date or today or datetime.now().strftime("%Y-%m-%d")

    picks = store.setdefault("picks", [])
    existing = {p["id"] for p in picks}
    added = 0
    for pk in analysis.picks:
        pid = make_pick_id(up.video_id, pk.ticker, pk.direction)
        if pid in existing:
            continue
        picks.append({
            "id": pid, "channel": up.channel, "channel_id": up.channel_id,
            "video_id": up.video_id, "video_title": up.title, "video_url": up.url,
            "ticker": pk.ticker, "direction": pk.direction,
            "price_target": pk.price_target, "horizon": pk.horizon,
            "snippet": pk.snippet, "timestamp_sec": pk.timestamp_sec, "deep_link": pk.deep_link,
            "mention_date": call_date, "price_at_mention": price_on_fn(pk.ticker, call_date),
        })
        existing.add(pid)
        added += 1
    return added


_DEFAULT_MATURE_DAYS = 90


def _horizon_to_days(horizon: Optional[str]) -> Optional[int]:
    """Approximate a stated horizon ('by end of year', '6 months', 'long-term') in days."""
    if not horizon:
        return None
    h = horizon.lower()
    m = re.search(r"(\d+)\s*(day|week|month|year)", h)
    if m:
        return int(m.group(1)) * {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)]
    if "year" in h:
        return 180 if ("end of" in h or "year-end" in h or "year end" in h) else 365
    if "quarter" in h:
        return 90
    if "month" in h:
        return 30
    if "week" in h:
        return 7
    if "long" in h:
        return 365
    if "short" in h:
        return 30
    return None


def grade_picks(picks: List[Dict],
                current_price_fn: Callable[[str], Optional[float]],
                bench_return_fn: Callable[[str], Optional[float]],
                peak_since_fn: Optional[Callable[[str, str], Optional[float]]] = None,
                today: Optional[str] = None) -> List[Dict]:
    """Annotate each stored pick with return, alpha vs SPY, and a *decided* win/loss.

    Grading respects the stated horizon: a call is only marked win/loss once the horizon has
    elapsed (default 90d when none was stated) — except a long call whose **price target was
    hit** counts as a win immediately. ``hold`` calls aren't directional, so they're excluded
    from win/loss. A short/sell call profits from a decline (directional return negated).
    ``bench_return_fn(date)`` and ``peak_since_fn(ticker, date)`` are point-in-time lookups.
    """
    today_dt = datetime.strptime(today, "%Y-%m-%d") if today else datetime.now()
    graded: List[Dict] = []
    for p in picks:
        rec = dict(p)
        entry = p.get("price_at_mention")
        direction = p.get("direction")
        mdate = p.get("mention_date", "")
        cur = current_price_fn(p["ticker"]) if entry else None

        hdays = _horizon_to_days(p.get("horizon"))
        try:
            age = (today_dt - datetime.strptime(mdate, "%Y-%m-%d")).days
        except Exception:
            age = None
        matured = age is not None and age >= (hdays if hdays is not None else _DEFAULT_MATURE_DAYS)
        rec["horizon_days"], rec["age_days"], rec["matured"] = hdays, age, matured

        target_hit = None
        tgt = p.get("price_target")
        if peak_since_fn is not None and tgt and direction in (None, "buy"):
            peak = peak_since_fn(p["ticker"], mdate)
            if peak is not None:
                target_hit = bool(peak >= tgt)
        rec["target_hit"] = target_hit

        if entry and cur and entry > 0:
            raw = (cur - entry) / entry
            directional = -raw if direction == "sell" else raw
            bench = bench_return_fn(mdate)
            rec["return_pct"] = round(raw * 100, 2)
            rec["directional_pct"] = round(directional * 100, 2)
            rec["alpha_pct"] = round((directional - bench) * 100, 2) if bench is not None else None
            if direction == "hold":
                rec["win"] = None            # not a directional call
            elif target_hit:
                rec["win"] = True            # reached the stated target → correct
            elif matured:
                rec["win"] = bool(directional > 0)
            else:
                rec["win"] = None            # still pending its horizon
            rec["status"] = "pending" if rec["win"] is None else ("win" if rec["win"] else "loss")
        else:
            rec["return_pct"] = rec["directional_pct"] = rec["alpha_pct"] = None
            rec["win"] = None
            rec["status"] = "ungraded"
        graded.append(rec)
    return graded


def creator_leaderboard(graded: List[Dict]) -> List[Dict]:
    """Per-creator scorecard: #picks, win rate, average alpha — ranked by alpha."""
    by: Dict[str, Dict] = {}
    for p in graded:
        c = by.setdefault(p["channel"], {"channel": p["channel"], "picks": 0, "wins": 0,
                                         "graded": 0, "alpha_sum": 0.0, "alpha_n": 0})
        c["picks"] += 1
        if p.get("win") is not None:  # only decided picks count toward the record
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
