"""News NLP: sentiment, event tagging, summarization and novelty.

Every model is an optional, lazily-loaded Hugging Face pipeline with a non-AI heuristic
fallback, so the app degrades gracefully when ``transformers`` / ``sentence-transformers``
(or the model weights) are unavailable.
"""

import difflib
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Candidate event categories for zero-shot classification, mapped to short display tags.
_EVENT_LABELS = {
    "earnings report": "earnings",
    "merger or acquisition": "M&A",
    "analyst upgrade": "upgrade",
    "analyst downgrade": "downgrade",
    "lawsuit or regulation": "legal",
    "guidance change": "guidance",
    "product launch": "product",
}
# Events that move sentiment magnitude (positive amplifiers / negative amplifiers).
_BULLISH_EVENTS = {"upgrade", "product"}
_BEARISH_EVENTS = {"downgrade", "legal"}
_KEYWORDS = {
    "earnings": ["earnings", "eps", "revenue", "quarterly results", "beats", "misses"],
    "M&A": ["acquire", "acquisition", "merger", "buyout", "takeover"],
    "upgrade": ["upgrade", "raised price target", "outperform", "overweight"],
    "downgrade": ["downgrade", "cut price target", "underperform", "underweight"],
    "legal": ["lawsuit", "sues", "investigation", "sec ", "probe", "fined", "regulat"],
    "guidance": ["guidance", "forecast", "outlook"],
    "product": ["launch", "unveil", "announces", "releases", "rollout"],
}


# ── Sentiment ───────────────────────────────────────────────────────────────────

class FinBERTSentimentAnalyzer:
    """Financial sentiment analyzer.

    Tries a fast distilled financial model first, then FinBERT, then a neutral stub.
    The class name is kept for backward compatibility with existing imports.
    """

    _MODELS = [
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "ProsusAI/finbert",
    ]

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.pipeline = None
        self.model_name = None
        for name in self._MODELS:
            try:
                from transformers import pipeline
                self.pipeline = pipeline("sentiment-analysis", model=name, tokenizer=name)
                self.model_name = name
                logger.info("Loaded sentiment model: %s", name)
                break
            except Exception:
                continue
        if self.pipeline is None:
            logger.info("No sentiment model available — using neutral stub")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        if self.pipeline is None:
            return {"label": "neutral", "score": 0.5, "note": "transformers not available"}
        try:
            result = self.pipeline(str(text)[:512])
        except Exception:
            return {"label": "neutral", "score": 0.5}
        if not result:
            return {"label": "neutral", "score": 0.0}
        raw = result[0]
        return {"label": str(raw.get("label", "neutral")).lower(), "score": float(raw.get("score", 0.0))}


# ── Event classification ─────────────────────────────────────────────────────────

class NewsEventClassifier:
    """Tag a headline with a market event type (zero-shot, with keyword fallback)."""

    def __init__(self) -> None:
        self.pipeline = None
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
            )
            logger.info("Loaded zero-shot event classifier")
        except Exception:
            logger.info("Event classifier unavailable — using keyword heuristics")

    def classify(self, headline: str) -> str:
        if self.pipeline is not None:
            try:
                out = self.pipeline(str(headline), list(_EVENT_LABELS.keys()))
                return _EVENT_LABELS.get(out["labels"][0], "other")
            except Exception:
                pass
        return self._classify_keywords(headline)

    @staticmethod
    def _classify_keywords(headline: str) -> str:
        low = str(headline).lower()
        for tag, words in _KEYWORDS.items():
            if any(w in low for w in words):
                return tag
        return "other"


# ── Summarization ────────────────────────────────────────────────────────────────

class NewsSummarizer:
    """Summarize a batch of headlines into a short digest (with join fallback)."""

    def __init__(self) -> None:
        self.pipeline = None
        try:
            from transformers import pipeline
            self.pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            logger.info("Loaded news summarizer")
        except Exception:
            logger.info("Summarizer unavailable — using headline join fallback")

    def summarize(self, headlines: List[str]) -> str:
        clean = [h for h in (headlines or []) if h]
        if not clean:
            return ""
        if self.pipeline is None:
            return " · ".join(clean[:3])
        try:
            text = ". ".join(clean)[:1024]
            out = self.pipeline(text, max_length=80, min_length=20, do_sample=False)
            return out[0]["summary_text"] if out else " · ".join(clean[:3])
        except Exception:
            return " · ".join(clean[:3])


# ── Transcript cleanup (punctuation + casing restoration) ─────────────────────────

def _recase(text: str) -> str:
    """Capitalize sentence starts + standalone 'i', and tighten spaces around punctuation.

    The punctuation model restores ``. , ? -`` etc. but keeps everything lowercase, so this adds
    the casing. Finance-specific casing (company names, ``$cashtags``) is applied separately by
    the YouTube ``_tidy`` step.
    """
    s = re.sub(r"\s+", " ", text or "").strip()
    if not s:
        return s
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)   # no space before punctuation
    s = re.sub(r"\bi\b", "I", s)             # standalone "i" -> "I"
    out, cap = [], True
    for ch in s:
        if cap and ch.isalpha():
            ch, cap = ch.upper(), False
        elif ch in ".!?":
            cap = True
        out.append(ch)
    return "".join(out)


class TranscriptCleaner:
    """Restore punctuation + casing to noisy ASR transcripts (lowercase, unpunctuated).

    Optional + local (mirrors the other AI features). Wraps the multilingual fullstop punctuation
    model via ``deepmultilingualpunctuation``. Falls back to **returning the input unchanged** when
    the package/model isn't installed or errors, so the caller's heuristic cleanup still applies.
    """

    def __init__(self) -> None:
        self.model = None
        try:
            from deepmultilingualpunctuation import PunctuationModel
            self.model = PunctuationModel()
            logger.info("Loaded transcript cleaner (punctuation restoration)")
        except Exception:
            logger.info("Transcript cleaner unavailable — heuristic tidy only")

    def clean(self, text: str) -> str:
        if not text or self.model is None:
            return text or ""
        try:
            return _recase(self.model.restore_punctuation(text))
        except Exception:
            return text


# ── Novelty / dedup ──────────────────────────────────────────────────────────────

class NewsNovelty:
    """Deduplicate near-identical headlines and score how much is genuinely new."""

    def __init__(self) -> None:
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Loaded news novelty embedder")
        except Exception:
            logger.info("Embedder unavailable — using difflib novelty fallback")

    def dedup(self, headlines: List[str], threshold: float = 0.85) -> Dict[str, Any]:
        clean = [h for h in (headlines or []) if h]
        if not clean:
            return {"unique": [], "unique_count": 0, "novelty": 0.0}
        unique = self._dedup_embeddings(clean, threshold) if self.model else self._dedup_difflib(clean)
        return {
            "unique": unique,
            "unique_count": len(unique),
            "novelty": round(len(unique) / len(clean), 4),
        }

    def _dedup_embeddings(self, headlines: List[str], threshold: float) -> List[str]:
        try:
            import numpy as np
            emb = self.model.encode(headlines, normalize_embeddings=True)
            kept_idx: List[int] = []
            for i in range(len(headlines)):
                if all(float(np.dot(emb[i], emb[j])) < threshold for j in kept_idx):
                    kept_idx.append(i)
            return [headlines[i] for i in kept_idx]
        except Exception:
            return self._dedup_difflib(headlines)

    @staticmethod
    def _dedup_difflib(headlines: List[str], threshold: float = 0.8) -> List[str]:
        unique: List[str] = []
        for h in headlines:
            if all(difflib.SequenceMatcher(None, h.lower(), u.lower()).ratio() < threshold for u in unique):
                unique.append(h)
        return unique


# ── Aggregation ──────────────────────────────────────────────────────────────────

def _ts_to_str(ts: float) -> str:
    from datetime import datetime as _dt
    try:
        return _dt.fromtimestamp(float(ts)).strftime("%b %d, %H:%M")
    except Exception:
        return ""


def _fetch_yf_news_items(symbol: str, max_items: int) -> List[Dict]:
    """Yahoo Finance news for a symbol (legacy + nested ``content`` schemas)."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        try:
            news = getattr(t, "news", None) or (t.get_news() if hasattr(t, "get_news") else [])
        except Exception:
            news = []
        items: List[Dict] = []
        for a in (news or [])[:max_items]:
            if not isinstance(a, dict):
                items.append({"title": str(a), "publisher": "", "url": "", "time": "", "_ts": 0})
                continue
            c = a.get("content") if isinstance(a.get("content"), dict) else a
            title = c.get("title") or a.get("title") or c.get("headline") or ""
            if not title:
                continue
            prov = c.get("provider")
            publisher = (prov.get("displayName", "") if isinstance(prov, dict) else "") or a.get("publisher", "")
            url = ""
            for key in ("canonicalUrl", "clickThroughUrl"):
                u = c.get(key)
                if isinstance(u, dict) and u.get("url"):
                    url = u["url"]
                    break
            url = url or a.get("link", "")
            ts = a.get("providerPublishTime") or 0
            items.append({"title": title, "publisher": publisher or "Yahoo Finance",
                          "url": url, "time": _ts_to_str(ts) if ts else "", "_ts": float(ts or 0)})
        return items
    except Exception:
        return []


def _fetch_google_news(query: str, max_items: int = 8) -> List[Dict]:
    """Free, key-less news from Google News RSS — aggregates most public outlets."""
    import ssl
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    try:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()
        url = ("https://news.google.com/rss/search?q="
               + urllib.parse.quote(query) + "&hl=en-US&gl=US&ceid=US:en")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SwingTradeApp)"})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            root = ET.fromstring(resp.read())
        items: List[Dict] = []
        for it in root.findall(".//item")[:max_items]:
            title = (it.findtext("title") or "").strip()
            if not title:
                continue
            link = (it.findtext("link") or "").strip()
            src_el = it.find("source")
            publisher = src_el.text.strip() if (src_el is not None and src_el.text) else "Google News"
            if publisher and title.endswith(f" - {publisher}"):
                title = title[: -(len(publisher) + 3)]
            ts = 0.0
            time_str = ""
            pub = it.findtext("pubDate")
            if pub:
                try:
                    dt = parsedate_to_datetime(pub)
                    ts = dt.timestamp()
                    time_str = dt.strftime("%b %d, %H:%M")
                except Exception:
                    pass
            items.append({"title": title, "publisher": publisher, "url": link,
                          "time": time_str, "_ts": ts})
        return items
    except Exception:
        return []


def _dedup_news(items: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for it in items:
        key = "".join(ch for ch in it["title"].lower() if ch.isalnum())[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def fetch_news_items(symbol: str, max_items: int = 8, company: Optional[str] = None) -> List[Dict[str, str]]:
    """Aggregated recent news for a ticker from Yahoo Finance **and** Google News.

    Each item: ``{title, publisher, url, time}``, deduped and newest-first.
    """
    query = f"{company} stock" if company else f"{symbol} stock"
    combined = _fetch_google_news(query, max_items) + _fetch_yf_news_items(symbol, max_items)
    combined = _dedup_news(combined)
    combined.sort(key=lambda x: x.get("_ts", 0), reverse=True)
    return [{k: v for k, v in it.items() if k != "_ts"} for it in combined[:max_items]]


# Broad finance queries for whole-market news (Google News aggregates the outlets).
_MARKET_QUERIES = ["stock market", "S&P 500", "Federal Reserve", "earnings", "Nasdaq", "US economy"]


def fetch_market_news(max_items: int = 20) -> List[Dict[str, str]]:
    """Broad market news from across free outlets (Google News), deduped + newest-first."""
    items: List[Dict] = []
    for q in _MARKET_QUERIES:
        items += _fetch_google_news(q, max_items=6)
    items = _dedup_news(items)
    items.sort(key=lambda x: x.get("_ts", 0), reverse=True)
    return [{k: v for k, v in it.items() if k != "_ts"} for it in items[:max_items]]


def _fetch_headlines(symbol: str, max_articles: int) -> List[str]:
    # Yahoo-only (fast) — used by the bulk per-symbol scan. On-demand views use
    # fetch_news_items() which also pulls Google News.
    return [it["title"] for it in _fetch_yf_news_items(symbol, max_articles)]


def aggregate_news_sentiment(
    symbol: str,
    analyzer: "FinBERTSentimentAnalyzer",
    max_articles: int = 5,
    event_classifier: Optional["NewsEventClassifier"] = None,
    novelty_scorer: Optional["NewsNovelty"] = None,
    headlines: Optional[List[str]] = None,
) -> Dict:
    """Fetch recent headlines for ``symbol`` and return aggregate sentiment.

    When ``headlines`` is provided (e.g. pre-fetched concurrently by a scan), no network
    call is made — only scoring runs, which keeps FinBERT usage on the caller's thread.
    When ``novelty_scorer`` is provided, recycled headlines are deduped before scoring.
    When ``event_classifier`` is provided, each headline is tagged with a market event
    and binary-risk flags (e.g. ``earnings_imminent``) are surfaced. Backward compatible:
    callers passing only (symbol, analyzer) get the original keys plus benign extras.
    """
    base = {"avg_score": 0.5, "positive_pct": 0.0, "count": 0, "headlines": [],
            "events": [], "event_flags": [], "novelty": 1.0, "unique_count": 0}
    try:
        if headlines is None:
            headlines = _fetch_headlines(symbol, max_articles)
        else:
            headlines = list(headlines)[:max_articles]

        novelty, unique_count = 1.0, len(headlines)
        if novelty_scorer is not None and headlines:
            nd = novelty_scorer.dedup(headlines)
            headlines = nd["unique"]
            novelty, unique_count = nd["novelty"], nd["unique_count"]

        if not headlines:
            base["novelty"] = novelty
            return base

        scores, positives, results, events = [], 0, [], []
        for h in headlines:
            r = analyzer.analyze_text(h)
            label = str(r.get("label", "neutral")).lower()
            score = float(r.get("score", 0.5))
            event = event_classifier.classify(h) if event_classifier is not None else None
            scores.append(score)
            if "pos" in label:
                positives += 1
            results.append({"headline": h, "label": label, "score": score, "event": event})
            if event and event != "other":
                events.append(event)

        event_flags = sorted(set(events))
        if "earnings" in event_flags:
            event_flags = ["earnings_imminent"] + [e for e in event_flags if e != "earnings"]

        # Light event-aware tilt to positive_pct so amplifying events move the needle.
        pos_pct = positives / len(scores) if scores else 0.0
        tilt = 0.05 * sum(e in _BULLISH_EVENTS for e in events) - 0.05 * sum(e in _BEARISH_EVENTS for e in events)
        pos_pct = float(min(1.0, max(0.0, pos_pct + tilt)))

        return {
            "avg_score": sum(scores) / len(scores) if scores else 0.5,
            "positive_pct": pos_pct,
            "count": len(scores),
            "headlines": results,
            "events": events,
            "event_flags": event_flags,
            "novelty": novelty,
            "unique_count": unique_count,
        }
    except Exception:
        return base
