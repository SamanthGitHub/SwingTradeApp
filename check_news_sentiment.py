"""Scan top signals and evaluate news headline sentiment as supporting evidence."""
import yfinance as yf

from swingtradeapp.config import TradingConfig
from swingtradeapp.nlp import FinBERTSentimentAnalyzer, aggregate_news_sentiment
from swingtradeapp.risk import BayesianKellySizer
from swingtradeapp.signals import TrendSignalGenerator
from swingtradeapp.universe import UniverseFilter


def fetch_history(symbol: str, days: int = 90):
    try:
        hist = yf.Ticker(symbol).history(period=f"{days}d")
    except Exception:
        return None, None
    if hist is None or hist.empty:
        return None, None
    return hist["Close"].tolist(), hist["Volume"].tolist()


def main(sample_size: int = 50, min_score: float = 0.45):
    cfg = TradingConfig.load_from_env()
    universe = UniverseFilter(cfg).fetch_screened_symbols()

    trend = TrendSignalGenerator(cfg)
    sizer = BayesianKellySizer(cfg)
    finbert = FinBERTSentimentAnalyzer(cfg)

    signals = []
    for s in universe[:sample_size]:
        closes, volumes = fetch_history(s, days=90)
        if not closes or len(closes) < 20:
            continue
        sig = trend.build_signal(s, closes, volumes)
        if sig and sig.score >= min_score:
            signals.append(sig)

    signals.sort(key=lambda x: x.score, reverse=True)
    top = signals[:10]
    if not top:
        print("No signals found for the sample.")
        return

    print(f"Found {len(signals)} signals; checking news sentiment for top {len(top)}")
    for sig in top:
        evidence = aggregate_news_sentiment(sig.ticker, finbert, max_articles=6)
        size = sizer.size_position(sig)
        print("-" * 80)
        print(f"{sig.ticker}: signal_score={sig.score:.2f}  allocation_frac={size.fraction:.3f}")
        print(f"News headlines analyzed: {evidence['count']}, avg_score={evidence['avg_score']:.3f}, positive_pct={evidence['positive_pct']:.2%}")
        for h in evidence["headlines"]:
            print(f"  - [{h['label']}] {h['score']:.3f} — {h['headline']}")


if __name__ == "__main__":
    main(sample_size=50, min_score=0.45)
