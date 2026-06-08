"""
Multi-timeframe confluence analysis.
Checks if signals align across 15m, 1h, daily, and weekly timeframes.
"""

from typing import Dict, List, Optional, Tuple

import yfinance as yf

from .signals import TrendSignalGenerator, Signal


class MultiTimeframeAnalyzer:
    """
    Analyzes signal confluence across multiple timeframes.
    Higher confidence when multiple timeframes agree.
    """

    def __init__(self, config):
        self.config = config
        self.signal_generator = TrendSignalGenerator(config)
        self.timeframes = {
            "15m": "15m",
            "1h": "60m",
            "4h": "240m",
            "daily": "1d",
            "weekly": "1wk",
        }

    def fetch_multitf_data(
        self,
        symbol: str,
        start: int = 100,
    ) -> Dict[str, Dict]:
        """
        Fetch OHLCV data for multiple timeframes.
        Returns dict: {timeframe: {'closes': [...], 'highs': [...], ...}}
        """
        data = {}

        try:
            ticker = yf.Ticker(symbol)

            # Daily data (easy, good quality)
            daily = ticker.history(period="300d")
            if not daily.empty:
                data["daily"] = {
                    "closes": daily["Close"].tolist(),
                    "highs": daily["High"].tolist(),
                    "lows": daily["Low"].tolist(),
                    "volumes": daily["Volume"].tolist(),
                }

            # Intra-day data (limited by free tier, may not be available)
            try:
                intraday = ticker.history(period="60d", interval="1h")
                if not intraday.empty and len(intraday) >= 20:
                    data["1h"] = {
                        "closes": intraday["Close"].tolist(),
                        "highs": intraday["High"].tolist(),
                        "lows": intraday["Low"].tolist(),
                        "volumes": intraday["Volume"].tolist(),
                    }
            except Exception:
                pass

            # Weekly (inferred from daily)
            if "daily" in data:
                closes = data["daily"]["closes"]
                if len(closes) >= 52:
                    # Simple weekly aggregation (not perfect, but functional)
                    weekly_closes = [closes[i] for i in range(0, len(closes), 5)][-52:]
                    data["weekly"] = {
                        "closes": weekly_closes,
                        "highs": weekly_closes,  # Simplified
                        "lows": weekly_closes,
                        "volumes": [0] * len(weekly_closes),
                    }
        except Exception as e:
            pass

        return data

    def generate_multitf_signals(
        self,
        symbol: str,
        data: Dict[str, Dict],
    ) -> Dict[str, Optional[Signal]]:
        """
        Generate signals for each timeframe.
        Returns: {'daily': Signal(...), '1h': Signal(...), ...}
        """
        signals = {}

        for tf, ohlcv in data.items():
            try:
                signal = self.signal_generator.build_signal(
                    symbol,
                    ohlcv["closes"],
                    ohlcv["volumes"],
                    highs=ohlcv["highs"],
                    lows=ohlcv["lows"],
                )
                signals[tf] = signal
            except Exception:
                signals[tf] = None

        return signals

    def score_confluence(
        self,
        signals: Dict[str, Optional[Signal]],
    ) -> Tuple[float, int, str]:
        """
        Score multi-timeframe confluence.
        Returns (confluence_score, signals_in_agreement, agreement_type)

        Example:
        - Daily bullish + 1h bullish + weekly bullish = 3x agreement = 1.0 score
        - Daily bullish + 1h bearish = 0x agreement = low score
        """
        bullish_count = 0
        bearish_count = 0

        for tf, signal in signals.items():
            if signal is None:
                continue
            if signal.signal_type == "long":
                bullish_count += 1
            elif signal.signal_type == "short":
                bearish_count += 1

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0, "NO_SIGNAL"

        # Confluence score: 0-1, higher when aligned
        max_agreement = max(bullish_count, bearish_count)
        confluence_score = (max_agreement / total) * (max_agreement / 3)  # Boost for 3+ agreement

        agreement_type = "BULLISH_CONFLUENCE" if bullish_count > bearish_count else "BEARISH_CONFLUENCE"
        if bullish_count == bearish_count:
            agreement_type = "CONFLICT"

        return float(confluence_score), max_agreement, agreement_type

    def adjust_signal_strength(
        self,
        base_signal: Signal,
        confluence_score: float,
    ) -> Signal:
        """
        Adjust signal score based on multi-timeframe confluence.
        Example: 0.6 base score × 1.5 confluence = 0.9 final score (capped at 1.0)
        """
        confluence_multiplier = 1.0 + (confluence_score * 0.5)  # Max 1.5x boost
        adjusted_score = min(base_signal.score * confluence_multiplier, 1.0)

        # Update metadata
        base_signal.metadata["confluence_score"] = confluence_score
        base_signal.metadata["adjusted_score"] = adjusted_score
        base_signal.score = adjusted_score

        return base_signal

    def should_enter_based_on_confluence(
        self,
        base_signal: Signal,
        signals: Dict[str, Optional[Signal]],
        require_higher_tf_agreement: bool = True,
    ) -> Tuple[bool, str]:
        """
        Decide whether to enter based on multi-timeframe rules.

        Rules:
        1. Base signal required (intra-day)
        2. If require_higher_tf_agreement: daily must agree with base
        3. Bonus: if weekly also agrees

        Returns (should_enter, reason)
        """
        base_type = base_signal.signal_type

        # Check daily
        daily_sig = signals.get("daily")
        if require_higher_tf_agreement and daily_sig:
            if daily_sig.signal_type != base_type:
                return False, f"Daily disagrees ({daily_sig.signal_type} vs {base_type})"

        # Check weekly (nice-to-have)
        weekly_sig = signals.get("weekly")
        if weekly_sig and weekly_sig.signal_type != base_type:
            return True, f"Daily agrees but weekly disagrees (lower priority)"

        return True, "Confluence OK"

    def generate_summary(self, signals: Dict[str, Optional[Signal]]) -> str:
        """Generate human-readable summary of multi-timeframe analysis."""
        summary = "Multi-timeframe analysis:\n"
        for tf, sig in signals.items():
            if sig:
                summary += f"  {tf:8s}: {sig.signal_type:6s} score={sig.score:.2f}\n"
            else:
                summary += f"  {tf:8s}: --\n"
        return summary
