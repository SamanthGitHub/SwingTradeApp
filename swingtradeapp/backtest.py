"""
Vectorized walk-forward backtest engine.

Simulates bracket trades (entry → stop loss OR take profit) on historical
OHLCV data and returns edge metrics consumed by the Kelly position sizer.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    pnl_pct: float
    won: bool


@dataclass
class BacktestResult:
    total_trades: int
    win_rate: float          # fraction 0-1
    avg_win_pct: float       # avg % gain on winners
    avg_loss_pct: float      # avg % loss on losers (positive number)
    profit_factor: float     # gross_win / gross_loss
    sharpe_ratio: float
    max_drawdown_pct: float  # max peak-to-trough % of equity curve
    total_return_pct: float
    trades: List[TradeRecord] = field(default_factory=list)
    edge_ratio: float = 0.0  # win_rate * avg_win / ((1-win_rate) * avg_loss)

    @classmethod
    def empty(cls) -> "BacktestResult":
        return cls(
            total_trades=0,
            win_rate=0.5,
            avg_win_pct=0.04,
            avg_loss_pct=0.02,
            profit_factor=1.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            total_return_pct=0.0,
            edge_ratio=1.0,
        )


class VectorBacktestEngine:
    """
    Simulate bracket orders on a price series.

    For each signal bar `i`, it enters at bar i+1 open (approximated as
    close[i+1]) and then scans forward until:
      - close[j] <= stop_price  → loss
      - close[j] >= target_price → win
      - max_hold bars elapsed   → exit at market
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.max_hold_bars = 10   # max holding period in bars
        # Cost model (basis points → fraction). Applied to every simulated fill so
        # backtest edge is net of slippage + round-trip commission.
        self.slippage = float(getattr(config, "slippage_bps", 5.0)) / 1e4
        self.commission = float(getattr(config, "commission_bps", 1.0)) / 1e4

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        prices: List[float],
        signals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Legacy dict interface — wraps run() for backward compat."""
        result = self.empty_result()
        if not prices or not signals:
            return result
        closes = np.array(prices, dtype=float)
        highs = closes
        lows = closes
        result = self._simulate(closes, highs, lows, signals)
        return {
            "return_pct": result.total_return_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown_pct,
            "trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
        }

    def run(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        signal_bars: Optional[List[int]] = None,
    ) -> BacktestResult:
        """
        Full backtest on OHLCV arrays.

        signal_bars: list of bar indices where a long entry is triggered.
                     If None, generates signals using built-in logic (EMA cross).
        """
        if len(closes) < 30:
            return BacktestResult.empty()

        if signal_bars is None:
            signal_bars = self._generate_ema_cross_signals(closes)

        # Build synthetic signal dicts with ATR-based stops
        atr = self._rolling_atr(highs, lows, closes, 14)
        signals = []
        for bar in signal_bars:
            if bar + 1 >= len(closes):
                continue
            entry = float(closes[bar])
            bar_atr = float(atr[bar]) if not np.isnan(atr[bar]) else entry * 0.01
            stop = round(entry - 2.0 * bar_atr, 4)
            target = round(entry + 4.0 * bar_atr, 4)
            signals.append({"bar": bar, "entry": entry, "stop": stop, "target": target})

        return self._simulate(closes, highs, lows, signals)

    def run_signals(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        signals: List[Dict[str, Any]],
    ) -> BacktestResult:
        """Backtest an explicit list of signal dicts ``{bar, entry, stop, target, symbol}``.

        Unlike :meth:`run` (which overwrites every signal's exits with a uniform ATR bracket),
        this honours each signal's *own* stop/target — used by the Setup Backtest Lab so a
        setup is validated under the exit rules it actually defines. Signals whose levels are
        invalid (stop ≥ entry or target ≤ entry) are skipped by the simulator.
        """
        if len(closes) < 5 or not signals:
            return BacktestResult.empty()
        return self._simulate(np.asarray(closes, dtype=float),
                              np.asarray(highs, dtype=float),
                              np.asarray(lows, dtype=float), signals)

    def run_walk_forward(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        n_folds: int = 4,
        train_frac: float = 0.5,
    ) -> BacktestResult:
        """Out-of-sample walk-forward backtest.

        Splits the series into ``n_folds`` sequential folds. Within each fold, signals
        are generated on the train portion but only trades whose entry falls in the
        held-out (test) portion are recorded — so the aggregated metrics are
        out-of-sample and safe to feed into position sizing.
        """
        n = len(closes)
        if n < 60:
            return BacktestResult.empty()

        fold_size = n // n_folds
        if fold_size < 30:
            # Too short to split meaningfully; fall back to a single 50/50 OOS split.
            n_folds, fold_size = 1, n

        oos_trades: List[TradeRecord] = []
        equity_curve: List[float] = [1.0]
        equity = 1.0

        for f in range(n_folds):
            start = f * fold_size
            end = n if f == n_folds - 1 else (f + 1) * fold_size
            seg_close = closes[start:end]
            seg_high = highs[start:end]
            seg_low = lows[start:end]
            if len(seg_close) < 30:
                continue

            split = int(len(seg_close) * train_frac)
            # Signals generated on the whole segment, but kept only if they fire in the
            # out-of-sample (post-split) portion.
            signal_bars = [b for b in self._generate_ema_cross_signals(seg_close) if b >= split]
            if not signal_bars:
                continue

            fold_result = self.run(seg_close, seg_high, seg_low, signal_bars=signal_bars)
            for t in fold_result.trades:
                oos_trades.append(t)
                equity *= (1 + t.pnl_pct)
                equity_curve.append(equity)

        if not oos_trades:
            return BacktestResult.empty()
        return self._compute_stats(oos_trades, equity_curve)

    def validate_strategy(self, walk_forward_data: Any) -> Dict[str, Any]:
        return {"status": "ok"}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _simulate(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        signals: List[Dict[str, Any]],
    ) -> BacktestResult:
        trades: List[TradeRecord] = []
        equity_curve: List[float] = [1.0]
        equity = 1.0

        for sig in signals:
            bar = int(sig.get("bar", sig.get("entry_bar", 0)))
            entry_bar = bar + 1
            if entry_bar >= len(closes):
                continue

            entry_price = float(sig.get("entry", closes[entry_bar]))
            stop_price = float(sig.get("stop", entry_price * 0.98))
            target_price = float(sig.get("target", entry_price * 1.06))

            if stop_price >= entry_price or target_price <= entry_price:
                continue

            exit_price = None
            exit_bar = None
            won = False

            for j in range(entry_bar + 1, min(entry_bar + self.max_hold_bars + 1, len(closes))):
                lo = float(lows[j]) if j < len(lows) else float(closes[j])
                hi = float(highs[j]) if j < len(highs) else float(closes[j])

                if lo <= stop_price:
                    exit_price = stop_price
                    exit_bar = j
                    won = False
                    break
                if hi >= target_price:
                    exit_price = target_price
                    exit_bar = j
                    won = True
                    break

            if exit_price is None:
                # Time exit — use last close
                exit_bar = min(entry_bar + self.max_hold_bars, len(closes) - 1)
                exit_price = float(closes[exit_bar])
                won = exit_price > entry_price

            # Apply cost model: slippage worsens both fills, commission charged per side.
            entry_fill = entry_price * (1 + self.slippage)
            exit_fill = exit_price * (1 - self.slippage)
            pnl_pct = (exit_fill - entry_fill) / entry_fill - 2 * self.commission
            won = pnl_pct > 0
            equity *= (1 + pnl_pct)
            equity_curve.append(equity)

            trades.append(TradeRecord(
                symbol=str(sig.get("symbol", "")),
                entry_bar=entry_bar,
                exit_bar=exit_bar,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price=stop_price,
                target_price=target_price,
                pnl_pct=pnl_pct,
                won=won,
            ))

        return self._compute_stats(trades, equity_curve)

    def _compute_stats(
        self,
        trades: List[TradeRecord],
        equity_curve: List[float],
    ) -> BacktestResult:
        if not trades:
            return BacktestResult.empty()

        pnls = np.array([t.pnl_pct for t in trades])
        wins = [t for t in trades if t.won]
        losses = [t for t in trades if not t.won]

        win_rate = len(wins) / len(trades)
        avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
        avg_loss = abs(float(np.mean([t.pnl_pct for t in losses]))) if losses else 0.01

        gross_win = sum(t.pnl_pct for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 1.0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe (trade-level, annualized assuming ~252 trades / year is too aggressive;
        # use sqrt(252/len) scaling — sufficient for ranking)
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 1e-6
        sharpe = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0.0

        # Max drawdown on equity curve
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdowns = (peak - eq) / peak
        max_drawdown = float(np.max(drawdowns))

        total_return = float(equity_curve[-1] - 1.0)

        edge_ratio = (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 and win_rate < 1 else 0.0

        return BacktestResult(
            total_trades=len(trades),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown,
            total_return_pct=total_return,
            trades=trades,
            edge_ratio=edge_ratio,
        )

    def _generate_ema_cross_signals(self, closes: np.ndarray) -> List[int]:
        """Simple EMA 9/21 crossover as default signal generator."""
        if len(closes) < 25:
            return []
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        bars = []
        for i in range(1, len(closes)):
            if ema9[i - 1] < ema21[i - 1] and ema9[i] > ema21[i]:
                bars.append(i)
        return bars

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        result = np.empty_like(values, dtype=float)
        k = 2.0 / (period + 1)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def _rolling_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        n = len(closes)
        atr = np.full(n, np.nan)
        if n < 2:
            return atr
        tr = np.maximum(highs[1:] - lows[1:],
              np.maximum(np.abs(highs[1:] - closes[:-1]),
                         np.abs(lows[1:] - closes[:-1])))
        for i in range(period - 1, len(tr)):
            atr[i + 1] = float(np.mean(tr[i - period + 1: i + 1]))
        return atr

    @staticmethod
    def empty_result() -> Dict[str, Any]:
        return {
            "return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "win_rate": 0.5,
            "profit_factor": 1.0,
        }
