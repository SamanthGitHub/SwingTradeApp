"""
Risk management: dynamic Kelly position sizing + portfolio-level guardrails.

Kelly fraction = (win_rate * (avg_win/avg_loss + 1) - 1) / (avg_win/avg_loss)
We use half-Kelly and cap at MAX_POSITION_FRACTION to protect against
estimation error in win rate / edge.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import TradingConfig
from .signals import Signal

logger = logging.getLogger(__name__)

MAX_POSITION_FRACTION = 0.10   # never risk more than 10% of account per trade
MAX_PORTFOLIO_HEAT = 0.25      # total portfolio at risk (sum of stop-distances) ≤ 25%
MAX_CONCURRENT_POSITIONS = 10  # hard cap on open positions
DAILY_LOSS_LIMIT_FRACTION = 0.03  # circuit-breaker: halt if day P&L < -3%


@dataclass
class PositionSize:
    fraction: float   # fraction of account
    dollars: float    # dollar allocation


@dataclass
class PortfolioState:
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    daily_pnl: float = 0.0
    halted: bool = False

    def position_count(self) -> int:
        return len(self.open_positions)

    def total_heat(self) -> float:
        """Sum of (entry - stop) / entry for all open positions, weighted by fraction."""
        heat = 0.0
        for pos in self.open_positions.values():
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_price", 0)
            frac = pos.get("fraction", 0)
            if entry > 0:
                heat += frac * (entry - stop) / entry
        return heat

    def update_pnl(self, pnl_dollars: float) -> None:
        self.daily_pnl += pnl_dollars

    def add_position(self, symbol: str, entry_price: float, stop_price: float,
                     target_price: float, fraction: float, dollars: float) -> None:
        self.open_positions[symbol] = {
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "fraction": fraction,
            "dollars": dollars,
        }

    def close_position(self, symbol: str, exit_price: float) -> float:
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return 0.0
        entry = pos.get("entry_price", exit_price)
        dollars = pos.get("dollars", 0.0)
        pnl = dollars * (exit_price - entry) / entry
        self.update_pnl(pnl)
        return pnl


class BayesianKellySizer:
    """
    Position sizer that combines:
    1. Backtested win rate / edge (if provided)
    2. Signal quality score (Bayesian update weight)
    3. Portfolio-level heat check
    4. Daily circuit breaker
    """

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        # Priors — updated by backtest results
        self.prior_win_rate = 0.50
        self.prior_avg_win = 0.04   # 4% avg win
        self.prior_avg_loss = 0.02  # 2% avg loss
        self.portfolio = PortfolioState()
        self._load_state()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_from_backtest(self, win_rate: float, avg_win_pct: float,
                             avg_loss_pct: float, n_trades: int = 30) -> None:
        """Feed backtested metrics so Kelly uses real edge estimates.

        The win rate is shrunk toward 0.5 with a Beta pseudo-count so small-sample
        (few-trade) backtests don't produce overconfident Kelly fractions.
        """
        if win_rate > 0 and avg_win_pct > 0 and avg_loss_pct > 0:
            prior_strength = 20.0  # equivalent pseudo-trades pulling toward 0.5
            self.prior_win_rate = (
                (win_rate * n_trades + 0.5 * prior_strength) / (n_trades + prior_strength)
            )
            self.prior_avg_win = avg_win_pct
            self.prior_avg_loss = avg_loss_pct
            logger.debug("Kelly priors updated: wr=%.2f (raw %.2f, n=%d) aw=%.3f al=%.3f",
                         self.prior_win_rate, win_rate, n_trades, avg_win_pct, avg_loss_pct)

    def size_position(
        self,
        signal: Signal,
        account_size: float = 100_000,
    ) -> PositionSize:
        """
        Compute half-Kelly position size, then apply portfolio guards.
        Returns zero allocation if circuit breaker is tripped or portfolio is full.
        """
        # Circuit breaker
        if self.portfolio.halted:
            logger.warning("Portfolio halted — daily loss limit hit")
            return PositionSize(fraction=0.0, dollars=0.0)

        daily_loss_limit = account_size * DAILY_LOSS_LIMIT_FRACTION
        if self.portfolio.daily_pnl < -daily_loss_limit:
            self.portfolio.halted = True
            logger.warning("Circuit breaker tripped: daily P&L = $%.2f", self.portfolio.daily_pnl)
            return PositionSize(fraction=0.0, dollars=0.0)

        # Position count
        if self.portfolio.position_count() >= MAX_CONCURRENT_POSITIONS:
            logger.info("Max concurrent positions reached (%d)", MAX_CONCURRENT_POSITIONS)
            return PositionSize(fraction=0.0, dollars=0.0)

        # Bayesian Kelly
        win_rate = self._score_adjusted_win_rate(signal.score)
        avg_win = self.prior_avg_win
        avg_loss = max(self.prior_avg_loss, 1e-6)
        odds = avg_win / avg_loss

        kelly = ((win_rate * (odds + 1)) - 1) / odds
        kelly = max(0.0, kelly)
        half_kelly = kelly * 0.5

        # Portfolio heat check
        projected_heat = self.portfolio.total_heat() + half_kelly * (
            (signal.entry_price - signal.stop_price) / signal.entry_price
            if signal.entry_price > 0 else 0.05
        )
        if projected_heat > MAX_PORTFOLIO_HEAT:
            # Scale down to fit within heat budget
            remaining_heat = max(0.0, MAX_PORTFOLIO_HEAT - self.portfolio.total_heat())
            risk_per_share = (signal.entry_price - signal.stop_price) / signal.entry_price if signal.entry_price > 0 else 0.05
            half_kelly = remaining_heat / risk_per_share if risk_per_share > 0 else 0.0

        fraction = min(half_kelly, MAX_POSITION_FRACTION)
        dollars = round(account_size * fraction, 2)
        return PositionSize(fraction=fraction, dollars=dollars)

    def evaluate_trade(self, signal: Signal, metrics: Dict[str, Any]) -> float:
        return self.size_position(signal).fraction

    def reset_daily(self) -> None:
        """Call at market open each day to reset circuit breaker."""
        self.portfolio.daily_pnl = 0.0
        self.portfolio.halted = False
        self._save_state()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _score_adjusted_win_rate(self, score: float) -> float:
        """
        Blend base win rate with signal quality.
        score=1.0 → use full prior_win_rate; score=0.4 → discount by ~20%.
        """
        discount = 0.8 + 0.2 * score  # range 0.88–1.0
        return min(self.prior_win_rate * discount, 0.95)

    def _state_path(self) -> Path:
        return Path(".data") / "portfolio_state.json"

    def _save_state(self) -> None:
        try:
            Path(".data").mkdir(exist_ok=True)
            with open(self._state_path(), "w") as f:
                json.dump({
                    "daily_pnl": self.portfolio.daily_pnl,
                    "halted": self.portfolio.halted,
                    "open_positions": self.portfolio.open_positions,
                }, f, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            if self._state_path().exists():
                with open(self._state_path()) as f:
                    data = json.load(f)
                self.portfolio.daily_pnl = data.get("daily_pnl", 0.0)
                self.portfolio.halted = data.get("halted", False)
                self.portfolio.open_positions = data.get("open_positions", {})
        except Exception:
            pass
