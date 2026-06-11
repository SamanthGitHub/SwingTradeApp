"""
Options analysis: IV rank, put/call ratio, Greeks, unusual activity alerts.
Uses free data from yfinance and simple Black-Scholes calculations.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

import numpy as np
import yfinance as yf


class OptionsAnalyzer:
    """Analyzes options chains for trading signals."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    # ── IV Analysis ────────────────────────────────────────────────────────────

    def fetch_current_iv(self, symbol: str) -> Optional[float]:
        """Current implied volatility: median call IV across the nearest 3 expirations.

        Shared by ``fetch_iv_rank`` (52-week positioning) and the options page (seeds the IV-crush
        estimate and the Greeks calculator). Returns None when no usable chains are available.
        """
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            if not options_dates:
                return None

            ivs = []
            for exp_date in options_dates[:3]:
                try:
                    chain = ticker.option_chain(exp_date)
                    calls = chain.calls
                    if not calls.empty:
                        mid_iv = calls["impliedVolatility"].median()
                        if mid_iv > 0:
                            ivs.append(mid_iv)
                except Exception:
                    pass

            if not ivs:
                return None
            mean_iv = float(np.mean(ivs))
            # Yahoo occasionally returns near-zero IV for a chain (bad/placeholder data); treat
            # anything below 1% as unavailable rather than reporting a misleading ~0 IV / IV rank.
            return mean_iv if mean_iv >= 0.01 else None
        except Exception:
            return None

    def fetch_iv_rank(self, symbol: str) -> Optional[float]:
        """
        Calculate IV Rank: where is current IV vs 52-week range?
        IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV) * 100
        High IV Rank (>70) = expensive premiums, sell strategies
        Low IV Rank (<30) = cheap premiums, buy strategies
        """
        try:
            current_iv = self.fetch_current_iv(symbol)
            if current_iv is None:
                return None

            # For 52-week range, we'd need historical volatility
            # As proxy: use historical price volatility
            hist = yf.Ticker(symbol).history(period="1y")
            hist_vol = hist["Close"].pct_change().std() * math.sqrt(252)

            # Very simplified: assume 52w IV range is 0.5x to 1.5x hist_vol
            iv_52w_low = hist_vol * 0.5
            iv_52w_high = hist_vol * 1.5

            if iv_52w_high == iv_52w_low:
                return 50.0

            iv_rank = (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
            return float(np.clip(iv_rank, 0, 100))
        except Exception:
            return None

    def fetch_put_call_ratio_symbol(self, symbol: str) -> Optional[float]:
        """
        Fetch put/call ratio from options chain.
        > 1.0 = more put volume (bearish sentiment)
        < 0.5 = more call volume (bullish sentiment)
        """
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            if not options_dates:
                return None

            # Use nearest expiration
            chain = ticker.option_chain(options_dates[0])
            put_volume = chain.puts["volume"].sum()
            call_volume = chain.calls["volume"].sum()

            if call_volume == 0:
                return None

            pc_ratio = put_volume / call_volume
            return float(pc_ratio)
        except Exception:
            return None

    # ── Greeks Estimation ──────────────────────────────────────────────────────

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Approximate normal CDF without scipy (using error function)."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def _black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Black-Scholes call price and delta.
        S = spot price
        K = strike price
        T = time to expiration (years)
        r = risk-free rate
        sigma = volatility (annualized)
        """
        if T <= 0:
            return max(S - K, 0), 1.0 if S > K else 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        norm_cdf = OptionsAnalyzer._norm_cdf
        call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)

        return call_price, delta

    def estimate_greeks(
        self,
        symbol: str,
        spot_price: float,
        strike: float,
        days_to_expiry: int,
        iv: float = 0.30,
    ) -> Optional[Dict[str, float]]:
        """
        Estimate call option Greeks using Black-Scholes.
        Very simplified; production systems use more sophisticated models.
        """
        if days_to_expiry <= 0 or iv == 0:
            return None

        T = days_to_expiry / 365.0
        call_price, delta = self._black_scholes_call(
            spot_price, strike, T, self.risk_free_rate, iv
        )

        # Simplified gamma, vega, theta (not exact, but directionally correct)
        gamma = (1 / (spot_price * iv * math.sqrt(2 * math.pi * T))) * math.exp(
            -0.5 * ((math.log(spot_price / strike) + self.risk_free_rate * T) / (iv * math.sqrt(T))) ** 2
        )
        vega = spot_price * math.exp(
            -0.5 * ((math.log(spot_price / strike) + self.risk_free_rate * T) / (iv * math.sqrt(T))) ** 2
        ) * math.sqrt(T) / 100.0  # Per 1% IV change
        theta = (
            -spot_price * (1 / math.sqrt(2 * math.pi * T)) * iv * math.exp(
                -0.5 * ((math.log(spot_price / strike) + self.risk_free_rate * T) / (iv * math.sqrt(T))) ** 2
            ) / (365 * 2)
        )

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "call_price": float(call_price),
        }

    # ── Unusual Activity Detection ─────────────────────────────────────────────

    def detect_unusual_volume(
        self,
        symbol: str,
        volume_threshold_pct: float = 2.0,
    ) -> Optional[Dict[str, any]]:
        """
        Detect unusually large options trades (block purchases).
        Indicator of smart money positioning.
        """
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            if not options_dates:
                return None

            chain = ticker.option_chain(options_dates[0])
            calls = chain.calls
            puts = chain.puts

            if calls.empty or puts.empty:
                return None

            # Find highest volume options
            high_vol_calls = calls.nlargest(3, "volume")
            high_vol_puts = puts.nlargest(3, "volume")

            unusual_calls = []
            unusual_puts = []

            for _, row in high_vol_calls.iterrows():
                vol = row["volume"]
                open_int = row["openInterest"]
                if vol > open_int * 0.1:  # Volume > 10% of open interest = unusual
                    unusual_calls.append({
                        "strike": row["strike"],
                        "volume": vol,
                        "openInterest": open_int,
                        "type": "call",
                    })

            for _, row in high_vol_puts.iterrows():
                vol = row["volume"]
                open_int = row["openInterest"]
                if vol > open_int * 0.1:
                    unusual_puts.append({
                        "strike": row["strike"],
                        "volume": vol,
                        "openInterest": open_int,
                        "type": "put",
                    })

            if unusual_calls or unusual_puts:
                return {
                    "unusual_calls": unusual_calls,
                    "unusual_puts": unusual_puts,
                    "signal": "calls" if len(unusual_calls) > len(unusual_puts) else "puts",
                }

            return None
        except Exception:
            return None

    def analyze_key_strikes(
        self,
        symbol: str,
        spot: Optional[float] = None,
    ) -> Optional[Dict[str, any]]:
        """Open-interest 'walls', max-pain, and OI put/call ratio for the nearest expiration.

        Answers *where* the options positioning sits — the put-OI wall (the strike with the most
        put open interest) often acts as **support** (sellers of those puts defend it), the call-OI
        wall as **resistance**, and **max pain** is the strike where the most option value expires
        worthless, which price often gravitates toward into expiration. Best-effort and None-safe.
        """
        try:
            ticker = yf.Ticker(symbol)
            dates = ticker.options
            if not dates:
                return None
            expiration = dates[0]
            chain = ticker.option_chain(expiration)
            calls, puts = chain.calls.copy(), chain.puts.copy()
            if calls.empty and puts.empty:
                return None

            for df in (calls, puts):
                for col in ("openInterest", "volume", "strike"):
                    if col in df.columns:
                        df[col] = df[col].fillna(0)

            if spot is None:
                try:
                    spot = float(ticker.fast_info["lastPrice"])
                except Exception:
                    spot = None

            call_oi = {float(r.strike): float(r.openInterest) for r in calls.itertuples()}
            put_oi = {float(r.strike): float(r.openInterest) for r in puts.itertuples()}
            strikes = sorted(set(call_oi) | set(put_oi))

            def _wall(df):
                if df.empty or df["openInterest"].max() <= 0:
                    return None
                row = df.loc[df["openInterest"].idxmax()]
                return {"strike": float(row["strike"]), "oi": int(row["openInterest"]),
                        "volume": int(row["volume"])}

            call_wall = _wall(calls)
            put_wall = _wall(puts)

            # Max pain: the strike that minimizes total intrinsic value paid to option holders
            # (calls below it + puts above it). Small chains, so the O(n²) sweep is cheap.
            def _pain(K: float) -> float:
                tot = 0.0
                for s, oi in call_oi.items():
                    if K > s:
                        tot += (K - s) * oi
                for s, oi in put_oi.items():
                    if K < s:
                        tot += (s - K) * oi
                return tot

            max_pain = min(strikes, key=_pain) if strikes else None

            total_call_oi = sum(call_oi.values())
            total_put_oi = sum(put_oi.values())
            pc_oi_ratio = (total_put_oi / total_call_oi) if total_call_oi else None

            return {
                "expiration": expiration,
                "spot": spot,
                "max_pain": max_pain,
                "call_wall": call_wall,
                "put_wall": put_wall,
                "pc_oi_ratio": pc_oi_ratio,
                "total_call_oi": int(total_call_oi),
                "total_put_oi": int(total_put_oi),
                "oi_by_strike": [
                    {"strike": s, "call_oi": call_oi.get(s, 0.0), "put_oi": put_oi.get(s, 0.0)}
                    for s in strikes
                ],
            }
        except Exception:
            return None

    # ── Earnings IV Crush Prediction ───────────────────────────────────────────

    def fetch_earnings_date(self, symbol: str) -> Optional[datetime]:
        """Get next earnings date."""
        try:
            info = yf.Ticker(symbol).info
            earn_date = info.get("nextEarningsDate") or info.get("lastEarningsDate")
            if isinstance(earn_date, (int, float)):
                return datetime.fromtimestamp(earn_date)
            return None
        except Exception:
            return None

    def estimate_iv_crush(self, symbol: str, current_iv: float) -> Optional[Dict[str, float]]:
        """
        Estimate potential IV crush post-earnings.
        Post-earnings IV typically drops 20-50% depending on implied move.
        """
        earn_date = self.fetch_earnings_date(symbol)
        if not earn_date:
            return None

        days_to_earnings = (earn_date - datetime.now()).days
        if days_to_earnings < 0 or days_to_earnings > 30:
            return None

        # Rule of thumb: earnings moves are ~1% × sqrt(days_to_earnings) / 100
        estimated_move_pct = (current_iv * math.sqrt(days_to_earnings / 252)) * 0.5
        post_earnings_iv = current_iv * 0.7  # 30% crush estimate

        return {
            "days_to_earnings": days_to_earnings,
            "estimated_move_pct": estimated_move_pct,
            "current_iv": current_iv,
            "post_earnings_iv_estimate": post_earnings_iv,
            "iv_crush_pct": (current_iv - post_earnings_iv) / current_iv * 100,
        }
