"""
Market structure analysis: swing levels, support/resistance, order flow.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class MarketStructure:
    """Identifies support/resistance, swing pivots, and market structure."""

    @staticmethod
    def find_swing_levels(
        highs: List[float],
        lows: List[float],
        lookback: int = 20,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Find recent swing high and swing low.
        Swing high: bar with higher highs on both sides
        Swing low: bar with lower lows on both sides
        """
        h = np.array(highs[-lookback:])
        lo = np.array(lows[-lookback:])

        swing_high = None
        swing_low = None

        # Find most recent swing high
        for i in range(2, len(h) - 2):
            if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > h[i - 2] and h[i] > h[i + 2]:
                swing_high = float(h[i])
                break

        # Find most recent swing low
        for i in range(2, len(lo) - 2):
            if lo[i] < lo[i - 1] and lo[i] < lo[i + 1] and lo[i] < lo[i - 2] and lo[i] < lo[i + 2]:
                swing_low = float(lo[i])
                break

        return swing_high, swing_low

    @staticmethod
    def find_consolidation(
        highs: List[float],
        lows: List[float],
        lookback: int = 20,
    ) -> Optional[Dict[str, float]]:
        """
        Identify consolidation (tight range for N bars).
        Returns dict with range_pct, consolidation_high, consolidation_low.
        """
        h = np.array(highs[-lookback:])
        lo = np.array(lows[-lookback:])

        cons_high = float(np.max(h))
        cons_low = float(np.min(lo))
        range_pct = (cons_high - cons_low) / cons_low

        # Consolidation is when range_pct < 3%
        if range_pct < 0.03:
            return {
                "consolidation_high": cons_high,
                "consolidation_low": cons_low,
                "range_pct": range_pct,
            }
        return None

    @staticmethod
    def is_breakout_setup(
        current_price: float,
        consolidation_high: float,
        consolidation_low: float,
        range_pct: float,
    ) -> Optional[str]:
        """
        Check if price is breaking out of consolidation.
        Returns 'long' (above high), 'short' (below low), or None.
        """
        above_high = current_price > consolidation_high * 1.005
        below_low = current_price < consolidation_low * 0.995

        if above_high:
            return "long"
        elif below_low:
            return "short"
        return None

    @staticmethod
    def identify_order_flow_imbalance(
        closes: List[float],
        volumes: List[float],
        lookback: int = 20,
    ) -> Optional[Dict[str, float]]:
        """
        Detect order flow imbalance: more volume on up moves than down.
        Returns dict with up_vol_ratio.
        """
        c = np.array(closes[-lookback:])
        v = np.array(volumes[-lookback:])

        up_volume = v[1:][np.diff(c) > 0].sum()
        down_volume = v[1:][np.diff(c) < 0].sum()
        total_volume = v[1:].sum()

        if total_volume == 0:
            return None

        up_vol_pct = up_volume / total_volume
        down_vol_pct = down_volume / total_volume

        # Imbalance when one side has >60% of volume
        if up_vol_pct > 0.60:
            return {"flow": "bullish", "up_vol_pct": up_vol_pct}
        elif down_vol_pct > 0.60:
            return {"flow": "bearish", "down_vol_pct": down_vol_pct}

        return None

    @staticmethod
    def calculate_supply_demand_levels(
        highs: List[float],
        lows: List[float],
        volumes: List[float],
        lookback: int = 60,
    ) -> Dict[str, List[float]]:
        """
        Identify supply (resistance) and demand (support) zones
        based on high-volume nodes and price rejection.
        """
        h = np.array(highs[-lookback:])
        lo = np.array(lows[-lookback:])
        v = np.array(volumes[-lookback:])

        # High volume nodes
        v_ma = np.mean(v)
        high_vol_bars = np.where(v > v_ma * 1.5)[0]

        supply_zones = []
        demand_zones = []

        for idx in high_vol_bars:
            if idx > 0:
                # If high-vol bar closes near high = supply
                if h[idx] > (h[idx] + lo[idx]) / 2:
                    supply_zones.append(float(h[idx]))
                # If high-vol bar closes near low = demand
                else:
                    demand_zones.append(float(lo[idx]))

        return {
            "supply": sorted(set(supply_zones), reverse=True)[:3],
            "demand": sorted(set(demand_zones))[:3],
        }

    @staticmethod
    def is_liquidity_zone(
        current_price: float,
        recent_highs: List[float],
        recent_lows: List[float],
        tolerance_pct: float = 0.02,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if current price is near previous lows (liquidity zones
        where stops cluster and orders accumulate).
        Returns (is_in_zone, zone_type: 'support' or 'resistance')
        """
        recent_h = np.array(recent_highs)
        recent_lo = np.array(recent_lows)

        # Check distance to recent swing lows (demand)
        min_dist_to_low = np.min(np.abs(current_price - recent_lo))
        is_near_low = min_dist_to_low / current_price < tolerance_pct
        if is_near_low:
            return True, "support"

        # Check distance to recent swing highs (supply)
        min_dist_to_high = np.min(np.abs(current_price - recent_h))
        is_near_high = min_dist_to_high / current_price < tolerance_pct
        if is_near_high:
            return True, "resistance"

        return False, None
