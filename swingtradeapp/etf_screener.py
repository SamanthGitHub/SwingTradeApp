"""
ETF screening with same signal logic as stocks.
Tracks sector rotation, volatility, and macro themes.
"""

from typing import Dict, List, Optional

import yfinance as yf

from .signals import TrendSignalGenerator, Signal


class ETFScreener:
    """Apply trading signals to ETFs for sector/macro trades."""

    # Major ETF universe by category
    SECTOR_ETFS = {
        "XLK": "Technology",
        "XLV": "Healthcare",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLRE": "Real Estate",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLC": "Communication Services",
    }

    BROAD_MARKET_ETFS = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq-100",
        "DIA": "Dow Jones",
        "IWM": "Russell 2000 Small-cap",
    }

    VOLATILITY_ETFS = {
        "^VIX": "VIX Index",
        "UVXY": "2x VIX Short-term",
        "SVXY": "-0.5x VIX",
    }

    COMMODITY_ETFS = {
        "GLD": "Gold",
        "SLV": "Silver",
        "USO": "Oil",
        "DBC": "Commodities",
    }

    BOND_ETFS = {
        "TLT": "20+ Year Treasuries",
        "BND": "Total Bond Market",
        "HYG": "High-Yield Bonds",
    }

    def __init__(self, config):
        self.config = config
        self.signal_generator = TrendSignalGenerator(config)

    def get_all_etfs(self) -> Dict[str, str]:
        """Return all ETFs organized by category."""
        return {
            **self.SECTOR_ETFS,
            **self.BROAD_MARKET_ETFS,
            **self.VOLATILITY_ETFS,
            **self.COMMODITY_ETFS,
            **self.BOND_ETFS,
        }

    def screen_sector_rotation(self) -> Optional[Dict]:
        """
        Compare sector ETF performance to detect rotation.
        Example: If XLV outperforming XLK, rotation to defensive.
        """
        try:
            sector_data = {}
            for etf, sector in self.SECTOR_ETFS.items():
                hist = yf.Ticker(etf).history(period="60d")
                if not hist.empty:
                    returns_30d = (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
                    sector_data[sector] = returns_30d

            if not sector_data:
                return None

            # Rank sectors
            ranked = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
            return {
                "best_sector": ranked[0][0],
                "worst_sector": ranked[-1][0],
                "best_return_pct": ranked[0][1] * 100,
                "worst_return_pct": ranked[-1][1] * 100,
                "rotation_signal": "growth" if "Technology" in ranked[0][0] else "defensive",
            }
        except Exception:
            return None

    def fetch_etf_signal(self, etf_symbol: str) -> Optional[Signal]:
        """Generate signal for single ETF."""
        try:
            hist = yf.Ticker(etf_symbol).history(period="120d")
            if hist.empty or len(hist) < 26:
                return None

            closes = hist["Close"].tolist()
            volumes = hist["Volume"].tolist()
            highs = hist["High"].tolist() if "High" in hist.columns else closes
            lows = hist["Low"].tolist() if "Low" in hist.columns else closes

            signal = self.signal_generator.build_signal(
                etf_symbol, closes, volumes, highs=highs, lows=lows
            )
            return signal
        except Exception:
            return None

    def screen_all_etfs(self, categories: Optional[List[str]] = None) -> Dict[str, Optional[Signal]]:
        """
        Generate signals for all or selected ETFs.
        categories: ['sectors', 'broad_market', 'volatility', 'commodities', 'bonds']
        """
        etf_map = {
            "sectors": self.SECTOR_ETFS,
            "broad_market": self.BROAD_MARKET_ETFS,
            "volatility": self.VOLATILITY_ETFS,
            "commodities": self.COMMODITY_ETFS,
            "bonds": self.BOND_ETFS,
        }

        etfs_to_scan = {}
        if categories is None:
            categories = list(etf_map.keys())

        for cat in categories:
            if cat in etf_map:
                etfs_to_scan.update(etf_map[cat])

        signals = {}
        for etf in etfs_to_scan.keys():
            signal = self.fetch_etf_signal(etf)
            signals[etf] = signal

        return signals

    def detect_macro_themes(self) -> Dict[str, Optional[Signal]]:
        """
        Scan macro themes: growth vs defensive, inflation, risk-off.
        Returns signals for key macro indicators.
        """
        macro_signals = {}

        # Growth indicator: QQQ vs TLT (tech vs bonds)
        qqq_sig = self.fetch_etf_signal("QQQ")
        tlt_sig = self.fetch_etf_signal("TLT")

        if qqq_sig and tlt_sig:
            if qqq_sig.signal_type == "long" and tlt_sig.signal_type == "short":
                macro_signals["growth_theme"] = qqq_sig
            elif tlt_sig.signal_type == "long" and qqq_sig.signal_type == "short":
                macro_signals["defensive_theme"] = tlt_sig

        # Inflation hedge: GLD (gold)
        gld_sig = self.fetch_etf_signal("GLD")
        if gld_sig and gld_sig.signal_type == "long":
            macro_signals["inflation_hedge"] = gld_sig

        # Risk-off: VIX
        vix_sig = self.fetch_etf_signal("^VIX")
        if vix_sig and vix_sig.signal_type == "long":
            macro_signals["risk_off_signal"] = vix_sig

        return macro_signals
