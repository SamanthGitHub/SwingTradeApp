"""
ETF screening with same signal logic as stocks.
Tracks sector rotation, volatility, and macro themes.
"""

from typing import Dict, List, Optional

import yfinance as yf

from .signals import TrendSignalGenerator, Signal


class ETFScreener:
    """Apply trading signals to ETFs for sector/macro trades."""

    # ── Major ETF universe by category (curated "famous ETFs", manually maintained) ──
    # Sector list spans every GICS sector plus the headline industry ETFs within each, across
    # the big providers (SPDR / Vanguard / iShares / Fidelity / VanEck / Global X).
    SECTOR_ETFS = {
        # Technology
        "XLK": "Technology — SPDR",
        "VGT": "Technology — Vanguard",
        "FTEC": "Technology — Fidelity",
        "IYW": "Technology — iShares US",
        "IGV": "Software — iShares Expanded Tech-Software",
        "SMH": "Semiconductors — VanEck",
        "SOXX": "Semiconductors — iShares",
        "PSI": "Semiconductors — Invesco",
        "SKYY": "Cloud Computing — First Trust",
        "HACK": "Cybersecurity — ETFMG",
        "CIBR": "Cybersecurity — First Trust",
        "FDN": "Internet — First Trust Dow Jones",
        # Communication Services
        "XLC": "Communication Services — SPDR",
        "VOX": "Communication Services — Vanguard",
        "FCOM": "Communication Services — Fidelity",
        # Healthcare
        "XLV": "Healthcare — SPDR",
        "VHT": "Healthcare — Vanguard",
        "IYH": "Healthcare — iShares US",
        "IBB": "Biotech — iShares Nasdaq",
        "XBI": "Biotech — SPDR S&P",
        "IHI": "Medical Devices — iShares",
        # Financials
        "XLF": "Financials — SPDR",
        "VFH": "Financials — Vanguard",
        "IYF": "Financials — iShares US",
        "KRE": "Regional Banks — SPDR",
        "KBE": "Banks — SPDR",
        "KIE": "Insurance — SPDR",
        # Energy
        "XLE": "Energy — SPDR",
        "VDE": "Energy — Vanguard",
        "XOP": "Oil & Gas E&P — SPDR",
        "OIH": "Oil Services — VanEck",
        "AMLP": "MLP / Midstream — Alerian",
        "ICLN": "Clean Energy — iShares Global",
        "TAN": "Solar — Invesco",
        "URA": "Uranium — Global X",
        # Consumer Discretionary
        "XLY": "Consumer Discretionary — SPDR",
        "VCR": "Consumer Discretionary — Vanguard",
        "XRT": "Retail — SPDR",
        "XHB": "Homebuilders — SPDR",
        "ITB": "Home Construction — iShares",
        # Consumer Staples
        "XLP": "Consumer Staples — SPDR",
        "VDC": "Consumer Staples — Vanguard",
        "FSTA": "Consumer Staples — Fidelity",
        # Industrials
        "XLI": "Industrials — SPDR",
        "VIS": "Industrials — Vanguard",
        "ITA": "Aerospace & Defense — iShares",
        "XAR": "Aerospace & Defense — SPDR",
        "JETS": "Airlines — US Global",
        "IYT": "Transportation — iShares",
        # Materials
        "XLB": "Materials — SPDR",
        "VAW": "Materials — Vanguard",
        "GDX": "Gold Miners — VanEck",
        "GDXJ": "Junior Gold Miners — VanEck",
        "XME": "Metals & Mining — SPDR",
        "LIT": "Lithium & Battery — Global X",
        "COPX": "Copper Miners — Global X",
        # Real Estate
        "XLRE": "Real Estate — SPDR",
        "VNQ": "Real Estate — Vanguard",
        "SCHH": "REITs — Schwab",
        "IYR": "Real Estate — iShares US",
        # Utilities
        "XLU": "Utilities — SPDR",
        "VPU": "Utilities — Vanguard",
        "FUTY": "Utilities — Fidelity",
    }

    BROAD_MARKET_ETFS = {
        "SPY": "S&P 500 — SPDR",
        "VOO": "S&P 500 — Vanguard",
        "IVV": "S&P 500 — iShares",
        "SPLG": "S&P 500 — SPDR Portfolio",
        "RSP": "S&P 500 Equal Weight — Invesco",
        "VTI": "Total US Market — Vanguard",
        "ITOT": "Total US Market — iShares",
        "QQQ": "Nasdaq-100 — Invesco",
        "QQQM": "Nasdaq-100 — Invesco (low-cost)",
        "DIA": "Dow Jones — SPDR",
        "IWB": "Russell 1000 — iShares",
        "IWF": "Russell 1000 Growth — iShares",
        "IWD": "Russell 1000 Value — iShares",
        "MDY": "S&P MidCap 400 — SPDR",
        "IJH": "S&P MidCap 400 — iShares",
        "IWM": "Russell 2000 Small-cap — iShares",
        "IJR": "S&P SmallCap 600 — iShares",
        "VT": "Total World — Vanguard",
    }

    DIVIDEND_FACTOR_ETFS = {
        "SCHD": "Dividend — Schwab US Dividend",
        "VYM": "High Dividend — Vanguard",
        "VIG": "Dividend Growth — Vanguard",
        "DVY": "Select Dividend — iShares",
        "HDV": "High Dividend — iShares",
        "NOBL": "Dividend Aristocrats — ProShares",
        "VTV": "Value — Vanguard",
        "VUG": "Growth — Vanguard",
        "MTUM": "Momentum — iShares",
        "QUAL": "Quality — iShares",
        "USMV": "Min Volatility — iShares",
        "SPLV": "Low Volatility — Invesco",
    }

    THEMATIC_ETFS = {
        "ARKK": "Innovation — ARK",
        "ARKW": "Next-Gen Internet — ARK",
        "ARKQ": "Autonomous & Robotics — ARK",
        "ARKF": "Fintech — ARK",
        "ARKG": "Genomics — ARK",
        "BOTZ": "Robotics & AI — Global X",
        "ROBO": "Robotics & Automation — ROBO",
        "AIQ": "AI & Technology — Global X",
        "DRIV": "EVs & Autonomous — Global X",
        "FINX": "Fintech — Global X",
        "CLOU": "Cloud — Global X",
        "WCLD": "Cloud — WisdomTree",
        "BLOK": "Blockchain — Amplify",
        "ESPO": "Video Games & Esports — VanEck",
        "MJ": "Cannabis — ETFMG",
    }

    INTERNATIONAL_ETFS = {
        "VEA": "Developed Markets — Vanguard",
        "EFA": "EAFE Developed — iShares",
        "IEFA": "Core EAFE — iShares",
        "VWO": "Emerging Markets — Vanguard",
        "EEM": "Emerging Markets — iShares",
        "IEMG": "Core EM — iShares",
        "VXUS": "Total International — Vanguard",
        "ACWI": "All-World — iShares",
        "FXI": "China Large-Cap — iShares",
        "MCHI": "China — iShares",
        "KWEB": "China Internet — KraneShares",
        "EWJ": "Japan — iShares",
        "EWZ": "Brazil — iShares",
        "INDA": "India — iShares",
        "EWG": "Germany — iShares",
        "EWU": "United Kingdom — iShares",
        "EWT": "Taiwan — iShares",
        "EWY": "South Korea — iShares",
    }

    VOLATILITY_ETFS = {
        "^VIX": "VIX Index",
        "VXX": "VIX Short-term — iPath",
        "VIXY": "VIX Short-term — ProShares",
        "UVXY": "1.5x VIX Short-term — ProShares",
        "SVXY": "-0.5x VIX — ProShares",
    }

    COMMODITY_ETFS = {
        "GLD": "Gold — SPDR",
        "IAU": "Gold — iShares",
        "SLV": "Silver — iShares",
        "PPLT": "Platinum — abrdn",
        "USO": "Crude Oil — US Oil Fund",
        "BNO": "Brent Crude — US Brent",
        "UNG": "Natural Gas — US Nat Gas",
        "DBC": "Broad Commodities — Invesco",
        "PDBC": "Broad Commodities — Invesco (no K-1)",
        "DBA": "Agriculture — Invesco",
        "CPER": "Copper — US Copper",
    }

    BOND_ETFS = {
        "AGG": "US Aggregate — iShares",
        "BND": "Total Bond Market — Vanguard",
        "BNDX": "International Bonds — Vanguard",
        "TLT": "20+ Year Treasuries — iShares",
        "IEF": "7-10 Year Treasuries — iShares",
        "SHY": "1-3 Year Treasuries — iShares",
        "BIL": "1-3 Month T-Bills — SPDR",
        "LQD": "Investment-Grade Corp — iShares",
        "VCIT": "Intermediate Corp — Vanguard",
        "HYG": "High-Yield — iShares",
        "JNK": "High-Yield — SPDR",
        "TIP": "TIPS (Inflation) — iShares",
        "MUB": "Municipal — iShares",
        "EMB": "Emerging-Market Bonds — iShares",
    }

    CRYPTO_ETFS = {
        "IBIT": "Bitcoin — iShares",
        "FBTC": "Bitcoin — Fidelity",
        "GBTC": "Bitcoin — Grayscale",
        "BITB": "Bitcoin — Bitwise",
        "BITO": "Bitcoin Strategy (futures) — ProShares",
        "ETHA": "Ethereum — iShares",
        "ETHE": "Ethereum — Grayscale",
    }

    LEVERAGED_ETFS = {
        "TQQQ": "3x Nasdaq-100 Bull — ProShares",
        "SQQQ": "3x Nasdaq-100 Bear — ProShares",
        "UPRO": "3x S&P 500 Bull — ProShares",
        "SPXU": "3x S&P 500 Bear — ProShares",
        "SOXL": "3x Semiconductors Bull — Direxion",
        "SOXS": "3x Semiconductors Bear — Direxion",
        "TNA": "3x Small-cap Bull — Direxion",
        "TMF": "3x 20Y Treasury Bull — Direxion",
    }

    def __init__(self, config):
        self.config = config
        self.signal_generator = TrendSignalGenerator(config)

    def get_all_etfs(self) -> Dict[str, str]:
        """Return all ETFs organized by category."""
        return {
            **self.BROAD_MARKET_ETFS,
            **self.SECTOR_ETFS,
            **self.DIVIDEND_FACTOR_ETFS,
            **self.THEMATIC_ETFS,
            **self.INTERNATIONAL_ETFS,
            **self.VOLATILITY_ETFS,
            **self.COMMODITY_ETFS,
            **self.BOND_ETFS,
            **self.CRYPTO_ETFS,
            **self.LEVERAGED_ETFS,
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
            "dividend_factor": self.DIVIDEND_FACTOR_ETFS,
            "thematic": self.THEMATIC_ETFS,
            "international": self.INTERNATIONAL_ETFS,
            "volatility": self.VOLATILITY_ETFS,
            "commodities": self.COMMODITY_ETFS,
            "bonds": self.BOND_ETFS,
            "crypto": self.CRYPTO_ETFS,
            "leveraged": self.LEVERAGED_ETFS,
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
