import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TradingConfig:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str
    offline_mode: bool = False

    min_market_cap: int = 500_000_000
    min_price: float = 10.0
    min_avg_volume: int = 2_000_000
    max_spread_pct: float = 0.001
    gap_pct_threshold: float = 0.04
    premarket_volume_threshold: int = 100_000
    low_float_threshold: int = 50_000_000
    trend_rsi_min: float = 40.0
    trend_rsi_max: float = 65.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.06
    latency_ms: int = 50
    backtest_slippage_ticks: int = 1
    # Backtest cost model (basis points). slippage applied to each fill; commission per side.
    slippage_bps: float = 5.0
    commission_bps: float = 1.0

    # Optional third-party data-provider keys (used only on their FREE tier, hard-capped by
    # swingtradeapp.ratelimit so we never exceed the free limit). Blank = provider disabled.
    polygon_api_key: str = ''

    @classmethod
    def load_from_env(cls) -> 'TradingConfig':
        env_path = Path('.').joinpath('.env')
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if v and k not in os.environ:
                        os.environ[k] = v

        cfg = cls(
            alpaca_api_key=os.environ.get('ALPACA_API_KEY', ''),
            alpaca_secret_key=os.environ.get('ALPACA_SECRET_KEY', ''),
            alpaca_base_url=os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
        )
        # Optional backtest cost overrides from env.
        for field_name in ('slippage_bps', 'commission_bps'):
            raw = os.environ.get(field_name.upper())
            if raw:
                try:
                    setattr(cfg, field_name, float(raw))
                except ValueError:
                    pass
        # Optional free-tier data-provider keys.
        cfg.polygon_api_key = os.environ.get('POLYGON_API_KEY', '')
        if not cfg.alpaca_api_key:
            cfg.offline_mode = True
        return cfg

    def validate(self) -> None:
        if self.offline_mode:
            return
        if not (self.alpaca_api_key and self.alpaca_secret_key):
            raise ValueError('Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.')
