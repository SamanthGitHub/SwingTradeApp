import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BracketOrder:
    symbol: str
    qty: int
    side: str
    limit_price: float
    stop_loss_price: float
    take_profit_price: float
    time_in_force: str = 'day'


class AlpacaExecutionBridge:
    def __init__(self, config: Any) -> None:
        self.config = config
        # Only attempt to initialize the Alpaca client when API credentials are present
        if getattr(config, 'alpaca_api_key', None) and getattr(config, 'alpaca_secret_key', None):
            try:
                import alpaca_trade_api as tradeapi

                self.client = tradeapi.REST(
                    key_id=config.alpaca_api_key,
                    secret_key=config.alpaca_secret_key,
                    base_url=config.alpaca_base_url,
                    api_version='v2',
                )
            except Exception:
                self.client = None
        else:
            self.client = None

    def get_account(self) -> Dict[str, Any]:
        if self.client is None:
            return {'error': 'alpaca_trade_api not installed'}
        return self.client.get_account()._raw

    def submit_bracket_order(self, order: BracketOrder) -> Dict[str, Any]:
        logger.debug('Submitting bracket order: %s', order)
        if self.client is None:
            return {'error': 'alpaca_trade_api not installed'}

        return self.client.submit_order(
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            type='limit',
            time_in_force=order.time_in_force,
            limit_price=order.limit_price,
            order_class='bracket',
            stop_loss={'stop_price': order.stop_loss_price},
            take_profit={'limit_price': order.take_profit_price},
        )._raw
