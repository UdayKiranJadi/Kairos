from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from app.core.config import settings

class PaperBroker:
    """
    Thin wrapper around Alpaca Trading API.
    Handles the actual network requests to place paper trades.
    """
    def __init__(self):
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise ValueError("Missing Alpaca API keys in .env")

        self.client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper
        )

    def get_account(self):
        return self.client.get_account()

    def get_open_position(self, symbol: str):
        try:
            return self.client.get_open_position(symbol)
        except Exception:
            # Alpaca throws an error if the position doesn't exist
            return None

    def submit_market_order(self, symbol: str, qty: float, side: OrderSide):
        """
        Submits a market order. We use market orders for intraday momentum 
        trading to guarantee execution, though slippage is a factor.
        """
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        return self.client.submit_order(order_data=request)

    def close_all_positions(self):
        """Emergency stop: liquidates everything."""
        return self.client.close_all_positions(cancel_orders=True)
    
    def get_all_positions(self):
        """Fetches all current open positions from Alpaca."""
        try:
            return self.client.get_all_positions()
        except Exception:
            return []