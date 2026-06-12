from datetime import datetime

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from app.core.config import settings


class MarketDataClient:
    """
    Thin wrapper around Alpaca market data.

    This class should only fetch external data.
    It should not write to the database.
    """

    def __init__(self):
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise ValueError(
                "Missing Alpaca API keys. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to your .env file."
            )

        self.client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    def get_intraday_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Minute,
    ):
        feed = DataFeed.IEX if settings.alpaca_data_feed.lower() == "iex" else DataFeed.SIP

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=feed,
        )

        return self.client.get_stock_bars(request)