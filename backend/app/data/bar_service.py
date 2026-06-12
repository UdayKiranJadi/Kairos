from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.market_data_client import MarketDataClient
from app.db.models import MarketBar, Symbol


def to_utc(dt: datetime) -> datetime:
    """
    Normalize every datetime to timezone-aware UTC.

    If a datetime is naive, we assume it is already UTC.
    This keeps our backend consistent for intraday trading.
    """

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


class BarService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.market_data_client = MarketDataClient()

    async def get_or_create_symbol(self, ticker: str) -> Symbol:
        ticker = ticker.strip().upper()

        result = await self.db.execute(
            select(Symbol).where(Symbol.ticker == ticker)
        )
        symbol = result.scalar_one_or_none()

        if symbol:
            return symbol

        symbol = Symbol(
            ticker=ticker,
            name=None,
            exchange=None,
            is_active=True,
        )
        self.db.add(symbol)
        await self.db.flush()

        return symbol

    async def store_intraday_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
    ) -> int:
        """
        Fetch bars from Alpaca and store them in market_bars.

        Returns the number of newly inserted bars.
        """

        symbols = [s.strip().upper() for s in symbols if s.strip()]
        start = to_utc(start)
        end = to_utc(end)

        bar_set = self.market_data_client.get_intraday_bars(
            symbols=symbols,
            start=start,
            end=end,
        )

        inserted_count = 0

        for ticker in symbols:
            symbol = await self.get_or_create_symbol(ticker)

            bars = bar_set.data.get(ticker, [])

            for bar in bars:
                bar_timestamp = to_utc(bar.timestamp)

                existing = await self.db.execute(
                    select(MarketBar).where(
                        MarketBar.symbol_id == symbol.id,
                        MarketBar.timeframe == timeframe,
                        MarketBar.timestamp == bar_timestamp,
                    )
                )
                existing_bar = existing.scalar_one_or_none()

                if existing_bar:
                    continue

                market_bar = MarketBar(
                    symbol_id=symbol.id,
                    timeframe=timeframe,
                    timestamp=bar_timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                )

                self.db.add(market_bar)
                inserted_count += 1

        await self.db.commit()
        return inserted_count

    async def list_recent_bars(
        self,
        ticker: str,
        limit: int = 100,
    ) -> list[MarketBar]:
        ticker = ticker.strip().upper()

        result = await self.db.execute(
            select(Symbol).where(Symbol.ticker == ticker)
        )
        symbol = result.scalar_one_or_none()

        if symbol is None:
            return []

        result = await self.db.execute(
            select(MarketBar)
            .where(MarketBar.symbol_id == symbol.id)
            .order_by(MarketBar.timestamp.desc())
            .limit(limit)
        )

        return list(result.scalars().all())