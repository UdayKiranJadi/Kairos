import asyncio

from sqlalchemy import select

from app.db.models import Symbol
from app.db.session import AsyncSessionLocal


STARTER_SYMBOLS = [
    {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
    {"ticker": "AMD", "name": "Advanced Micro Devices Inc.", "exchange": "NASDAQ"},
]


async def main():
    async with AsyncSessionLocal() as session:
        for item in STARTER_SYMBOLS:
            existing = await session.execute(
                select(Symbol).where(Symbol.ticker == item["ticker"])
            )
            symbol = existing.scalar_one_or_none()

            if symbol is None:
                session.add(Symbol(**item))

        await session.commit()

    print("Starter symbols seeded.")


if __name__ == "__main__":
    asyncio.run(main())