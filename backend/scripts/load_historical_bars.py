import argparse
import asyncio
from datetime import datetime

from app.data.bar_service import BarService
from app.db.session import AsyncSessionLocal


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols, example: AAPL,MSFT,NVDA",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime, example: 2026-06-01T13:30:00",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End datetime, example: 2026-06-01T20:00:00",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    async with AsyncSessionLocal() as session:
        service = BarService(session)
        inserted = await service.store_intraday_bars(
            symbols=symbols,
            start=start,
            end=end,
        )

    print(f"Inserted {inserted} bars for {symbols}.")


if __name__ == "__main__":
    asyncio.run(main())