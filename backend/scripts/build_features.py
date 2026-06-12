import argparse
import asyncio
from datetime import datetime

from app.db.session import AsyncSessionLocal
from app.features.feature_builder import FeatureBuilder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols, example: AAPL,MSFT,NVDA",
    )
    parser.add_argument(
        "--start",
        required=False,
        default=None,
        help="Optional start datetime, example: 2026-06-01T13:30:00",
    )
    parser.add_argument(
        "--end",
        required=False,
        default=None,
        help="Optional end datetime, example: 2026-06-01T20:00:00",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = datetime.fromisoformat(args.start) if args.start else None
    end = datetime.fromisoformat(args.end) if args.end else None

    async with AsyncSessionLocal() as session:
        builder = FeatureBuilder(session)

        for symbol in symbols:
            inserted = await builder.store_features_for_symbol(
                ticker=symbol,
                start=start,
                end=end,
            )

            print(f"Inserted {inserted} feature rows for {symbol}.")


if __name__ == "__main__":
    asyncio.run(main())