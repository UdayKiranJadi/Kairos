"""
run_autonomous_bot.py

WHY asyncio.gather():
The WS stream and the trading bot are two separate async
tasks that need to run concurrently on the same event loop.

  stream_bars() — wakes on each incoming bar (~every 60s)
  bot.start()   — wakes every 60s to run the pipeline

gather() starts both and keeps them alive. If one crashes,
the other keeps running. Exceptions are caught inside each.
"""

import argparse
import asyncio

from app.data.stream_client import stream_bars
from app.execution.trading_bot import AutonomousBot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL,NVDA",
        help="Comma-separated symbols to trade.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    bot = AutonomousBot(symbols=symbols)

    # Run stream + bot loop concurrently.
    # If market is closed, stream connects but receives no bars.
    # Bot loop detects closed market and sleeps — no wasted work.
    await asyncio.gather(
        stream_bars(symbols),   # Layer 1: live data → Redis
        bot.start(),            # Layers 2-4: predict → risk → execute
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKairos stopped.")