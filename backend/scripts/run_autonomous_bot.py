import argparse
import asyncio
from app.execution.trading_bot import AutonomousBot

def parse_args():
    parser = argparse.ArgumentParser(description="Start the autonomous trading bot loop.")
    parser.add_argument(
        "--symbols", 
        type=str, 
        default="AAPL,MSFT,NVDA",
        help="Comma-separated list of symbols to trade."
    )
    return parser.parse_args()

async def main():
    args = parse_args()
    symbols_list = [s.strip() for s in args.symbols.split(",")]
    
    bot = AutonomousBot(symbols=symbols_list)
    await bot.start()

if __name__ == "__main__":
    # Ensure graceful exit on Ctrl+C
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")