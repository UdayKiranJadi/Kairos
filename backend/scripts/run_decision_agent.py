import argparse
import asyncio

from app.agents.decision_agent import DecisionAgent
from app.agents.portfolio_agent import PortfolioAgent
from app.db.session import AsyncSessionLocal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--trading-mode", default="paper")
    return parser.parse_args()

async def main():
    args = parse_args()

    async with AsyncSessionLocal() as session:
        # 1. Sync live data first
        portfolio_agent = PortfolioAgent(session)
        live_state = await portfolio_agent.sync_and_get_state()
        
        # 2. Pass live data into the decision agent
        decision_agent = DecisionAgent(session)
        result = await decision_agent.evaluate_latest_prediction(
            ticker=args.symbol,
            portfolio_value=live_state["portfolio_value"],
            daily_loss_pct=live_state["daily_loss_pct"],
            total_drawdown_pct=live_state["total_drawdown_pct"],
            trades_today=0, # We will track trades_today in the main loop tomorrow
            open_positions=live_state["open_positions"],
            trading_mode=args.trading_mode,
        )

    print("Decision evaluated using LIVE PAPER DATA")
    print("----------------------------------------")
    print(f"Live Equity: ${live_state['portfolio_value']:.2f}")
    print(f"Open Positions: {live_state['open_positions']}")
    print(f"Action: {result['trade_decision']['action']}")
    print(f"Risk approved: {result['risk_decision']['approved']}")
    print(f"Risk reason: {result['risk_decision']['reason']}")

if __name__ == "__main__":
    asyncio.run(main())