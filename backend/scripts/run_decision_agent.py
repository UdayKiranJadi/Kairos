import argparse
import asyncio

from app.agents.decision_agent import DecisionAgent
from app.db.session import AsyncSessionLocal


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--symbol", required=True)
    parser.add_argument("--portfolio-value", type=float, default=10_000)
    parser.add_argument("--daily-loss-pct", type=float, default=0.0)
    parser.add_argument("--total-drawdown-pct", type=float, default=0.0)
    parser.add_argument("--trades-today", type=int, default=0)
    parser.add_argument("--open-positions", type=int, default=0)
    parser.add_argument("--trading-mode", default="paper")

    return parser.parse_args()


async def main():
    args = parse_args()

    async with AsyncSessionLocal() as session:
        agent = DecisionAgent(session)

        result = await agent.evaluate_latest_prediction(
            ticker=args.symbol,
            portfolio_value=args.portfolio_value,
            daily_loss_pct=args.daily_loss_pct,
            total_drawdown_pct=args.total_drawdown_pct,
            trades_today=args.trades_today,
            open_positions=args.open_positions,
            trading_mode=args.trading_mode,
        )

    print("Decision evaluated")
    print("------------------")
    print(f"Symbol: {result['symbol']}")
    print(f"Probability up: {result['prediction']['probability_up']:.4f}")
    print(f"Confidence: {result['trade_decision']['confidence']:.4f}")
    print(f"Action: {result['trade_decision']['action']}")
    print(f"Decision reason: {result['trade_decision']['reason']}")
    print(f"Risk approved: {result['risk_decision']['approved']}")
    print(f"Risk reason: {result['risk_decision']['reason']}")
    print(f"Max position value: {result['risk_decision']['max_position_value']}")


if __name__ == "__main__":
    asyncio.run(main())