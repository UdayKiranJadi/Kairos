import argparse
import asyncio

from app.agents.prediction_agent import PredictionAgent
from app.db.session import AsyncSessionLocal


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        required=True,
        help="Symbol to predict, example: AAPL",
    )
    parser.add_argument(
        "--model-path",
        required=False,
        default=None,
        help="Optional model artifact path.",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    async with AsyncSessionLocal() as session:
        agent = PredictionAgent(session)

        prediction = await agent.predict_latest(
            ticker=args.symbol,
            model_path=args.model_path,
        )

    print("Prediction stored")
    print("-----------------")
    print(f"Symbol: {args.symbol.upper()}")
    print(f"Timestamp: {prediction.timestamp}")
    print(f"Model: {prediction.model_name}")
    print(f"Version: {prediction.model_version}")
    print(f"Horizon minutes: {prediction.horizon_minutes}")
    print(f"Probability up: {prediction.probability_up:.4f}")
    print(f"Predicted class: {prediction.predicted_class}")
    print(f"Confidence: {prediction.confidence:.4f}")
    print(f"Predicted return proxy: {prediction.predicted_return:.4f}")


if __name__ == "__main__":
    asyncio.run(main())