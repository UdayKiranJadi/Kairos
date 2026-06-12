from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.prediction_agent import PredictionAgent
from app.db.session import get_db

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/run/latest")
async def run_latest_prediction(
    symbol: str = Query(..., description="Example: AAPL"),
    model_path: str | None = Query(
        default=None,
        description="Optional model path. Defaults to artifacts/models/{symbol}_intraday_direction_v0_1.joblib",
    ),
    db: AsyncSession = Depends(get_db),
):
    agent = PredictionAgent(db)

    prediction = await agent.predict_latest(
        ticker=symbol,
        model_path=model_path,
    )

    return {
        "status": "ok",
        "symbol": symbol.upper(),
        "prediction": {
            "timestamp": prediction.timestamp,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "horizon_minutes": prediction.horizon_minutes,
            "probability_up": prediction.probability_up,
            "predicted_class": prediction.predicted_class,
            "predicted_return": prediction.predicted_return,
            "confidence": prediction.confidence,
        },
    }


@router.get("/{symbol}")
async def get_recent_predictions(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    agent = PredictionAgent(db)
    predictions = await agent.list_recent_predictions(symbol, limit)

    return {
        "symbol": symbol.upper(),
        "count": len(predictions),
        "predictions": [
            {
                "timestamp": item.timestamp,
                "model_name": item.model_name,
                "model_version": item.model_version,
                "horizon_minutes": item.horizon_minutes,
                "probability_up": item.probability_up,
                "predicted_class": item.predicted_class,
                "predicted_return": item.predicted_return,
                "confidence": item.confidence,
            }
            for item in predictions
        ],
    }