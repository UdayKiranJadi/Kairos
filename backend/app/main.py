from fastapi import FastAPI
from sqlalchemy import text

from app.api.routes_decisions import router as decisions_router
from app.api.routes_features import router as features_router
from app.api.routes_market_data import router as market_data_router
from app.api.routes_predictions import router as predictions_router
from app.db.session import AsyncSessionLocal

app = FastAPI(title="TradeOps AI")

app.include_router(market_data_router)
app.include_router(features_router)
app.include_router(predictions_router)
app.include_router(decisions_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "tradeops-ai-backend",
        "mode": "paper",
    }


@app.get("/health/db")
async def database_health_check():
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT 1"))
        value = result.scalar()

    return {
        "status": "ok",
        "database": "connected",
        "result": value,
    }