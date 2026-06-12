from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.decision_agent import DecisionAgent
from app.db.session import get_db

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.post("/evaluate/latest")
async def evaluate_latest_decision(
    symbol: str = Query(..., description="Example: AAPL"),
    portfolio_value: float = Query(default=10_000),
    daily_loss_pct: float = Query(default=0.0),
    total_drawdown_pct: float = Query(default=0.0),
    trades_today: int = Query(default=0),
    open_positions: int = Query(default=0),
    trading_mode: str = Query(default="paper"),
    db: AsyncSession = Depends(get_db),
):
    agent = DecisionAgent(db)

    result = await agent.evaluate_latest_prediction(
        ticker=symbol,
        portfolio_value=portfolio_value,
        daily_loss_pct=daily_loss_pct,
        total_drawdown_pct=total_drawdown_pct,
        trades_today=trades_today,
        open_positions=open_positions,
        trading_mode=trading_mode,
    )

    return {
        "status": "ok",
        **result,
    }