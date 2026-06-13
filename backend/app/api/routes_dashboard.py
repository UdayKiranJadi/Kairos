from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.db.models import PortfolioSnapshot, PaperOrder, Symbol

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/summary")
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    # 1. Get the most recent portfolio snapshot
    snap_result = await db.execute(
        select(PortfolioSnapshot).order_by(PortfolioSnapshot.timestamp.desc()).limit(1)
    )
    latest_snapshot = snap_result.scalar_one_or_none()

    # 2. Get the 10 most recent paper orders
    orders_result = await db.execute(
        select(PaperOrder, Symbol.ticker)
        .join(Symbol, PaperOrder.symbol_id == Symbol.id)
        .order_by(PaperOrder.timestamp.desc())
        .limit(10)
    )
    
    recent_orders = []
    for order, ticker in orders_result.all():
        recent_orders.append({
            "id": order.id,
            "ticker": ticker,
            "side": order.side,
            "qty": order.quantity,
            "status": order.status,
            "price": order.submitted_price,
            "timestamp": order.timestamp.isoformat()
        })

    return {
        "portfolio": {
            "equity": latest_snapshot.equity if latest_snapshot else 0.0,
            "cash": latest_snapshot.cash if latest_snapshot else 0.0,
            "daily_pnl": latest_snapshot.daily_pnl if latest_snapshot else 0.0,
            "drawdown_pct": latest_snapshot.drawdown_pct if latest_snapshot else 0.0,
        },
        "recent_orders": recent_orders
    }