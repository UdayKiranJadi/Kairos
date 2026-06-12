from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.bar_service import BarService
from app.db.session import get_db

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.post("/historical-bars/load")
async def load_historical_bars(
    symbols: str = Query(..., description="Comma-separated symbols, example: AAPL,MSFT,NVDA"),
    start: datetime = Query(..., description="Start datetime, example: 2026-06-01T13:30:00"),
    end: datetime = Query(..., description="End datetime, example: 2026-06-01T20:00:00"),
    db: AsyncSession = Depends(get_db),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    service = BarService(db)
    inserted_count = await service.store_intraday_bars(
        symbols=symbol_list,
        start=start,
        end=end,
    )

    return {
        "status": "ok",
        "symbols": symbol_list,
        "inserted_bars": inserted_count,
    }


@router.get("/historical-bars/{symbol}")
async def get_recent_bars(
    symbol: str,
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    service = BarService(db)
    bars = await service.list_recent_bars(symbol, limit)

    return {
        "symbol": symbol.upper(),
        "count": len(bars),
        "bars": [
            {
                "timestamp": bar.timestamp,
                "timeframe": bar.timeframe,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ],
    }