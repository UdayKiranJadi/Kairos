from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.features.feature_builder import FeatureBuilder

router = APIRouter(prefix="/features", tags=["features"])


@router.post("/build")
async def build_features(
    symbols: str = Query(..., description="Comma-separated symbols, example: AAPL,MSFT,NVDA"),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    builder = FeatureBuilder(db)

    results = {}

    for symbol in symbol_list:
        inserted_count = await builder.store_features_for_symbol(
            ticker=symbol,
            start=start,
            end=end,
        )
        results[symbol] = inserted_count

    return {
        "status": "ok",
        "inserted_features": results,
    }


@router.get("/{symbol}")
async def get_recent_features(
    symbol: str,
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    builder = FeatureBuilder(db)
    features = await builder.list_recent_features(symbol, limit)

    return {
        "symbol": symbol.upper(),
        "count": len(features),
        "features": [
            {
                "timestamp": item.timestamp,
                "return_1m": item.return_1m,
                "return_5m": item.return_5m,
                "volatility_10m": item.volatility_10m,
                "volume_zscore": item.volume_zscore,
                "price_vs_vwap": item.price_vs_vwap,
            }
            for item in features
        ],
    }