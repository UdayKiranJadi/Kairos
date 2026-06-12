from datetime import datetime

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FeatureSnapshot, MarketBar, Symbol


FEATURE_COLUMNS = [
    "return_1m",
    "return_5m",
    "volatility_10m",
    "volume_zscore",
    "price_vs_vwap",
]


class PredictionDatasetBuilder:
    """
    Builds supervised intraday prediction datasets.

    Target:
    whether future return over the next N minutes is positive.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_symbol(self, ticker: str) -> Symbol | None:
        result = await self.db.execute(
            select(Symbol).where(Symbol.ticker == ticker.upper())
        )
        return result.scalar_one_or_none()

    async def load_feature_price_frame(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        symbol = await self.get_symbol(ticker)

        if symbol is None:
            return pd.DataFrame()

        query = (
            select(
                FeatureSnapshot.timestamp,
                FeatureSnapshot.return_1m,
                FeatureSnapshot.return_5m,
                FeatureSnapshot.volatility_10m,
                FeatureSnapshot.volume_zscore,
                FeatureSnapshot.price_vs_vwap,
                MarketBar.close,
            )
            .join(
                MarketBar,
                and_(
                    MarketBar.symbol_id == FeatureSnapshot.symbol_id,
                    MarketBar.timestamp == FeatureSnapshot.timestamp,
                ),
            )
            .where(FeatureSnapshot.symbol_id == symbol.id)
            .order_by(FeatureSnapshot.timestamp.asc())
        )

        if start is not None:
            query = query.where(FeatureSnapshot.timestamp >= start)

        if end is not None:
            query = query.where(FeatureSnapshot.timestamp <= end)

        result = await self.db.execute(query)
        rows = result.all()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "return_1m",
                "return_5m",
                "volatility_10m",
                "volume_zscore",
                "price_vs_vwap",
                "close",
            ],
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def build_dataset(
        self,
        df: pd.DataFrame,
        horizon_minutes: int = 5,
        min_abs_future_return: float = 0.0,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        data = df.copy()
        data = data.sort_values("timestamp").reset_index(drop=True)

        data["future_close"] = data["close"].shift(-horizon_minutes)
        data["future_return"] = (data["future_close"] / data["close"]) - 1.0

        data["target_up"] = (
            data["future_return"] > min_abs_future_return
        ).astype(int)

        data = data.dropna(
            subset=[
                *FEATURE_COLUMNS,
                "future_close",
                "future_return",
                "target_up",
            ]
        )

        return data[
            [
                "timestamp",
                *FEATURE_COLUMNS,
                "close",
                "future_close",
                "future_return",
                "target_up",
            ]
        ]