import math
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FeatureSnapshot, MarketBar, Symbol


def safe_float(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


class FeatureBuilder:
    """
    Builds causal intraday features from stored market bars.

    Causal means:
    each feature row only uses information available at or before that timestamp.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_symbol(self, ticker: str) -> Symbol | None:
        result = await self.db.execute(
            select(Symbol).where(Symbol.ticker == ticker.upper())
        )
        return result.scalar_one_or_none()

    async def load_bars(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        symbol = await self.get_symbol(ticker)

        if symbol is None:
            return pd.DataFrame()

        query = select(MarketBar).where(MarketBar.symbol_id == symbol.id)

        if start is not None:
            query = query.where(MarketBar.timestamp >= start)

        if end is not None:
            query = query.where(MarketBar.timestamp <= end)

        query = query.order_by(MarketBar.timestamp.asc())

        result = await self.db.execute(query)
        bars = list(result.scalars().all())

        if not bars:
            return pd.DataFrame()

        rows = [
            {
                "symbol_id": bar.symbol_id,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]

        return pd.DataFrame(rows)

    def build_features_from_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return bars

        df = bars.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["trading_day"] = df["timestamp"].dt.date

        # Return features
        df["return_1m"] = df["close"].pct_change(periods=1)
        df["return_5m"] = df["close"].pct_change(periods=5)

        # Recent volatility based on 1-minute returns
        df["volatility_10m"] = df["return_1m"].rolling(window=10).std()

        # Volume z-score using recent volume window
        rolling_volume_mean = df["volume"].rolling(window=20).mean()
        rolling_volume_std = df["volume"].rolling(window=20).std()

        df["volume_zscore"] = (
            (df["volume"] - rolling_volume_mean) / rolling_volume_std.replace(0, np.nan)
        )

        # VWAP resets every trading day
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["typical_price_volume"] = typical_price * df["volume"]

        df["cum_tpv"] = df.groupby("trading_day")["typical_price_volume"].cumsum()
        df["cum_volume"] = df.groupby("trading_day")["volume"].cumsum()

        df["vwap"] = df["cum_tpv"] / df["cum_volume"].replace(0, np.nan)
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

        return df[
            [
                "symbol_id",
                "timestamp",
                "return_1m",
                "return_5m",
                "volatility_10m",
                "volume_zscore",
                "price_vs_vwap",
            ]
        ]

    async def store_features_for_symbol(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        bars = await self.load_bars(ticker=ticker, start=start, end=end)

        if bars.empty:
            return 0

        features = self.build_features_from_bars(bars)

        inserted_count = 0

        for row in features.to_dict(orient="records"):
            existing = await self.db.execute(
                select(FeatureSnapshot).where(
                    FeatureSnapshot.symbol_id == row["symbol_id"],
                    FeatureSnapshot.timestamp == row["timestamp"].to_pydatetime(),
                )
            )
            existing_feature = existing.scalar_one_or_none()

            if existing_feature:
                continue

            snapshot = FeatureSnapshot(
                symbol_id=int(row["symbol_id"]),
                timestamp=row["timestamp"].to_pydatetime(),
                return_1m=safe_float(row["return_1m"]),
                return_5m=safe_float(row["return_5m"]),
                volatility_10m=safe_float(row["volatility_10m"]),
                volume_zscore=safe_float(row["volume_zscore"]),
                price_vs_vwap=safe_float(row["price_vs_vwap"]),
            )

            self.db.add(snapshot)
            inserted_count += 1

        await self.db.commit()
        return inserted_count

    async def list_recent_features(
        self,
        ticker: str,
        limit: int = 100,
    ) -> list[FeatureSnapshot]:
        symbol = await self.get_symbol(ticker)

        if symbol is None:
            return []

        result = await self.db.execute(
            select(FeatureSnapshot)
            .where(FeatureSnapshot.symbol_id == symbol.id)
            .order_by(FeatureSnapshot.timestamp.desc())
            .limit(limit)
        )

        return list(result.scalars().all())