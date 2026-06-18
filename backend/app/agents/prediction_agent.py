from pathlib import Path
from datetime import timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FeatureSnapshot, Prediction, Symbol
from app.models.prediction_dataset import FEATURE_COLUMNS
from app.models.prediction_model import IntradayPredictionModel


class PredictionAgent:
    """
    The Prediction Agent loads model artifacts, reads feature snapshots,
    generates model predictions, and stores them in the database.

    This agent does not trade.
    This agent does not call the broker.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_symbol(self, ticker: str) -> Symbol | None:
        result = await self.db.execute(
            select(Symbol).where(Symbol.ticker == ticker.upper())
        )
        return result.scalar_one_or_none()

    async def get_latest_feature_snapshot(
        self,
        symbol_id: int,
    ) -> FeatureSnapshot | None:
        result = await self.db.execute(
            select(FeatureSnapshot)
            .where(FeatureSnapshot.symbol_id == symbol_id)
            .order_by(FeatureSnapshot.timestamp.desc())
            .limit(1)
        )

        return result.scalar_one_or_none()

    def build_feature_dict(self, snapshot: FeatureSnapshot) -> dict:
     return {
        "return_1m":      snapshot.return_1m,
        "return_5m":      snapshot.return_5m,
        "volatility_10m": snapshot.volatility_10m,
        "volume_zscore":  snapshot.volume_zscore,
        "price_vs_vwap":  snapshot.price_vs_vwap,
        "rsi_14":         snapshot.rsi_14,
        "macd_signal":    snapshot.macd_signal,
        "obv_zscore":     snapshot.obv_zscore,
    }

    def has_missing_features(self, features: dict) -> bool:
     return any(features.get(col) is None for col in FEATURE_COLUMNS)

    async def store_prediction(
        self,
        symbol_id: int,
        timestamp,
        prediction_result: dict,
    ) -> Prediction:
        probability_up = float(prediction_result["probability_up"])
        predicted_class = int(prediction_result["predicted_class"])

        confidence = max(probability_up, 1.0 - probability_up)

        # Temporary directional score.
        # Later we will replace this with a true expected-return model.
        predicted_return_proxy = probability_up - 0.5

        clean_timestamp = timestamp

        if clean_timestamp.tzinfo is not None:
            clean_timestamp = clean_timestamp.astimezone(timezone.utc).replace(tzinfo=None)

        prediction = Prediction(
            symbol_id=symbol_id,
            timestamp=clean_timestamp,
            model_name=prediction_result["model_name"],
            model_version=prediction_result["model_version"],
            horizon_minutes=prediction_result["horizon_minutes"],
            probability_up=probability_up,
            predicted_class=predicted_class,
            predicted_return=predicted_return_proxy,
            confidence=confidence,
        )

        self.db.add(prediction)
        await self.db.commit()
        await self.db.refresh(prediction)

        return prediction

    async def predict_latest(
        self,
        ticker: str,
        model_path: str | Path | None = None,
    ) -> Prediction:
        ticker = ticker.upper()

        symbol = await self.get_symbol(ticker)

        if symbol is None:
            raise ValueError(f"Symbol {ticker} does not exist. Load bars first.")

        snapshot = await self.get_latest_feature_snapshot(symbol.id)

        if snapshot is None:
            raise ValueError(
                f"No feature snapshot found for {ticker}. Build features first."
            )

        features = self.build_feature_dict(snapshot)

        if self.has_missing_features(features):
            raise ValueError(
                f"Latest feature snapshot for {ticker} has missing values. "
                "Load more bars or use a later timestamp."
            )

        if model_path is None:
            model_path = (
                Path("artifacts/models")
                / f"{ticker.lower()}_intraday_direction_v0_1.joblib"
            )

        model = IntradayPredictionModel(model_path)
        prediction_result = model.predict_one(features)

        prediction = await self.store_prediction(
            symbol_id=symbol.id,
            timestamp=snapshot.timestamp,
            prediction_result=prediction_result,
        )

        return prediction

    async def list_recent_predictions(
        self,
        ticker: str,
        limit: int = 50,
    ) -> list[Prediction]:
        symbol = await self.get_symbol(ticker)

        if symbol is None:
            return []

        result = await self.db.execute(
            select(Prediction)
            .where(Prediction.symbol_id == symbol.id)
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
        )

        return list(result.scalars().all())