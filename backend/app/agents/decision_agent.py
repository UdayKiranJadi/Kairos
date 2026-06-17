from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Prediction, RiskCheck, Symbol
from app.risk.risk_engine import RiskEngine
from app.risk.risk_policy import RiskPolicy
from app.schemas.trading import TradeAction, TradeDecision


class DecisionAgent:
    """
    Converts predictions into trade decisions, then sends them to RiskEngine.

    Day 7 upgrade: accepts optional sentiment_signal from SentimentAgent.
    Sentiment gate blocks ENTER_LONG if news is strongly bearish.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.risk_engine = RiskEngine(RiskPolicy())

    async def get_symbol(self, ticker: str) -> Symbol | None:
        result = await self.db.execute(
            select(Symbol).where(Symbol.ticker == ticker.upper())
        )
        return result.scalar_one_or_none()

    async def get_latest_prediction(self, symbol_id: int) -> Prediction | None:
        result = await self.db.execute(
            select(Prediction)
            .where(Prediction.symbol_id == symbol_id)
            .order_by(Prediction.timestamp.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def prediction_to_trade_decision(
        self,
        ticker: str,
        prediction: Prediction,
    ) -> TradeDecision:
        probability_up = prediction.probability_up or 0.0
        confidence = prediction.confidence or 0.0

        if probability_up >= 0.70 and confidence >= 0.70:
            action = TradeAction.ENTER_LONG
            reason = (
                f"Probability up is {probability_up:.4f}, "
                "above entry threshold."
            )
        elif probability_up <= 0.45:
            action = TradeAction.EXIT_POSITION
            reason = (
                f"Probability up is {probability_up:.4f}, "
                "below exit threshold."
            )
        else:
            action = TradeAction.HOLD
            reason = (
                f"Probability up is {probability_up:.4f}, "
                "inside no-trade zone."
            )

        return TradeDecision(
            symbol=ticker.upper(),
            action=action,
            confidence=confidence,
            predicted_return=prediction.predicted_return,
            reason=reason,
        )

    async def evaluate_latest_prediction(
        self,
        ticker: str,
        portfolio_value: float = 10_000,
        daily_loss_pct: float = 0.0,
        total_drawdown_pct: float = 0.0,
        trades_today: int = 0,
        open_positions: int = 0,
        trading_mode: str = "paper",
        sentiment_signal: dict | None = None,
    ) -> dict:
        ticker = ticker.upper()

        symbol = await self.get_symbol(ticker)
        if symbol is None:
            raise ValueError(f"Symbol {ticker} does not exist.")

        prediction = await self.get_latest_prediction(symbol.id)
        if prediction is None:
            raise ValueError(
                f"No prediction found for {ticker}. Run PredictionAgent first."
            )

        trade_decision = self.prediction_to_trade_decision(
            ticker=ticker,
            prediction=prediction,
        )

        # ── Sentiment gate ────────────────────────────────────────
        # Blocks ENTER_LONG when news is strongly bearish.
        # The RL + LogReg models see price action only.
        # This layer sees news context they cannot.
        #
        # Example: LogReg says ENTER because RSI is oversold,
        # but GPT-4o says bearish because "Apple raising prices
        # due to chip crunch" just hit the wire. We hold.
        #
        # Gate conditions (all must be true to block):
        #   - sentiment bias is bearish
        #   - confidence > 0.65 (not just weak signal)
        #   - news_impact is high or medium (not noise)
        if (
            sentiment_signal is not None
            and trade_decision.action == TradeAction.ENTER_LONG
        ):
            sentiment_bias = sentiment_signal.get("bias", "neutral")
            sentiment_conf = sentiment_signal.get("confidence", 0.5)
            news_impact    = sentiment_signal.get("news_impact", "low")

            if (
                sentiment_bias == "bearish"
                and sentiment_conf > 0.65
                and news_impact in ("high", "medium")
            ):
                trade_decision = TradeDecision(
                    symbol=ticker,
                    action=TradeAction.HOLD,
                    confidence=trade_decision.confidence,
                    predicted_return=trade_decision.predicted_return,
                    reason=(
                        f"Sentiment gate blocked entry — "
                        f"{sentiment_signal.get('reasoning', 'bearish news')} "
                        f"(score={sentiment_signal.get('aggregate_score', 0):+.3f})"
                    ),
                )

        # ── Risk check ────────────────────────────────────────────
        risk_decision = self.risk_engine.evaluate(
            decision=trade_decision,
            portfolio_value=portfolio_value,
            daily_loss_pct=daily_loss_pct,
            total_drawdown_pct=total_drawdown_pct,
            trades_today=trades_today,
            open_positions=open_positions,
            trading_mode=trading_mode,
        )

        risk_check = RiskCheck(
            symbol_id=symbol.id,
            timestamp=prediction.timestamp,
            action=trade_decision.action.value,
            approved=risk_decision.approved,
            reason=risk_decision.reason,
            portfolio_value=portfolio_value,
            daily_loss_pct=daily_loss_pct,
            total_drawdown_pct=total_drawdown_pct,
        )
        self.db.add(risk_check)
        await self.db.commit()

        return {
            "symbol": ticker,
            "prediction": {
                "timestamp":        prediction.timestamp,
                "probability_up":   prediction.probability_up,
                "confidence":       prediction.confidence,
                "predicted_return": prediction.predicted_return,
                "model_name":       prediction.model_name,
                "model_version":    prediction.model_version,
            },
            "trade_decision": {
                "action":           trade_decision.action.value,
                "confidence":       trade_decision.confidence,
                "predicted_return": trade_decision.predicted_return,
                "reason":           trade_decision.reason,
            },
            "risk_decision": {
                "approved":           risk_decision.approved,
                "reason":             risk_decision.reason,
                "max_position_value": risk_decision.max_position_value,
            },
            "sentiment": sentiment_signal,
        }