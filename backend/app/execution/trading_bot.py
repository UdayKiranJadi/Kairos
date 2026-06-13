"""
trading_bot.py — Autonomous trading loop.

WHAT CHANGED FROM ORIGINAL (Day 1):
  Before: every 60s, fetch 2 hours of bars via Alpaca REST,
          rebuild features, predict, decide, execute.

  After:  every 60s, read latest bar from Redis (written by
          stream_client.py WebSocket). Falls back to REST if
          Redis is empty (first run or stream not started).

WHY THIS MATTERS:
  REST fetch = ~500ms + 120 API calls/hour per symbol.
  Redis read = ~0.1ms + 0 API calls.

  More importantly: the bot is now decoupled from the data
  source. stream_client runs independently. If the stream
  drops, the bot falls back gracefully. If the bot crashes,
  the stream keeps running.

  Separation of concerns = easier to debug.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.decision_agent import DecisionAgent
from app.agents.execution_agent import ExecutionAgent
from app.agents.portfolio_agent import PortfolioAgent
from app.agents.prediction_agent import PredictionAgent
from app.data.bar_service import BarService
from app.data.stream_client import get_bar_window, get_latest_bar
from app.db.session import AsyncSessionLocal
from app.execution.paper_broker import PaperBroker
from app.features.feature_builder import FeatureBuilder
from app.schemas.trading import RiskDecision, TradeAction, TradeDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Kairos.Bot")


class AutonomousBot:
    """
    Master orchestration loop.

    Each 60-second cycle:
      1. Check market is open
      2. Sync portfolio state from Alpaca
      3. For each symbol:
         a. Get latest bar (Redis → REST fallback)
         b. Store bar + build features
         c. Run prediction model
         d. Risk-check the decision
         e. Execute if approved
      4. Sleep until top of next minute
    """

    def __init__(self, symbols: list[str]):
        self.symbols = [s.strip().upper() for s in symbols]
        self.broker = PaperBroker()

    def is_market_open(self) -> bool:
        clock = self.broker.client.get_clock()
        return clock.is_open

    async def _get_current_price(
        self,
        ticker: str,
        db: AsyncSession,
    ) -> float | None:
        """
        Get current price. Tries Redis first, falls back to Postgres.

        WHY THIS ORDER:
        Redis has the bar the WS just pushed (sub-second fresh).
        Postgres has whatever was last stored (may be minutes old).
        We want the freshest price for position sizing.
        """
        # 1. Try Redis — fastest, freshest
        bar = await get_latest_bar(ticker)
        if bar:
            return bar["close"]

        # 2. Fall back to Postgres
        bar_service = BarService(db)
        recent = await bar_service.list_recent_bars(ticker, limit=1)
        if recent:
            return recent[0].close

        return None

    async def run_pipeline_for_symbol(
        self,
        db: AsyncSession,
        symbol: str,
        live_state: dict,
    ) -> None:
        """
        Full pipeline for one symbol per cycle.

        KEY CHANGE from original:
        We no longer fetch 2 hours of bars every cycle.
        stream_client.py is writing bars to Redis as they arrive.
        We just read what's already there.

        We still call store_intraday_bars as a FALLBACK for
        the first run (Redis empty) or if stream is down.
        """
        bar_service = BarService(db)
        feature_builder = FeatureBuilder(db)

        # --- 1. Get latest bar ---
        # Check Redis first. If empty (stream not started yet
        # or first minute of market open), fall back to REST.
        latest_bar = await get_latest_bar(symbol)

        if latest_bar is None:
            # WHY FALLBACK:
            # On first startup the WS stream may not have delivered
            # a bar yet. We fetch the last 30 mins via REST so the
            # bot can still operate while the stream warms up.
            logger.info(f"[{symbol}] Redis empty — fetching via REST fallback")
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=30)
            await bar_service.store_intraday_bars(
                [symbol], start=start_time, end=end_time
            )
        else:
            # WHY STORE EVEN THOUGH WE HAVE IT:
            # Redis is ephemeral — it loses data on restart.
            # Postgres is our source of truth for training data.
            # We write every bar to both. Redis = fast reads,
            # Postgres = durable history.
            logger.info(
                f"[{symbol}] Bar from Redis | "
                f"close={latest_bar['close']:.2f} | "
                f"t={latest_bar['timestamp']}"
            )
            # Store the bar to Postgres for historical record
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=2)
            await bar_service.store_intraday_bars(
                [symbol], start=start_time, end=end_time
            )

        # --- 2. Build features ---
        # Always build from Postgres so we have the full
        # rolling window (not just what's in Redis).
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)
        await feature_builder.store_features_for_symbol(
            symbol, start=start_time, end=end_time
        )

        # --- 3. Predict ---
        prediction_agent = PredictionAgent(db)
        try:
            prediction = await prediction_agent.predict_latest(symbol)
        except ValueError as e:
            logger.warning(f"[{symbol}] Prediction skipped: {e}")
            return

        # --- 4. Decide + risk check ---
        decision_agent = DecisionAgent(db)
        eval_result = await decision_agent.evaluate_latest_prediction(
            ticker=symbol,
            portfolio_value=live_state["portfolio_value"],
            daily_loss_pct=live_state["daily_loss_pct"],
            total_drawdown_pct=live_state["total_drawdown_pct"],
            trades_today=0,
            open_positions=live_state["open_positions"],
            trading_mode="paper",
        )

        trade_dict = eval_result["trade_decision"]
        risk_dict = eval_result["risk_decision"]

        decision_obj = TradeDecision(
            symbol=symbol,
            action=TradeAction(trade_dict["action"]),
            confidence=trade_dict["confidence"],
            predicted_return=trade_dict["predicted_return"],
            reason=trade_dict["reason"],
        )
        risk_obj = RiskDecision(
            approved=risk_dict["approved"],
            reason=risk_dict["reason"],
            max_position_value=risk_dict.get("max_position_value"),
        )

        # --- 5. Execute ---
        current_price = await self._get_current_price(symbol, db)
        if current_price is None:
            logger.warning(f"[{symbol}] No price available for sizing. Skipping.")
            return

        execution_agent = ExecutionAgent(db)
        exec_result = await execution_agent.execute_decision(
            decision_obj, risk_obj, current_price
        )

        logger.info(
            f"[{symbol}] "
            f"action={decision_obj.action.value} | "
            f"risk={'✓' if risk_obj.approved else '✗'} {risk_obj.reason} | "
            f"exec={exec_result['status']} | "
            f"price={current_price:.2f}"
        )

    async def start(self) -> None:
        """
        Infinite loop — heartbeat of the system.

        stream_client.stream_bars() runs ALONGSIDE this loop
        via asyncio.gather() in main.py (added Day 1).
        They share the same event loop but don't block each other:
        - stream_bars() wakes up when Alpaca pushes a bar
        - start() wakes up every 60 seconds to run the pipeline
        """
        logger.info(f"Kairos bot starting | symbols: {self.symbols}")

        while True:
            try:
                if not self.is_market_open():
                    logger.info("Market closed. Sleeping 60s...")
                    await asyncio.sleep(60)
                    continue

                logger.info("─── Trading cycle start ───")

                async with AsyncSessionLocal() as session:
                    portfolio_agent = PortfolioAgent(session)
                    live_state = await portfolio_agent.sync_and_get_state()
                    logger.info(
                        f"Portfolio | equity=${live_state['portfolio_value']:.2f} | "
                        f"positions={live_state['open_positions']}"
                    )

                    for symbol in self.symbols:
                        await self.run_pipeline_for_symbol(
                            session, symbol, live_state
                        )

                # Sleep until top of next minute
                now = datetime.now()
                sleep_secs = 60 - now.second
                logger.info(f"Cycle done. Sleeping {sleep_secs}s...")
                await asyncio.sleep(sleep_secs)

            except Exception as e:
                logger.error(f"Bot loop error: {e}", exc_info=True)
                await asyncio.sleep(60)