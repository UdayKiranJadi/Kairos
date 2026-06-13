import asyncio
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.data.bar_service import BarService
from app.features.feature_builder import FeatureBuilder
from app.agents.prediction_agent import PredictionAgent
from app.agents.portfolio_agent import PortfolioAgent
from app.agents.decision_agent import DecisionAgent
from app.agents.execution_agent import ExecutionAgent
from app.execution.paper_broker import PaperBroker
from app.schemas.trading import TradeDecision, TradeAction, RiskDecision

# Set up basic logging for the bot loop
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingBot")

class AutonomousBot:
    """
    The master orchestration loop. Runs continuously, syncing data, 
    generating predictions, and executing trades during market hours.
    """
    def __init__(self, symbols: list[str]):
        self.symbols = [s.strip().upper() for s in symbols]
        self.broker = PaperBroker()

    def is_market_open(self) -> bool:
        """Checks Alpaca to see if the US equity market is currently open."""
        clock = self.broker.client.get_clock()
        return clock.is_open

    async def run_pipeline_for_symbol(self, db: AsyncSession, symbol: str, live_state: dict):
        """Runs the entire Fetch -> Predict -> Decide -> Execute pipeline for a single symbol."""
        # 1. Fetch latest bars (last 2 hours to ensure enough data for rolling features)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=2)
        
        bar_service = BarService(db)
        await bar_service.store_intraday_bars([symbol], start=start_time, end=end_time)

        # 2. Build Features
        feature_builder = FeatureBuilder(db)
        await feature_builder.store_features_for_symbol(symbol, start=start_time, end=end_time)

        # 3. Predict
        prediction_agent = PredictionAgent(db)
        try:
            prediction = await prediction_agent.predict_latest(symbol)
        except ValueError as e:
            logger.warning(f"[{symbol}] Prediction skipped: {str(e)}")
            return

        # 4. Decide & Risk Check
        decision_agent = DecisionAgent(db)
        eval_result = await decision_agent.evaluate_latest_prediction(
            ticker=symbol,
            portfolio_value=live_state["portfolio_value"],
            daily_loss_pct=live_state["daily_loss_pct"],
            total_drawdown_pct=live_state["total_drawdown_pct"],
            trades_today=0, # Advanced logic can track trades_today via DB queries
            open_positions=live_state["open_positions"],
            trading_mode="paper" # Will allow actual paper execution
        )

        trade_dict = eval_result["trade_decision"]
        risk_dict = eval_result["risk_decision"]

        # Reconstruct Pydantic models for the Execution Agent
        decision_obj = TradeDecision(
            symbol=symbol,
            action=TradeAction(trade_dict["action"]),
            confidence=trade_dict["confidence"],
            predicted_return=trade_dict["predicted_return"],
            reason=trade_dict["reason"]
        )
        risk_obj = RiskDecision(
            approved=risk_dict["approved"],
            reason=risk_dict["reason"],
            max_position_value=risk_dict.get("max_position_value")
        )

        # 5. Execute
        # Fetch the latest closing price from the DB to calculate share sizing
        latest_bars = await bar_service.list_recent_bars(symbol, limit=1)
        if not latest_bars:
            logger.warning(f"[{symbol}] No bars found for execution pricing.")
            return
            
        current_price = latest_bars[0].close

        execution_agent = ExecutionAgent(db)
        exec_result = await execution_agent.execute_decision(decision_obj, risk_obj, current_price)
        
        logger.info(f"[{symbol}] Action: {decision_obj.action.value} | Exec: {exec_result['status']} | Reason: {risk_obj.reason}")


    async def start(self):
        """The infinite loop that acts as the heartbeat of the trading system."""
        logger.info(f"Starting Autonomous Bot for symbols: {self.symbols}")
        
        while True:
            try:
                # 1. Market Open Check
                if not self.is_market_open():
                    logger.info("Market is closed. Sleeping for 60 seconds...")
                    await asyncio.sleep(60)
                    continue

                logger.info("--- Starting Trading Cycle ---")
                
                async with AsyncSessionLocal() as session:
                    # 2. Sync Portfolio State ONCE per minute cycle
                    portfolio_agent = PortfolioAgent(session)
                    live_state = await portfolio_agent.sync_and_get_state()
                    logger.info(f"Live Equity: ${live_state['portfolio_value']:.2f}")
                    
                    # 3. Run Pipeline for each symbol consecutively
                    for symbol in self.symbols:
                        await self.run_pipeline_for_symbol(session, symbol, live_state)

                # 4. Sleep until the top of the next minute
                now = datetime.now()
                sleep_seconds = 60 - now.second
                logger.info(f"Cycle complete. Sleeping for {sleep_seconds} seconds until next minute bar.")
                await asyncio.sleep(sleep_seconds)

            except Exception as e:
                logger.error(f"Critical error in main loop: {str(e)}")
                # Sleep briefly to avoid rapid crash-looping if an API goes down
                await asyncio.sleep(60)