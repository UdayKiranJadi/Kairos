import math
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PaperOrder, Symbol, OrderStatus
from app.schemas.trading import RiskDecision, TradeDecision, TradeAction
from app.execution.paper_broker import PaperBroker
from alpaca.trading.enums import OrderSide
import logging

logger = logging.getLogger(__name__)

class ExecutionAgent:
    """
    Takes approved risk decisions, calculates position sizing, 
    executes via the broker, and logs to the database.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
        self.broker = PaperBroker()

    async def get_symbol(self, ticker: str) -> Symbol | None:
        result = await self.db.execute(
            select(Symbol).where(Symbol.ticker == ticker.upper())
        )
        return result.scalar_one_or_none()

    async def execute_decision(
        self, 
        decision: TradeDecision, 
        risk: RiskDecision, 
        current_price: float
    ) -> dict:
        
        if not risk.approved:
            logger.info(f"Execution skipped for {decision.symbol}: Risk not approved.")
            return {"status": "skipped", "reason": "risk_rejected"}

        if decision.action == TradeAction.HOLD:
            return {"status": "skipped", "reason": "hold_action"}

        symbol_record = await self.get_symbol(decision.symbol)
        if not symbol_record:
            raise ValueError(f"Symbol {decision.symbol} not found in DB.")

        alpaca_order = None
        qty = 0.0
        side = None

        try:
            if decision.action == TradeAction.ENTER_LONG:
                # Calculate how many shares we can buy with the max_position_value
                qty = math.floor(risk.max_position_value / current_price)
                if qty <= 0:
                    return {"status": "skipped", "reason": "insufficient_funds_for_1_share"}
                
                side = OrderSide.BUY
                logger.info(f"Submitting BUY order for {qty} shares of {decision.symbol}")
                alpaca_order = self.broker.submit_market_order(decision.symbol, qty, side)

            elif decision.action == TradeAction.EXIT_POSITION:
                # Find out how many shares we currently hold to sell them all
                position = self.broker.get_open_position(decision.symbol)
                if not position:
                    return {"status": "skipped", "reason": "no_open_position_to_close"}
                
                qty = float(position.qty)
                side = OrderSide.SELL
                logger.info(f"Submitting SELL order for {qty} shares of {decision.symbol}")
                alpaca_order = self.broker.submit_market_order(decision.symbol, qty, side)

            # 3. Log the order in our database
            if alpaca_order:
                db_order = PaperOrder(
                    broker_order_id=str(alpaca_order.id),
                    symbol_id=symbol_record.id,
                    side=side.value,
                    quantity=qty,
                    order_type="market",
                    status=OrderStatus.SUBMITTED.value,
                    submitted_price=current_price
                )
                self.db.add(db_order)
                await self.db.commit()
                
                return {
                    "status": "executed",
                    "broker_order_id": str(alpaca_order.id),
                    "qty": qty,
                    "side": side.value
                }

        except Exception as e:
            logger.error(f"Failed to execute order for {decision.symbol}: {str(e)}")
            return {"status": "error", "reason": str(e)}