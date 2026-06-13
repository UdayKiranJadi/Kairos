import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import PortfolioSnapshot
from app.execution.paper_broker import PaperBroker
from app.core.config import settings

logger = logging.getLogger(__name__)

class PortfolioAgent:
    """
    Syncs live account balances and positions from Alpaca 
    to the local database for the Risk Engine to consume.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
        self.broker = PaperBroker()

    async def sync_and_get_state(self) -> dict:
        """
        Fetches live account data, saves a snapshot to the database,
        and returns the metrics needed by the RiskEngine.
        """
        try:
            # 1. Fetch live account data from Alpaca
            account = self.broker.get_account()
            
            equity = float(account.equity)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            
            # Alpaca tracks last equity (yesterday's close) automatically
            last_equity = float(account.last_equity) if account.last_equity else equity
            
            # 2. Calculate PnL and Drawdown metrics
            daily_pnl = equity - last_equity
            daily_loss_pct = daily_pnl / last_equity if last_equity > 0 else 0.0
            
            total_pnl = equity - settings.starting_capital
            
            # Simplified drawdown: currently based on starting capital. 
            # (In the future, you can upgrade this to calculate from an all-time High Water Mark).
            drawdown_pct = min(0.0, (equity - settings.starting_capital) / settings.starting_capital)

            # 3. Fetch open positions count
            positions = self.broker.get_all_positions()
            open_positions_count = len(positions)

            # 4. Save a snapshot to the database for historical auditing
            snapshot = PortfolioSnapshot(
                cash=cash,
                equity=equity,
                buying_power=buying_power,
                daily_pnl=daily_pnl,
                total_pnl=total_pnl,
                drawdown_pct=drawdown_pct
            )
            
            self.db.add(snapshot)
            await self.db.commit()

            logger.info(f"Portfolio Synced: Equity=${equity:.2f}, Open Positions={open_positions_count}")

            # 5. Return the exact kwargs needed by the RiskEngine
            return {
                "portfolio_value": equity,
                "daily_loss_pct": daily_loss_pct,
                "total_drawdown_pct": drawdown_pct,
                "open_positions": open_positions_count,
            }

        except Exception as e:
            logger.error(f"Failed to sync portfolio: {str(e)}")
            # Fail gracefully by returning strict conservative defaults
            return {
                "portfolio_value": settings.starting_capital,
                "daily_loss_pct": -1.0,  # Forces Risk Engine to halt trading on error
                "total_drawdown_pct": -1.0, 
                "open_positions": 999,
            }