"""
trading_bot.py — Day 4 upgrade.

WHAT CHANGED:
  1. Circuit breaker checked BEFORE processing any symbol.
     If daily loss exceeds limit, entire cycle is skipped.

  2. ATR stop-loss checked for open positions each cycle.
     If price hits stop, EXIT is forced regardless of model.

  3. RL ensemble: LogReg + RL model vote together.
     Agreement → higher confidence signal.
     Disagreement → HOLD (skip the trade).

  4. trades_today tracked properly via DB query.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.decision_agent import DecisionAgent
from app.agents.execution_agent import ExecutionAgent
from app.agents.portfolio_agent import PortfolioAgent
from app.agents.prediction_agent import PredictionAgent
from app.data.bar_service import BarService
from app.data.stream_client import get_latest_bar
from app.db.models import PaperOrder
from app.db.session import AsyncSessionLocal
from app.execution.paper_broker import PaperBroker
from app.features.feature_builder import FeatureBuilder
from app.risk.risk_engine import RiskEngine
from app.risk.risk_policy import RiskPolicy
from app.schemas.trading import RiskDecision, TradeAction, TradeDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Kairos.Bot")

# Try to load RL agent — graceful fallback if model not found
_rl_agent = None
try:
    from app.rl.rl_agent import RLAgent
    _rl_agent = RLAgent()
    logger.info("RL agent loaded — ensemble mode active")
except Exception as e:
    logger.info(f"RL agent not loaded ({e}) — using LogReg only")


class AutonomousBot:
    """
    Fully autonomous trading loop with:
    - Live WebSocket data (Day 1)
    - LogReg + RL ensemble signals (Day 3/4)
    - ATR stop-loss per position (Day 4)
    - Circuit breaker on daily drawdown (Day 4)
    - Accurate trades_today tracking (Day 4)
    """

    def __init__(self, symbols: list[str]):
        self.symbols = [s.strip().upper() for s in symbols]
        self.broker = PaperBroker()
        self.risk_engine = RiskEngine(RiskPolicy())

    def is_market_open(self) -> bool:
        return self.broker.client.get_clock().is_open

    async def _trades_today(self, db: AsyncSession) -> int:
        """
        Count orders submitted today via DB query.

        WHY NOT A COUNTER:
        A counter resets on bot restart. The DB persists
        across restarts. This ensures we never exceed
        max_trades_per_day even after a crash + restart.
        """
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        result = await db.execute(
            select(func.count(PaperOrder.id)).where(
                PaperOrder.timestamp >= today_start
            )
        )
        return result.scalar() or 0

    async def _compute_atr(self, symbol: str, db: AsyncSession, period: int = 14) -> float:
        """
        Compute ATR from recent bars in DB.

        ATR = average of True Range over last N bars.
        True Range = max(H-L, |H-prev_C|, |L-prev_C|)

        Returns 0.0 if insufficient data.
        """
        bar_service = BarService(db)
        bars = await bar_service.list_recent_bars(symbol, limit=period + 1)

        if len(bars) < 2:
            return 0.0

        # bars are returned newest-first, reverse for calculation
        bars = list(reversed(bars))

        true_ranges = []
        for i in range(1, len(bars)):
            high  = bars[i].high
            low   = bars[i].low
            prev_close = bars[i-1].close
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low  - prev_close),
            )
            true_ranges.append(tr)

        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    def _apply_rl_ensemble(
        self,
        logreg_action: str,
        logreg_confidence: float,
        latest_features: dict,
    ) -> tuple[str, float, str]:
        """
        Combine LogReg and RL signals.

        Logic:
          No RL model → use LogReg alone
          RL bearish + LogReg EXIT → EXIT (both agree, high conf)
          RL bullish + LogReg ENTER → ENTER (both agree)
          Disagreement → HOLD

        Returns (final_action, final_confidence, reason)
        """
        if _rl_agent is None:
            return logreg_action, logreg_confidence, "LogReg only (no RL model)"

        try:
            rl_result = _rl_agent.predict(latest_features)
            rl_bearish = rl_result["action"] == 1  # SELL = bearish
            rl_conf = rl_result["confidence"]

            if logreg_action == "EXIT_POSITION" and rl_bearish:
                # Both agree: exit
                combined_conf = (logreg_confidence + rl_conf) / 2
                return "EXIT_POSITION", combined_conf, \
                    f"Ensemble EXIT: LogReg={logreg_confidence:.2f} RL={rl_conf:.2f}"

            if logreg_action == "ENTER_LONG" and not rl_bearish:
                # Both agree: enter
                combined_conf = (logreg_confidence + rl_conf) / 2
                return "ENTER_LONG", combined_conf, \
                    f"Ensemble ENTER: LogReg={logreg_confidence:.2f} RL={rl_conf:.2f}"

            # Disagree → hold
            return "HOLD", 0.5, \
                f"Ensemble HOLD (disagreement): LogReg={logreg_action} RL={'SELL' if rl_bearish else 'HOLD'}"

        except Exception as e:
            logger.warning(f"RL inference failed: {e} — using LogReg only")
            return logreg_action, logreg_confidence, "LogReg only (RL error)"

    async def run_pipeline_for_symbol(
        self,
        db: AsyncSession,
        symbol: str,
        live_state: dict,
        trades_today: int,
    ) -> None:
        bar_service = BarService(db)

        # ── 1. Get latest bar (Redis → REST fallback) ─────────────
        latest_bar = await get_latest_bar(symbol)

        if latest_bar is None:
            logger.info(f"[{symbol}] No Redis bar — REST fallback")
            end_time  = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=30)
            await bar_service.store_intraday_bars([symbol], start_time, end_time)
        else:
            logger.info(f"[{symbol}] Redis bar | close={latest_bar['close']:.2f}")
            end_time  = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=2)
            await bar_service.store_intraday_bars([symbol], start_time, end_time)

        # ── 2. ATR stop-loss check for open positions ─────────────
        # Check BEFORE prediction so we exit bad trades immediately
        position = self.broker.get_open_position(symbol)
        if position:
            entry_price   = float(position.avg_entry_price)
            current_price = float(position.current_price or 0)
            atr = await self._compute_atr(symbol, db)

            should_stop, stop_reason = self.risk_engine.check_stop_loss(
                entry_price=entry_price,
                current_price=current_price,
                atr=atr,
            )

            if should_stop:
                logger.warning(f"[{symbol}] STOP LOSS: {stop_reason}")
                # Force exit via broker
                try:
                    qty = float(position.qty)
                    from alpaca.trading.enums import OrderSide
                    order = self.broker.submit_market_order(symbol, qty, OrderSide.SELL)
                    logger.warning(f"[{symbol}] Emergency exit order: {order.id}")
                except Exception as e:
                    logger.error(f"[{symbol}] Stop-loss execution failed: {e}")
                return

        # ── 3. Build features ─────────────────────────────────────
        end_time  = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)
        feature_builder = FeatureBuilder(db)
        await feature_builder.store_features_for_symbol(symbol, start_time, end_time)

        # ── 4. LogReg prediction ──────────────────────────────────
        prediction_agent = PredictionAgent(db)
        try:
            prediction = await prediction_agent.predict_latest(symbol)
        except ValueError as e:
            logger.warning(f"[{symbol}] Prediction skipped: {e}")
            return

        # ── 5. Ensemble: LogReg + RL ──────────────────────────────
        # Get latest features for RL observation
        feature_b = FeatureBuilder(db)
        feature_list = await feature_b.list_recent_features(symbol, limit=1)

        if feature_list:
            f = feature_list[0]
            latest_features = {
                "return_1m":      f.return_1m      or 0.0,
                "return_5m":      f.return_5m      or 0.0,
                "volatility_10m": f.volatility_10m or 0.0,
                "volume_zscore":  f.volume_zscore  or 0.0,
                "price_vs_vwap":  f.price_vs_vwap  or 0.0,
            }
        else:
            latest_features = {}

        # Get LogReg decision first
        decision_agent = DecisionAgent(db)
        eval_result = await decision_agent.evaluate_latest_prediction(
            ticker=symbol,
            portfolio_value=live_state["portfolio_value"],
            daily_loss_pct=live_state["daily_loss_pct"],
            total_drawdown_pct=live_state["total_drawdown_pct"],
            trades_today=trades_today,
            open_positions=live_state["open_positions"],
            trading_mode="paper",
        )

        trade_dict = eval_result["trade_decision"]
        logreg_action = trade_dict["action"]
        logreg_conf   = trade_dict["confidence"]

        # Apply ensemble
        final_action, final_conf, ensemble_reason = self._apply_rl_ensemble(
            logreg_action, logreg_conf, latest_features
        )

        logger.info(f"[{symbol}] {ensemble_reason}")

        decision_obj = TradeDecision(
            symbol=symbol,
            action=TradeAction(final_action),
            confidence=final_conf,
            predicted_return=trade_dict["predicted_return"],
            reason=ensemble_reason,
        )

        risk_dict = eval_result["risk_decision"]
        risk_obj = RiskDecision(
            approved=risk_dict["approved"] and final_action != "HOLD",
            reason=risk_dict["reason"],
            max_position_value=risk_dict.get("max_position_value"),
        )

        # ── 6. Execute ────────────────────────────────────────────
        bar_service2 = BarService(db)
        recent = await bar_service2.list_recent_bars(symbol, limit=1)
        current_price = (
            recent[0].close if recent
            else (latest_bar["close"] if latest_bar else None)
        )

        if current_price is None:
            logger.warning(f"[{symbol}] No price for sizing — skip")
            return

        execution_agent = ExecutionAgent(db)
        exec_result = await execution_agent.execute_decision(
            decision_obj, risk_obj, current_price
        )

        logger.info(
            f"[{symbol}] "
            f"action={final_action} | "
            f"risk={'✓' if risk_obj.approved else '✗'} | "
            f"exec={exec_result['status']} | "
            f"price={current_price:.2f}"
        )

    async def start(self) -> None:
        logger.info(f"Kairos starting | symbols={self.symbols}")
        logger.info(f"RL ensemble: {'active' if _rl_agent else 'inactive (LogReg only)'}")

        while True:
            try:
                if not self.is_market_open():
                    logger.info("Market closed. Sleeping 60s...")
                    await asyncio.sleep(60)
                    continue

                logger.info("─── Cycle start ───")

                async with AsyncSessionLocal() as session:
                    # Portfolio sync
                    portfolio_agent = PortfolioAgent(session)
                    live_state = await portfolio_agent.sync_and_get_state()

                    # ── CIRCUIT BREAKER ───────────────────────────
                    # Check portfolio health BEFORE any trading
                    should_halt, halt_reason = self.risk_engine.check_circuit_breaker(
                        daily_loss_pct=live_state["daily_loss_pct"],
                        total_drawdown_pct=live_state["total_drawdown_pct"],
                    )

                    if should_halt:
                        logger.warning(f"🛑 {halt_reason}")
                        await asyncio.sleep(60)
                        continue

                    # Accurate trades_today count
                    trades_today = await self._trades_today(session)

                    logger.info(
                        f"Portfolio | "
                        f"equity=${live_state['portfolio_value']:.2f} | "
                        f"positions={live_state['open_positions']} | "
                        f"trades_today={trades_today}"
                    )

                    for symbol in self.symbols:
                        await self.run_pipeline_for_symbol(
                            session, symbol, live_state, trades_today
                        )

                now = datetime.now()
                sleep_secs = 60 - now.second
                logger.info(f"Cycle done. Sleeping {sleep_secs}s...")
                await asyncio.sleep(sleep_secs)

            except Exception as e:
                logger.error(f"Bot loop error: {e}", exc_info=True)
                await asyncio.sleep(60)