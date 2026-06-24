"""
trading_bot.py — Kairos autonomous trading loop.

AGENT STACK (in order of execution each cycle):
  1. PortfolioAgent   — sync live equity + positions from Alpaca
  2. RiskEngine       — circuit breaker check (portfolio level)
  3. BarService       — get latest bar (Redis → REST fallback)
  4. RiskEngine       — ATR stop-loss check (position level)
  5. FeatureBuilder   — compute 8 features from recent bars
  6. SentimentAgent   — FinBERT + GPT-4o news sentiment
  7. PredictionAgent  — LogReg direction prediction
  8. RLAgent          — PPO policy signal
  9. DecisionAgent    — ensemble + sentiment gate + risk check
  10. ExecutionAgent  — size + submit order via PaperBroker

PARALLEL PROCESSING (Day 7 upgrade):
  Symbols now run concurrently via asyncio.gather().
  Sequential: 2 symbols × 3s = 6s per cycle.
  Parallel:  20 symbols × 3s = 3s per cycle (bottleneck = slowest symbol).
  One symbol failing does not crash others (return_exceptions=True).
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
from app.agents.sentiment_agent import SentimentAgent
from app.core.alerting import AlertLevel, alert
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

# ── Module-level singletons ───────────────────────────────────────
_sentiment_agent = SentimentAgent()

_rl_agent = None
try:
    from app.rl.rl_agent import RLAgent
    _rl_agent = RLAgent()
    logger.info("RL agent loaded — ensemble mode active")
except Exception as e:
    logger.info(f"RL agent not loaded ({e}) — using LogReg only")


class AutonomousBot:

    def __init__(self, symbols: list[str]):
        self.symbols = [s.strip().upper() for s in symbols]
        self.broker = PaperBroker()
        self.risk_engine = RiskEngine(RiskPolicy())

    def is_market_open(self) -> bool:
        return self.broker.client.get_clock().is_open

    async def _trades_today(self, db: AsyncSession) -> int:
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        result = await db.execute(
            select(func.count(PaperOrder.id)).where(
                PaperOrder.timestamp >= today_start
            )
        )
        return result.scalar() or 0

    async def _compute_atr(
        self, symbol: str, db: AsyncSession, period: int = 14
    ) -> float:
        bar_service = BarService(db)
        bars = await bar_service.list_recent_bars(symbol, limit=period + 1)
        if len(bars) < 2:
            return 0.0
        bars = list(reversed(bars))
        true_ranges = []
        for i in range(1, len(bars)):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            true_ranges.append(tr)
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    def _apply_rl_ensemble(
        self,
        logreg_action: str,
        logreg_confidence: float,
        latest_features: dict,
        unrealized_pnl: float = 0.0,
    ) -> tuple[str, float, str]:
        if _rl_agent is None:
            return logreg_action, logreg_confidence, "LogReg only (no RL model)"

        try:
            rl_result = _rl_agent.predict(latest_features, unrealized_pnl=unrealized_pnl)
            rl_bearish = rl_result["action"] == 1
            rl_conf = rl_result["confidence"]

            if logreg_action == "EXIT_POSITION" and rl_bearish:
                combined = (logreg_confidence + rl_conf) / 2
                return "EXIT_POSITION", combined, \
                    f"Ensemble EXIT: LogReg={logreg_confidence:.2f} RL={rl_conf:.2f}"

            if logreg_action == "ENTER_LONG" and not rl_bearish:
                combined = (logreg_confidence + rl_conf) / 2
                return "ENTER_LONG", combined, \
                    f"Ensemble ENTER: LogReg={logreg_confidence:.2f} RL={rl_conf:.2f}"

            return "HOLD", 0.5, \
                f"Ensemble HOLD (disagreement): " \
                f"LogReg={logreg_action} RL={'SELL' if rl_bearish else 'HOLD'}"

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

        # ── 1. Latest bar (Redis → REST fallback) ─────────────────
        latest_bar = await get_latest_bar(symbol)
        if latest_bar is None:
            logger.info(f"[{symbol}] No Redis bar — REST fallback")
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=30)
            await bar_service.store_intraday_bars([symbol], start_time, end_time)
        else:
            logger.info(f"[{symbol}] Redis bar | close={latest_bar['close']:.2f}")
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=2)
            await bar_service.store_intraday_bars([symbol], start_time, end_time)

        # ── 2. ATR stop-loss check ────────────────────────────────
        position = self.broker.get_open_position(symbol)
        unrealized_pnl_pct = 0.0
        if position and position.avg_entry_price and float(position.avg_entry_price) > 0:
            entry = float(position.avg_entry_price)
            current = float(position.current_price or 0)
            unrealized_pnl_pct = (current - entry) / entry
        if position:
            entry_price = float(position.avg_entry_price)
            current_price = float(position.current_price or 0)
            atr = await self._compute_atr(symbol, db)

            should_stop, stop_reason = self.risk_engine.check_stop_loss(
                entry_price=entry_price,
                current_price=current_price,
                atr=atr,
            )

            if should_stop:
                logger.warning(f"[{symbol}] STOP LOSS: {stop_reason}")
                await alert(
                    f"ATR STOP-LOSS: {symbol}\n{stop_reason}",
                    level=AlertLevel.WARNING,
                    subject=f"Stop-loss triggered — {symbol}",
                )
                try:
                    from alpaca.trading.enums import OrderSide
                    qty = float(position.qty)
                    order = self.broker.submit_market_order(
                        symbol, qty, OrderSide.SELL
                    )
                    logger.warning(f"[{symbol}] Emergency exit: {order.id}")
                except Exception as e:
                    logger.error(f"[{symbol}] Stop-loss execution failed: {e}")
                return

        # ── 3. Build features ─────────────────────────────────────
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)
        feature_builder = FeatureBuilder(db)
        await feature_builder.store_features_for_symbol(
            symbol, start_time, end_time
        )

        # ── 4. Get latest features ────────────────────────────────
        feature_list = await feature_builder.list_recent_features(
            symbol, limit=1
        )

        latest_features = {}
        price_context = {}
        current_price = None

        if feature_list:
            f = feature_list[0]
            latest_features = {
                "return_1m":      f.return_1m or 0.0,
                "return_5m":      f.return_5m or 0.0,
                "volatility_10m": f.volatility_10m or 0.0,
                "volume_zscore":  f.volume_zscore or 0.0,
                "price_vs_vwap":  f.price_vs_vwap or 0.0,
                "rsi_14":         f.rsi_14 or 0.0,
                "macd_signal":    f.macd_signal or 0.0,
                "obv_zscore":     f.obv_zscore or 0.0,
            }
            recent_bars = await bar_service.list_recent_bars(symbol, limit=1)
            current_price = recent_bars[0].close if recent_bars else (
                latest_bar["close"] if latest_bar else None
            )
            price_context = {
                "close":         current_price or 0,
                "rsi_14":        f.rsi_14 or 0.0,
                "macd_signal":   f.macd_signal or 0.0,
                "price_vs_vwap": f.price_vs_vwap or 0.0,
                "volume_zscore": f.volume_zscore or 0.0,
            }
        else:
            recent_bars = await bar_service.list_recent_bars(symbol, limit=1)
            current_price = recent_bars[0].close if recent_bars else (
                latest_bar["close"] if latest_bar else None
            )

        # ── 5. Sentiment signal ───────────────────────────────────
        try:
            sentiment_signal = await asyncio.to_thread(
                _sentiment_agent.get_signal_sync,
                symbol,
                price_context,
            )
            logger.info(
                f"[{symbol}] Sentiment | "
                f"bias={sentiment_signal['bias']} | "
                f"score={sentiment_signal['aggregate_score']:+.3f} | "
                f"source={sentiment_signal['source']} | "
                f"{sentiment_signal['reasoning'][:60]}"
            )
        except Exception as e:
            logger.warning(f"[{symbol}] Sentiment failed: {e} — skipping")
            sentiment_signal = None

        # ── 6. LogReg prediction ──────────────────────────────────
        prediction_agent = PredictionAgent(db)
        try:
            await prediction_agent.predict_latest(symbol)
        except ValueError as e:
            logger.warning(f"[{symbol}] Prediction skipped: {e}")
            return

        # ── 7. DecisionAgent ──────────────────────────────────────
        decision_agent = DecisionAgent(db)
        eval_result = await decision_agent.evaluate_latest_prediction(
            ticker=symbol,
            portfolio_value=live_state["portfolio_value"],
            daily_loss_pct=live_state["daily_loss_pct"],
            total_drawdown_pct=live_state["total_drawdown_pct"],
            trades_today=trades_today,
            open_positions=live_state["open_positions"],
            trading_mode="paper",
            sentiment_signal=sentiment_signal,
        )

        trade_dict = eval_result["trade_decision"]
        logreg_action = trade_dict["action"]
        logreg_conf = trade_dict["confidence"]

        # ── 8. RL ensemble ────────────────────────────────────────
        final_action, final_conf, ensemble_reason = self._apply_rl_ensemble(
            logreg_action, logreg_conf, latest_features, unrealized_pnl=unrealized_pnl_pct
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

        # ── 9. Execute ────────────────────────────────────────────
        if current_price is None:
            logger.warning(f"[{symbol}] No price for sizing — skip")
            return

        execution_agent = ExecutionAgent(db)
        exec_result = await execution_agent.execute_decision(
            decision_obj, risk_obj, current_price
        )

        if exec_result.get("status") == "executed":
            await alert(
                f"TRADE EXECUTED: {symbol}\n"
                f"Side: {exec_result.get('side', '').upper()}\n"
                f"Qty: {exec_result.get('qty')}\n"
                f"Price: ${current_price:.2f}\n"
                f"Portfolio equity: ${live_state['portfolio_value']:.2f}",
                level=AlertLevel.INFO,
                subject=f"Trade executed — {symbol}",
            )

        logger.info(
            f"[{symbol}] "
            f"action={final_action} | "
            f"risk={'✓' if risk_obj.approved else '✗'} | "
            f"exec={exec_result['status']} | "
            f"price={current_price:.2f}"
        )

    async def _run_symbol_with_own_session(
        self, sym: str, live_state: dict, trades_today: int
    ) -> None:
        async with AsyncSessionLocal() as session:
            await self.run_pipeline_for_symbol(session, sym, live_state, trades_today)

    async def start(self) -> None:
        logger.info(f"Kairos starting | symbols={self.symbols}")
        logger.info(
            f"RL ensemble: {'active' if _rl_agent else 'inactive (LogReg only)'}"
        )

        # Preload FinBERT once at startup so the first trading cycle
        # doesn't block for ~10s waiting for model weights to load.
        logger.info("Preloading FinBERT...")
        await asyncio.to_thread(_sentiment_agent._build_signal, self.symbols[0], {})
        logger.info("FinBERT ready")

        while True:
            try:
                if not self.is_market_open():
                    logger.info("Market closed. Sleeping 60s...")
                    await asyncio.sleep(60)
                    continue

                logger.info("─── Cycle start ───")

                # Portfolio sync gets its own short-lived session.
                async with AsyncSessionLocal() as session:
                    portfolio_agent = PortfolioAgent(session)
                    live_state = await portfolio_agent.sync_and_get_state()

                    # ── Circuit breaker ───────────────────────────
                    should_halt, halt_reason = self.risk_engine.check_circuit_breaker(
                        daily_loss_pct=live_state["daily_loss_pct"],
                        total_drawdown_pct=live_state["total_drawdown_pct"],
                    )

                    trades_today = await self._trades_today(session)

                if should_halt:
                    logger.warning(f"CIRCUIT BREAKER: {halt_reason}")
                    await alert(
                        f"CIRCUIT BREAKER FIRED\n"
                        f"{halt_reason}\n"
                        f"Equity: ${live_state['portfolio_value']:.2f}",
                        level=AlertLevel.CRITICAL,
                        subject="Circuit breaker — trading halted",
                    )
                    await asyncio.sleep(60)
                    continue

                logger.info(
                    f"Portfolio | "
                    f"equity=${live_state['portfolio_value']:.2f} | "
                    f"positions={live_state['open_positions']} | "
                    f"trades_today={trades_today}"
                )

                # ── Parallel symbol processing ─────────────────
                # Each symbol gets its own AsyncSession so concurrent
                # coroutines don't share state across one session.
                # return_exceptions=True: one symbol failing never
                # blocks the others.
                results = await asyncio.gather(
                    *[
                        self._run_symbol_with_own_session(
                            sym, live_state, trades_today
                        )
                        for sym in self.symbols
                    ],
                    return_exceptions=True,
                )

                # Log per-symbol errors without crashing the cycle
                for sym, result in zip(self.symbols, results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"[{sym}] Pipeline error: {result}"
                        )

                now = datetime.now()
                sleep_secs = 60 - now.second
                logger.info(f"Cycle done. Sleeping {sleep_secs}s...")
                await asyncio.sleep(sleep_secs)

            except Exception as e:
                logger.error(f"Bot loop error: {e}", exc_info=True)
                await asyncio.sleep(60)