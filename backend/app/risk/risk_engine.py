"""
risk_engine.py — Upgraded for Day 4.

WHAT CHANGED:
  Before: checks entry conditions only. Once in a trade,
          no monitoring. No circuit breaker.

  After:  adds two new methods:
    - check_stop_loss(): ATR-based per-position stop
    - check_circuit_breaker(): daily drawdown halt

WHY ATR-BASED STOPS:
Fixed dollar stops ($50 down = exit) are brittle.
ATR (Average True Range) adapts to current volatility.
High volatility day → wider stop (don't get shaken out).
Low volatility day → tighter stop (protect profit).
2× ATR is industry standard for intraday stops.

WHY CIRCUIT BREAKER:
If the bot loses 3% in a day, something is wrong —
market regime changed, model is wrong, or there's a bug.
The circuit breaker halts ALL trading for the rest of
the day. Human (you) reviews logs before next open.
This is the single most important safety feature.
"""

from app.risk.risk_policy import RiskPolicy
from app.schemas.trading import RiskDecision, TradeAction, TradeDecision


class RiskEngine:
    def __init__(self, policy: RiskPolicy):
        self.policy = policy

    def evaluate(
        self,
        decision: TradeDecision,
        portfolio_value: float,
        daily_loss_pct: float,
        total_drawdown_pct: float,
        trades_today: int,
        open_positions: int,
        trading_mode: str,
    ) -> RiskDecision:
        """Entry risk check — unchanged from original."""

        if trading_mode != "paper":
            return RiskDecision(
                approved=False,
                reason="Auto execution only in paper mode.",
            )

        if decision.action == TradeAction.HOLD:
            return RiskDecision(
                approved=True,
                reason="Hold approved.",
            )

        if decision.confidence < self.policy.min_confidence:
            return RiskDecision(
                approved=False,
                reason=f"Confidence {decision.confidence:.2f} below minimum {self.policy.min_confidence}.",
            )

        if daily_loss_pct <= -self.policy.max_daily_loss_pct:
            return RiskDecision(
                approved=False,
                reason=f"Daily loss {daily_loss_pct:.3%} hit limit {self.policy.max_daily_loss_pct:.3%}.",
            )

        if total_drawdown_pct <= -self.policy.max_total_drawdown_pct:
            return RiskDecision(
                approved=False,
                reason=f"Total drawdown {total_drawdown_pct:.3%} hit limit.",
            )

        if trades_today >= self.policy.max_trades_per_day:
            return RiskDecision(
                approved=False,
                reason=f"Max trades/day ({self.policy.max_trades_per_day}) reached.",
            )

        if decision.action == TradeAction.ENTER_LONG:
            if open_positions >= self.policy.max_open_positions:
                return RiskDecision(
                    approved=False,
                    reason=f"Max positions ({self.policy.max_open_positions}) reached.",
                )
            max_position_value = portfolio_value * self.policy.max_position_pct
            return RiskDecision(
                approved=True,
                reason="Entry approved by conservative risk policy.",
                max_position_value=max_position_value,
            )

        if decision.action == TradeAction.EXIT_POSITION:
            return RiskDecision(
                approved=True,
                reason="Exit approved.",
            )

        return RiskDecision(approved=False, reason="Unknown action.")

    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
    ) -> tuple[bool, str]:
        """
        ATR-based stop-loss check for open positions.

        Called every cycle for each open position.
        Returns (should_exit, reason).

        WHY ATR NOT FIXED PCT:
        AAPL on a calm day has ATR ~$0.50.
        AAPL on earnings day has ATR ~$3.00.
        A fixed 1% stop on earnings day = guaranteed stop-out.
        ATR-based stop respects current market conditions.

        Formula: stop_price = entry - (ATR_MULTIPLIER × ATR)
        Default multiplier = 2.0 (from RiskPolicy)
        """
        if entry_price <= 0 or atr <= 0:
            return False, ""

        stop_price = entry_price - (self.policy.atr_stop_multiplier * atr)

        if current_price <= stop_price:
            return True, (
                f"ATR stop triggered: price {current_price:.2f} ≤ "
                f"stop {stop_price:.2f} "
                f"(entry={entry_price:.2f}, ATR={atr:.4f}, "
                f"mult={self.policy.atr_stop_multiplier}×)"
            )

        return False, ""

    def check_circuit_breaker(
        self,
        daily_loss_pct: float,
        total_drawdown_pct: float,
    ) -> tuple[bool, str]:
        """
        Portfolio-level circuit breaker.

        Called once per cycle BEFORE any symbol processing.
        If triggered, the bot skips all symbols for that cycle.

        WHY SEPARATE FROM evaluate():
        evaluate() is per-trade. The circuit breaker is
        portfolio-wide — it doesn't matter which symbol
        caused the loss, everything halts.

        Returns (should_halt, reason).
        """
        if daily_loss_pct <= -self.policy.max_daily_loss_pct:
            return True, (
                f"CIRCUIT BREAKER: daily loss {daily_loss_pct:.3%} "
                f"exceeds limit {self.policy.max_daily_loss_pct:.3%}. "
                f"Halting all trading until market close."
            )

        if total_drawdown_pct <= -self.policy.max_total_drawdown_pct:
            return True, (
                f"CIRCUIT BREAKER: total drawdown {total_drawdown_pct:.3%} "
                f"exceeds limit {self.policy.max_total_drawdown_pct:.3%}. "
                f"Halting all trading."
            )

        return False, ""