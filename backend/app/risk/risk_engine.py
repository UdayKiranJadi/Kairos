from app.risk.risk_policy import RiskPolicy
from app.schemas.trading import TradeAction, TradeDecision, RiskDecision


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
        if trading_mode != "paper":
            return RiskDecision(
                approved=False,
                reason="Automatic execution is allowed only in paper trading mode.",
            )

        if decision.action == TradeAction.HOLD:
            return RiskDecision(
                approved=True,
                reason="Hold action approved. No order required.",
            )

        if decision.confidence < self.policy.min_confidence:
            return RiskDecision(
                approved=False,
                reason="Model confidence is below minimum threshold.",
            )

        if daily_loss_pct <= -self.policy.max_daily_loss_pct:
            return RiskDecision(
                approved=False,
                reason="Daily loss limit reached.",
            )

        if total_drawdown_pct <= -self.policy.max_total_drawdown_pct:
            return RiskDecision(
                approved=False,
                reason="Total drawdown limit reached.",
            )

        if trades_today >= self.policy.max_trades_per_day:
            return RiskDecision(
                approved=False,
                reason="Max trades per day reached.",
            )

        if decision.action == TradeAction.ENTER_LONG:
            if open_positions >= self.policy.max_open_positions:
                return RiskDecision(
                    approved=False,
                    reason="Max open positions reached.",
                )

            max_position_value = portfolio_value * self.policy.max_position_pct

            return RiskDecision(
                approved=True,
                reason="Enter long approved by conservative risk policy.",
                max_position_value=max_position_value,
            )

        if decision.action == TradeAction.EXIT_POSITION:
            return RiskDecision(
                approved=True,
                reason="Exit position approved.",
            )

        return RiskDecision(
            approved=False,
            reason="Unknown action.",
        )