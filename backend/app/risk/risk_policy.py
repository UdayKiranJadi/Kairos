from dataclasses import dataclass


@dataclass(frozen=True)
class RiskPolicy:
    name: str = "LEVEL_0_CONSERVATIVE"
    max_position_pct: float = 0.02
    max_daily_loss_pct: float = 0.005
    max_total_drawdown_pct: float = 0.02
    max_trades_per_day: int = 3
    max_open_positions: int = 1
    min_confidence: float = 0.70
    allow_shorting: bool = False
    allow_margin: bool = False
    allow_live_auto_trading: bool = False
    # NEW: ATR stop-loss multiplier
    # 2.0 = stop at entry minus 2× ATR
    atr_stop_multiplier: float = 2.0