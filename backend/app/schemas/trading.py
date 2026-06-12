from enum import Enum
from pydantic import BaseModel, Field


class TradeAction(str, Enum):
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"
    EXIT_POSITION = "EXIT_POSITION"


class TradeDecision(BaseModel):
    symbol: str
    action: TradeAction
    confidence: float = Field(ge=0.0, le=1.0)
    predicted_return: float
    reason: str


class RiskDecision(BaseModel):
    approved: bool
    reason: str
    max_position_value: float | None = None