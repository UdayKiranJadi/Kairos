from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TradeAction(str, Enum):
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"
    EXIT_POSITION = "EXIT_POSITION"


class OrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Symbol(Base):
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    bars: Mapped[list["MarketBar"]] = relationship(back_populates="symbol")


class MarketBar(Base):
    __tablename__ = "market_bars"
    __table_args__ = (
        UniqueConstraint("symbol_id", "timeframe", "timestamp", name="uq_market_bar"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)

    timeframe: Mapped[str] = mapped_column(String(16), default="1Min")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    symbol: Mapped["Symbol"] = relationship(back_populates="bars")

class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"
    __table_args__ = (
        UniqueConstraint("symbol_id", "timestamp", name="uq_feature_snapshot"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )

    # Original 5 features
    return_1m:      Mapped[float | None] = mapped_column(Float, nullable=True)
    return_5m:      Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_10m: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_zscore:  Mapped[float | None] = mapped_column(Float, nullable=True)
    price_vs_vwap:  Mapped[float | None] = mapped_column(Float, nullable=True)

    # New features (RSI, MACD, OBV)
    rsi_14:      Mapped[float | None] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    obv_zscore:  Mapped[float | None] = mapped_column(Float, nullable=True)

    symbol: Mapped["Symbol"] = relationship()


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50))

    horizon_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    probability_up: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_class: Mapped[int | None] = mapped_column(Integer, nullable=True)

    predicted_return: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)

    symbol: Mapped["Symbol"] = relationship()


class RLAction(Base):
    __tablename__ = "rl_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )

    action: Mapped[str] = mapped_column(String(32))
    reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    policy_name: Mapped[str] = mapped_column(String(100))
    policy_version: Mapped[str] = mapped_column(String(50))

    observation_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    symbol: Mapped["Symbol"] = relationship()


class RiskCheck(Base):
    __tablename__ = "risk_checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int | None] = mapped_column(
        ForeignKey("symbols.id"),
        nullable=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )

    action: Mapped[str] = mapped_column(String(32))
    approved: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(Text)

    portfolio_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    daily_loss_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_drawdown_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    symbol: Mapped["Symbol"] = relationship()


class PaperOrder(Base):
    __tablename__ = "paper_orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    broker_order_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    side: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[float] = mapped_column(Float)
    order_type: Mapped[str] = mapped_column(String(32), default="market")
    status: Mapped[str] = mapped_column(String(32), default=OrderStatus.SUBMITTED.value)

    submitted_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    symbol: Mapped["Symbol"] = relationship()


class PaperFill(Base):
    __tablename__ = "paper_fills"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("paper_orders.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    fill_price: Mapped[float] = mapped_column(Float)
    fill_quantity: Mapped[float] = mapped_column(Float)

    order: Mapped["PaperOrder"] = relationship()


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), index=True)

    quantity: Mapped[float] = mapped_column(Float)
    avg_entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)

    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    is_open: Mapped[bool] = mapped_column(Boolean, default=True)

    symbol: Mapped["Symbol"] = relationship()


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )

    cash: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)
    buying_power: Mapped[float] = mapped_column(Float)
    daily_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    drawdown_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(100))
    message: Mapped[str] = mapped_column(Text)

    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)