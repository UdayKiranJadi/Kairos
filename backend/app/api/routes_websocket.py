"""
routes_websocket.py — Real-time WebSocket push to dashboard.

WHY WEBSOCKET NOT SSE:
Server-Sent Events (SSE) are simpler but one-way and
HTTP/1.1 only. WebSocket is bidirectional and works
better with React. FastAPI supports both natively.

PUSH INTERVAL: 1 second.
WHY 1s NOT FASTER:
Alpaca bars arrive every 60s. Portfolio state changes
on order fills. 1s gives smooth UI updates without
hammering the DB. Sub-second would be wasteful.

WHAT WE PUSH EACH TICK:
{
  timestamp:     ISO string
  portfolio:     equity, cash, daily_pnl, drawdown_pct
  positions:     list of open positions with unrealized PnL
  recent_orders: last 10 orders
  agent_state:   last decision, confidence, ensemble reason
  risk_state:    circuit_breaker_active, daily_loss_pct
  stream_state:  last bar received per ticker
}
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select, desc

from app.db.session import AsyncSessionLocal
from app.db.models import PortfolioSnapshot, PaperOrder, Symbol, Position
from app.core.redis_client import get_redis

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store of last agent decision per symbol
# Updated by the bot loop, read by the WS push
_last_agent_state: dict = {}


def update_agent_state(symbol: str, state: dict) -> None:
    """Called by the bot after each decision cycle."""
    _last_agent_state[symbol] = {
        **state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _build_payload(db) -> dict:
    """Build the full dashboard payload from DB + Redis."""

    # ── Portfolio snapshot ────────────────────────────────────────
    snap_result = await db.execute(
        select(PortfolioSnapshot)
        .order_by(desc(PortfolioSnapshot.timestamp))
        .limit(1)
    )
    snap = snap_result.scalar_one_or_none()

    portfolio = {
        "equity":       snap.equity       if snap else 0.0,
        "cash":         snap.cash         if snap else 0.0,
        "daily_pnl":    snap.daily_pnl    if snap else 0.0,
        "total_pnl":    snap.total_pnl    if snap else 0.0,
        "drawdown_pct": snap.drawdown_pct if snap else 0.0,
        "buying_power": snap.buying_power if snap else 0.0,
    }

    # ── Open positions ────────────────────────────────────────────
    pos_result = await db.execute(
        select(Position, Symbol.ticker)
        .join(Symbol, Symbol.id == Position.symbol_id)
        .where(Position.is_open == True)
    )
    positions = [
        {
            "ticker":          row[1],
            "quantity":        row[0].quantity,
            "avg_entry_price": row[0].avg_entry_price,
            "current_price":   row[0].current_price,
            "unrealized_pnl":  row[0].unrealized_pnl,
        }
        for row in pos_result.all()
    ]

    # ── Recent orders ─────────────────────────────────────────────
    order_result = await db.execute(
        select(PaperOrder, Symbol.ticker)
        .join(Symbol, Symbol.id == PaperOrder.symbol_id)
        .order_by(desc(PaperOrder.timestamp))
        .limit(10)
    )
    orders = [
        {
            "id":              row[0].id,
            "ticker":          row[1],
            "side":            row[0].side,
            "qty":             row[0].quantity,
            "price":           row[0].submitted_price,
            "status":          row[0].status,
            "timestamp":       row[0].timestamp.isoformat(),
        }
        for row in order_result.all()
    ]

    # ── Stream state from Redis ───────────────────────────────────
    try:
        r = await get_redis()
        stream_state = {}
        for ticker in ["AAPL", "NVDA"]:
            bar = await r.hgetall(f"kairos:bar:{ticker}")
            if bar:
                stream_state[ticker] = {
                    "close":     float(bar.get("close", 0)),
                    "volume":    float(bar.get("volume", 0)),
                    "timestamp": bar.get("timestamp", ""),
                }
    except Exception:
        stream_state = {}

    # ── Equity curve (last 50 snapshots) ─────────────────────────
    curve_result = await db.execute(
        select(PortfolioSnapshot.timestamp, PortfolioSnapshot.equity)
        .order_by(desc(PortfolioSnapshot.timestamp))
        .limit(50)
    )
    equity_curve = [
        {
            "t": row[0].isoformat(),
            "equity": row[1],
        }
        for row in reversed(curve_result.all())
    ]

    return {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "portfolio":   portfolio,
        "positions":   positions,
        "orders":      orders,
        "agent_state": _last_agent_state,
        "stream":      stream_state,
        "equity_curve": equity_curve,
    }


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Push live dashboard state to connected browser.
    Sends every 1 second.
    Handles disconnects gracefully.
    """
    await websocket.accept()
    client = websocket.client
    logger.info(f"Dashboard WS connected: {client}")

    try:
        while True:
            async with AsyncSessionLocal() as db:
                payload = await _build_payload(db)

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"Dashboard WS disconnected: {client}")
    except Exception as e:
        logger.error(f"WS error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass