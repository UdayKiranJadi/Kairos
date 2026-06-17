"""
routes_websocket.py — Real-time WebSocket push to dashboard.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select, desc

from app.core.redis_client import get_redis
from app.db.models import PaperOrder, Position, PortfolioSnapshot, Symbol
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)
router = APIRouter()

_last_agent_state: dict = {}


def update_agent_state(symbol: str, state: dict) -> None:
    _last_agent_state[symbol] = {
        **state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _build_payload(db) -> dict:

    # ── Portfolio from Alpaca live ────────────────────────────────
    from app.agents.portfolio_agent import PortfolioAgent

    try:
        async with AsyncSessionLocal() as live_session:
            pa = PortfolioAgent(live_session)
            live = await pa.sync_and_get_state()
        equity = live["portfolio_value"]
        daily_pnl = live["daily_loss_pct"] * equity
        drawdown_pct = live["total_drawdown_pct"]
        total_pnl = equity - 100_000.0
    except Exception as e:
        logger.warning(f"Portfolio sync failed: {e}")
        equity = 0.0
        daily_pnl = 0.0
        drawdown_pct = 0.0
        total_pnl = 0.0

    portfolio = {
        "equity":       equity,
        "cash":         equity,
        "daily_pnl":    daily_pnl,
        "total_pnl":    total_pnl,
        "drawdown_pct": drawdown_pct,
        "buying_power": equity * 4,
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
            "id":        row[0].id,
            "ticker":    row[1],
            "side":      row[0].side,
            "qty":       row[0].quantity,
            "price":     row[0].submitted_price,
            "status":    row[0].status,
            "timestamp": row[0].timestamp.isoformat(),
        }
        for row in order_result.all()
    ]

    # ── Stream state from Redis ───────────────────────────────────
    stream_state = {}
    try:
        r = await get_redis()
        for ticker in ["AAPL", "NVDA"]:
            bar = await r.hgetall(f"kairos:bar:{ticker}")
            if bar:
                stream_state[ticker] = {
                    "close":     float(bar.get("close", 0)),
                    "volume":    float(bar.get("volume", 0)),
                    "timestamp": bar.get("timestamp", ""),
                }
    except Exception:
        pass

    # ── Equity curve ──────────────────────────────────────────────
    equity_curve = []
    try:
        curve_result = await db.execute(
            select(PortfolioSnapshot.timestamp, PortfolioSnapshot.equity)
            .order_by(desc(PortfolioSnapshot.timestamp))
            .limit(50)
        )
        rows = curve_result.all()
        equity_curve = [
            {"t": row[0].isoformat(), "equity": row[1]}
            for row in reversed(rows)
        ]
    except Exception:
        pass

    # ── Sharpe ratio ──────────────────────────────────────────────
    sharpe = 0.0
    try:
        sharpe_result = await db.execute(
            select(PortfolioSnapshot.equity)
            .order_by(desc(PortfolioSnapshot.timestamp))
            .limit(100)
        )
        equity_vals = [row[0] for row in sharpe_result.all()]
        if len(equity_vals) >= 10:
            vals = list(reversed(equity_vals))
            returns = np.diff(vals) / np.array(vals[:-1])
            std = returns.std()
            if std > 0:
                sharpe = float(returns.mean() / std * np.sqrt(252 * 390))
    except Exception:
        pass

    return {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "portfolio":    portfolio,
        "positions":    positions,
        "orders":       orders,
        "agent_state":  _last_agent_state,
        "stream":       stream_state,
        "equity_curve": equity_curve,
        "sharpe":       sharpe,
    }


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"Dashboard WS connected: {websocket.client}")

    try:
        while True:
            async with AsyncSessionLocal() as db:
                payload = await _build_payload(db)
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info(f"Dashboard WS disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WS error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass