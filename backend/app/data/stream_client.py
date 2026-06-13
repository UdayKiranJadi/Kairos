"""
stream_client.py — Alpaca WebSocket live bar stream.

WHY THIS EXISTS:
The trading bot currently calls Alpaca REST every 60 seconds
to fetch the last 2 hours of bars. That's wasteful and slow.

This module opens ONE persistent WebSocket connection.
Alpaca pushes a bar the moment each 1-minute candle closes.
We write it to Redis immediately.

The bot then reads from Redis instead of calling Alpaca.
Result: the bot always has fresh data with zero extra API calls.

FLOW:
  Alpaca WS → on_bar() → Redis (kairos:bar:{TICKER})
                       → Postgres via bar_service (for history)
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone

import redis.asyncio as aioredis
import websockets

from app.core.config import settings

logger = logging.getLogger(__name__)

# How many bars to keep in memory per ticker.
# FeatureBuilder needs ~10 bars minimum for rolling calcs.
# 60 bars = 1 hour of 1-min data. Safe buffer.
BAR_WINDOW_SIZE = 60

# In-memory rolling window per ticker.
# deque auto-drops oldest when full — no manual cleanup needed.
_bar_windows: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=BAR_WINDOW_SIZE)
)

# Module-level Redis connection — created once, reused.
_redis: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


async def _write_bar_to_redis(ticker: str, bar: dict) -> None:
    """
    Write the latest bar to Redis as a simple hash.

    Key: kairos:bar:AAPL
    Fields: open, high, low, close, volume, timestamp

    WHY HASH NOT STRING:
    A hash lets the bot read individual fields (e.g. just
    "close") without deserializing a full JSON blob.
    hgetall() returns all fields in one O(1) call.

    We also append to a Redis List (kairos:bars:AAPL) so
    the feature builder can read the last N bars from memory
    instead of hitting Postgres every cycle.
    """
    r = await _get_redis()
    key_latest = f"kairos:bar:{ticker}"
    key_window = f"kairos:bars:{ticker}"

    bar_str = json.dumps(bar)

    pipe = r.pipeline()
    # Overwrite latest bar snapshot
    pipe.hset(key_latest, mapping={k: str(v) for k, v in bar.items()})
    # Append to rolling list and cap at BAR_WINDOW_SIZE
    pipe.rpush(key_window, bar_str)
    pipe.ltrim(key_window, -BAR_WINDOW_SIZE, -1)
    await pipe.execute()

    logger.info(
        f"Redis updated | {ticker} | close={bar['close']:.2f} | "
        f"vol={bar['volume']:.0f} | t={bar['timestamp']}"
    )


async def _authenticate(ws, symbols: list[str]) -> None:
    """
    Alpaca WS requires auth then subscription before data flows.
    Step 1: send credentials.
    Step 2: subscribe to 1-min bars for our symbols.
    """
    # Auth
    await ws.send(json.dumps({
        "action": "auth",
        "key": settings.alpaca_api_key,
        "secret": settings.alpaca_secret_key,
    }))
    resp = json.loads(await ws.recv())
    logger.info(f"WS auth response: {resp}")

    # Subscribe to bars
    await ws.send(json.dumps({
        "action": "subscribe",
        "bars": symbols,
    }))
    resp = json.loads(await ws.recv())
    logger.info(f"WS subscription response: {resp}")


async def stream_bars(symbols: list[str]) -> None:
    """
    Main streaming loop. Connects to Alpaca WS and processes bars.

    WHY RECONNECT LOOP WITH BACKOFF:
    WebSocket connections drop — network blips, Alpaca maintenance,
    timeouts. A silent drop is worse than a noisy crash.
    Exponential backoff (1s → 2s → 4s → max 60s) avoids
    hammering Alpaca's servers during an outage.
    """
    # Alpaca WS URL — IEX feed is free tier, SIP requires paid
    ws_url = "wss://stream.data.alpaca.markets/v2/iex"
    backoff = 1

    while True:
        try:
            logger.info(f"Connecting to Alpaca WS for: {symbols}")
            async with websockets.connect(ws_url) as ws:
                backoff = 1  # reset on successful connect
                await _authenticate(ws, symbols)
                logger.info("Streaming live bars...")

                async for raw in ws:
                    messages = json.loads(raw)
                    for msg in messages:
                        # "T": "b" means this is a bar message
                        if msg.get("T") != "b":
                            continue

                        ticker = msg["S"]
                        bar = {
                            "ticker":    ticker,
                            "open":      float(msg["o"]),
                            "high":      float(msg["h"]),
                            "low":       float(msg["l"]),
                            "close":     float(msg["c"]),
                            "volume":    float(msg["v"]),
                            "timestamp": msg["t"],
                        }

                        # Write to Redis immediately
                        await _write_bar_to_redis(ticker, bar)

        except websockets.ConnectionClosed as e:
            logger.warning(f"WS closed: {e}. Reconnecting in {backoff}s...")
        except Exception as e:
            logger.error(f"WS error: {e}. Reconnecting in {backoff}s...")

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


async def get_latest_bar(ticker: str) -> dict | None:
    """
    Read the latest bar for a ticker from Redis.
    Called by the trading bot instead of hitting Alpaca REST.

    Returns None if no bar exists yet (stream not started
    or market hasn't opened).
    """
    r = await _get_redis()
    key = f"kairos:bar:{ticker}"
    data = await r.hgetall(key)
    if not data:
        return None
    return {
        "ticker":    data["ticker"],
        "open":      float(data["open"]),
        "high":      float(data["high"]),
        "low":       float(data["low"]),
        "close":     float(data["close"]),
        "volume":    float(data["volume"]),
        "timestamp": data["timestamp"],
    }


async def get_bar_window(ticker: str, n: int = BAR_WINDOW_SIZE) -> list[dict]:
    """
    Read the last N bars for a ticker from Redis.
    Used by FeatureBuilder to compute rolling indicators
    without a Postgres query.

    WHY REDIS NOT POSTGRES:
    Postgres query for 60 rows takes ~5ms with index.
    Redis list read takes ~0.1ms.
    In a 60-second cycle, this is a minor win — but when
    we move to sub-minute decisions on Day 5+, it matters.
    """
    r = await _get_redis()
    key = f"kairos:bars:{ticker}"
    raw_list = await r.lrange(key, -n, -1)
    return [json.loads(b) for b in raw_list]