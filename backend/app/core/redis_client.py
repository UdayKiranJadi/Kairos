"""
redis_client.py — Shared Redis helpers used across the app.

WHY A SEPARATE MODULE:
stream_client.py owns the bar data in Redis.
Other parts of the system (dashboard, risk monitor, future
WebSocket push) also need Redis. This module provides a
shared connection pool so we don't open multiple connections.

One pool, many users.
"""

import redis.asyncio as aioredis
from app.core.config import settings

_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """
    Returns the shared async Redis client.
    Creates it on first call, reuses on all subsequent calls.
    Thread-safe for asyncio — one event loop, one pool.
    """
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _pool


async def ping() -> bool:
    """Health check — returns True if Redis is reachable."""
    try:
        r = await get_redis()
        return await r.ping()
    except Exception:
        return False