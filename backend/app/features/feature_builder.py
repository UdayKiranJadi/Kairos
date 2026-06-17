import math
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FeatureSnapshot, MarketBar, Symbol


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


class FeatureBuilder:
    """
    Builds causal intraday features from stored market bars.

    Causal means each feature row only uses information available
    at or before that timestamp — no lookahead bias.

    Features (8 total):
      Original 5: return_1m, return_5m, volatility_10m,
                  volume_zscore, price_vs_vwap
      New 3:      rsi_14, macd_signal, obv_zscore

    WHY THESE THREE:
      RSI(14)   — momentum oscillator. Tells the agent whether
                  the asset is overbought/oversold. The single
                  most widely used intraday indicator.

      MACD signal line — difference between fast and slow EMA,
                  smoothed. Captures trend direction AND momentum
                  changes. The crossover is a classical entry signal.

      OBV z-score — On-Balance Volume tracks whether volume is
                  flowing into or out of the asset. High OBV on
                  an up move = institutional buying. This is the
                  one feature that sees what smart money is doing.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_symbol(self, ticker: str) -> Symbol | None:
        result = await self.db.execute(
            select(Symbol).where(Symbol.ticker == ticker.upper())
        )
        return result.scalar_one_or_none()

    async def load_bars(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        symbol = await self.get_symbol(ticker)
        if symbol is None:
            return pd.DataFrame()

        query = select(MarketBar).where(MarketBar.symbol_id == symbol.id)
        if start is not None:
            query = query.where(MarketBar.timestamp >= start)
        if end is not None:
            query = query.where(MarketBar.timestamp <= end)
        query = query.order_by(MarketBar.timestamp.asc())

        result = await self.db.execute(query)
        bars = list(result.scalars().all())
        if not bars:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "symbol_id": bar.symbol_id,
                "timestamp": bar.timestamp,
                "open":      bar.open,
                "high":      bar.high,
                "low":       bar.low,
                "close":     bar.close,
                "volume":    bar.volume,
            }
            for bar in bars
        ])

    # ── Private indicator helpers ────────────────────────────────

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        """
        Exponential moving average.
        adjust=False matches how trading platforms compute EMA —
        each value depends only on past values (causal).
        """
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI(14) — normalized to [-1, +1] so the RL agent's
        observation space stays bounded.

        Raw RSI: 0 = max oversold, 100 = max overbought.
        Normalized: -1 = max oversold, 0 = neutral, +1 = max overbought.

        WHY NORMALIZE: The RL observation space is bounded [-5, 5].
        Raw RSI (0-100) would dominate every other feature numerically.
        """
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        # Wilder smoothing (equivalent to EMA with alpha=1/period)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Normalize: (RSI - 50) / 50 → [-1, +1]
        return (rsi - 50) / 50

    @staticmethod
    def _macd_signal(close: pd.Series,
                     fast: int = 12,
                     slow: int = 26,
                     signal: int = 9) -> pd.Series:
        """
        MACD signal line, normalized by a rolling std.

        We return the SIGNAL LINE (9-period EMA of MACD), not the
        raw MACD line. The signal line is smoother and generates
        fewer false positives — better for a 1-minute timeframe.

        Normalized by rolling std so the value is scale-independent
        across different price levels (AAPL $200 vs SPY $500).
        """
        ema_fast   = close.ewm(span=fast,   adjust=False).mean()
        ema_slow   = close.ewm(span=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Normalize: divide by rolling std of MACD line
        rolling_std = macd_line.rolling(window=50).std().replace(0, np.nan)
        return (signal_line / rolling_std).clip(-3, 3)

    @staticmethod
    def _obv_zscore(close: pd.Series,
                    volume: pd.Series,
                    window: int = 20) -> pd.Series:
        """
        On-Balance Volume z-score.

        OBV accumulates volume: add volume on up days, subtract on
        down days. Captures whether volume is confirming price moves.

        Rising price + rising OBV = confirmed uptrend (institutions buying).
        Rising price + falling OBV = unconfirmed move (weak, likely to fail).

        We z-score over a 20-bar window so the value is stationary
        and comparable across different time periods.
        """
        direction = np.sign(close.diff()).fillna(0)
        obv       = (direction * volume).cumsum()

        obv_mean = obv.rolling(window=window).mean()
        obv_std  = obv.rolling(window=window).std().replace(0, np.nan)

        return ((obv - obv_mean) / obv_std).clip(-3, 3)

    # ── Main feature computation ─────────────────────────────────

    def build_features_from_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return bars

        df = bars.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"]   = pd.to_datetime(df["timestamp"])
        df["trading_day"] = df["timestamp"].dt.date

        # ── Original 5 features (unchanged) ──────────────────────

        df["return_1m"]      = df["close"].pct_change(periods=1)
        df["return_5m"]      = df["close"].pct_change(periods=5)
        df["volatility_10m"] = df["return_1m"].rolling(window=10).std()

        rolling_vol_mean = df["volume"].rolling(window=20).mean()
        rolling_vol_std  = df["volume"].rolling(window=20).std()
        df["volume_zscore"] = (
            (df["volume"] - rolling_vol_mean)
            / rolling_vol_std.replace(0, np.nan)
        )

        typical_price             = (df["high"] + df["low"] + df["close"]) / 3
        df["typical_price_volume"] = typical_price * df["volume"]
        df["cum_tpv"]    = df.groupby("trading_day")["typical_price_volume"].cumsum()
        df["cum_volume"] = df.groupby("trading_day")["volume"].cumsum()
        df["vwap"]       = df["cum_tpv"] / df["cum_volume"].replace(0, np.nan)
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

        # ── New 3 features ────────────────────────────────────────

        df["rsi_14"]      = self._rsi(df["close"], period=14)
        df["macd_signal"] = self._macd_signal(df["close"])
        df["obv_zscore"]  = self._obv_zscore(df["close"], df["volume"])

        return df[[
            "symbol_id",
            "timestamp",
            "return_1m",
            "return_5m",
            "volatility_10m",
            "volume_zscore",
            "price_vs_vwap",
            "rsi_14",        # NEW
            "macd_signal",   # NEW
            "obv_zscore",    # NEW
        ]]

    async def store_features_for_symbol(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        bars = await self.load_bars(ticker=ticker, start=start, end=end)
        if bars.empty:
            return 0

        features = self.build_features_from_bars(bars)
        inserted_count = 0

        for row in features.to_dict(orient="records"):
            existing = await self.db.execute(
                select(FeatureSnapshot).where(
                    FeatureSnapshot.symbol_id == row["symbol_id"],
                    FeatureSnapshot.timestamp == row["timestamp"].to_pydatetime(),
                )
            )
            if existing.scalar_one_or_none():
                continue

            snapshot = FeatureSnapshot(
                symbol_id=int(row["symbol_id"]),
                timestamp=row["timestamp"].to_pydatetime(),
                return_1m=safe_float(row["return_1m"]),
                return_5m=safe_float(row["return_5m"]),
                volatility_10m=safe_float(row["volatility_10m"]),
                volume_zscore=safe_float(row["volume_zscore"]),
                price_vs_vwap=safe_float(row["price_vs_vwap"]),
                rsi_14=safe_float(row["rsi_14"]),         # NEW
                macd_signal=safe_float(row["macd_signal"]), # NEW
                obv_zscore=safe_float(row["obv_zscore"]),   # NEW
            )
            self.db.add(snapshot)
            inserted_count += 1

        await self.db.commit()
        return inserted_count

    async def list_recent_features(
        self,
        ticker: str,
        limit: int = 100,
    ) -> list[FeatureSnapshot]:
        symbol = await self.get_symbol(ticker)
        if symbol is None:
            return []

        result = await self.db.execute(
            select(FeatureSnapshot)
            .where(FeatureSnapshot.symbol_id == symbol.id)
            .order_by(FeatureSnapshot.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())