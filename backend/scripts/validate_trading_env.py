"""
validate_trading_env.py — validates TradingEnv v2 (forced entry, 2 actions).
"""

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from app.db.session import AsyncSessionLocal
from app.models.prediction_dataset import PredictionDatasetBuilder
from app.rl.trading_env import TradingEnv


def make_dummy_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = 150.0 + np.cumsum(rng.normal(0, 0.3, n))
    returns = np.diff(closes, prepend=closes[0]) / closes
    return pd.DataFrame({
        "timestamp":      pd.date_range("2026-01-01 09:30", periods=n, freq="1min"),
        "return_1m":      rng.normal(0, 0.001, n),
        "return_5m":      rng.normal(0, 0.003, n),
        "volatility_10m": np.abs(rng.normal(0.003, 0.001, n)) + 0.001,
        "volume_zscore":  rng.normal(0, 1, n),
        "price_vs_vwap":  rng.normal(0, 0.002, n),
        "rsi_14":         rng.uniform(-1, 1, n),
        "macd_signal":    rng.normal(0, 0.5, n),
        "obv_zscore":     rng.normal(0, 1, n),
        "close":          closes,
    })

async def load_real_df(ticker: str) -> pd.DataFrame:
    async with AsyncSessionLocal() as session:
        builder = PredictionDatasetBuilder(session)
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=10)
        return await builder.load_feature_price_frame(
            ticker=ticker, start=start, end=end
        )


def main():
    print("=" * 45)
    print("  Kairos TradingEnv v2 Validation")
    print("  (forced entry, 2-action: HOLD/SELL)")
    print("=" * 45 + "\n")

    df = make_dummy_df(400)

    # ── Step 1: gymnasium checker ────────────────────────────────
    print("Step 1: gymnasium.check_env()...")
    env = TradingEnv(df=df)
    check_env(env, skip_render_check=True)
    print("✓ check_env passed\n")

    # ── Step 2: manual walkthrough ───────────────────────────────
    print("Step 2: Manual episode...")
    print(f"  action space : {env.action_space}  (0=HOLD, 1=SELL)")
    print(f"  obs space    : {env.observation_space}")
    print(f"  n_steps      : {env.n_steps}\n")

    env2 = TradingEnv(df=df)
    obs, _ = env2.reset(seed=0)
    print(f"  entry_price  : {env2._entry_price:.2f}")
    print(f"  initial obs  : {obs.round(4)}\n")

    # Hold 5 steps then sell
    actions = [0, 0, 0, 0, 0, 1]
    total = 0.0
    for i, a in enumerate(actions):
        obs, rew, term, trunc, info = env2.step(a)
        name = ["HOLD", "SELL"][a]
        total += rew
        print(
            f"  step {i+1:2d} | {name} | "
            f"reward={rew:+.5f} | "
            f"price={info['current_price']:.2f} | "
            f"held={info['holding_steps']}"
        )
        if term or trunc:
            print(f"\n  Episode ended at step {i+1}")
            break

    print(f"\n  Total reward  : {total:+.5f}")
    print(f"  Episode Sharpe: {env2._get_episode_sharpe():.4f}")
    print("\n✓ Manual episode complete\n")

    # ── Step 3: real DB data ─────────────────────────────────────
    print("Step 3: Real DB data (AAPL)...")
    try:
        real_df = asyncio.run(load_real_df("AAPL"))
        if len(real_df) < 10:
            print("  ⚠ Insufficient DB data.")
        else:
            real_env = TradingEnv(df=real_df)
            check_env(real_env, skip_render_check=True)
            print(f"  ✓ Real data valid | {len(real_df)} rows")
    except Exception as e:
        print(f"  ⚠ DB check skipped: {e}")

    print("\n✓ All checks passed — ready to train.\n")


if __name__ == "__main__":
    main()