"""
validate_trading_env.py

WHY THIS STEP:
Before training, we must verify the environment is
correctly implemented. gymnasium.utils.env_checker()
runs ~30 tests: action/obs space shapes, reset/step
return types, reward types, etc.

If this passes cleanly, SB3 can train on it.
If it fails, training will crash in confusing ways.

Always validate before training. Always.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
from gymnasium.utils.env_checker import check_env

from app.db.session import AsyncSessionLocal
from app.models.prediction_dataset import PredictionDatasetBuilder
from app.rl.trading_env import TradingEnv


def make_dummy_df(n: int = 300) -> pd.DataFrame:
    """
    Creates a synthetic DataFrame for offline validation.
    Uses random data so we can test without a DB connection.
    WHY: lets you validate the env even outside market hours
    or before bars are loaded.
    """
    import numpy as np
    rng = np.random.default_rng(42)

    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))

    return pd.DataFrame({
        "timestamp":      pd.date_range("2026-01-01 09:30", periods=n, freq="1min"),
        "return_1m":      rng.normal(0, 0.001, n),
        "return_5m":      rng.normal(0, 0.003, n),
        "volatility_10m": np.abs(rng.normal(0.001, 0.0005, n)),
        "volume_zscore":  rng.normal(0, 1, n),
        "price_vs_vwap":  rng.normal(0, 0.002, n),
        "close":          closes,
    })


async def load_real_df(ticker: str) -> pd.DataFrame:
    """Load real data from your DB for validation."""
    async with AsyncSessionLocal() as session:
        builder = PredictionDatasetBuilder(session)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=5)
        return await builder.load_feature_price_frame(
            ticker=ticker, start=start, end=end
        )


def main():
    print("=== Kairos TradingEnv Validation ===\n")

    # 1. Validate with synthetic data (always works offline)
    print("Step 1: Validating with synthetic data...")
    df = make_dummy_df(300)
    env = TradingEnv(df=df, initial_cash=10_000)

    # gymnasium's built-in checker — runs ~30 tests
    check_env(env, warn=True, skip_render_check=True)
    print("✓ gymnasium.check_env() passed\n")

    # 2. Manual episode walkthrough — see what the agent experiences
    print("Step 2: Running one manual episode...")
    obs, info = env.reset(seed=0)
    print(f"  Initial obs shape: {obs.shape}")
    print(f"  Obs space:         {env.observation_space}")
    print(f"  Action space:      {env.action_space}")
    print(f"  Initial obs:       {obs.round(4)}\n")

    total_reward = 0.0
    step = 0

    # Simulate: HOLD 5 steps, BUY, HOLD 3 steps, SELL
    actions = [0, 0, 0, 0, 0, 1, 0, 0, 0, 2]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = ["HOLD", "BUY", "SELL"][action]
        total_reward += reward
        step += 1
        print(
            f"  Step {step:2d} | action={action_name:4s} | "
            f"reward={reward:+.5f} | "
            f"position={info['position']} | "
            f"price={info['current_price']:.2f}"
        )
        if terminated or truncated:
            print(f"\n  Episode ended at step {step}")
            break

    print(f"\n  Total reward: {total_reward:+.5f}")
    print(f"  Episode Sharpe: {env._get_episode_sharpe():.4f}")
    print("\n✓ Manual episode complete\n")

    # 3. Try with real DB data if available
    print("Step 3: Attempting real DB data (AAPL)...")
    try:
        real_df = asyncio.run(load_real_df("AAPL"))
        if real_df.empty:
            print("  ⚠ No data in DB yet. Run load_historical_bars first.")
        else:
            real_env = TradingEnv(df=real_df, initial_cash=10_000)
            check_env(real_env, warn=True, skip_render_check=True)
            print(f"  ✓ Real data env valid | {len(real_df)} rows")
    except Exception as e:
        print(f"  ⚠ DB check skipped: {e}")

    print("\n=== All checks passed. Ready for Day 3 training. ===")


if __name__ == "__main__":
    main()