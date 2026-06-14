"""
validate_trading_env.py — Validates TradingEnv before training.
Run this before every training run.
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
    """Synthetic data — no DB needed. Volatility floored so no zeros."""
    rng = np.random.default_rng(42)
    closes = 150.0 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({
        "timestamp":      pd.date_range("2026-01-01 09:30", periods=n, freq="1min"),
        "return_1m":      rng.normal(0, 0.001, n),
        "return_5m":      rng.normal(0, 0.003, n),
        "volatility_10m": np.abs(rng.normal(0.003, 0.001, n)) + 0.001,
        "volume_zscore":  rng.normal(0, 1, n),
        "price_vs_vwap":  rng.normal(0, 0.002, n),
        "close":          closes,
    })


async def load_real_df(ticker: str) -> pd.DataFrame:
    async with AsyncSessionLocal() as session:
        builder = PredictionDatasetBuilder(session)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=10)
        return await builder.load_feature_price_frame(
            ticker=ticker, start=start, end=end
        )


def run_manual_episode(env: TradingEnv) -> None:
    obs, _ = env.reset(seed=0)
    print(f"  obs shape : {obs.shape}")
    print(f"  obs space : {env.observation_space}")
    print(f"  act space : {env.action_space}")
    print(f"  n_steps   : {env.n_steps}")
    print(f"  initial   : {obs.round(4)}\n")

    actions = [0, 0, 0, 0, 0, 1, 0, 0, 0, 2]
    total = 0.0
    for i, a in enumerate(actions):
        obs, rew, term, trunc, info = env.step(a)
        name = ["HOLD", "BUY ", "SELL"][a]
        total += rew
        print(
            f"  step {i+1:2d} | {name} | "
            f"reward={rew:+.5f} | "
            f"pos={info['position']} | "
            f"price={info['current_price']:.2f}"
        )
        if term or trunc:
            print(f"\n  Episode ended at step {i+1}")
            break

    print(f"\n  Total reward  : {total:+.5f}")
    sharpe = env._get_episode_sharpe()
    print(f"  Episode Sharpe: {sharpe:.4f}")


def main():
    print("=" * 45)
    print("  Kairos TradingEnv Validation")
    print("=" * 45 + "\n")

    # ── Step 1: gymnasium checker on synthetic data ──────────────
    print("Step 1: gymnasium.check_env() on synthetic data...")
    df = make_dummy_df(400)
    env = TradingEnv(df=df)
    check_env(env, skip_render_check=True)
    print("✓ check_env passed\n")

    # ── Step 2: manual episode walkthrough ───────────────────────
    print("Step 2: Manual episode walkthrough...")
    env2 = TradingEnv(df=df)
    run_manual_episode(env2)
    print("\n✓ Manual episode complete\n")

    # ── Step 3: real DB data (optional) ──────────────────────────
    print("Step 3: Real DB data check (AAPL)...")
    try:
        real_df = asyncio.run(load_real_df("AAPL"))
        if real_df.empty or len(real_df) < 10:
            print("  ⚠ No/insufficient DB data. Run load_historical_bars first.")
        else:
            real_env = TradingEnv(df=real_df)
            check_env(real_env, skip_render_check=True)
            print(f"  ✓ Real data env valid | {len(real_df)} rows")
    except Exception as e:
        print(f"  ⚠ DB check skipped: {e}")

    print("\n✓ All checks passed — ready to train.\n")


if __name__ == "__main__":
    main()