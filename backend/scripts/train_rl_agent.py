"""
train_rl_agent.py — Offline PPO training on historical AAPL/NVDA data.
"""

import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from app.db.session import AsyncSessionLocal
from app.models.prediction_dataset import PredictionDatasetBuilder
from app.rl.trading_env import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Kairos.Train")

MODEL_DIR = Path("artifacts/rl")
LOG_DIR   = Path("artifacts/rl/logs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "return_1m", "return_5m", "volatility_10m",
    "volume_zscore", "price_vs_vwap",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",       default="AAPL")
    p.add_argument("--timesteps",    type=int, default=200_000)
    p.add_argument("--days-history", type=int, default=30)
    return p.parse_args()


async def load_data(ticker: str, days: int) -> pd.DataFrame:
    async with AsyncSessionLocal() as session:
        builder = PredictionDatasetBuilder(session)
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        df = await builder.load_feature_price_frame(
            ticker=ticker, start=start, end=end
        )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    df["close"] = df["close"].ffill().bfill().fillna(100.0)
    return df.reset_index(drop=True)


def dummy_df(n: int = 1000) -> pd.DataFrame:
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


def make_env(df: pd.DataFrame, seed: int = 0):
    def _init():
        return Monitor(TradingEnv(df=df))
    return _init


class RewardLogger(BaseCallback):
    def __init__(self, freq: int = 10_000):
        super().__init__()
        self.freq = freq
        self._rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._rewards.append(info["episode"]["r"])
        if self.num_timesteps % self.freq == 0 and self._rewards:
            mean = np.mean(self._rewards[-50:])
            logger.info(
                f"step={self.num_timesteps:>8,} | "
                f"ep_rew_mean={mean:+.4f} | "
                f"n_episodes={len(self._rewards)}"
            )
        return True


async def main():
    args   = parse_args()
    symbol = args.symbol.upper()

    logger.info(f"Loading {args.days_history}d of {symbol} from DB...")
    df = await load_data(symbol, args.days_history)

    if df.empty or len(df) < 100:
        logger.warning(f"Only {len(df)} rows in DB — using synthetic data")
        df = dummy_df(1000)
    else:
        logger.info(f"Loaded {len(df)} rows")

    df = clean(df)
    logger.info(f"Clean rows: {len(df)}")

    split    = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)
    logger.info(f"Train={len(train_df)} | Eval={len(eval_df)}")

    train_env = DummyVecEnv([make_env(train_df, seed=0)])
    eval_env  = DummyVecEnv([make_env(eval_df,  seed=1)])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        seed=42,
    )

    logger.info(f"Training PPO on {symbol} for {args.timesteps:,} steps...")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[
            RewardLogger(freq=10_000),
            CheckpointCallback(
                save_freq=50_000,
                save_path=str(MODEL_DIR / "checkpoints"),
                name_prefix=f"ppo_{symbol.lower()}",
                verbose=1,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(MODEL_DIR / "best"),
                log_path=str(LOG_DIR),
                eval_freq=20_000,
                n_eval_episodes=10,
                deterministic=True,
                verbose=1,
            ),
        ],
        progress_bar=True,
    )

    path = MODEL_DIR / f"ppo_{symbol.lower()}_final"
    model.save(str(path))
    logger.info(f"\n✓ Model saved → {path}.zip")

    # ── Final eval ───────────────────────────────────────────────
    logger.info("Running final evaluation (20 episodes)...")
    obs = eval_env.reset()
    rewards, cur = [], 0.0
    done_count = 0

    while done_count < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = eval_env.step(action)
        cur += float(rew[0])
        if done[0]:
            rewards.append(cur)
            cur = 0.0
            done_count += 1
            obs = eval_env.reset()

    mean_r = np.mean(rewards)
    logger.info(f"Eval mean={mean_r:+.4f} | "
                f"min={min(rewards):+.4f} | "
                f"max={max(rewards):+.4f}")

    if mean_r > 0:
        logger.info("✓ Agent profitable on eval. Ready for Day 4.")
    else:
        logger.info("⚠ Not yet profitable — but training pipeline works.")
        logger.info("  Load more data or increase timesteps if needed.")


if __name__ == "__main__":
    asyncio.run(main())