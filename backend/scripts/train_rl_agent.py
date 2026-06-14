"""
train_rl_agent.py — Offline PPO training on historical AAPL/NVDA data.

FLOW:
  Postgres (FeatureSnapshot + MarketBar)
    → clean DataFrame
      → TradingEnv (gymnasium)
        → PPO (stable-baselines3)
          → artifacts/rl/ppo_{symbol}_final.zip

HOW TO KNOW IT'S LEARNING:
  Watch ep_rew_mean in logs. Should trend upward.
  Target: ep_rew_mean > 0 after 100k steps.
  If flat: increase --timesteps or load more history.

AFTER TRAINING:
  artifacts/rl/ppo_aapl_final.zip    ← final model
  artifacts/rl/best/best_model.zip   ← best eval checkpoint
  artifacts/rl/checkpoints/          ← saves every 25k steps
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
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("artifacts/rl/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Args ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total PPO training steps. 100k ≈ 10-20 min on CPU."
    )
    parser.add_argument(
        "--days-history", type=int, default=30,
        help="Days of historical data to load from DB."
    )
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────

async def load_training_data(ticker: str, days: int) -> pd.DataFrame:
    """
    Load feature + price data from Postgres using your existing builder.
    Same DataFrame the LogReg model was trained on — no duplication.
    """
    async with AsyncSessionLocal() as session:
        builder = PredictionDatasetBuilder(session)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        return await builder.load_feature_price_frame(
            ticker=ticker, start=start, end=end
        )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN in any feature or price column.

    WHY DROP NOT FILL:
    NaNs occur at session open when rolling windows aren't warm yet
    (first 10-20 bars of each trading day). Filling with 0 or mean
    invents market data. Dropping is honest — we lose 38 rows out
    of 1995 (1.9%). Acceptable cost for clean training data.

    A single NaN in an observation propagates through the neural
    network as NaN → NaN logits → Categorical crash. This is
    exactly the crash we saw. Prevention is here.
    """
    cols = [
        "return_1m", "return_5m", "volatility_10m",
        "volume_zscore", "price_vs_vwap", "close",
    ]
    before = len(df)
    df = df.dropna(subset=cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} NaN rows → {len(df)} clean rows remain")

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} clean rows after NaN removal. "
            "Load more historical data: python scripts/load_historical_bars.py"
        )
    return df


def make_dummy_df(n: int = 1000) -> pd.DataFrame:
    """
    Synthetic data fallback when DB has no bars.
    Uses realistic volatility values — no NaN issues.
    """
    rng = np.random.default_rng(42)
    closes = 150.0 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({
        "timestamp":      pd.date_range("2026-01-01 09:30", periods=n, freq="1min"),
        "return_1m":      rng.normal(0, 0.001, n),
        "return_5m":      rng.normal(0, 0.003, n),
        "volatility_10m": np.abs(rng.normal(0.002, 0.001, n)) + 0.0005,
        "volume_zscore":  rng.normal(0, 1, n),
        "price_vs_vwap":  rng.normal(0, 0.002, n),
        "close":          closes,
    })


# ── Environment factory ───────────────────────────────────────────────────

def make_env(df: pd.DataFrame, seed: int = 0):
    """
    Factory for a monitored TradingEnv.

    WHY Monitor: SB3 needs it to compute ep_rew_mean in logs.
    WHY DummyVecEnv: SB3 requires vectorized env. Single env
    is simplest — add SubprocVecEnv later for parallel training.
    """
    def _init():
        env = TradingEnv(df=df, initial_cash=10_000)
        env = Monitor(env)
        return env
    return _init


# ── Callbacks ─────────────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    Logs mean episode reward every N steps.

    WHY: SB3 only logs ep_rew_mean when episodes complete,
    which can be sparse. This gives a smooth running view
    so you spot problems early (flat reward = bad shaping).
    """

    def __init__(self, log_freq: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self.log_freq == 0 and self._episode_rewards:
            recent = self._episode_rewards[-50:]
            mean_r = np.mean(recent)
            logger.info(
                f"Step {self.num_timesteps:>8,} | "
                f"ep_rew_mean={mean_r:+.5f} | "
                f"total_episodes={len(self._episode_rewards)}"
            )
        return True


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()
    symbol = args.symbol.upper()

    # 1. Load data
    logger.info(f"Loading training data for {symbol} ({args.days_history} days)...")
    df = await load_training_data(symbol, args.days_history)

    if df.empty or len(df) < 100:
        logger.warning(
            f"Insufficient DB data ({len(df)} rows). Using synthetic data.\n"
            "To train on real data:\n"
            "  python scripts/load_historical_bars.py "
            "--symbols AAPL,NVDA --start 2026-05-01T13:30:00 --end 2026-06-13T20:00:00\n"
            "  python scripts/build_features.py --symbols AAPL,NVDA"
        )
        df = make_dummy_df(1000)
    else:
        logger.info(f"Loaded {len(df)} rows from DB for {symbol}")
        df = clean_dataframe(df)   # ← removes NaN rows that crash training

    # 2. Sanity check — no NaNs should reach the env
    nan_count = df[TradingEnv.FEATURE_COLS + ["close"]].isna().sum().sum()
    if nan_count > 0:
        raise RuntimeError(
            f"DataFrame still has {nan_count} NaN values after cleaning. "
            "Check feature builder output."
        )
    logger.info(f"NaN check passed — 0 NaN values in {len(df)} rows")

    # 3. Train/eval split — same 80/20 as your LogReg trainer
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    eval_df  = df.iloc[split:].reset_index(drop=True)

    logger.info(
        f"Train rows: {len(train_df)} | "
        f"Eval rows:  {len(eval_df)} | "
        f"Timesteps:  {args.timesteps:,}"
    )

    # 4. Build vectorized environments
    train_env = DummyVecEnv([make_env(train_df, seed=0)])
    eval_env  = DummyVecEnv([make_env(eval_df,  seed=1)])

    # 5. Build PPO model
    # Hyperparameter reasoning:
    #   n_steps=2048    collect 2048 steps before each policy update
    #   batch_size=64   mini-batch size for gradient update
    #   n_epochs=10     passes over collected data before discarding
    #   learning_rate=3e-4  Adam default, stable for most RL problems
    #   gamma=0.99      discount — values future rewards almost as much as now
    #   ent_coef=0.01   entropy bonus prevents premature convergence to one action
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        seed=42,
    )

    logger.info(
        "\nStarting PPO training...\n"
        "Watch ep_rew_mean — should trend upward.\n"
        "Target: ep_rew_mean > 0 after 100k steps.\n"
    )

    # 6. Callbacks
    reward_logger = RewardLoggerCallback(log_freq=5_000)

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix=f"ppo_{symbol.lower()}",
        verbose=1,
    )

    # EvalCallback: evaluate on held-out data every 10k steps
    # If eval reward drops while train reward rises → overfitting
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(LOG_DIR),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # 7. Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=[reward_logger, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # 8. Save final model
    final_path = MODEL_DIR / f"ppo_{symbol.lower()}_final"
    model.save(str(final_path))
    logger.info(f"\n✓ Model saved → {final_path}.zip")

    # 9. Final evaluation — 20 episodes on held-out data
    logger.info("\nRunning final evaluation (20 episodes)...")
    obs = eval_env.reset()
    episode_rewards: list[float] = []
    current_ep_reward = 0.0

    while len(episode_rewards) < 20:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        current_ep_reward += float(reward[0])
        if done[0]:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0.0
            obs = eval_env.reset()

    mean_r = np.mean(episode_rewards)
    std_r  = np.std(episode_rewards)

    logger.info(f"Eval mean reward : {mean_r:+.5f} ± {std_r:.5f}")
    logger.info(f"Eval min / max   : {min(episode_rewards):+.5f} / {max(episode_rewards):+.5f}")

    if mean_r > 0:
        logger.info(
            "\n✓ Agent is profitable on held-out data.\n"
            "  Commit Day 3 and move to Day 4 — wiring RL into live bot."
        )
    else:
        logger.info(
            "\n⚠ Agent not yet profitable. Options:\n"
            "  1. Increase --timesteps 200000\n"
            "  2. Load more real historical data\n"
            "  3. Review reward shaping with mentor on Day 4"
        )

    logger.info(f"\nArtifacts saved to: {MODEL_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())