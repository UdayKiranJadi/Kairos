"""
trading_env.py — Kairos RL trading environment v2.

KEY DESIGN CHANGE FROM v1:
v1 problem: agent learns to never buy (entropy collapses,
            ep_len grows to 600+, policy_gradient ≈ 0).

v2 solution: FORCED ENTRY. Every episode starts with the
agent already in a long position at the current bar's price.
The agent only decides: hold or sell (2 actions, not 3).

WHY THIS WORKS:
The "never trade" collapse happens because BUY has a cost
and the agent finds safety in never paying it. By forcing
entry, we remove that escape route. The agent must learn
WHEN to exit — which is the harder and more valuable skill.
This mirrors how real momentum strategies work: signals
trigger entry, the model manages exit timing.

OBSERVATION SPACE (7 dimensions):
  [0] return_1m        — recent momentum
  [1] return_5m        — medium momentum  
  [2] volatility_10m   — market volatility
  [3] volume_zscore    — volume signal
  [4] price_vs_vwap    — price vs average
  [5] holding_steps    — how long in trade (normalized)
  [6] unrealized_pnl   — current P&L (normalized)

ACTION SPACE (2 discrete):
  0 = HOLD  — stay in position
  1 = SELL  — exit position, end episode

REWARD:
  On HOLD: tiny reward = unrealized_pnl direction signal
  On SELL: clipped percentage return × 100
  Force-close at max_holding_steps with same reward
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):

    metadata = {"render_modes": []}

    
    # NEW — 8 features
    FEATURE_COLS = [
    "return_1m",
    "return_5m",
    "volatility_10m",
    "volume_zscore",
    "price_vs_vwap",
    "rsi_14",
    "macd_signal",
    "obv_zscore",
]

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 10_000.0,
        max_holding_steps: int = 30,
        transaction_cost_pct: float = 0.001,
    ):
        super().__init__()

        missing = [c for c in self.FEATURE_COLS + ["close"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df = df.copy().reset_index(drop=True)
        self.df[self.FEATURE_COLS] = self.df[self.FEATURE_COLS].fillna(0.0)
        self.df["close"] = self.df["close"].ffill().bfill().fillna(100.0)
        self.n_steps = len(self.df)

        if self.n_steps < 10:
            raise ValueError(f"Too few rows: {self.n_steps}")

        self.initial_cash = initial_cash
        self.max_holding_steps = max_holding_steps
        self.transaction_cost_pct = transaction_cost_pct

        # 2 actions: 0=HOLD, 1=SELL
        self.action_space = spaces.Discrete(2)

        # 7-dim observation
        # NEW
        self.observation_space = spaces.Box(
    low=-5.0, high=5.0, shape=(10,), dtype=np.float32
)

        self._step = 0
        self._entry_price = 0.0
        self._holding_steps = 0
        self._returns: list[float] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start anywhere in first 60% of data
        # Need at least max_holding_steps rows remaining
        max_start = max(0, int(self.n_steps * 0.6) - self.max_holding_steps - 1)
        self._step = int(self.np_random.integers(0, max(1, max_start)))

        # FORCE ENTRY: agent starts already in a position
        # Entry price includes transaction cost
        entry_bar = self.df.iloc[self._step]
        self._entry_price = float(entry_bar["close"]) * (1.0 + self.transaction_cost_pct)
        self._holding_steps = 0
        self._returns = []

        # Advance one step so first obs is AFTER entry
        self._step += 1

        return self._obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        if self._step >= self.n_steps:
            return self._obs(), 0.0, False, True, {}

        row = self.df.iloc[self._step]
        price = float(row["close"])
        terminated = False
        reward = 0.0

        if action == 1:  # SELL
            exit_price = price * (1.0 - self.transaction_cost_pct)
            trade_return = (exit_price - self._entry_price) / self._entry_price
            self._returns.append(trade_return)

            # Reward: percentage return scaled to [-1, 1]
            # 1% profit → +1.0, 1% loss → -1.0
            reward = float(np.clip(trade_return * 100.0, -1.0, 1.0))
            terminated = True

        else:  # HOLD
            self._holding_steps += 1
            unrealized = (price - self._entry_price) / self._entry_price

            # Small signal: positive when winning, negative when losing
            # Teaches agent to cut losses and let winners run
            reward = float(np.clip(unrealized * 5.0, -0.05, 0.05))

            # Force-close at max_holding_steps
            if self._holding_steps >= self.max_holding_steps:
                exit_price = price * (1.0 - self.transaction_cost_pct)
                trade_return = (exit_price - self._entry_price) / self._entry_price
                self._returns.append(trade_return)
                reward = float(np.clip(trade_return * 100.0, -1.0, 1.0))
                terminated = True

        self._step += 1
        truncated = self._step >= self.n_steps

        return self._obs(), reward, bool(terminated), bool(truncated), self._info(price)

    def _obs(self) -> np.ndarray:
        idx = min(self._step, self.n_steps - 1)
        row = self.df.iloc[idx]

        market = np.array(
            [float(row[c]) for c in self.FEATURE_COLS],
            dtype=np.float32,
        )

        price = float(row["close"])
        unrealized = 0.0
        if self._entry_price > 0:
            unrealized = float(np.clip(
                (price - self._entry_price) / self._entry_price * 10.0,
                -1.0, 1.0
            ))

        position = np.array(
            [
                float(self._holding_steps) / float(self.max_holding_steps),
                unrealized,
            ],
            dtype=np.float32,
        )

        return np.clip(
            np.concatenate([market, position]), -5.0, 5.0
        ).astype(np.float32)

    def _info(self, price: float) -> dict:
        return {
            "step":          self._step,
            "holding_steps": self._holding_steps,
            "entry_price":   self._entry_price,
            "current_price": price,
        }

    def _get_episode_sharpe(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        r = np.array(self._returns)
        if r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))