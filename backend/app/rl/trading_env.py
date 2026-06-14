"""
trading_env.py — Gymnasium trading environment for Kairos RL agent.

DESIGN PRINCIPLES:
- Rewards bounded to [-1, +1] so value function learns stably
- NaN cleaned at construction time, not observation time
- No dropna — fill instead, so n_steps never shrinks unexpectedly
- One trade per episode: buy → hold → sell → terminate
- Force-close after max_holding_steps (intraday rule)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):

    metadata = {"render_modes": []}

    FEATURE_COLS = [
        "return_1m",
        "return_5m",
        "volatility_10m",
        "volume_zscore",
        "price_vs_vwap",
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
            raise ValueError(f"DataFrame missing columns: {missing}")

        # Fill NaN with 0 — never drop rows (keeps n_steps stable)
        # WHY FILL NOT DROP: dropping rows shrinks the dataset unpredictably
        # and can leave n_steps=1 which breaks gymnasium's checker.
        self.df = df.copy().reset_index(drop=True)
        self.df[self.FEATURE_COLS] = self.df[self.FEATURE_COLS].fillna(0.0)
        self.df["close"] = self.df["close"].ffill().bfill().fillna(100.0)
        self.n_steps = len(self.df)

        if self.n_steps < 10:
            raise ValueError(
                f"DataFrame too small: {self.n_steps} rows. Need at least 10."
            )

        self.initial_cash = initial_cash
        self.max_holding_steps = max_holding_steps
        self.transaction_cost_pct = transaction_cost_pct

        # 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # 8 dimensions: 5 market features + 3 position features
        # All bounded [-5, 5] — neural nets work best with bounded inputs
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(8,),
            dtype=np.float32,
        )

        # State variables — reset in reset()
        self._current_step: int = 0
        self._position: int = 0        # 0=flat, 1=long
        self._entry_price: float = 0.0
        self._holding_steps: int = 0
        self._cash: float = initial_cash
        self._returns: list[float] = []

    # ── Gymnasium interface ──────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start at a random point in first 70% of data
        # Leave 30% tail so there's always room to run episodes
        # WHY RANDOM START: prevents overfitting to session-open patterns
        max_start = max(0, int(self.n_steps * 0.7) - self.max_holding_steps - 1)
        if max_start <= 0:
            self._current_step = 0
        else:
            self._current_step = int(self.np_random.integers(0, max_start))

        self._position = 0
        self._entry_price = 0.0
        self._holding_steps = 0
        self._cash = self.initial_cash
        self._returns = []

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Guard: if we somehow ran out of data, truncate
        if self._current_step >= self.n_steps:
            return self._get_obs(), 0.0, False, True, {}

        row = self.df.iloc[self._current_step]
        price = float(row["close"])
        reward = 0.0
        terminated = False

        # ── BUY ─────────────────────────────────────────────────
        if action == 1:
            if self._position == 0:
                # Enter long. Entry price includes transaction cost.
                self._position = 1
                self._entry_price = price * (1.0 + self.transaction_cost_pct)
                self._holding_steps = 0
                # Small entry cost — agent must earn this back
                reward = -0.01
            else:
                # Already holding — invalid action
                reward = -0.01

        # ── SELL ────────────────────────────────────────────────
        elif action == 2:
            if self._position == 1:
                exit_price = price * (1.0 - self.transaction_cost_pct)
                trade_return = (exit_price - self._entry_price) / self._entry_price

                self._cash = self.initial_cash * (1.0 + trade_return)
                self._position = 0
                self._entry_price = 0.0
                self._holding_steps = 0
                self._returns.append(trade_return)

                # Reward = clipped return × 100
                # 1% gain  → reward +1.0
                # 1% loss  → reward -1.0
                # WHY × 100: raw return (0.01) is too small for the value
                # function to learn from. Scale makes signal clear.
                reward = float(np.clip(trade_return * 100.0, -1.0, 1.0))

                # Asymmetric nudge: reward winners, punish losers
                if trade_return > 0.001:
                    reward += 0.1
                elif trade_return < -0.001:
                    reward -= 0.1

                reward = float(np.clip(reward, -1.0, 1.0))
                terminated = True

            else:
                # Nothing to sell — invalid action
                reward = -0.01

        # ── HOLD ────────────────────────────────────────────────
        else:  # action == 0
            if self._position == 1:
                self._holding_steps += 1
                unrealized = (price - self._entry_price) / self._entry_price
                # Small reward tracks P&L direction while holding
                reward = float(np.clip(unrealized * 10.0, -0.1, 0.1))

                # Force-close at max holding steps (intraday rule)
                if self._holding_steps >= self.max_holding_steps:
                    exit_price = price * (1.0 - self.transaction_cost_pct)
                    trade_return = (exit_price - self._entry_price) / self._entry_price
                    self._cash = self.initial_cash * (1.0 + trade_return)
                    self._position = 0
                    self._holding_steps = 0
                    self._returns.append(trade_return)
                    reward = float(np.clip(trade_return * 100.0, -1.0, 1.0))
                    terminated = True
            else:
                # Flat: tiny penalty to incentivize seeking trades
                # Small enough (-0.001) that it won't cause panic-buying
                reward = -0.001

        self._current_step += 1
        truncated = self._current_step >= self.n_steps

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._get_info(price),
        )

    # ── Private helpers ──────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        idx = min(self._current_step, self.n_steps - 1)
        row = self.df.iloc[idx]

        market = np.array(
            [float(row[c]) for c in self.FEATURE_COLS],
            dtype=np.float32,
        )

        position = np.array(
            [
                float(self._position),
                float(self._holding_steps) / float(self.max_holding_steps),
                self._unrealized(row),
            ],
            dtype=np.float32,
        )

        return np.clip(
            np.concatenate([market, position]), -5.0, 5.0
        ).astype(np.float32)

    def _unrealized(self, row) -> float:
        if self._position == 0 or self._entry_price == 0.0:
            return 0.0
        pnl = (float(row["close"]) - self._entry_price) / self._entry_price
        return float(np.clip(pnl * 10.0, -1.0, 1.0))

    def _get_info(self, price: float) -> dict:
        return {
            "step":          self._current_step,
            "position":      self._position,
            "holding_steps": self._holding_steps,
            "entry_price":   self._entry_price,
            "current_price": price,
            "cash":          self._cash,
        }

    def _get_episode_sharpe(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        r = np.array(self._returns)
        std = r.std()
        if std == 0.0:
            return 0.0
        return float(r.mean() / std * np.sqrt(252))