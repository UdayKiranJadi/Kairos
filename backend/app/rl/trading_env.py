"""
trading_env.py — Gymnasium trading environment for Kairos RL agent.

OBSERVATION SPACE (8 numbers):
  [0] return_1m        — 1-minute price return
  [1] return_5m        — 5-minute price return
  [2] volatility_10m   — 10-min rolling volatility
  [3] volume_zscore    — volume vs recent average
  [4] price_vs_vwap    — position relative to VWAP
  [5] position_held    — are we currently in a trade? (0 or 1)
  [6] holding_steps    — how long have we held? (normalized 0→1)
  [7] unrealized_pnl   — current trade P&L (normalized)

ACTION SPACE (3 discrete):
  0 = HOLD
  1 = BUY  (enter long if flat)
  2 = SELL (exit if holding)

REWARD:
  Buy:       small transaction cost penalty
  Sell:      Sharpe-adjusted return (trade_return / volatility)
  Hold long: tiny unrealized P&L signal
  Hold flat: zero
  Invalid:   small penalty (buy when holding, sell when flat)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Single-asset intraday trading environment.

    Args:
        df: DataFrame with columns [return_1m, return_5m,
            volatility_10m, volume_zscore, price_vs_vwap, close].
            Produced by PredictionDatasetBuilder.load_feature_price_frame().
            Must be pre-cleaned (no NaNs).

        initial_cash: Starting capital per episode.
        max_holding_steps: Force-close after this many steps (intraday rule).
        transaction_cost_pct: Slippage + commission per trade (0.001 = 0.1%).
    """

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

        # Validate columns
        missing = [c for c in self.FEATURE_COLS + ["close"] if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_cash = initial_cash
        self.max_holding_steps = max_holding_steps
        self.transaction_cost_pct = transaction_cost_pct

        # Action: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation: 5 market features + 3 position features
        # Bounded to [-5, 5] — prevents gradient explosion in PPO
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(8,),
            dtype=np.float32,
        )

        # State variables — initialized properly in reset()
        self._current_step = 0
        self._position = 0
        self._entry_price = 0.0
        self._holding_steps = 0
        self._cash = initial_cash
        self._returns: list[float] = []

    # ── Gymnasium interface ──────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reset to start of a new episode.

        WHY RANDOM START:
        Always starting at step 0 causes the agent to overfit
        to early-session patterns. Random start exposes it to
        all market conditions (open, mid-session, close).
        """
        super().reset(seed=seed)

        # Start in first 80% so there's always room to run
        max_start = max(1, int(self.n_steps * 0.8))
        self._current_step = int(self.np_random.integers(0, max_start))

        self._position = 0
        self._entry_price = 0.0
        self._holding_steps = 0
        self._cash = self.initial_cash
        self._returns = []

        return self._get_observation(), {}

    def step(self, action: int):
        """
        Execute one timestep.

        Returns: (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        row = self.df.iloc[self._current_step]
        current_price = float(row["close"])
        reward = 0.0
        terminated = False

        # ── BUY ─────────────────────────────────────────────────────────
        if action == 1:
            if self._position == 0:
                # Enter long — pay transaction cost
                self._position = 1
                self._entry_price = current_price
                self._holding_steps = 0
                self._cash = 0.0
                # Cost penalty — agent must earn this back to profit
                reward = -self.transaction_cost_pct

            else:
                # Already holding — invalid action
                # WHY PENALIZE: without penalty, agent spams BUY
                # with zero consequence and never learns state awareness
                reward = -0.005

        # ── SELL ─────────────────────────────────────────────────────────
        elif action == 2:
            if self._position == 1:
                # Exit long — compute Sharpe-adjusted reward
                exit_price = current_price * (1 - self.transaction_cost_pct)
                trade_return = (exit_price - self._entry_price) / self._entry_price

                self._cash = self.initial_cash * (1 + trade_return)
                self._position = 0
                self._entry_price = 0.0
                self._holding_steps = 0

                # Sharpe adjustment: same return is better if achieved
                # with lower volatility — teaches risk-adjusted thinking
                # Floor at 0.0005 prevents division explosion on tiny vol
                volatility = max(
                    float(row["volatility_10m"]) if not pd.isna(row["volatility_10m"]) else 0.001,
                    0.0005,
                )
                reward = float(np.clip(trade_return / volatility, -10.0, 10.0))

                # Asymmetric bonus/penalty mirrors your RiskPolicy loss aversion
                if trade_return > 0:
                    reward += 0.01
                else:
                    reward -= 0.02

                self._returns.append(trade_return)
                terminated = True  # one trade = one episode

            else:
                # Tried to sell when flat — invalid
                reward = -0.005

        # ── HOLD ─────────────────────────────────────────────────────────
        else:
            if self._position == 1:
                self._holding_steps += 1

                # Small unrealized P&L signal while holding
                unrealized = (current_price - self._entry_price) / self._entry_price
                reward = float(unrealized * 0.001)

                # Force-close at max holding steps (intraday rule)
                # WHY: prevents agent learning to hold indefinitely
                if self._holding_steps >= self.max_holding_steps:
                    exit_price = current_price * (1 - self.transaction_cost_pct)
                    trade_return = (exit_price - self._entry_price) / self._entry_price
                    self._cash = self.initial_cash * (1 + trade_return)
                    self._position = 0
                    self._holding_steps = 0
                    self._returns.append(trade_return)
                    terminated = True
            else:
                # Flat and waiting — zero reward
                # Agent must find opportunities; idle = neutral
                reward = 0.0

        # Advance time
        self._current_step += 1
        truncated = self._current_step >= self.n_steps

        obs = self._get_observation()
        info = self._get_info(current_price)

        return obs, float(reward), terminated, truncated, info

    # ── Private helpers ──────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Build 8-dimensional observation vector.
        NaN-safe: missing values replaced with 0.0.
        All values clipped to [-5, 5] to match observation_space.
        """
        if self._current_step >= self.n_steps:
            return np.zeros(8, dtype=np.float32)

        row = self.df.iloc[self._current_step]

        # 5 market features — NaN-safe extraction
        # pd.isna() catches both float('nan') and None
        market_obs = np.array(
            [
                0.0 if pd.isna(v := row.get(col, 0.0)) else float(v)
                for col in self.FEATURE_COLS
            ],
            dtype=np.float32,
        )

        # 3 position features — agent's self-knowledge
        position_obs = np.array(
            [
                float(self._position),                          # 0=flat, 1=long
                self._holding_steps / self.max_holding_steps,   # 0.0 → 1.0
                self._get_unrealized_pnl(row),                  # fraction of entry
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([market_obs, position_obs])
        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    def _get_unrealized_pnl(self, row) -> float:
        """Unrealized P&L as fraction of entry price. 0.0 if flat."""
        if self._position == 0 or self._entry_price == 0:
            return 0.0
        current_price = float(row["close"])
        return float(np.clip(
            (current_price - self._entry_price) / self._entry_price,
            -5.0, 5.0
        ))

    def _get_info(self, current_price: float) -> dict:
        """Debug info returned with each step. Not used by SB3 for training."""
        return {
            "step":          self._current_step,
            "position":      self._position,
            "holding_steps": self._holding_steps,
            "entry_price":   self._entry_price,
            "current_price": current_price,
            "cash":          self._cash,
        }

    def get_episode_sharpe(self) -> float:
        """
        Sharpe ratio for completed episode.
        Target before going live: > 1.0 consistently.
        """
        if len(self._returns) < 2:
            return 0.0
        r = np.array(self._returns)
        if r.std() == 0:
            return 0.0
        # Annualized: sqrt(252 trading days * 390 minutes/day)
        return float(r.mean() / r.std() * np.sqrt(252 * 390))