"""
trading_env.py — Gymnasium trading environment for Kairos RL agent.

THE CORE IDEA:
We wrap the trading problem as a standard RL environment.
At each timestep the agent sees market state, picks an action,
and receives a reward. Over thousands of episodes it learns
which actions lead to profit.

WHY GYMNASIUM NOT CUSTOM:
Gymnasium is the industry standard for RL environments.
Using it means we get:
  - Compatible with any SB3 algorithm (PPO, SAC, TD3)
  - Built-in validation via gymnasium.utils.env_checker
  - Easy to swap algorithms without changing the env

OBSERVATION SPACE (what the agent sees — 8 numbers):
  [0] return_1m        — 1-minute price return
  [1] return_5m        — 5-minute price return
  [2] volatility_10m   — 10-min rolling volatility
  [3] volume_zscore    — volume vs recent average
  [4] price_vs_vwap    — position relative to VWAP
  [5] position_held    — are we currently in a trade? (0 or 1)
  [6] holding_steps    — how long have we held? (normalized)
  [7] unrealized_pnl   — current trade P&L (normalized)

WHY ADD [5][6][7]:
The market features alone are not enough. The agent also
needs to know its own state. Without [5], it can't know
whether to EXIT (no point exiting if not holding).
Without [6], it might hold forever. Without [7], it can't
decide when a trade is profitable enough to close.

ACTION SPACE (what the agent can do — 3 discrete choices):
  0 = HOLD    — do nothing
  1 = BUY     — enter a long position (if not already in one)
  2 = SELL    — exit the position (if holding one)

WHY DISCRETE NOT CONTINUOUS:
Continuous actions (position sizing) are harder to train
and less stable. We start discrete: in or out, one position
at a time. This mirrors your existing RiskPolicy (max 1 position).
We can add continuous sizing later (Day 6+).

REWARD FUNCTION:
  Per-step: small negative reward for holding (time decay)
  On sell:  Sharpe-adjusted return = return / volatility
  On bad action: small penalty (buy when already holding, etc.)

WHY SHARPE-ADJUSTED:
Raw P&L reward → agent learns to gamble (high variance wins).
Sharpe reward  → agent learns consistency (profit per unit risk).
This is exactly what your RiskPolicy enforces manually —
we're teaching the agent to internalize that preference.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Single-asset intraday trading environment.

    Args:
        df: DataFrame with columns [timestamp, return_1m, return_5m,
            volatility_10m, volume_zscore, price_vs_vwap, close]
            ordered oldest → newest. This is exactly what
            PredictionDatasetBuilder.load_feature_price_frame() returns.

        initial_cash: Starting capital for the episode.
        max_holding_steps: Force-close position after this many steps.
            Prevents the agent from holding overnight (intraday rule).
        transaction_cost_pct: Slippage + commission per trade.
            0.001 = 0.1% round-trip (realistic for paper trading).
    """

    metadata = {"render_modes": []}

    # Feature columns from your existing FeatureBuilder
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

        # Validate input
        missing = [c for c in self.FEATURE_COLS + ["close"] if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_cash = initial_cash
        self.max_holding_steps = max_holding_steps
        self.transaction_cost_pct = transaction_cost_pct

        # --- Action space ---
        # 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # --- Observation space ---
        # 5 market features + 3 position features = 8 dimensions
        # All bounded to [-5, 5] after normalization.
        # WHY BOUNDED: SB3 algorithms work best with bounded obs spaces.
        # Unbounded spaces cause gradient explosion during training.
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(8,),
            dtype=np.float32,
        )

        # State is initialized in reset()
        self._current_step = 0
        self._position = 0          # 0=flat, 1=long
        self._entry_price = 0.0
        self._holding_steps = 0
        self._cash = initial_cash
        self._portfolio_value = initial_cash
        self._returns: list[float] = []  # track per-step returns for Sharpe

    # ── Gymnasium interface ─────────────────────────────────────

    def reset(self, seed=None, options=None):
       super().reset(seed=seed)

    # Guarantee at least (max_holding_steps + 20) steps remaining
    # after the random start. This prevents immediate truncation
    # which breaks gymnasium's determinism checker.
       min_steps_needed = self.max_holding_steps + 20
       max_start = max(0, self.n_steps - min_steps_needed)

       if max_start == 0:
        # DataFrame too short — always start at 0
        self._current_step = 0
       else:
        self._current_step = int(self.np_random.integers(0, max_start))

        self._position = 0
        self._entry_price = 0.0
        self._holding_steps = 0
        self._cash = self.initial_cash
        self._portfolio_value = self.initial_cash
        self._returns = []

        return self._get_observation(), {}

    def step(self, action: int):
        """
        Execute one timestep.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        row = self.df.iloc[self._current_step]
        current_price = float(row["close"])

        reward = 0.0
        terminated = False

        # --- Execute action ---
        if action == 1:  # BUY
            if self._position == 0:
                # Enter long: spend all cash on shares
                # Transaction cost simulates slippage + commission
                cost = current_price * (1 + self.transaction_cost_pct)
                shares = self._cash / cost

                self._position = 1
                self._entry_price = current_price
                self._holding_steps = 0
                self._cash = 0.0

                # Small penalty for buying — agent must earn it back
                # WHY: prevents the agent from churning in/out constantly
                reward = -self.transaction_cost_pct

            else:
                # Tried to buy when already holding → invalid action penalty
                # WHY PENALTY NOT IGNORE: if we ignore it, the agent learns
                # to spam BUY with no consequence. Penalty teaches it to
                # check its state before acting.
                reward = -0.005

        elif action == 2:  # SELL
            if self._position == 1:
                # Exit long: calculate return
                exit_price = current_price * (1 - self.transaction_cost_pct)
                trade_return = (exit_price - self._entry_price) / self._entry_price

                self._cash = self.initial_cash * (1 + trade_return)
                self._position = 0
                self._entry_price = 0.0
                self._holding_steps = 0

                # Sharpe-adjusted reward
                # WHY: raw return rewards gambling; dividing by volatility
                # rewards the same return achieved with less risk
                volatility = float(row["volatility_10m"]) or 0.001
                reward = trade_return / (volatility + 1e-8)

                # Bonus for positive trade, extra penalty for loss
                # This asymmetry mimics your RiskPolicy loss aversion
                if trade_return > 0:
                    reward += 0.01
                else:
                    reward -= 0.02

                self._returns.append(trade_return)
                terminated = True  # end episode after each trade

            else:
                # Tried to sell when flat → penalty
                reward = -0.005

        else:  # HOLD (action == 0)
            if self._position == 1:
                # Holding: small time-decay penalty
                # WHY: without this, agent learns to hold forever.
                # Penalizing time teaches it to take profits.
                self._holding_steps += 1
                unrealized = (current_price - self._entry_price) / self._entry_price
                reward = unrealized * 0.001  # tiny reward for floating profit

                # Force close if held too long (intraday rule)
                if self._holding_steps >= self.max_holding_steps:
                    exit_price = current_price * (1 - self.transaction_cost_pct)
                    trade_return = (exit_price - self._entry_price) / self._entry_price
                    self._cash = self.initial_cash * (1 + trade_return)
                    self._position = 0
                    self._holding_steps = 0
                    self._returns.append(trade_return)
                    terminated = True
            else:
                # Flat and holding: zero reward
                # Agent must find opportunities, not sit idle forever
                reward = 0.0

        # Advance time
        self._current_step += 1
        truncated = self._current_step >= self.n_steps

        obs = self._get_observation()
        info = self._get_info(current_price)

        return obs, reward, terminated, truncated, info

    # ── Private helpers ─────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Build the 8-dimensional observation vector.

        Market features come from your FeatureBuilder columns.
        Position features tell the agent its own state.
        All values clipped to [-5, 5] to match observation_space.
        """
        # Guard: if we've run out of data, return zeros
        if self._current_step >= self.n_steps:
            return np.zeros(8, dtype=np.float32)

        row = self.df.iloc[self._current_step]

        # 5 market features (already normalized by FeatureBuilder)
        market_obs = np.array(
            [float(row.get(col, 0.0) or 0.0) for col in self.FEATURE_COLS],
            dtype=np.float32,
        )

        # 3 position features (normalize to [-1, 1] range)
        position_obs = np.array(
            [
                float(self._position),                              # 0 or 1
                self._holding_steps / self.max_holding_steps,       # 0→1
                self._get_unrealized_pnl(row),                      # float
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([market_obs, position_obs])
        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    def _get_unrealized_pnl(self, row) -> float:
        """Current unrealized P&L as a fraction of entry price."""
        if self._position == 0 or self._entry_price == 0:
            return 0.0
        current_price = float(row["close"])
        return (current_price - self._entry_price) / self._entry_price

    def _get_info(self, current_price: float) -> dict:
        """
        Extra info returned with each step.
        Not used by SB3 for training but useful for logging + debugging.
        """
        return {
            "step":           self._current_step,
            "position":       self._position,
            "holding_steps":  self._holding_steps,
            "entry_price":    self._entry_price,
            "current_price":  current_price,
            "cash":           self._cash,
        }

    def _get_episode_sharpe(self) -> float:
        """
        Sharpe ratio for the completed episode.
        Used after training to evaluate agent quality.
        Higher = better. >1.0 is our target before going live.
        """
        if len(self._returns) < 2:
            return 0.0
        r = np.array(self._returns)
        if r.std() == 0:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252))