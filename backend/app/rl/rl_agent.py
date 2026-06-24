"""
rl_agent.py — Wraps the trained PPO model for live inference.

WHY A WRAPPER:
The PPO model expects a numpy observation vector.
The bot has a database row of features.
This class bridges the two: fetches latest features,
builds the observation, runs the model, returns a signal.

ENSEMBLE WITH LOGREG:
We don't replace the LogReg model. We run both and combine:

  RL says BUY  + LogReg says UP   → ENTER_LONG (high confidence)
  RL says SELL + LogReg says DOWN → EXIT (high confidence)
  They disagree                   → HOLD (uncertainty)

WHY ENSEMBLE:
RL is trained on 6 months of data. LogReg is trained on the
same data with different methodology. Agreement = signal is
robust across two independent models. Disagreement = the
market signal is ambiguous, better to wait.
"""

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class RLAgent:
    """
    Wraps a trained PPO model for live trading inference.

    Args:
        model_path: Path to the saved .zip model file.
                    Defaults to artifacts/rl/best/best_model.zip
                    (saved by EvalCallback during training).
    """

    # Matches TradingEnv.FEATURE_COLS
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

    def __init__(self, model_path: str | Path | None = None):
        if model_path is None:
            # Try best model first (saved by EvalCallback)
            # Fall back to final model
            candidates = [
                Path("artifacts/rl/best/best_model.zip"),
                Path("artifacts/rl/ppo_aapl_final.zip"),
            ]
            model_path = next(
                (p for p in candidates if p.exists()), None
            )

        if model_path is None or not Path(model_path).exists():
            raise FileNotFoundError(
                "No RL model found. Run train_rl_agent.py first.\n"
                f"Looked in: {candidates}"
            )

        self.model = PPO.load(str(model_path))
        self.model_path = Path(model_path)
        logger.info(f"RL agent loaded from: {self.model_path}")

    def build_observation(
        self,
        features: dict,
        holding_steps: int = 0,
        max_holding_steps: int = 30,
        unrealized_pnl: float = 0.0,
    ) -> np.ndarray:
        """
        Build 10-dim observation vector from feature dict.

        Matches TradingEnv v2 observation space exactly:
          [8 market features, holding_steps_normalized, unrealized_pnl_normalized]
        """
        market = np.array(
            [float(features.get(col, 0.0) or 0.0) for col in self.FEATURE_COLS],
            dtype=np.float32,
        )

        position = np.array(
            [
                float(holding_steps) / float(max_holding_steps),
                float(np.clip(unrealized_pnl * 10.0, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate([market, position])
        return np.clip(obs, -5.0, 5.0).astype(np.float32)

    def predict(
        self,
        features: dict,
        holding_steps: int = 0,
        unrealized_pnl: float = 0.0,
        deterministic: bool = True,
    ) -> dict:
        """
        Run inference on latest features.

        Returns:
            {
                "action": 0 (HOLD) or 1 (SELL),
                "action_name": "HOLD" or "SELL",
                "confidence": float,  # probability of predicted action
            }

        NOTE: TradingEnv v2 uses forced entry (HOLD/SELL only).
        For the ensemble, we interpret:
          HOLD → bullish (agent wants to stay in position)
          SELL → bearish (agent wants to exit)
        """
        obs = self.build_observation(features, holding_steps, unrealized_pnl=unrealized_pnl)

        # predict() returns (action, state)
        # We use deterministic=True for live trading
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action = int(action)

        # Get action probabilities for confidence score
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        with __import__("torch").no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]

        confidence = float(probs[action])

        return {
            "action":      action,
            "action_name": ["HOLD", "SELL"][action],
            "confidence":  confidence,
            "probs":       probs.tolist(),
        }

    def is_bearish(self, features: dict, threshold: float = 0.6) -> bool:
        """
        Returns True if RL model is bearish (wants to sell).
        Used in ensemble logic.
        """
        result = self.predict(features)
        return result["action"] == 1 and result["confidence"] >= threshold