from pathlib import Path

import joblib
import pandas as pd


class IntradayPredictionModel:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.bundle = joblib.load(self.model_path)

        self.symbol = self.bundle["symbol"]
        self.model_name = self.bundle["model_name"]
        self.model_version = self.bundle["model_version"]
        self.horizon_minutes = self.bundle["horizon_minutes"]
        self.feature_columns = self.bundle["feature_columns"]
        self.model = self.bundle["model"]

    def predict_one(self, features: dict) -> dict:
        X = pd.DataFrame([features], columns=self.feature_columns)

        probability_up = float(self.model.predict_proba(X)[0, 1])
        predicted_class = int(probability_up >= 0.5)

        return {
            "symbol": self.symbol,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "horizon_minutes": self.horizon_minutes,
            "probability_up": probability_up,
            "predicted_class": predicted_class,
        }