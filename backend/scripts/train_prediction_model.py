import argparse
import asyncio
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.db.session import AsyncSessionLocal
from app.models.prediction_dataset import FEATURE_COLUMNS, PredictionDatasetBuilder


MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        required=True,
        help="Symbol to train on, example: AAPL",
    )
    parser.add_argument(
        "--start",
        required=False,
        default=None,
        help="Optional start datetime, example: 2026-06-01T13:30:00",
    )
    parser.add_argument(
        "--end",
        required=False,
        default=None,
        help="Optional end datetime, example: 2026-06-01T20:00:00",
    )
    parser.add_argument(
        "--horizon-minutes",
        required=False,
        default=5,
        type=int,
        help="Prediction horizon in minutes.",
    )
    parser.add_argument(
        "--test-size",
        required=False,
        default=0.2,
        type=float,
        help="Fraction of latest data used for testing.",
    )

    return parser.parse_args()


def time_series_split(df, test_size: float):
    split_index = int(len(df) * (1.0 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


async def main():
    args = parse_args()

    start = datetime.fromisoformat(args.start) if args.start else None
    end = datetime.fromisoformat(args.end) if args.end else None
    symbol = args.symbol.upper()

    async with AsyncSessionLocal() as session:
        builder = PredictionDatasetBuilder(session)
        frame = await builder.load_feature_price_frame(
            ticker=symbol,
            start=start,
            end=end,
        )
        dataset = builder.build_dataset(
            frame,
            horizon_minutes=args.horizon_minutes,
        )

    if dataset.empty:
        raise ValueError(
            f"No training dataset available for {symbol}. "
            "Load bars and build features first."
        )

    if len(dataset) < 100:
        raise ValueError(
            f"Dataset has only {len(dataset)} rows. "
            "Load more intraday history before training."
        )

    train_df, test_df = time_series_split(dataset, test_size=args.test_size)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_up"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target_up"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = float("nan")

    print("\nTraining complete")
    print("-----------------")
    print(f"Symbol: {symbol}")
    print(f"Rows: {len(dataset)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Horizon minutes: {args.horizon_minutes}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}" if np.isfinite(roc_auc) else "ROC AUC: unavailable")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    model_bundle = {
        "symbol": symbol,
        "model_name": "logistic_regression_intraday_direction",
        "model_version": "v0.1",
        "horizon_minutes": args.horizon_minutes,
        "feature_columns": FEATURE_COLUMNS,
        "model": model,
        "metrics": {
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else None,
            "rows": int(len(dataset)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
    }

    output_path = MODEL_DIR / f"{symbol.lower()}_intraday_direction_v0_1.joblib"
    joblib.dump(model_bundle, output_path)

    print(f"\nSaved model to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())