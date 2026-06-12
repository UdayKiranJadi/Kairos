import argparse
from pathlib import Path

from app.models.prediction_model import IntradayPredictionModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model artifact.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = IntradayPredictionModel(Path(args.model_path))

    sample_features = {
        "return_1m": 0.0003,
        "return_5m": 0.0012,
        "volatility_10m": 0.0008,
        "volume_zscore": 1.4,
        "price_vs_vwap": 0.0021,
    }

    prediction = model.predict_one(sample_features)

    print(prediction)


if __name__ == "__main__":
    main()