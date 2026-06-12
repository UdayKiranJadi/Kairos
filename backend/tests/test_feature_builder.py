import pandas as pd
import pytest

from app.features.feature_builder import FeatureBuilder


def test_build_features_from_bars():
    bars = pd.DataFrame(
        [
            {
                "symbol_id": 1,
                "timestamp": "2026-06-01T13:30:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
            },
            {
                "symbol_id": 1,
                "timestamp": "2026-06-01T13:31:00",
                "open": 100.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.0,
                "volume": 1200,
            },
            {
                "symbol_id": 1,
                "timestamp": "2026-06-01T13:32:00",
                "open": 101.0,
                "high": 103.0,
                "low": 101.0,
                "close": 102.0,
                "volume": 1400,
            },
        ]
    )

    builder = FeatureBuilder(db=None)
    features = builder.build_features_from_bars(bars)

    assert len(features) == 3
    assert "return_1m" in features.columns
    assert "price_vs_vwap" in features.columns
    assert features.iloc[1]["return_1m"] == pytest.approx(0.01)
