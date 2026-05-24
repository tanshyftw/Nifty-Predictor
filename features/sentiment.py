"""
News sentiment feature computation.
Reads sentiment data from the database and computes features.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def compute_sentiment_features(
    sentiment_data: dict | None,
) -> dict:
    """
    Extract sentiment features from a sentiment record.

    Input: dict from news_sentiment table (or None if no data)
    Output: dict with 8 sentiment feature values
    """
    if sentiment_data is None:
        return _neutral_features()

    return {
        "news_count_24h": sentiment_data.get("news_count", 0),
        "vader_compound_mean": sentiment_data.get("vader_compound_mean", 0.0),
        "vader_compound_max": sentiment_data.get("vader_compound_max", 0.0),
        "vader_compound_min": sentiment_data.get("vader_compound_min", 0.0),
        "vader_positive_ratio": sentiment_data.get("vader_positive_ratio", 0.0),
        "vader_negative_ratio": sentiment_data.get("vader_negative_ratio", 0.0),
        "textblob_polarity": sentiment_data.get("textblob_polarity", 0.0),
        "textblob_subjectivity": sentiment_data.get("textblob_subjectivity", 0.0),
    }


def _neutral_features() -> dict:
    """Return neutral sentiment features (no news available)."""
    return {
        "news_count_24h": 0,
        "vader_compound_mean": 0.0,
        "vader_compound_max": 0.0,
        "vader_compound_min": 0.0,
        "vader_positive_ratio": 0.0,
        "vader_negative_ratio": 0.0,
        "textblob_polarity": 0.0,
        "textblob_subjectivity": 0.0,
    }


def get_sentiment_feature_names() -> list[str]:
    """Return list of sentiment feature column names."""
    return [
        "news_count_24h",
        "vader_compound_mean",
        "vader_compound_max",
        "vader_compound_min",
        "vader_positive_ratio",
        "vader_negative_ratio",
        "textblob_polarity",
        "textblob_subjectivity",
    ]
