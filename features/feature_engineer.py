"""
Master feature engineering orchestrator.
Combines technical, sentiment, fundamental, macro, and temporal features
into a single feature vector per stock per day.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np
from loguru import logger

from config.settings import SETTINGS
from config.nifty50_tickers import get_symbols, get_sector
from data.storage.db_manager import DBManager
from features.technical import compute_technical_features, get_technical_feature_names
from features.sentiment import compute_sentiment_features, get_sentiment_feature_names
from features.fundamental import (
    compute_fundamental_features,
    get_fundamental_feature_names,
)
from features.macro import (
    compute_macro_features,
    compute_macro_features_with_history,
    get_macro_feature_names,
)
from features.temporal import compute_temporal_features, get_temporal_feature_names


class FeatureEngineer:
    """Assembles all features for the prediction model."""

    def __init__(self, db: DBManager):
        self.db = db

    def compute_features_for_date(
        self, target_date: str
    ) -> pd.DataFrame:
        """
        Compute complete feature vectors for all Nifty 50 stocks on a given date.

        Returns DataFrame with one row per stock, ~100 feature columns,
        plus 'symbol' and 'date' columns.
        """
        symbols = get_symbols()
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

        # Lookback window for technical indicators
        lookback_start = (target_dt - timedelta(days=SETTINGS.FEATURE_LOOKBACK_DAYS * 2)).isoformat()

        # Get macro data for this date
        macro_records = self.db.get_macro(start_date=lookback_start, end_date=target_date)
        macro_for_date = None
        macro_idx = -1
        for i, rec in enumerate(macro_records):
            if rec["date"] == target_date:
                macro_for_date = rec
                macro_idx = i
                break
        # If exact date not found, use most recent
        if macro_for_date is None and macro_records:
            macro_for_date = macro_records[-1]
            macro_idx = len(macro_records) - 1

        # Compute macro features (with history for rolling features)
        if macro_idx >= 0:
            macro_features = compute_macro_features_with_history(macro_records, macro_idx)
        else:
            macro_features = compute_macro_features(None)

        # Temporal features
        temporal_features = compute_temporal_features(target_dt)

        # Get all fundamentals for sector-relative features
        all_fundamentals = []
        for sym in symbols:
            fund = self.db.get_latest_fundamentals(sym)
            if fund:
                all_fundamentals.append(fund)

        # Process each stock
        all_rows = []
        for symbol in symbols:
            try:
                row = self._compute_stock_features(
                    symbol, target_date, lookback_start,
                    macro_features, temporal_features, all_fundamentals
                )
                if row is not None:
                    all_rows.append(row)
            except Exception as e:
                logger.warning(f"Feature computation failed for {symbol}: {e}")
                continue

        if not all_rows:
            logger.error(f"No features computed for {target_date}")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        # Forward fill then fill remaining NaN with 0
        feature_cols = [c for c in df.columns if c not in ("symbol", "date")]
        df[feature_cols] = df[feature_cols].ffill().fillna(0)

        logger.info(
            f"Computed {len(df)} stock feature vectors with "
            f"{len(feature_cols)} features for {target_date}"
        )
        return df

    def _compute_stock_features(
        self,
        symbol: str,
        target_date: str,
        lookback_start: str,
        macro_features: dict,
        temporal_features: dict,
        all_fundamentals: list[dict],
    ) -> dict | None:
        """Compute complete feature vector for a single stock."""
        # Get OHLCV history
        ohlcv_records = self.db.get_ohlcv(symbol, lookback_start, target_date)
        if not ohlcv_records or len(ohlcv_records) < 30:
            logger.warning(f"Insufficient OHLCV data for {symbol}: {len(ohlcv_records)} rows")
            return None

        ohlcv_df = pd.DataFrame(ohlcv_records)

        # Compute technical features
        tech_df = compute_technical_features(ohlcv_df)

        # Get the last row (target date or most recent)
        last_row = tech_df.iloc[-1]

        # Extract technical features
        tech_feature_names = get_technical_feature_names()
        features = {"symbol": symbol, "date": target_date}
        for col in tech_feature_names:
            features[col] = float(last_row.get(col, 0)) if col in last_row.index else 0.0

        # Sentiment features
        sentiment_data = self.db.get_sentiment(symbol, target_date)
        sentiment_features = compute_sentiment_features(sentiment_data)
        features.update(sentiment_features)

        # Fundamental features
        fund_data = self.db.get_latest_fundamentals(symbol)
        fund_features = compute_fundamental_features(fund_data, all_fundamentals)
        features.update(fund_features)

        # Macro features (same for all stocks on this date)
        features.update(macro_features)

        # Temporal features (same for all stocks on this date)
        features.update(temporal_features)

        # Sector encoding
        sector = get_sector(symbol)
        features["sector_encoded"] = _encode_sector(sector)

        return features

    def compute_training_features(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Compute features for all stocks across a date range (for training).
        Caches results to the DB — subsequent calls only compute new dates.
        Returns DataFrame with columns: date, symbol, + all features.
        """
        # Get all unique OHLCV dates in range
        all_ohlcv = self.db.get_all_ohlcv(start_date, end_date)
        if not all_ohlcv:
            logger.error("No OHLCV data found for training period")
            return pd.DataFrame()

        all_dates = sorted(set(r["date"] for r in all_ohlcv))

        # Skip first FEATURE_LOOKBACK_DAYS dates — not enough history for indicators
        min_trading_days = max(SETTINGS.FEATURE_LOOKBACK_DAYS, 60)
        if len(all_dates) > min_trading_days:
            all_dates = all_dates[min_trading_days:]

        # Check which dates are already cached in the DB
        cached_dates = self.db.get_cached_feature_dates(all_dates[0], all_dates[-1])
        missing_dates = [d for d in all_dates if d not in cached_dates]

        logger.info(
            f"Feature cache: {len(cached_dates)} dates cached, "
            f"{len(missing_dates)} dates to compute"
        )

        # Compute features only for missing dates
        if missing_dates:
            for i, date_str in enumerate(missing_dates):
                if i % 50 == 0:
                    logger.info(
                        f"Computing features {i + 1}/{len(missing_dates)}: {date_str}"
                    )
                try:
                    df = self.compute_features_for_date(date_str)
                    if df.empty:
                        continue

                    # Save each stock's features to the DB cache
                    batch = []
                    for _, row in df.iterrows():
                        feat = {
                            k: v for k, v in row.items()
                            if k not in ("symbol", "date")
                        }
                        batch.append((row["date"], row["symbol"], feat))
                    self.db.insert_features_batch(batch)

                except Exception as e:
                    logger.warning(f"Failed to compute features for {date_str}: {e}")
                    continue

            logger.info(f"Computed and cached features for {len(missing_dates)} new dates")

        # Load all features from DB cache
        result = self.db.get_all_features(all_dates[0], all_dates[-1])

        if result.empty:
            return pd.DataFrame()

        logger.info(
            f"Total training features: {len(result)} rows x {len(result.columns)} cols"
        )
        return result


def _encode_sector(sector: str) -> int:
    """Simple label encoding for sectors."""
    sector_map = {
        "Auto": 0,
        "Banking": 1,
        "Cement": 2,
        "Chemicals": 3,
        "Conglomerate": 4,
        "Consumer Goods": 5,
        "Financial Services": 6,
        "FMCG": 7,
        "Healthcare": 8,
        "Infrastructure": 9,
        "Insurance": 10,
        "IT": 11,
        "Metals": 12,
        "Mining": 13,
        "Oil & Gas": 14,
        "Pharma": 15,
        "Power": 16,
        "Telecom": 17,
    }
    return sector_map.get(sector, -1)


def get_all_feature_names() -> list[str]:
    """Return complete list of all feature names."""
    names = []
    names.extend(get_technical_feature_names())
    names.extend(get_sentiment_feature_names())
    names.extend(get_fundamental_feature_names())
    names.extend(get_macro_feature_names())
    names.extend(get_temporal_feature_names())
    names.append("sector_encoded")
    return names
