"""
NSE India data fetcher for FII/DII trading activity.
Uses nselib derivatives.participant_wise_trading_volume for institutional flow data.
"""

from __future__ import annotations

import time
from datetime import date, timedelta, datetime

import pandas as pd
from loguru import logger


class NSEFetcher:
    """Fetches FII/DII data from NSE India."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_fii_dii(
        self, start_date: str, end_date: str = None
    ) -> list[dict]:
        """
        Fetch FII/DII trading activity data across a date range.
        Uses nselib's participant_wise_trading_volume (per-day API).
        Returns list of dicts with date, fii_net_buy, dii_net_buy.
        """
        if end_date is None:
            end_date = date.today().isoformat()

        start_dt = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date[:10], "%Y-%m-%d").date()

        records = []
        current = start_dt

        while current <= end_dt:
            # Skip weekends
            if current.weekday() < 5:
                record = self._fetch_single_day(current)
                if record is not None:
                    records.append(record)

            current += timedelta(days=1)

        logger.info(f"Fetched {len(records)} FII/DII records")
        return records

    def _fetch_single_day(self, trade_date: date) -> dict | None:
        """Fetch FII/DII data for a single trading day."""
        date_str_nse = trade_date.strftime("%d-%m-%Y")

        for attempt in range(self.max_retries):
            try:
                from nselib import derivatives
                data = derivatives.participant_wise_trading_volume(date_str_nse)

                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                    return None

                return self._parse_participant_data(data, trade_date.isoformat())

            except ImportError:
                logger.warning("nselib not available, skipping FII/DII fetch")
                return None
            except Exception as e:
                err_msg = str(e)
                if "No data available" in err_msg:
                    # Holiday or no data — skip silently
                    return None
                logger.debug(f"FII/DII fetch for {date_str_nse} attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def _parse_participant_data(self, df: pd.DataFrame, date_str: str) -> dict | None:
        """Parse participant_wise_trading_volume DataFrame into FII/DII net activity."""
        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Find client type column
        type_col = None
        for candidate in ["client_type", "category", "client"]:
            if candidate in df.columns:
                type_col = candidate
                break
        if type_col is None:
            logger.debug(f"No client type column found. Columns: {list(df.columns)}")
            return None

        # Net position = Total Long - Total Short
        long_col = None
        short_col = None
        for col in df.columns:
            if "total_long" in col:
                long_col = col
            elif "total_short" in col:
                short_col = col

        fii_net = 0.0
        dii_net = 0.0

        for _, row in df.iterrows():
            client_type = str(row.get(type_col, "")).upper().strip()

            if long_col and short_col:
                net = _to_float(row.get(long_col, 0)) - _to_float(row.get(short_col, 0))
            else:
                # Fallback: sum all long columns minus all short columns
                longs = sum(
                    _to_float(row.get(c, 0))
                    for c in df.columns if "long" in c and c != type_col
                )
                shorts = sum(
                    _to_float(row.get(c, 0))
                    for c in df.columns if "short" in c and c != type_col
                )
                net = longs - shorts

            if client_type == "FII" or client_type == "FPI":
                fii_net = net
            elif client_type == "DII":
                dii_net = net

        return {
            "date": date_str,
            "fii_net_buy": fii_net,
            "dii_net_buy": dii_net,
        }

    def fetch_recent_fii_dii(self, lookback_days: int = 5) -> list[dict]:
        """Fetch recent FII/DII data for the daily pipeline."""
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=lookback_days)).isoformat()
        return self.fetch_fii_dii(start_date, end_date)


def _to_float(val) -> float:
    """Safely convert a value to float."""
    if val is None:
        return 0.0
    try:
        if isinstance(val, str):
            val = val.replace(",", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return 0.0
