"""
Global indices and currency data fetcher.
Fetches S&P 500, NASDAQ, Dow, FTSE, Nikkei, Hang Seng, India VIX, USD/INR.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

from config.nifty50_tickers import GLOBAL_INDICES


class GlobalFetcher:
    """Fetches global index and currency data from Yahoo Finance."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_all_indices(
        self, start_date: str, end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch historical data for all global indices and compute daily returns.
        Returns a DataFrame indexed by date with return columns for each index.
        """
        if end_date is None:
            end_date = (date.today() + timedelta(days=1)).isoformat()

        tickers = list(GLOBAL_INDICES.values())
        ticker_str = " ".join(tickers)

        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    ticker_str,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )

                records = []
                # Extract close prices for each index
                close_dict = {}
                for name, ticker in GLOBAL_INDICES.items():
                    try:
                        if len(GLOBAL_INDICES) == 1:
                            close_series = data["Close"]
                        else:
                            close_series = data[ticker]["Close"]
                        close_dict[name] = close_series
                    except (KeyError, TypeError):
                        logger.warning(f"No data for {name} ({ticker})")
                        continue

                if not close_dict:
                    logger.error("No index data fetched")
                    return pd.DataFrame()

                close_df = pd.DataFrame(close_dict)
                returns_df = close_df.pct_change()

                # Build macro records per date
                for idx in close_df.index:
                    date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]

                    record = {"date": date_str}

                    # Nifty 50
                    record["nifty50_close"] = _safe_float(close_df, idx, "NIFTY50")
                    record["nifty50_ret"] = _safe_float(returns_df, idx, "NIFTY50")

                    # Bank Nifty
                    record["bank_nifty_ret"] = _safe_float(returns_df, idx, "BANKNIFTY")

                    # India VIX
                    record["india_vix"] = _safe_float(close_df, idx, "INDIAVIX")
                    record["india_vix_change"] = _safe_float(returns_df, idx, "INDIAVIX")

                    # Global indices
                    record["sp500_ret"] = _safe_float(returns_df, idx, "SP500")
                    record["nasdaq_ret"] = _safe_float(returns_df, idx, "NASDAQ")
                    record["dow_ret"] = _safe_float(returns_df, idx, "DOW")
                    record["ftse_ret"] = _safe_float(returns_df, idx, "FTSE")
                    record["nikkei_ret"] = _safe_float(returns_df, idx, "NIKKEI")
                    record["hangseng_ret"] = _safe_float(returns_df, idx, "HANGSENG")

                    # Currency
                    record["usdinr"] = _safe_float(close_df, idx, "USDINR")
                    record["usdinr_change"] = _safe_float(returns_df, idx, "USDINR")

                    # FII/DII will be filled by nse_fetcher
                    record["fii_net_buy"] = None
                    record["dii_net_buy"] = None

                    records.append(record)

                logger.info(f"Fetched {len(records)} days of global index data")
                return pd.DataFrame(records)

            except Exception as e:
                logger.warning(f"Global fetch attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        logger.error("All global index fetch attempts failed")
        return pd.DataFrame()

    def fetch_recent_indices(self, period: str = "5d") -> list[dict]:
        """Fetch recent global index data for the daily pipeline."""
        tickers = list(GLOBAL_INDICES.values())
        ticker_str = " ".join(tickers)

        try:
            data = yf.download(
                ticker_str,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            close_dict = {}
            for name, ticker in GLOBAL_INDICES.items():
                try:
                    if len(GLOBAL_INDICES) == 1:
                        close_dict[name] = data["Close"]
                    else:
                        close_dict[name] = data[ticker]["Close"]
                except (KeyError, TypeError):
                    continue

            close_df = pd.DataFrame(close_dict)
            returns_df = close_df.pct_change()

            records = []
            for idx in close_df.index:
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                record = {
                    "date": date_str,
                    "nifty50_close": _safe_float(close_df, idx, "NIFTY50"),
                    "nifty50_ret": _safe_float(returns_df, idx, "NIFTY50"),
                    "bank_nifty_ret": _safe_float(returns_df, idx, "BANKNIFTY"),
                    "india_vix": _safe_float(close_df, idx, "INDIAVIX"),
                    "india_vix_change": _safe_float(returns_df, idx, "INDIAVIX"),
                    "sp500_ret": _safe_float(returns_df, idx, "SP500"),
                    "nasdaq_ret": _safe_float(returns_df, idx, "NASDAQ"),
                    "dow_ret": _safe_float(returns_df, idx, "DOW"),
                    "ftse_ret": _safe_float(returns_df, idx, "FTSE"),
                    "nikkei_ret": _safe_float(returns_df, idx, "NIKKEI"),
                    "hangseng_ret": _safe_float(returns_df, idx, "HANGSENG"),
                    "usdinr": _safe_float(close_df, idx, "USDINR"),
                    "usdinr_change": _safe_float(returns_df, idx, "USDINR"),
                    "fii_net_buy": None,
                    "dii_net_buy": None,
                }
                records.append(record)

            return records

        except Exception as e:
            logger.error(f"Failed to fetch recent indices: {e}")
            return []


def _safe_float(df: pd.DataFrame, idx, col: str) -> float | None:
    """Safely extract a float value from a DataFrame, handling NaN."""
    try:
        val = df.loc[idx, col]
        if pd.isna(val):
            return None
        return float(val)
    except (KeyError, TypeError, ValueError):
        return None
