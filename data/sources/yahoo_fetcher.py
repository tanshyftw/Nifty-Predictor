"""
Yahoo Finance data fetcher using yfinance.
Handles OHLCV data, fundamentals, and earnings calendar.
Compatible with yfinance 0.2.x and 1.x APIs.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger

from config.nifty50_tickers import (
    get_yahoo_tickers,
    get_symbols,
    symbol_to_yahoo,
    NIFTY50_STOCKS,
)

YFINANCE_TIMEOUT_SECONDS = 8


class YahooFetcher:
    """Fetches stock data from Yahoo Finance."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_ohlcv_batch(
        self, start_date: str, end_date: str = None
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all Nifty 50 stocks in a single batch call.
        Returns dict mapping NSE symbol -> DataFrame.
        Falls back to individual downloads if batch parsing fails.
        """
        tickers_list = get_yahoo_tickers()
        tickers_str = " ".join(tickers_list)
        if end_date is None:
            end_date = (date.today() + timedelta(days=1)).isoformat()

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching OHLCV batch (attempt {attempt + 1})")
                data = yf.download(
                    tickers_str,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    threads=True,
                    timeout=YFINANCE_TIMEOUT_SECONDS,
                )

                if data.empty:
                    logger.warning("Empty batch download result")
                    break

                result = {}
                for symbol, (yahoo_ticker, _, _) in NIFTY50_STOCKS.items():
                    stock_df = self._extract_stock_from_batch(data, yahoo_ticker)
                    if stock_df is not None and not stock_df.empty:
                        result[symbol] = stock_df

                if len(result) >= 10:
                    logger.info(f"Fetched OHLCV for {len(result)} stocks")
                    return result
                else:
                    logger.warning(
                        f"Only {len(result)} stocks parsed from batch, "
                        f"falling back to individual downloads"
                    )
                    break

            except Exception as e:
                logger.warning(f"Batch fetch attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        # Fallback: download stocks individually
        return self._fetch_ohlcv_individually(start_date, end_date)

    def _extract_stock_from_batch(
        self, data: pd.DataFrame, yahoo_ticker: str
    ) -> pd.DataFrame | None:
        """Extract a single stock's OHLCV from a batch download DataFrame.
        Handles both old and new yfinance MultiIndex formats."""
        stock_df = None

        try:
            if isinstance(data.columns, pd.MultiIndex):
                level_0 = data.columns.get_level_values(0).unique().tolist()
                level_1 = data.columns.get_level_values(1).unique().tolist()

                if yahoo_ticker in level_0:
                    # Format: (Ticker, Field) — group_by="ticker"
                    stock_df = data[yahoo_ticker].copy()
                elif yahoo_ticker in level_1:
                    # Format: (Field, Ticker) — default yfinance MultiIndex
                    stock_df = data.xs(yahoo_ticker, level=1, axis=1).copy()
                else:
                    # Case-insensitive fallback
                    ticker_upper = yahoo_ticker.upper()
                    for val in level_0:
                        if str(val).upper() == ticker_upper:
                            stock_df = data[val].copy()
                            break
                    if stock_df is None:
                        for val in level_1:
                            if str(val).upper() == ticker_upper:
                                stock_df = data.xs(val, level=1, axis=1).copy()
                                break
            else:
                # Single ticker download — flat columns
                if len(NIFTY50_STOCKS) == 1:
                    stock_df = data.copy()
        except Exception:
            return None

        if stock_df is None or stock_df.empty:
            return None

        # Flatten any remaining MultiIndex columns
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = [
                str(c[0]) if isinstance(c, tuple) else str(c)
                for c in stock_df.columns
            ]

        # Find Close column (case-insensitive)
        close_col = None
        for col in stock_df.columns:
            if str(col).lower() == "close":
                close_col = col
                break

        if close_col is None:
            return None

        stock_df = stock_df.dropna(subset=[close_col])
        if stock_df.empty:
            return None

        stock_df = stock_df.reset_index()
        stock_df.columns = [
            str(c).lower().replace(" ", "_") for c in stock_df.columns
        ]

        # Ensure adj_close exists
        if "adj_close" not in stock_df.columns:
            if "adj close" in stock_df.columns:
                stock_df = stock_df.rename(columns={"adj close": "adj_close"})
            elif "close" in stock_df.columns:
                # In newer yfinance with auto_adjust=True (default),
                # Close IS the adjusted close
                stock_df["adj_close"] = stock_df["close"]

        return stock_df

    def _fetch_ohlcv_individually(
        self, start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """Fallback: download each stock individually with rate limiting."""
        logger.info("Downloading stocks individually (this may take a few minutes)...")
        result = {}
        symbols = list(NIFTY50_STOCKS.keys())

        for i, symbol in enumerate(symbols):
            if i > 0 and i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(symbols)} stocks downloaded")
                time.sleep(1)

            df = self.fetch_ohlcv_single(symbol, start_date, end_date)
            if df is not None and not df.empty:
                result[symbol] = df

        logger.info(f"Individual download: fetched {len(result)}/{len(symbols)} stocks")
        return result

    def fetch_ohlcv_single(
        self, symbol: str, start_date: str, end_date: str = None
    ) -> pd.DataFrame | None:
        """Fetch OHLCV data for a single stock."""
        yahoo_ticker = symbol_to_yahoo(symbol)
        if end_date is None:
            end_date = (date.today() + timedelta(days=1)).isoformat()

        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    yahoo_ticker,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    progress=False,
                    timeout=YFINANCE_TIMEOUT_SECONDS,
                )
                if data.empty:
                    return None

                # Flatten MultiIndex columns if present (single-ticker download)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [
                        str(c[0]) if isinstance(c, tuple) else str(c)
                        for c in data.columns
                    ]

                data = data.reset_index()
                data.columns = [
                    str(c).lower().replace(" ", "_") for c in data.columns
                ]
                if "adj_close" not in data.columns:
                    if "adj close" in data.columns:
                        data = data.rename(columns={"adj close": "adj_close"})
                    elif "close" in data.columns:
                        data["adj_close"] = data["close"]
                return data

            except Exception as e:
                logger.warning(
                    f"Fetch {symbol} attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def fetch_recent_ohlcv(self, period: str = "5d") -> dict[str, pd.DataFrame]:
        """Fetch recent OHLCV data (last N days) for all stocks."""
        tickers_str = " ".join(get_yahoo_tickers())

        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    tickers_str,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    threads=True,
                    timeout=YFINANCE_TIMEOUT_SECONDS,
                )

                if data.empty:
                    break

                result = {}
                for symbol, (yahoo_ticker, _, _) in NIFTY50_STOCKS.items():
                    stock_df = self._extract_stock_from_batch(data, yahoo_ticker)
                    if stock_df is not None and not stock_df.empty:
                        result[symbol] = stock_df

                if result:
                    return result
                break

            except Exception as e:
                logger.warning(f"Recent fetch attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        # Fallback to individual downloads
        start = (date.today() - timedelta(days=int(period.replace("d", "")) + 2)).isoformat()
        end = (date.today() + timedelta(days=1)).isoformat()
        return self._fetch_ohlcv_individually(start, end)

    def fetch_fundamentals(self, symbol: str) -> dict | None:
        """Fetch fundamental data for a single stock."""
        yahoo_ticker = symbol_to_yahoo(symbol)

        try:
            ticker = yf.Ticker(yahoo_ticker)
            info = ticker.info

            return {
                "symbol": symbol,
                "updated_date": date.today().isoformat(),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "market_cap": info.get("marketCap"),
                "dividend_yield": info.get("dividendYield"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margin": info.get("profitMargins"),
                "sector": info.get("sector", NIFTY50_STOCKS.get(symbol, (None, None, "Unknown"))[2]),
                "industry": info.get("industry", ""),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
            return None

    def fetch_all_fundamentals(self, delay: float = 1.0) -> list[dict]:
        """Fetch fundamentals for all Nifty 50 stocks with rate limiting."""
        results = []
        symbols = get_symbols()

        for i, symbol in enumerate(symbols):
            logger.info(f"Fetching fundamentals {i + 1}/{len(symbols)}: {symbol}")
            data = self.fetch_fundamentals(symbol)
            if data:
                results.append(data)
            if i < len(symbols) - 1:
                time.sleep(delay)

        logger.info(f"Fetched fundamentals for {len(results)}/{len(symbols)} stocks")
        return results

    @staticmethod
    def _read_fast_info(fi, attrs: tuple[str, ...]):
        """Pull the first non-null value from a yfinance fast_info object.

        fast_info exposes both attribute and dict-style access depending on
        the yfinance version, and the field names changed (`last_price` vs
        `lastPrice`). Try every variant before giving up.
        """
        for a in attrs:
            try:
                v = fi.get(a) if hasattr(fi, "get") else None
            except Exception:
                v = None
            if v is None:
                v = getattr(fi, a, None)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    def fetch_live_quotes(
        self, yahoo_tickers: list[str], max_workers: int = 16
    ) -> dict[str, dict]:
        """Fetch live last price + prior-day close via Yahoo's fast_info.

        Why this matters: `yf.download(period="1mo", interval="1d")` returns
        the daily candle, and Yahoo's daily candle for Indian tickers can lag
        by a session — meaning iloc[-1] is yesterday's close and the dashboard
        shows yesterday's DoD. fast_info hits the live-quote endpoint instead,
        which is updated through the session, so it's the right source for
        "today's price" and "previous-day's settled close".

        Returns: {yahoo_ticker: {"last_price", "previous_close"}}.
        Tickers that fail are silently dropped — the caller should fall back
        to the historical OHLCV path.
        """
        from concurrent.futures import ThreadPoolExecutor

        last_attrs = (
            "last_price", "lastPrice",
            "regular_market_price", "regularMarketPrice",
        )
        prev_attrs = (
            "previous_close", "previousClose",
            "regular_market_previous_close", "regularMarketPreviousClose",
        )

        def _one(yt: str):
            try:
                fi = yf.Ticker(yt).fast_info
                last = self._read_fast_info(fi, last_attrs)
                prev = self._read_fast_info(fi, prev_attrs)
                if last and prev and last > 0 and prev > 0:
                    return yt, {"last_price": last, "previous_close": prev}
            except Exception:
                pass
            return yt, None

        out: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for yt, q in ex.map(_one, yahoo_tickers):
                if q:
                    out[yt] = q
        return out

    def ohlcv_to_records(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Convert a DataFrame to list of dicts for DB insertion."""
        records = []
        for _, row in df.iterrows():
            date_val = row.get("date")
            if hasattr(date_val, "strftime"):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)[:10]

            records.append({
                "date": date_str,
                "symbol": symbol,
                "open": float(row.get("open", 0)) if pd.notna(row.get("open")) else None,
                "high": float(row.get("high", 0)) if pd.notna(row.get("high")) else None,
                "low": float(row.get("low", 0)) if pd.notna(row.get("low")) else None,
                "close": float(row.get("close", 0)) if pd.notna(row.get("close")) else None,
                "adj_close": float(row.get("adj_close", row.get("close", 0)))
                    if pd.notna(row.get("adj_close", row.get("close"))) else None,
                "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else None,
            })
        return records
