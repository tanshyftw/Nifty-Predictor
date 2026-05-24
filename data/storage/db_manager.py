"""
SQLite database manager for the Nifty Predictor system.
Handles schema creation, connection management, and core CRUD operations.
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
from loguru import logger


SCHEMA_SQL = """
-- Core OHLCV data per stock per day
CREATE TABLE IF NOT EXISTS stock_ohlcv (
    date       TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    adj_close  REAL,
    volume     INTEGER,
    PRIMARY KEY (date, symbol)
);

-- Computed features per stock per day (stored as JSON blob)
CREATE TABLE IF NOT EXISTS stock_features (
    date         TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    feature_json TEXT,
    PRIMARY KEY (date, symbol)
);

-- Fundamental data per stock (updated weekly)
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol          TEXT NOT NULL,
    updated_date    TEXT NOT NULL,
    trailing_pe     REAL,
    forward_pe      REAL,
    price_to_book   REAL,
    market_cap      REAL,
    dividend_yield  REAL,
    roe             REAL,
    debt_to_equity  REAL,
    earnings_growth REAL,
    revenue_growth  REAL,
    profit_margin   REAL,
    sector          TEXT,
    industry        TEXT,
    PRIMARY KEY (symbol, updated_date)
);

-- Index and macro data per day
CREATE TABLE IF NOT EXISTS macro_data (
    date              TEXT PRIMARY KEY,
    nifty50_close     REAL,
    nifty50_ret       REAL,
    bank_nifty_ret    REAL,
    india_vix         REAL,
    india_vix_change  REAL,
    sp500_ret         REAL,
    nasdaq_ret        REAL,
    dow_ret           REAL,
    ftse_ret          REAL,
    nikkei_ret        REAL,
    hangseng_ret      REAL,
    usdinr            REAL,
    usdinr_change     REAL,
    fii_net_buy       REAL,
    dii_net_buy       REAL
);

-- News sentiment per stock per day
CREATE TABLE IF NOT EXISTS news_sentiment (
    date                  TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    news_count            INTEGER,
    vader_compound_mean   REAL,
    vader_compound_max    REAL,
    vader_compound_min    REAL,
    vader_positive_ratio  REAL,
    vader_negative_ratio  REAL,
    textblob_polarity     REAL,
    textblob_subjectivity REAL,
    PRIMARY KEY (date, symbol)
);

-- Model predictions and signals (audit trail)
CREATE TABLE IF NOT EXISTS predictions (
    date           TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    prob_up        REAL,
    prob_down      REAL,
    prob_flat      REAL,
    signal         TEXT,
    confidence     REAL,
    position_size  REAL,
    PRIMARY KEY (date, symbol)
);

-- Actual outcomes for backtesting
CREATE TABLE IF NOT EXISTS outcomes (
    date           TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    actual_ret     REAL,
    actual_class   INTEGER,
    PRIMARY KEY (date, symbol)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON stock_ohlcv(symbol);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON stock_ohlcv(date);
CREATE INDEX IF NOT EXISTS idx_features_date ON stock_features(date);
CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_data(date);
CREATE INDEX IF NOT EXISTS idx_sentiment_date ON news_sentiment(date);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date);
"""


class DBManager:
    """SQLite database manager with connection pooling and schema management."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.debug(f"Database initialized at {self.db_path}")

    @contextmanager
    def connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_ohlcv(self, records: list[dict]):
        """Insert OHLCV records. Skips duplicates."""
        if not records:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO stock_ohlcv
                   (date, symbol, open, high, low, close, adj_close, volume)
                   VALUES (:date, :symbol, :open, :high, :low, :close, :adj_close, :volume)""",
                records,
            )
        logger.debug(f"Inserted {len(records)} OHLCV records")

    def insert_macro(self, records: list[dict]):
        """Insert macro data records. Updates on conflict."""
        if not records:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO macro_data
                   (date, nifty50_close, nifty50_ret, bank_nifty_ret,
                    india_vix, india_vix_change, sp500_ret, nasdaq_ret, dow_ret,
                    ftse_ret, nikkei_ret, hangseng_ret, usdinr, usdinr_change,
                    fii_net_buy, dii_net_buy)
                   VALUES (:date, :nifty50_close, :nifty50_ret, :bank_nifty_ret,
                    :india_vix, :india_vix_change, :sp500_ret, :nasdaq_ret, :dow_ret,
                    :ftse_ret, :nikkei_ret, :hangseng_ret, :usdinr, :usdinr_change,
                    :fii_net_buy, :dii_net_buy)""",
                records,
            )

    def insert_fundamentals(self, records: list[dict]):
        """Insert fundamental data records."""
        if not records:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO fundamentals
                   (symbol, updated_date, trailing_pe, forward_pe, price_to_book,
                    market_cap, dividend_yield, roe, debt_to_equity,
                    earnings_growth, revenue_growth, profit_margin, sector, industry)
                   VALUES (:symbol, :updated_date, :trailing_pe, :forward_pe, :price_to_book,
                    :market_cap, :dividend_yield, :roe, :debt_to_equity,
                    :earnings_growth, :revenue_growth, :profit_margin, :sector, :industry)""",
                records,
            )

    def insert_sentiment(self, records: list[dict]):
        """Insert news sentiment records."""
        if not records:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO news_sentiment
                   (date, symbol, news_count, vader_compound_mean, vader_compound_max,
                    vader_compound_min, vader_positive_ratio, vader_negative_ratio,
                    textblob_polarity, textblob_subjectivity)
                   VALUES (:date, :symbol, :news_count, :vader_compound_mean,
                    :vader_compound_max, :vader_compound_min, :vader_positive_ratio,
                    :vader_negative_ratio, :textblob_polarity, :textblob_subjectivity)""",
                records,
            )

    def insert_features(self, date: str, symbol: str, features: dict):
        """Insert computed features as JSON."""
        with self.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO stock_features (date, symbol, feature_json)
                   VALUES (?, ?, ?)""",
                (date, symbol, json.dumps(features)),
            )

    def insert_features_batch(self, rows: list[tuple]):
        """Bulk insert features. Each row is (date, symbol, features_dict)."""
        if not rows:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO stock_features (date, symbol, feature_json)
                   VALUES (?, ?, ?)""",
                [(d, s, json.dumps(f)) for d, s, f in rows],
            )

    def get_cached_feature_dates(self, start_date: str, end_date: str) -> set[str]:
        """Return set of dates that already have features for ALL 50 stocks cached."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT date, COUNT(DISTINCT symbol) as n
                   FROM stock_features
                   WHERE date >= ? AND date <= ?
                   GROUP BY date
                   HAVING n >= 40""",
                (start_date, end_date),
            ).fetchall()
            return {r["date"] for r in rows}

    def get_all_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load all cached feature rows for a date range into a DataFrame."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT date, symbol, feature_json
                   FROM stock_features
                   WHERE date >= ? AND date <= ?
                   ORDER BY date, symbol""",
                (start_date, end_date),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            d = json.loads(r["feature_json"])
            d["date"] = r["date"]
            d["symbol"] = r["symbol"]
            records.append(d)
        return pd.DataFrame(records)

    def insert_predictions(self, records: list[dict]):
        """Insert model predictions."""
        if not records:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO predictions
                   (date, symbol, prob_up, prob_down, prob_flat,
                    signal, confidence, position_size)
                   VALUES (:date, :symbol, :prob_up, :prob_down, :prob_flat,
                    :signal, :confidence, :position_size)""",
                records,
            )

    def insert_outcomes(self, records: list[dict]):
        """Insert actual outcomes for backtesting."""
        if not records:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO outcomes
                   (date, symbol, actual_ret, actual_class)
                   VALUES (:date, :symbol, :actual_ret, :actual_class)""",
                records,
            )

    def get_ohlcv(self, symbol: str, start_date: str = None,
                  end_date: str = None) -> list[dict]:
        """Retrieve OHLCV data for a symbol within a date range."""
        query = "SELECT * FROM stock_ohlcv WHERE symbol = ?"
        params = [symbol]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"

        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_all_ohlcv(self, start_date: str = None,
                      end_date: str = None) -> list[dict]:
        """Retrieve OHLCV data for all symbols within a date range."""
        query = "SELECT * FROM stock_ohlcv WHERE 1=1"
        params = []
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date, symbol"

        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_macro(self, start_date: str = None,
                  end_date: str = None) -> list[dict]:
        """Retrieve macro data within a date range."""
        query = "SELECT * FROM macro_data WHERE 1=1"
        params = []
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"

        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_latest_fundamentals(self, symbol: str) -> dict | None:
        """Get most recent fundamentals for a symbol."""
        with self.connect() as conn:
            row = conn.execute(
                """SELECT * FROM fundamentals WHERE symbol = ?
                   ORDER BY updated_date DESC LIMIT 1""",
                (symbol,),
            ).fetchone()
            return dict(row) if row else None

    def get_sentiment(self, symbol: str, date: str) -> dict | None:
        """Get sentiment data for a symbol on a specific date."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM news_sentiment WHERE symbol = ? AND date = ?",
                (symbol, date),
            ).fetchone()
            return dict(row) if row else None

    def get_features(self, symbol: str, date: str) -> dict | None:
        """Get computed features for a symbol on a specific date."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT feature_json FROM stock_features WHERE symbol = ? AND date = ?",
                (symbol, date),
            ).fetchone()
            if row:
                return json.loads(row["feature_json"])
            return None

    def get_latest_date(self, table: str = "stock_ohlcv") -> str | None:
        """Get the most recent date in a table."""
        with self.connect() as conn:
            row = conn.execute(f"SELECT MAX(date) as max_date FROM {table}").fetchone()
            return row["max_date"] if row else None

    def get_predictions(self, date: str) -> list[dict]:
        """Get all predictions for a specific date."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE date = ? ORDER BY confidence DESC",
                (date,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_previous_predictions(self, n_days: int = 5) -> list[dict]:
        """Get predictions from the last n trading days for performance tracking."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.*, o.actual_ret, o.actual_class
                   FROM predictions p
                   LEFT JOIN outcomes o ON p.date = o.date AND p.symbol = o.symbol
                   WHERE p.date IN (
                       SELECT DISTINCT date FROM predictions ORDER BY date DESC LIMIT ?
                   )
                   ORDER BY p.date DESC, p.confidence DESC""",
                (n_days,),
            ).fetchall()
            return [dict(r) for r in rows]
